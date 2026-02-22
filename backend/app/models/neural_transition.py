"""
Neural transition model for nonlinear macro-state dynamics.

Replaces the linear transition  x_{t+1} = A x_t + B u_t + noise
with a learned MLP:             x_{t+1} = f_θ(x_t, u_t, regime_t) + noise

The model is trained on smoothed Kalman state trajectories produced
by the existing estimation pipeline, creating a supervised dataset of
(x_t, u_t, regime_t) → x_{t+1} transitions.

Architecture:
    Input  → Linear(in, H) → GELU → [LayerNorm] →
             Linear(H, H)  → GELU → [LayerNorm] →
             Linear(H, state_dim)

Inherits from ``BaseStateTransition`` so it can be swapped in-place
with ``LinearStateTransition`` via config toggle.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from app.config import settings
from app.models.base_transition import BaseStateTransition

logger = logging.getLogger(__name__)

# ── Lazy torch import (graceful fallback) ─────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — neural transition model unavailable.")


# ──────────────────────────────────────────────────────────────────
# Dataset construction
# ──────────────────────────────────────────────────────────────────

def build_transition_dataset(
    smoothed_states: np.ndarray,
    actions: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (input, target) arrays from a smoothed state trajectory.

    Args:
        smoothed_states: (T, state_dim) array from Kalman smoother.
        actions: (T,) scalar actions in bps.  If *None*, zeros are used.
        regimes: (T,) integer regime labels {0,1,2}.  If *None*, zeros
                 are used and one-hot will be all-zero except index 0.

    Returns:
        inputs:  (T-1, state_dim + 1 + 3)  — [x_t, u_t, regime_onehot]
        targets: (T-1, state_dim)           — x_{t+1}
    """
    T, state_dim = smoothed_states.shape

    if actions is None:
        actions = np.zeros(T, dtype=np.float32)
    if regimes is None:
        regimes = np.zeros(T, dtype=np.int64)

    inputs_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    for t in range(T - 1):
        x_t = smoothed_states[t].astype(np.float32)
        u_t = np.array([actions[t]], dtype=np.float32)

        # One-hot regime
        regime_oh = np.zeros(3, dtype=np.float32)
        regime_oh[int(regimes[t])] = 1.0

        inp = np.concatenate([x_t, u_t, regime_oh])
        inputs_list.append(inp)
        targets_list.append(smoothed_states[t + 1].astype(np.float32))

    return np.array(inputs_list), np.array(targets_list)


def generate_synthetic_actions(T: int, seed: int = 42) -> np.ndarray:
    """
    Generate plausible synthetic policy actions for dataset augmentation.

    Produces a mix of hold (0), ease (−50), and tighten (+50) with
    occasional custom magnitudes drawn uniformly from [−100, 100].
    """
    rng = np.random.default_rng(seed)
    actions = np.zeros(T, dtype=np.float32)
    for i in range(T):
        r = rng.random()
        if r < 0.5:
            actions[i] = 0.0      # hold
        elif r < 0.7:
            actions[i] = -50.0    # ease
        elif r < 0.9:
            actions[i] = 50.0     # tighten
        else:
            actions[i] = rng.uniform(-100, 100)
    return actions


def assign_regimes(
    stress_series: np.ndarray,
    fragile_threshold: float = 0.5,
    crisis_threshold: float = 1.5,
) -> np.ndarray:
    """Classify each timestep into a regime based on z-scored stress."""
    mean = np.mean(stress_series)
    std = np.std(stress_series) + 1e-10
    z = (stress_series - mean) / std
    regimes = np.zeros(len(z), dtype=np.int64)
    regimes[z >= fragile_threshold] = 1
    regimes[z >= crisis_threshold] = 2
    return regimes


# ──────────────────────────────────────────────────────────────────
# MLP Model
# ──────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class MacroTransitionMLP(nn.Module):
        """
        Two-layer MLP that predicts next latent state.

        Input:  [x_t (state_dim), u_t (1), regime_onehot (3)] → state_dim+4
        Output: x_{t+1} (state_dim)
        """

        def __init__(
            self,
            input_dim: int = 8,   # 4 state + 1 action + 3 regime
            hidden_dim: int = 64,
            output_dim: int = 4,  # 4-dim state
            use_layer_norm: bool = True,
        ):
            super().__init__()
            layers: list[nn.Module] = []

            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.GELU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.Linear(hidden_dim, output_dim))

            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x)


# ──────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────

def train_neural_transition(
    inputs: np.ndarray,
    targets: np.ndarray,
    hidden_dim: int | None = None,
    use_layer_norm: bool | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    epochs: int | None = None,
    patience: int | None = None,
    train_split: float | None = None,
    seed: int | None = None,
    checkpoint_path: Path | str | None = None,
) -> dict:
    """
    Train the neural transition model and save a checkpoint.

    Uses a temporal 80/20 split (NOT random) and MSE loss + L2 decay.

    Returns:
        dict with train_losses, val_losses, best_epoch, checkpoint_path.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for neural transition training.")

    # Resolve defaults from settings
    hidden_dim = hidden_dim or settings.neural_hidden_dim
    use_layer_norm = use_layer_norm if use_layer_norm is not None else settings.neural_use_layer_norm
    lr = lr or settings.neural_lr
    weight_decay = weight_decay or settings.neural_weight_decay
    epochs = epochs or settings.neural_epochs
    patience = patience or settings.neural_early_stopping_patience
    train_split = train_split or settings.neural_train_split
    seed = seed if seed is not None else settings.neural_seed
    checkpoint_path = Path(checkpoint_path or settings.neural_checkpoint_path)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Temporal split
    T = len(inputs)
    split_idx = int(T * train_split)

    X_train = torch.tensor(inputs[:split_idx], dtype=torch.float32)
    Y_train = torch.tensor(targets[:split_idx], dtype=torch.float32)
    X_val = torch.tensor(inputs[split_idx:], dtype=torch.float32)
    Y_val = torch.tensor(targets[split_idx:], dtype=torch.float32)

    input_dim = inputs.shape[1]
    output_dim = targets.shape[1]

    model = MacroTransitionMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        use_layer_norm=use_layer_norm,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=256, shuffle=True
    )

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0

    logger.info(
        f"Training neural transition: {T} samples "
        f"(train={split_idx}, val={T - split_idx}), "
        f"input_dim={input_dim}, hidden={hidden_dim}, epochs={epochs}"
    )

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Save best
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": output_dim,
                "use_layer_norm": use_layer_norm,
                "epoch": epoch,
                "val_loss": val_loss,
            }, checkpoint_path)
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs} — "
                f"train={avg_train:.6f}  val={val_loss:.6f}  "
                f"best={best_val:.6f}@{best_epoch+1}"
            )

        if epochs_no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1} (patience={patience})")
            break

    logger.info(
        f"Neural transition training complete. "
        f"Best val loss={best_val:.6f} at epoch {best_epoch+1}. "
        f"Checkpoint: {checkpoint_path}"
    )

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "checkpoint_path": str(checkpoint_path),
    }


# ──────────────────────────────────────────────────────────────────
# Inference wrapper
# ──────────────────────────────────────────────────────────────────

class NeuralTransitionModel(BaseStateTransition):
    """
    Stateless wrapper for loading and querying the trained MLP.

    Inherits from ``BaseStateTransition`` for seamless swap with
    ``LinearStateTransition``.

    Usage:
        model = NeuralTransitionModel()
        model.load()                       # loads from checkpoint
        x_next = model.predict(x_t, action=0.25, regime=1)
    """

    def __init__(self, checkpoint_path: Path | str | None = None):
        self.checkpoint_path = Path(
            checkpoint_path or settings.neural_checkpoint_path
        )
        self._model: Optional["MacroTransitionMLP"] = None
        self._loaded = False
        self._output_dim: int = settings.state_dim  # default 4

    @property
    def state_dim(self) -> int:
        return self._output_dim

    @property
    def is_available(self) -> bool:
        return TORCH_AVAILABLE and self.checkpoint_path.exists()

    def load(self) -> None:
        """Load model weights from checkpoint."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed.")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Neural transition checkpoint not found: {self.checkpoint_path}"
            )

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        self._model = MacroTransitionMLP(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            output_dim=ckpt["output_dim"],
            use_layer_norm=ckpt.get("use_layer_norm", True),
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        self._loaded = True
        self._output_dim = ckpt["output_dim"]
        logger.info(
            f"Neural transition loaded from {self.checkpoint_path} "
            f"(epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f}, "
            f"output_dim={self._output_dim})"
        )

    def predict(
        self,
        x_t: np.ndarray,
        action: float = 0.0,
        regime: int = 0,
    ) -> np.ndarray:
        """
        Predict deterministic next state (without noise).

        Args:
            x_t: current state (state_dim,)
            action: Δ rate in % points
            regime: current regime index {0, 1, 2}

        Returns:
            x_{t+1}: predicted next state (state_dim,)
        """
        if not self._loaded:
            self.load()

        regime_oh = np.zeros(3, dtype=np.float32)
        regime_oh[regime] = 1.0
        inp = np.concatenate([
            x_t.astype(np.float32),
            np.array([action], dtype=np.float32),
            regime_oh,
        ])

        with torch.no_grad():
            t_inp = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
            t_out = self._model(t_inp)
        return t_out.squeeze(0).numpy()


# ──────────────────────────────────────────────────────────────────
# CLI entry-point for standalone training
# ──────────────────────────────────────────────────────────────────

def train_from_pipeline() -> dict:
    """
    End-to-end: run data pipeline → Kalman → build dataset → train MLP.

    Callable from ``python -m app.models.neural_transition``.
    """
    from app.data.pipeline import DataPipeline
    from app.data.manager import DataManager
    from app.models.structure_features import compute_structure_features
    from app.models.kalman_em import KalmanEM

    logger.info("=== Neural transition: building dataset from pipeline ===")

    dm = DataManager()

    # Always force a fresh pipeline run to ensure full historical data.
    # The pipeline will fetch from FRED (or generate synthetic) and rebuild
    # the processed cache, giving us ~8000+ daily rows back to 1990.
    df = dm.download_data(force=True)
    if df is None or len(df) < 500:
        logger.warning(f"Real data too small ({0 if df is None else len(df)} rows) — falling back to synthetic")
        df = dm.download_data(synthetic=True)

    logger.info(f"Training data: {len(df)} daily rows, {df.index.min().date()} → {df.index.max().date()}")

    struct = compute_structure_features(df, window=settings.rolling_cov_window)
    Z_history = struct["Z_history"].values

    kalman = KalmanEM(latent_dim=settings.kalman_latent_dim)
    kalman.fit(Z_history)

    smoothed_3d = kalman.smoothed_means  # (T, 3)

    # Build 4-dim state by appending inflation gap
    smoothed_4d = dm.build_state_matrix(smoothed_3d)

    stress = smoothed_4d[:, 0]
    regimes = assign_regimes(stress)
    actions = generate_synthetic_actions(len(smoothed_4d), seed=settings.neural_seed)

    inputs, targets = build_transition_dataset(smoothed_4d, actions, regimes)
    logger.info(f"Dataset: {inputs.shape[0]} transitions, input_dim={inputs.shape[1]}")

    result = train_neural_transition(inputs, targets)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    train_from_pipeline()
