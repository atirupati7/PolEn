"""
API routes for RL training & evaluation control.

Allows the frontend to:
  - Start / stop PPO training
  - Poll training progress
  - Run RL-vs-heuristic evaluation
  - Get latest training metrics
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/rl", tags=["rl"])

# ── Mutable server-side training state ──────────────────────────

_training_state: dict = {
    "status": "idle",           # idle | training | completed | failed
    "progress": 0.0,           # 0..1
    "total_timesteps": 0,
    "current_timesteps": 0,
    "start_time": None,
    "elapsed_seconds": 0,
    "reward_history": [],       # list of (timestep, mean_reward)
    "loss_history": [],         # list of (timestep, loss_value)
    "error": None,
    "checkpoint_path": None,
}

_eval_cache: Optional[dict] = None


def get_training_state() -> dict:
    return _training_state


# ── Request models ──────────────────────────────────────────────

class TrainRequest(BaseModel):
    total_timesteps: int = Field(default=100_000, ge=10_000, le=5_000_000)
    learning_rate: float = Field(default=3e-4, gt=0, le=0.1)
    seed: int = Field(default=42)
    net_arch: list = Field(default=[64, 64])
    gamma: float = Field(default=0.99, ge=0.9, le=1.0)
    clip_range: float = Field(default=0.2, ge=0.05, le=0.5)
    ent_coef: float = Field(default=0.01, ge=0.0, le=0.1)


class EvalRequest(BaseModel):
    n_episodes: int = Field(default=20, ge=5, le=200)
    seed: int = Field(default=42)


# ── Endpoints ───────────────────────────────────────────────────

@router.get("/status")
async def training_status():
    """Return current training status, progress, and metrics."""
    state = _training_state.copy()
    if state["start_time"] and state["status"] == "training":
        state["elapsed_seconds"] = round(time.time() - state["start_time"], 1)
    return state


@router.post("/train")
async def start_training(req: TrainRequest):
    """
    Start PPO training in a background thread.
    
    Returns immediately; poll /api/rl/status for progress.
    """
    global _training_state

    if _training_state["status"] == "training":
        raise HTTPException(status_code=409, detail="Training already in progress")

    _training_state = {
        "status": "training",
        "progress": 0.0,
        "total_timesteps": req.total_timesteps,
        "current_timesteps": 0,
        "start_time": time.time(),
        "elapsed_seconds": 0,
        "reward_history": [],
        "loss_history": [],
        "error": None,
        "checkpoint_path": None,
    }

    # Run training in a background task
    asyncio.get_event_loop().run_in_executor(
        None,
        _run_training,
        req.total_timesteps,
        req.learning_rate,
        req.seed,
        req.net_arch,
        req.gamma,
        req.clip_range,
        req.ent_coef,
    )

    return {
        "status": "training",
        "message": f"Training started with {req.total_timesteps} timesteps",
    }


@router.post("/stop")
async def stop_training():
    """Request graceful stop of running training."""
    global _training_state
    if _training_state["status"] != "training":
        return {"status": _training_state["status"], "message": "No active training"}
    _training_state["status"] = "stopping"
    return {"status": "stopping", "message": "Stop requested — will finish current batch"}


@router.post("/evaluate")
async def evaluate_models(req: EvalRequest):
    """
    Run full RL-vs-heuristic evaluation.

    Returns detailed comparison metrics.
    """
    global _eval_cache
    try:
        from app.rl.evaluate import evaluate
        result = evaluate(n_episodes=req.n_episodes, seed=req.seed)
        _eval_cache = result
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model checkpoint not found. Train first: {e}",
        )
    except Exception as e:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluate/cached")
async def get_cached_evaluation():
    """Return the last cached evaluation result."""
    if _eval_cache is None:
        raise HTTPException(status_code=404, detail="No evaluation has been run yet")
    return _eval_cache


@router.get("/model/info")
async def model_info():
    """Return info about the current model checkpoint."""
    ckpt = settings.ppo_checkpoint_path
    exists = ckpt.exists()
    size_mb = round(ckpt.stat().st_size / 1_048_576, 2) if exists else 0
    modified = round(ckpt.stat().st_mtime, 0) if exists else None

    return {
        "checkpoint_exists": exists,
        "checkpoint_path": str(ckpt),
        "size_mb": size_mb,
        "last_modified": modified,
        "config": {
            "net_arch": settings.ppo_net_arch,
            "gamma": settings.ppo_gamma,
            "lr": settings.ppo_lr,
            "clip_range": settings.ppo_clip_range,
            "ent_coef": settings.ppo_ent_coef,
            "episode_length": settings.gym_episode_length,
            "max_rate_step": settings.gym_max_rate_step,
            "state_dim": settings.gym_state_dim,
        },
    }


@router.get("/config")
async def rl_config():
    """Return the current RL / reward / environment configuration."""
    return {
        "ppo": {
            "total_timesteps": settings.ppo_total_timesteps,
            "n_steps": settings.ppo_n_steps,
            "batch_size": settings.ppo_batch_size,
            "gamma": settings.ppo_gamma,
            "gae_lambda": settings.ppo_gae_lambda,
            "lr": settings.ppo_lr,
            "clip_range": settings.ppo_clip_range,
            "ent_coef": settings.ppo_ent_coef,
            "net_arch": settings.ppo_net_arch,
            "activation": settings.ppo_activation,
        },
        "reward": {
            "w_stress": settings.reward_w_stress,
            "w_inflation": settings.reward_w_inflation,
            "w_crisis": settings.reward_w_crisis,
            "w_rate_change": settings.reward_w_rate_change,
            "w_taylor": settings.reward_w_taylor,
            "inflation_target": settings.reward_inflation_target,
        },
        "environment": {
            "episode_length": settings.gym_episode_length,
            "max_rate_step": settings.gym_max_rate_step,
            "zero_lower_bound": settings.gym_zero_lower_bound,
            "state_dim": settings.gym_state_dim,
        },
        "transition": {
            "model_type": settings.transition_model_type,
            "inflation_persistence": settings.inflation_persistence,
        },
    }


# ── Background training worker ─────────────────────────────────

def _run_training(
    total_timesteps: int,
    lr: float,
    seed: int,
    net_arch: list,
    gamma: float,
    clip_range: float,
    ent_coef: float,
):
    """Synchronous training worker (runs in thread pool)."""
    global _training_state
    try:
        import torch.nn as nn
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback

        from app.rl.train_ppo import _build_env

        class ProgressCallback(BaseCallback):
            """Reports progress back to _training_state."""

            def __init__(self):
                super().__init__()
                self._last_report = 0

            def _on_step(self) -> bool:
                # Check for stop request
                if _training_state.get("status") == "stopping":
                    logger.info("Training stop requested by user")
                    return False

                current = self.num_timesteps
                _training_state["current_timesteps"] = current
                _training_state["progress"] = min(current / total_timesteps, 1.0)

                # Log every 5000 steps
                if current - self._last_report >= 5000:
                    self._last_report = current
                    # Collect reward info from logger if available
                    if len(self.model.ep_info_buffer) > 0:
                        mean_reward = sum(
                            ep["r"] for ep in self.model.ep_info_buffer
                        ) / len(self.model.ep_info_buffer)
                        _training_state["reward_history"].append(
                            {"timestep": current, "mean_reward": round(float(mean_reward), 4)}
                        )
                    # Collect loss from logger
                    if hasattr(self.model, "logger") and self.model.logger is not None:
                        try:
                            name_to_value = getattr(self.model.logger, "name_to_value", {})
                            loss_val = name_to_value.get("train/loss", None)
                            if loss_val is not None:
                                _training_state["loss_history"].append(
                                    {"timestep": current, "loss": round(float(loss_val), 6)}
                                )
                        except Exception:
                            pass

                return True

        logger.info(f"Background training starting: {total_timesteps} timesteps")
        env = _build_env()

        act_map = {"Tanh": nn.Tanh, "ReLU": nn.ReLU, "GELU": nn.GELU}
        activation_fn = act_map.get(settings.ppo_activation, nn.Tanh)

        model = PPO(
            "MlpPolicy",
            env,
            n_steps=settings.ppo_n_steps,
            batch_size=settings.ppo_batch_size,
            gamma=gamma,
            gae_lambda=settings.ppo_gae_lambda,
            learning_rate=lr,
            clip_range=clip_range,
            ent_coef=ent_coef,
            policy_kwargs=dict(net_arch=net_arch, activation_fn=activation_fn),
            seed=seed,
            verbose=0,
        )

        callback = ProgressCallback()
        model.learn(total_timesteps=total_timesteps, callback=callback)

        # Save checkpoint
        ckpt = settings.ppo_checkpoint_path
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(ckpt).removesuffix(".zip"))

        _training_state["status"] = "completed"
        _training_state["progress"] = 1.0
        _training_state["current_timesteps"] = total_timesteps
        _training_state["elapsed_seconds"] = round(time.time() - _training_state["start_time"], 1)
        _training_state["checkpoint_path"] = str(ckpt)
        logger.info(f"Training completed → {ckpt}")

        env.close()

    except Exception as e:
        logger.exception("Training failed")
        _training_state["status"] = "failed"
        _training_state["error"] = str(e)
        _training_state["elapsed_seconds"] = round(
            time.time() - (_training_state["start_time"] or time.time()), 1
        )
