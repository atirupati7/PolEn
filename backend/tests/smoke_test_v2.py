"""Smoke test for all new modules in the extended macro-policy system."""
import sys

def main():
    errors = []
    
    # 1. Config
    try:
        from app.config import settings
        assert settings.state_dim == 4
        assert settings.kalman_latent_dim == 3
        assert settings.gym_state_dim == 12
        assert "CPIAUCSL" in settings.fred_series
        assert "FEDFUNDS" in settings.fred_series
        print("✓ Config system: all settings loaded correctly")
    except Exception as e:
        errors.append(f"Config: {e}")
        print(f"✗ Config: {e}")

    # 2. DataManager
    try:
        from app.data.manager import DataManager
        dm = DataManager()
        print("✓ DataManager: instantiated")
    except Exception as e:
        errors.append(f"DataManager: {e}")
        print(f"✗ DataManager: {e}")

    # 3. BaseStateTransition
    try:
        from app.models.base_transition import BaseStateTransition
        print("✓ BaseStateTransition: imported")
    except Exception as e:
        errors.append(f"BaseStateTransition: {e}")
        print(f"✗ BaseStateTransition: {e}")

    # 4. LinearStateTransition
    try:
        import numpy as np
        from app.models.linear_transition import LinearStateTransition
        A = np.eye(3) * 0.98
        B = np.array([0.003, 0.006, -0.004])
        lt = LinearStateTransition(A=A, B=B)
        x = np.array([0.1, -0.2, 0.5, 0.01])
        x_next = lt.predict(x, action=0.25, regime=0)
        assert lt.state_dim == 4
        assert lt.A.shape == (4, 4)
        assert x_next.shape == (4,)
        print(f"✓ LinearStateTransition: predict {x.tolist()} → {x_next.tolist()}")
    except Exception as e:
        errors.append(f"LinearStateTransition: {e}")
        print(f"✗ LinearStateTransition: {e}")

    # 5. NeuralTransitionModel inherits BaseStateTransition
    try:
        from app.models.neural_transition import NeuralTransitionModel
        from app.models.base_transition import BaseStateTransition
        assert issubclass(NeuralTransitionModel, BaseStateTransition)
        print("✓ NeuralTransitionModel: inherits BaseStateTransition")
    except Exception as e:
        errors.append(f"NeuralTransitionModel: {e}")
        print(f"✗ NeuralTransitionModel: {e}")

    # 6. RewardModule
    try:
        from app.environment.reward import RewardModule, RewardComponents
        rm = RewardModule()
        rc = rm.compute(stress=0.5, inflation=0.03, crisis_prob=0.0, rate_change=0.1)
        assert isinstance(rc, RewardComponents)
        assert rc.total < 0  # reward should be negative (penalty)
        print(f"✓ RewardModule: total={rc.total:.4f} (stress_pen={rc.stress_penalty:.4f}, infl={rc.inflation_penalty:.4f})")
    except Exception as e:
        errors.append(f"RewardModule: {e}")
        print(f"✗ RewardModule: {e}")

    # 7. MacroPolicyEnvV2
    try:
        from app.environment.macro_env import MacroPolicyEnvV2, GYM_AVAILABLE
        if GYM_AVAILABLE:
            print("✓ MacroPolicyEnvV2: class imported (gymnasium available)")
        else:
            print("⚠ MacroPolicyEnvV2: class imported (gymnasium NOT installed)")
    except Exception as e:
        errors.append(f"MacroPolicyEnvV2: {e}")
        print(f"✗ MacroPolicyEnvV2: {e}")

    # 8. Synthetic data with CPI + FEDFUNDS
    try:
        from app.data.fred_client import generate_synthetic_data
        df = generate_synthetic_data()
        assert "CPIAUCSL" in df.columns
        assert "FEDFUNDS" in df.columns
        assert len(df) == 9000
        print(f"✓ Synthetic data: {len(df)} days, cols={list(df.columns)}")
    except Exception as e:
        errors.append(f"Synthetic data: {e}")
        print(f"✗ Synthetic data: {e}")

    # 9. Pipeline with inflation
    try:
        from app.data.pipeline import DataPipeline
        pipeline = DataPipeline()
        df = pipeline.refresh(synthetic=True)
        assert "inflation_yoy" in df.columns or "cpi_level" in df.columns
        assert "fed_rate" in df.columns
        print(f"✓ Pipeline: {len(df)} rows, has inflation+fed_rate columns")
    except Exception as e:
        errors.append(f"Pipeline: {e}")
        print(f"✗ Pipeline: {e}")

    # 10. DataManager end-to-end
    try:
        from app.data.manager import DataManager
        dm = DataManager()
        df = dm.download_data(synthetic=True)
        inflation = dm.get_inflation_series()
        fed_rate = dm.get_fed_rate_series()
        gap = dm.get_inflation_gap()
        print(f"✓ DataManager E2E: inflation series len={len(inflation)}, fed_rate len={len(fed_rate)}")
    except Exception as e:
        errors.append(f"DataManager E2E: {e}")
        print(f"✗ DataManager E2E: {e}")

    # 11. Full env construction (if gymnasium available)
    try:
        from app.environment.macro_env import GYM_AVAILABLE
        if GYM_AVAILABLE:
            import numpy as np
            from app.models.linear_transition import LinearStateTransition
            from app.models.regime import RegimeModel
            from app.environment.macro_env import MacroPolicyEnvV2
            from app.environment.reward import RewardModule

            A = np.eye(3) * 0.98
            B = np.array([0.003, 0.006, -0.004])
            transition = LinearStateTransition(A=A, B=B)
            regime = RegimeModel()
            
            hist_states = np.random.randn(100, 4).astype(np.float32) * 0.3
            hist_eigen = np.random.rand(100, 3).astype(np.float32)
            hist_fed = np.full(100, 0.03, dtype=np.float32)

            env = MacroPolicyEnvV2(
                transition_model=transition,
                regime_model=regime,
                historical_states=hist_states,
                eigenvalues_history=hist_eigen,
                fed_rate_history=hist_fed,
                crisis_threshold=2.0,
                reward_module=RewardModule(),
            )
            obs, info = env.reset()
            assert obs.shape == (12,)  # 4 + 3 + 1 + 1 + 3
            
            action = np.array([0.3], dtype=np.float32)
            obs2, reward, term, trunc, info2 = env.step(action)
            assert obs2.shape == (12,)
            assert "inflation_gap" in info2
            assert "fed_rate" in info2
            assert "delta_rate" in info2
            print(f"✓ EnvV2 full test: obs={obs.shape}, reward={reward:.4f}, fed_rate={info2['fed_rate']:.4f}")
        else:
            print("⚠ Skipped env full test (gymnasium not available)")
    except Exception as e:
        errors.append(f"Env full test: {e}")
        print(f"✗ Env full test: {e}")

    # 12. API routes import
    try:
        from app.api.routes_state import router as state_router
        from app.api.routes_policy import router as policy_router
        print("✓ API routes: imported successfully")
    except Exception as e:
        errors.append(f"API routes: {e}")
        print(f"✗ API routes: {e}")

    # 13. FastAPI app
    try:
        from app.main import app
        print("✓ FastAPI app: instantiated")
    except Exception as e:
        errors.append(f"FastAPI app: {e}")
        print(f"✗ FastAPI app: {e}")

    # Summary
    print(f"\n{'='*60}")
    if errors:
        print(f"FAILED: {len(errors)} error(s)")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("ALL SMOKE TESTS PASSED ✓")
        sys.exit(0)


if __name__ == "__main__":
    main()
