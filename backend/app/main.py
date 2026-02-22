"""
MacroState Control Room — FastAPI Application Entry Point.

Serves:
- REST API for data status, state refresh, policy recommendation
- WebSocket for live Monte Carlo streaming
- CORS enabled for frontend at localhost:3000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_state import router as state_router
from app.api.routes_policy import router as policy_router
from app.api.routes_rl import router as rl_router
from app.api.routes_historical import router as historical_router
from app.api.routes_agents import router as agents_router
from app.api.routes_timeseries import router as timeseries_router
from app.api.routes_gemini import router as gemini_router
from app.ws.sim_stream import handle_simulation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup, attempt to load cached state; on shutdown, clean up."""
    logger.info("MacroState Control Room backend starting...")
    # Try loading cached/processed data on startup
    try:
        from app.api.routes_state import refresh_state
        await refresh_state(synthetic=False)
        logger.info("State loaded from cache/FRED on startup")
    except Exception as e:
        logger.warning(f"Could not auto-load state on startup: {e}")
        logger.info("State will be loaded on first /api/state/refresh call")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="MacroState Control Room",
    description="Cross-Asset Macro State + Live Monte Carlo Policy Engine",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST routers
app.include_router(state_router)
app.include_router(policy_router)
app.include_router(rl_router)
app.include_router(historical_router)
app.include_router(agents_router)
app.include_router(timeseries_router)
app.include_router(gemini_router)


# WebSocket endpoint
@app.websocket("/ws/simulate")
async def ws_simulate(websocket: WebSocket):
    await handle_simulation(websocket)
