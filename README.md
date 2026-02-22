# MacroState Control Room

**Cross-Asset Macro State + Live Monte Carlo Policy Engine**

A real-time interactive dashboard that infers latent macroeconomic states from cross-asset relationships and recommends monetary policy (Ease/Hold/Tighten) by simulating forward under uncertainty with live Monte Carlo streaming.

## Architecture

- **Backend**: FastAPI + WebSocket (Python 3.11)
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS + D3 + Recharts
- **Data**: FRED API with local parquet caching
- **Math**: Kalman Filter + EM, Regime Switching, Numba-accelerated Monte Carlo

## Prerequisites

- Docker & Docker Compose (recommended)
- OR: Python 3.11+, Node.js 18+

## Quick Start (Docker)

```bash
# Set your FRED API key (optional - synthetic mode available)
export FRED_API_KEY=your_key_here

# Build and run
docker compose up --build

# Open browser
open http://localhost:3000
```

## Quick Start (Local)

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export FRED_API_KEY=your_key_here
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:3000

## Without FRED API Key

The app works without a FRED API key. Toggle "Demo Synthetic Mode" in the UI or simply run without setting `FRED_API_KEY`. The synthetic mode generates realistic cross-asset data for demonstration.

## Running Tests

```bash
cd backend
pip install -r requirements.txt
pytest tests/ -v
```

## Demo Script

```bash
cd backend
python -m scripts.demo
```

## Data Sources (FRED Series)

| Series | Description | Transform |
|--------|-------------|-----------|
| SP500 | S&P 500 Index | log returns |
| DGS2 | 2-Year Treasury | level + changes |
| DGS10 | 10-Year Treasury | level + changes |
| BAA | BAA Corporate Yield | level; credit spread = BAA - DGS10 |
| VIXCLS | VIX Index | log changes |
| DTWEXBGS | Trade-Weighted Dollar | log returns |
| DGS5 | 5-Year Treasury (optional) | curvature = DGS10 - 2*DGS5 + DGS2 |

## Features

- **Real-time Kalman Filter** state estimation from cross-asset covariance structure
- **Live Monte Carlo** streaming 5,000–10,000 paths over WebSocket
- **Policy Recommendation** via stochastic loss minimization with tail risk (ES95)
- **Regime Switching** (Normal/Fragile/Crisis) with Markov transitions
- **Shock Injection** (Credit, Volatility, Rate) with adjustable intensity
- **Compare Mode** to overlay two policy scenarios
- **Interactive Controls**: sliders, buttons, speed control, pause/resume

## API Endpoints

- `GET /api/health` — Health check
- `GET /api/data/status` — Data cache status
- `POST /api/state/refresh` — Fetch data, run pipeline, estimate model
- `GET /api/state/current` — Current macro state + structure metrics
- `POST /api/policy/recommend` — Run recommendation engine
- `WS /ws/simulate` — Live Monte Carlo streaming
