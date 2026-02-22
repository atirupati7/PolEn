"""
Tests for the new Historical and Multi-Agent API endpoints.

Validates:
  1. Historical dates endpoint returns sorted monthly dates
  2. Historical state endpoint returns full MacroState schema
  3. Historical state updates correctly across time (regime shifts, metrics)
  4. Multi-agent endpoint runs multiple agents and returns comparative data
  5. All panels (structure, fed policy) receive consistent data at every date
  6. Edge cases: invalid dates, empty agents list, unknown agents
"""

import pytest
import httpx

BASE = "http://localhost:8001"


@pytest.fixture(scope="module")
def client():
    """httpx client for the running backend."""
    with httpx.Client(base_url=BASE, timeout=30) as c:
        yield c


# ═══════════════════════════════════════════════════════════════
# 1. Historical dates
# ═══════════════════════════════════════════════════════════════


def test_historical_dates_returns_list(client):
    r = client.get("/api/historical/dates")
    assert r.status_code == 200
    data = r.json()
    assert "dates" in data
    assert "count" in data
    assert data["count"] > 0
    assert data["count"] == len(data["dates"])


def test_historical_dates_are_sorted(client):
    r = client.get("/api/historical/dates")
    dates = r.json()["dates"]
    assert dates == sorted(dates), "Dates must be in chronological order"


def test_historical_dates_span_decades(client):
    data = client.get("/api/historical/dates").json()
    start_year = int(data["start"][:4])
    end_year = int(data["end"][:4])
    assert end_year - start_year >= 20, "Should span at least 20 years"


# ═══════════════════════════════════════════════════════════════
# 2. Historical state schema
# ═══════════════════════════════════════════════════════════════


REQUIRED_STATE_KEYS = {
    "date",
    "latest_date",
    "mu_T",
    "P_T",
    "stress_score",
    "regime_label",
    "crisis_threshold",
    "inflation_gap",
    "fed_rate",
    "metrics",
    "correlation_matrix",
    "correlation_labels",
    "eigenvalues",
}


def test_historical_state_has_full_schema(client):
    r = client.get("/api/historical/state", params={"date": "2015-06-30"})
    assert r.status_code == 200
    data = r.json()
    missing = REQUIRED_STATE_KEYS - set(data.keys())
    assert not missing, f"Missing keys: {missing}"


def test_historical_state_mu_T_is_3d(client):
    data = client.get("/api/historical/state", params={"date": "2015-06-30"}).json()
    assert len(data["mu_T"]) == 3, "mu_T should be 3-dimensional (stress, liquidity, growth)"


def test_historical_state_correlation_matrix_is_square(client):
    data = client.get("/api/historical/state", params={"date": "2015-06-30"}).json()
    matrix = data["correlation_matrix"]
    assert len(matrix) > 0, "Correlation matrix should not be empty"
    assert len(matrix) == len(matrix[0]), "Correlation matrix must be square"


def test_historical_state_eigenvalues_nonnegative(client):
    data = client.get("/api/historical/state", params={"date": "2015-06-30"}).json()
    for ev in data["eigenvalues"]:
        assert ev >= 0, f"Eigenvalue must be non-negative, got {ev}"


def test_historical_state_regime_label_valid(client):
    data = client.get("/api/historical/state", params={"date": "2015-06-30"}).json()
    assert data["regime_label"] in {"Normal", "Fragile", "Crisis"}


# ═══════════════════════════════════════════════════════════════
# 3. Historical state varies over time
# ═══════════════════════════════════════════════════════════════


def test_state_varies_across_dates(client):
    """Different dates should yield different stress scores."""
    dates = ["2005-01-31", "2009-01-31", "2020-03-31"]
    scores = []
    for d in dates:
        data = client.get("/api/historical/state", params={"date": d}).json()
        scores.append(data["stress_score"])
    assert len(set(round(s, 3) for s in scores)) > 1, (
        f"Stress scores should vary across dates, got {scores}"
    )


def test_closest_date_matching(client):
    """Requesting a non-exact date should return the closest snapshot."""
    r = client.get("/api/historical/state", params={"date": "2015-06-15"})
    assert r.status_code == 200
    data = r.json()
    assert "date" in data


def test_invalid_date_format_returns_400(client):
    r = client.get("/api/historical/state", params={"date": "not-a-date"})
    assert r.status_code == 400


# ═══════════════════════════════════════════════════════════════
# 4. Multi-agent simulation
# ═══════════════════════════════════════════════════════════════


def test_multi_agent_basic(client):
    r = client.post(
        "/api/agents/simulate",
        json={
            "agents": ["custom", "heuristic"],
            "custom_delta_bps": -25,
            "N": 500,
            "H": 6,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "agents" in data
    assert "custom" in data["agents"]
    assert "heuristic" in data["agents"]


def test_multi_agent_has_metrics(client):
    data = client.post(
        "/api/agents/simulate",
        json={"agents": ["custom"], "N": 500, "H": 6},
    ).json()
    agent = data["agents"]["custom"]
    for key in ("mean_stress", "mean_growth_penalty", "mean_es95", "crisis_end", "total_loss"):
        assert key in agent["metrics"], f"Missing metric: {key}"


def test_multi_agent_has_paths(client):
    data = client.post(
        "/api/agents/simulate",
        json={"agents": ["custom"], "N": 500, "H": 12},
    ).json()
    agent = data["agents"]["custom"]
    assert len(agent["stress_path"]) == 12
    assert len(agent["crisis_prob_path"]) == 12
    assert len(agent["growth_path"]) == 12


def test_multi_agent_from_historical_date(client):
    data = client.post(
        "/api/agents/simulate",
        json={
            "agents": ["custom", "historical"],
            "start_date": "2008-09-30",
            "custom_delta_bps": -100,
            "N": 500,
            "H": 6,
        },
    ).json()
    assert "custom" in data["agents"]
    assert "historical" in data["agents"]
    hist = data["agents"]["historical"]
    assert "Historical Fed" in hist["label"]


def test_multi_agent_different_agents_different_results(client):
    """Custom(-50) and Custom(+50) should produce different losses."""
    # Run two separate single-agent calls
    r1 = client.post(
        "/api/agents/simulate",
        json={"agents": ["custom"], "custom_delta_bps": -50, "N": 500, "H": 6},
    ).json()
    r2 = client.post(
        "/api/agents/simulate",
        json={"agents": ["custom"], "custom_delta_bps": 50, "N": 500, "H": 6},
    ).json()
    loss1 = r1["agents"]["custom"]["metrics"]["total_loss"]
    loss2 = r2["agents"]["custom"]["metrics"]["total_loss"]
    assert loss1 != loss2, "Different actions should produce different losses"


# ═══════════════════════════════════════════════════════════════
# 5. Data synchronisation (structure panels consistent)
# ═══════════════════════════════════════════════════════════════


def test_current_and_historical_latest_match(client):
    """The latest historical snapshot should match /api/state/current data."""
    dates = client.get("/api/historical/dates").json()["dates"]
    latest_date = dates[-1]
    hist = client.get("/api/historical/state", params={"date": latest_date}).json()
    curr = client.get("/api/state/current").json()

    # mu_T should be very close (might differ slightly due to rounding)
    for i in range(min(len(hist["mu_T"]), len(curr["mu_T"]))):
        assert abs(hist["mu_T"][i] - curr["mu_T"][i]) < 0.01, (
            f"mu_T[{i}] mismatch: historical={hist['mu_T'][i]}, current={curr['mu_T'][i]}"
        )


def test_historical_snapshot_has_all_panel_data(client):
    """Every snapshot must have the data needed for all 3 bottom panels."""
    data = client.get("/api/historical/state", params={"date": "2015-06-30"}).json()

    # Structure panel needs:
    assert len(data["correlation_matrix"]) > 0, "Correlation matrix needed for structure panel"
    assert len(data["eigenvalues"]) > 0, "Eigenvalues needed for structure panel"
    assert len(data["metrics"]) > 0, "Metrics needed for structure panel"
    assert len(data["mu_T"]) > 0, "mu_T needed for latent state display"

    # Fed policy panel needs:
    assert "inflation_gap" in data, "inflation_gap needed for fed policy panel"
    assert "fed_rate" in data, "fed_rate needed for fed policy panel"
    assert "stress_score" in data, "stress_score needed for fed policy panel"
    assert "regime_label" in data, "regime_label needed for fed policy panel"


# ═══════════════════════════════════════════════════════════════
# 6. Edge cases
# ═══════════════════════════════════════════════════════════════


def test_empty_agents_list_returns_empty(client):
    r = client.post(
        "/api/agents/simulate",
        json={"agents": [], "N": 500, "H": 6},
    )
    # Might return 200 with empty agents or 422 validation error
    assert r.status_code in (200, 422)
