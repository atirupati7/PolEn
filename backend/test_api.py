#!/usr/bin/env python3
"""Quick integration test: start server, hit endpoints, report results."""
import subprocess, time, sys, json, os, signal, requests

BACKEND = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(BACKEND, "venv", "bin", "python")
BASE = "http://127.0.0.1:8000"

# Kill anything on port 8000
os.system("lsof -ti:8000 | xargs kill -9 2>/dev/null")
time.sleep(1)

# Start server
print("Starting server...")
proc = subprocess.Popen(
    [PYTHON, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000",
     "--app-dir", BACKEND],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    preexec_fn=os.setsid
)

# Wait for server to be ready
for i in range(30):
    try:
        r = requests.get(f"{BASE}/api/health", timeout=1)
        if r.status_code == 200:
            break
    except Exception:
        pass
    time.sleep(1)
else:
    print("FAIL: Server did not start in 30s")
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    sys.exit(1)

print("Server ready!\n")
results = []

# ---- Test 1: Health ----
try:
    r = requests.get(f"{BASE}/api/health")
    d = r.json()
    ok = d.get("ok") is True
    results.append(("GET /api/health", ok, d))
except Exception as e:
    results.append(("GET /api/health", False, str(e)))

# ---- Test 2: Current State ----
try:
    r = requests.get(f"{BASE}/api/state/current")
    d = r.json()
    ok = all(k in d for k in ["regime_label", "eigenvalues", "correlation_matrix", "mu_T", "stress_score"])
    summary = {
        "latest_date": d.get("latest_date"),
        "regime_label": d.get("regime_label"),
        "stress_score": round(d.get("stress_score", 0), 4),
        "num_eigenvalues": len(d.get("eigenvalues", [])),
        "corr_shape": f"{len(d.get('correlation_matrix',[]))}x{len(d.get('correlation_matrix',[[]])[0])}",
        "latent_state_dim": len(d.get("mu_T", [])),
        "is_synthetic": d.get("is_synthetic"),
    }
    results.append(("GET /api/state/current", ok, summary))
except Exception as e:
    results.append(("GET /api/state/current", False, str(e)))

# ---- Test 3: Data Status ----
try:
    r = requests.get(f"{BASE}/api/data/status")
    d = r.json()
    ok = "series" in d and "has_fred_key" in d
    summary = {
        "has_fred_key": d.get("has_fred_key"),
        "num_series": len(d.get("series", {})),
        "series_names": list(d.get("series", {}).keys()),
    }
    results.append(("GET /api/data/status", ok, summary))
except Exception as e:
    results.append(("GET /api/data/status", False, str(e)))

# ---- Test 4: Policy Recommend ----
try:
    payload = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0, "lambda_reg": 1.0, "N": 1000, "H": 12}
    r = requests.post(f"{BASE}/api/policy/recommend", json=payload)
    d = r.json()
    ok = "recommended_action" in d and "comparison" in d and len(d["comparison"]) == 3
    summary = {
        "recommended": f"{d.get('recommended_action')} ({d.get('recommended_bps', 0):+d} bps)",
        "num_actions": len(d.get("comparison", [])),
    }
    for c in d.get("comparison", []):
        summary[f"  {c['action']}"] = f"loss={c['total_loss']:.4f}, ES95={c.get('es_95','n/a')}"
    results.append(("POST /api/policy/recommend", ok, summary))
except Exception as e:
    results.append(("POST /api/policy/recommend", False, str(e)))

# ---- Test 5: State Refresh ----
try:
    r = requests.post(f"{BASE}/api/state/refresh?synthetic=true")
    d = r.json()
    ok = d.get("status") == "ok" and d.get("data_points", 0) > 0
    summary = {
        "status": d.get("status"),
        "latest_date": d.get("latest_date"),
        "regime_label": d.get("regime_label"),
        "data_points": d.get("data_points"),
        "is_synthetic": d.get("is_synthetic"),
    }
    results.append(("POST /api/state/refresh", ok, summary))
except Exception as e:
    results.append(("POST /api/state/refresh", False, str(e)))

# ---- Test 6: OpenAPI Docs ----
try:
    r = requests.get(f"{BASE}/docs")
    ok = r.status_code == 200 and "swagger" in r.text.lower()
    results.append(("GET /docs (Swagger UI)", ok, f"status={r.status_code}, has_swagger=True"))
except Exception as e:
    results.append(("GET /docs (Swagger UI)", False, str(e)))

# Print results
print("=" * 60)
print("API INTEGRATION TEST RESULTS")
print("=" * 60)
passed = 0
for name, ok, detail in results:
    status = "✅ PASS" if ok else "❌ FAIL"
    if ok:
        passed += 1
    print(f"\n{status}  {name}")
    if isinstance(detail, dict):
        for k, v in detail.items():
            print(f"        {k}: {v}")
    else:
        print(f"        {detail}")

print(f"\n{'=' * 60}")
print(f"Result: {passed}/{len(results)} endpoints passed")
print("=" * 60)

# Cleanup
os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
proc.wait(timeout=5)
print("\nServer stopped.")
