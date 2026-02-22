#!/bin/bash
cd /Users/delithosdenvorn/Documents/Python/hacklytics26/PolEn/backend
nohup /Users/delithosdenvorn/Documents/Python/hacklytics26/PolEn/backend/venv/bin/python \
  -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload \
  > /tmp/polen_backend.log 2>&1 &
echo "Backend PID: $!"
