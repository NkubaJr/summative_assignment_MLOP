#!/bin/sh
# Start the API in the background, wait until it answers, then run the UI in the foreground
uvicorn app.api:app --host 0.0.0.0 --port 8000 &
for i in $(seq 1 60); do
  python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null && break
  sleep 1
done
exec streamlit run app/ui.py --server.port "${PORT:-7860}" --server.address 0.0.0.0 --server.headless true
