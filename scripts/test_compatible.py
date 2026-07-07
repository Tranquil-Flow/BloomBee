#!/usr/bin/env python3
"""Quick smoke test for /compatible endpoint."""
import json, sys, time, subprocess, urllib.request

# Start server
proc = subprocess.Popen(
    [sys.executable, "mvp_capabilities/join_http_server.py",
     "--host", "0.0.0.0", "--port", "8787",
     "--coordinator", "http://localhost:8787"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(2)

try:
    # Health check
    data = json.loads(urllib.request.urlopen("http://localhost:8787/healthz").read())
    assert data.get("ok"), f"Health check failed: {data}"
    print("✅ healthz OK")
    
    # Compatible endpoint
    data = json.loads(urllib.request.urlopen("http://localhost:8787/compatible?token=*").read())
    models = data.get("compatible_models", [])
    best = data.get("best_model", {})
    print(f"✅ /compatible: {len(models)} models, best={best.get('model_id', 'none')}")
    for m in models[:5]:
        print(f"   {m['status']:12s} {m['model_id']} ({m['params_b']}B)")
except Exception as e:
    print(f"❌ Error: {e}")
    stderr = proc.stderr.read().decode()
    if stderr:
        print(f"Server stderr:\n{stderr[:500]}")
finally:
    proc.terminate()
    proc.wait()
