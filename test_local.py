import socket
import time
import subprocess
import requests
import sys
import os

if __name__ == "__main__":
    env = os.environ.copy()
    env["BUGINSIGHT_WEBHOOK_SECRET"] = "test-secret"
    env["BUGINSIGHT_PR_MODE"] = "COMMENT"
    env["LIVE_MODE_ENABLED_REPOS"] = "BIGREASONS/buginsight-actions-test"
    env["GITHUB_MODE"] = "LIVE"

    print("Starting uvicorn server...")
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "swarm.api:app", "--port", "8002"], env=env)

    print("Waiting for server to open port 8002...")
    started = False
    for _ in range(60):
        try:
            with socket.create_connection(("127.0.0.1", 8002), timeout=1):
                started = True
                break
        except OSError:
            time.sleep(2)

    if not started:
        print("Server failed to start in time.")
        server.terminate()
        sys.exit(1)
        
    print("Server is up! Triggering webhook...")
    time.sleep(2)  # Give it a tiny bit more time to fully initialize

    headers = {"Authorization": "Bearer test-secret", "Content-Type": "application/json"}
    payload = {"repo_url": "https://github.com/BIGREASONS/buginsight-actions-test", "pr_number": 1, "branch": "vulnerable-auth-2"}

    try:
        res = requests.post("http://127.0.0.1:8002/api/v1/analyze_pr", json=payload, headers=headers)
        print("API Response:", res.status_code, res.text)
        if res.status_code == 200:
            job_id = res.json().get("job_id")
            print(f"Monitoring SSE stream for job {job_id}...")
            sse = requests.get(f"http://127.0.0.1:8002/api/swarm/stream/{job_id}", stream=True, timeout=120)
            for line in sse.iter_lines():
                if line:
                    print(line.decode("utf-8"))
    except Exception as e:
        print(f"Request error: {e}")

    print("Terminating server...")
    server.terminate()
