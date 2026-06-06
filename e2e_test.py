import os
import time
import requests
import threading
import subprocess
import uvicorn

def start_server():
    os.environ["BUGINSIGHT_WEBHOOK_SECRET"] = "test-secret"
    os.environ["BUGINSIGHT_PR_MODE"] = "COMMENT"
    os.environ["LIVE_MODE_ENABLED_REPOS"] = "buginsight-actions-test"
    
    import copy
    env = copy.deepcopy(os.environ)
    if "GITHUB_TOKEN" in env:
        del env["GITHUB_TOKEN"]
    token = subprocess.check_output(["gh", "auth", "token"], env=env).decode("utf-8").strip()
    os.environ["GITHUB_TOKEN"] = token
    
    from swarm.api import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

def run_e2e():
    print("Waiting for server to start...")
    time.sleep(5)
    
    print("Triggering webhook...")
    headers = {
        "Authorization": "Bearer test-secret",
        "Content-Type": "application/json"
    }
    payload = {
        "repo_url": "https://github.com/BIGREASONS/buginsight-actions-test",
        "pr_number": 1,
        "branch": "main"
    }
    
    try:
        res = requests.post("http://127.0.0.1:8000/api/v1/analyze_pr", json=payload, headers=headers)
        print("Webhook response:", res.status_code, res.json())
        job_id = res.json().get("job_id")
    except Exception as e:
        print("Webhook failed:", e)
        os._exit(1)
        
    print(f"Job queued successfully. Job ID: {job_id}")
    print("Connecting to SSE stream to monitor progress...")
    
    try:
        response = requests.get(f"http://127.0.0.1:8000/api/swarm/stream/{job_id}", stream=True)
        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))
    except Exception as e:
        print("Error reading stream:", e)
        
    print("E2E Test Script Finished.")
    os._exit(0)

if __name__ == "__main__":
    try:
        t = threading.Thread(target=start_server, daemon=True)
        t.start()
        run_e2e()
    except Exception as e:
        print(f"Failed with exception: {e}")
        os._exit(1)
