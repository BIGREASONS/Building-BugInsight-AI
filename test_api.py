import requests
import time
import json
from sseclient import SSEClient

API_URL = "http://127.0.0.1:8002"
WEBHOOK_SECRET = "test-secret"

headers = {
    "Authorization": f"Bearer {WEBHOOK_SECRET}",
    "Content-Type": "application/json"
}
payload = {
    "repo_url": "https://github.com/BIGREASONS/buginsight-benchmark",
    "pr_number": 1,
    "branch": "main",
    "target_file": "vulnerable/sql_injection.py"
}

res = requests.post(f"{API_URL}/api/v1/analyze_pr", json=payload, headers=headers)
job_id = res.json()["job_id"]
print(f"Job ID: {job_id}")

sse_url = f"{API_URL}/api/swarm/stream/{job_id}"
sse_response = requests.get(sse_url, stream=True)
messages = SSEClient(sse_response)

for msg in messages.events():
    if not msg.data: continue
    event = json.loads(msg.data)
    if event["type"] == "swarm_complete":
        final_data = event["data"]
        print("PATCH:", repr(final_data.get("patch")))
        print("RAW_FIX_OUTPUT:", repr(final_data.get("raw_fix_output")))
        print("GENERATED LINES:", final_data.get("generated_patch_lines"))
        break
