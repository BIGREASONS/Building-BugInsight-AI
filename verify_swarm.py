import os
import sys
import requests
import json
import time

# Fix Windows console charmap errors for emojis/arrows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from swarm.repo_agent import clone_and_index_repo, retrieve_context
from swarm.workflow import swarm_app

# --- Offline Simulation ---
# Ensure no OpenAI API keys are present in the environment
os.environ.pop("OPENAI_API_KEY", None)

print("="*60)
print("BUGINSIGHT SWARM: VERIFICATION CHECKLIST")
print("="*60)

# --- Test 1: CodeBERT Endpoint ---
print("\n--- Test 1: CodeBERT API ---")
try:
    start_time = time.time()
    res = requests.post("http://localhost:8000/predict_severity", json={"issue_text": "Authentication bypass allows unauthorized access."})
    elapsed = time.time() - start_time
    
    if res.status_code == 200:
        data = res.json()
        print(f"[PASS] Severity: {data['severity']}, Confidence: {data['confidence']}")
        print(f"[PASS] Latency: {elapsed:.2f} seconds")
        if elapsed > 2.0:
            print("[WARN] Latency exceeded 2 seconds.")
    else:
        print(f"[FAIL] CodeBERT API returned status {res.status_code}")
except Exception as e:
    print(f"[FAIL] Could not connect to CodeBERT API. Is uvicorn running? Error: {e}")


# --- Test 2: Repository Agent (ChromaDB) ---
print("\n--- Test 2: Repository Agent (ChromaDB) ---")

local_repo = os.path.dirname(os.path.abspath(__file__))

try:
    print("Indexing local BugInsight repository for verification...")
    # Use the current project directory
    clone_and_index_repo(local_repo, "verify_job")
    
    queries = {
        "chromadb cache": "repo_agent.py",
        "sse stream endpoints": "api.py",
        "react strict mode agent timings": "page.tsx"
    }

    passed_retrieval = 0
    for query, expected_file in queries.items():
        res = retrieve_context(query, "verify_job")
        if expected_file in res:
            print(f"[PASS] Query '{query}' correctly retrieved {expected_file}")
            passed_retrieval += 1
        else:
            print(f"[FAIL] Query '{query}' failed to retrieve {expected_file}. Got:\n{repr(res)[:1000]}...")

    print(f"Retrieval Score: {passed_retrieval}/3")
except Exception as e:
    print(f"[FAIL] Repository indexing or retrieval failed: {e}")


# --- Test 6/7/8: Full Workflow (Mock PR) ---
print("\n--- Test 6 & 7: LangGraph Full Workflow & Mock PR ---")
initial_state = {
    "issue_url": None,
    "repo_url": local_repo,
    "issue_text": "SQL injection vulnerability in buginsight-demo authentication system. Users can bypass login by injecting SQL into the username field.",
    "trace_logs": []
}

try:
    print("Executing Swarm...")
    final_state = None
    for output in swarm_app.stream(initial_state):
        for node_name, state in output.items():
            print(f" -> Completed: {node_name}")
            final_state = state
            
    print("\n[PASS] Workflow completed without crashing.")
    
    print("\n--- Test 3: Root Cause Agent JSON Stability ---")
    print(f"Suspected Files: {final_state.get('suspected_files')}")
    print(f"Reasoning: {final_state.get('root_cause')}")
    if "auth.py" in str(final_state.get("suspected_files")):
        print("[PASS] Identified auth.py correctly.")
    else:
        print("[FAIL] Did not identify auth.py.")
        
    print("\n--- Test 4: Fix Agent JSON Stability ---")
    patch = final_state.get("patch", "")
    print(f"Patch Length: {len(patch)} characters")
    if len(patch) > 10:
        print("[PASS] Patch generated.")
    else:
        print("[FAIL] Empty or invalid patch.")
        
    print("\n--- Test 5: Sprint Agent JSON Stability ---")
    print(final_state.get('sprint_recommendation', ''))
    if "Estimated developer time saved" in final_state.get('sprint_recommendation', ''):
        print("[PASS] Business impact metric present.")
    else:
        print("[FAIL] Missing business impact metric.")
        
    print("\n--- Test 7: Mock PR Verification ---")
    pr_url = final_state.get("pr_url", "")
    print(f"PR URL: {pr_url}")
    if pr_url.startswith("http"):
        print("[PASS] Valid Mock PR URL generated.")
    else:
        print("[FAIL] Invalid or missing PR URL.")

    print("\n--- Test 8: Validation Agent ---")
    print(f"Validation Score: {final_state.get('validation_score')}")
    print(f"Validation Approved: {final_state.get('is_patch_valid')}")
    print(f"Reasoning: {final_state.get('validation_reasoning')}")

    print("\n--- Test 9: Test Execute Agent ---")
    print(f"Tests Passed: {final_state.get('tests_passed')}")

    print("\n--- Test 10: Auto-Rescan Agent ---")
    print(f"Rescan Passed (0 findings): {final_state.get('rescan_passed')}")

except Exception as e:
    print(f"[FAIL] LangGraph workflow crashed: {e}")

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)
