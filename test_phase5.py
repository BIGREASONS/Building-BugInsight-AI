import asyncio
import uuid
import time
from swarm.workflow import swarm_app

repos = [
    "https://github.com/tiangolo/fastapi",
    "https://github.com/pallets/flask",
    "https://github.com/BIGREASONS/Building-BugInsight-AI"
]

def run_repo(repo_url):
    print(f"\n========================================")
    print(f"Testing: {repo_url}")
    print(f"========================================")
    
    start = time.time()
    
    initial_state = {
        "job_id": uuid.uuid4().hex[:12],
        "issue_url": None,
        "repo_url": repo_url,
        "issue_text": "Random bug testing",
        "trace_logs": []
    }
    
    final_state = None
    try:
        for output in swarm_app.stream(initial_state):
            for node_name, state in output.items():
                print(f" -> Completed: {node_name}")
                final_state = state
                
        elapsed = time.time() - start
        
        # Extract metrics
        index_stats = final_state.get("index_stats", {})
        
        print("\n--- Validation Results ---")
        print(f"Runtime: {elapsed:.2f} seconds")
        print(f"Files Indexed: {index_stats.get('files_indexed', 0)}")
        print(f"Languages: {index_stats.get('language_count', {})}")
        print(f"Size: {index_stats.get('size_mb', 0)} MB")
        print(f"Clone & Indexing Status: {index_stats.get('status', 'Unknown')}")
        
        # Verify Context Retrieval
        repo_context = final_state.get('repo_context', '')
        print(f"Context Retrieval: {'PASS' if len(repo_context) > 0 and 'FILE:' in repo_context else 'FAIL'}")
        
        # Verify Root Cause Generation
        print(f"Root Cause Generated: {'PASS' if final_state.get('root_cause') else 'FAIL'}")
        
        # Verify Patch Generation
        print(f"Patch Generated: {'PASS' if final_state.get('patch') else 'FAIL'}")
        
        # Verify Sprint Planning
        print(f"Sprint Planning Generated: {'PASS' if final_state.get('sprint_recommendation') else 'FAIL'}")
        
    except Exception as e:
        print(f"FAILED with error: {e}")

if __name__ == "__main__":
    for r in repos:
        run_repo(r)
