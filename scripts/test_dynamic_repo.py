import asyncio
import json
import uuid
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from swarm.workflow import swarm_app
from swarm.repo_agent import MAX_REPO_SIZE_MB, MAX_FILES

# Mock minimal states for testing repo node specifically
async def run_repo_test(repo_url: str, test_name: str, issue_text: str = "Test"):
    print(f"\n{'='*50}\nStarting Test: {test_name}\nURL: {repo_url}\n{'='*50}")
    
    job_id = uuid.uuid4().hex[:12]
    initial_state = {
        "job_id": job_id,
        "issue_url": None,
        "repo_url": repo_url,
        "issue_text": issue_text,
        "trace_logs": []
    }
    
    print("Running Repository Agent...")
    try:
        # We can just call repo_node directly to save time, since we only care about repo indexing here
        # or we can run the whole workflow up to repo_agent. Since LangGraph executes node by node,
        # we can just run the graph and break after repo_agent or root_cause_agent.
        from swarm.workflow import repo_node
        
        result_state = repo_node(initial_state)
        
        index_stats = result_state.get('index_stats')
        repo_context = result_state.get('repo_context')
        
        if index_stats and index_stats.get("status") == "Success":
            print(f"[PASS] Indexing Succeeded!")
            print(f"  Files Indexed: {index_stats.get('files_indexed')}")
            print(f"  Languages: {index_stats.get('language_count')}")
            print(f"  Size: {index_stats.get('size_mb')} MB")
        else:
            print(f"[FAIL] Indexing Failed / No Stats Returned")
            print(f"Context / Error: {repo_context[:500]}...")
            return False
            
        if "FILE:" in repo_context:
            print(f"[PASS] Context Retrieval Works. Context length: {len(repo_context)}")
        else:
            print(f"[FAIL] Context Retrieval Failed. Output: {repo_context}")
            return False
            
        return True
    except Exception as e:
        print(f"[FAIL] Exception caught: {str(e)}")
        return False

async def main():
    print(f"Running limits: Max {MAX_FILES} files, Max {MAX_REPO_SIZE_MB} MB")
    
    # Test 1: FastAPI
    res1 = await run_repo_test("https://github.com/tiangolo/fastapi", "Test 1 (FastAPI)", "authentication middleware")
    
    # Test 2: Flask
    res2 = await run_repo_test("https://github.com/pallets/flask", "Test 2 (Flask)", "routing issues")
    
    # Test 3: My Repo
    res3 = await run_repo_test("https://github.com/BIGREASONS/Building-BugInsight-AI", "Test 3 (My Repo)", "severity prediction CodeBERT")
    
    # Test 4: Invalid Repo
    res4 = await run_repo_test("https://github.com/this/repo/does/not/exist", "Test 4 (Invalid Repo)")
    
    # Test 5: Oversized Repo (Linux)
    res5 = await run_repo_test("https://github.com/torvalds/linux", "Test 5 (Linux)")

if __name__ == "__main__":
    asyncio.run(main())
