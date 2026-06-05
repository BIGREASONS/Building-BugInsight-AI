from swarm.repo_agent import clone_and_index_repo, retrieve_context
import os
local_repo = os.path.dirname(os.path.abspath(__file__))
clone_and_index_repo(local_repo, "test_job")
queries = ["chromadb cache", "sse stream endpoints", "react strict mode agent timings"]
for q in queries:
    res = retrieve_context(q, "test_job")
    print(f"QUERY: {q}\n{repr(res)[:200]}\n")
