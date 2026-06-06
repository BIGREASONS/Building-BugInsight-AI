import sys
import os
import json
from swarm.test_execute_agent import test_execute_agent

def run_test(name, malicious_code):
    print(f"\n--- Running Verification: {name} ---")
    state = {
        "repo_url": "https://github.com/BIGREASONS/BugInsightSwarm-Vanguard",
        "affected_file": "swarm/agents.py",
        "patched_code": "def placeholder(): pass\n",
        "regression_tests": malicious_code,
        "trace_logs": []
    }
    
    result = test_execute_agent(state)
    print("Tests Passed:", result.get("tests_passed"))
    print("Duration:", result.get("test_results", {}).get("duration"))
    print("Stderr:", result.get("test_results", {}).get("stderr"))
    print("Stdout:", result.get("test_results", {}).get("stdout"))
    return result

if __name__ == "__main__":
    # Test A: Infinite Loop (Timeout)
    test_a_code = """
def test_infinite_loop():
    while True:
        pass
"""
    res_a = run_test("A. Infinite Loop Timeout", test_a_code)
    
    # Test B: Network Access
    test_b_code = """
import urllib.request
def test_network():
    r = urllib.request.urlopen("https://google.com", timeout=2)
    assert r.getcode() == 200
"""
    res_b = run_test("B. Network Access", test_b_code)
    
    # Test C: Environment Variables
    test_c_code = """
import os
def test_env():
    # If GITHUB_TOKEN is present, it means secrets leaked into container
    assert "GITHUB_TOKEN" not in os.environ
    print("ENV:", os.environ)
    assert False, "Fail on purpose to see output"
"""
    # We pass some dummy env variable to the main process and see if it leaks
    os.environ["GITHUB_TOKEN"] = "SUPER_SECRET_TOKEN"
    os.environ["LIVE_MODE_ENABLED_REPOS"] = "BIGREASONS/buginsight-demo"
    res_c = run_test("C. Secrets Exposure", test_c_code)
    
    print("\n--- Summary ---")
    print("A. Timeout Triggered:", res_a.get("tests_passed") == False and "timed out" in res_a.get("test_results", {}).get("stderr", ""))
    print("B. Network Error:", "NameResolutionError" in res_b.get("test_results", {}).get("stdout", "") or "URLError" in res_b.get("test_results", {}).get("stdout", "") or "Failed to establish" in res_b.get("test_results", {}).get("stdout", ""))
    print("C. Secrets Exposure:", "SUPER_SECRET_TOKEN" in res_c.get("test_results", {}).get("stdout", ""))
