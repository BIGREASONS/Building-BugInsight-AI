import json
import asyncio
from swarm.state import SwarmState
from swarm.validation_agent import validation_agent
from swarm.test_execute_agent import test_execute_agent
from swarm.auto_rescan_agent import auto_rescan_agent

repo_url = "https://github.com/BIGREASONS/buginsight-demo"
affected_file = "buginsight-demo/auth.py"

findings = [
    {
        "tool": "semgrep",
        "rule": "python.lang.security.audit.sqli.string-formatting.string-formatting",
        "severity": "HIGH",
        "file": affected_file,
        "line": 42,
        "description": "SQL Injection"
    }
]

root_cause = "Direct string formatting is used to build SQL query, leading to SQL Injection."

def test_validation_rejects_bad_patch():
    print("--- 1. Testing Validation Agent (Bad Patch) ---")
    state: SwarmState = {
        "scanner_findings": findings,
        "root_cause": root_cause,
        "patched_code": "query = f\"SELECT * FROM users WHERE username = '{username}'\" # Just added a comment",
        "trace_logs": []
    }
    state = validation_agent(state)
    print(f"Score: {state.get('validation_score')}")
    print(f"Approved: {state.get('is_patch_valid')}")
    print(f"Reasoning: {state.get('validation_reasoning')}\n")
    return state.get('is_patch_valid') == False

def test_test_execute_fails_bad_test():
    print("--- 2. Testing Test Execution Agent (Failing Test) ---")
    bad_test_code = """
def test_fail():
    assert False, "This test should fail"
"""
    # Provide a decent patched code so syntax is fine, but test fails
    good_patched_code = "import sqlite3\ndef login(username, password):\n    pass\n"
    state: SwarmState = {
        "repo_url": repo_url,
        "regression_tests": bad_test_code,
        "affected_file": affected_file,
        "patched_code": good_patched_code,
        "trace_logs": []
    }
    state = test_execute_agent(state)
    print(f"Tests Passed: {state.get('tests_passed')}")
    results = state.get("test_results", {})
    print(f"Passed: {results.get('passed')}, Failed: {results.get('failed')}\n")
    return state.get('tests_passed') == False

def test_auto_rescan_catches_remaining_vuln():
    print("--- 3. Testing Auto-Rescan Agent (Vulnerability Remains) ---")
    # A vulnerable patched code
    vulnerable_code = """
import sqlite3
def unsafe_query(username):
    query = "SELECT * FROM users WHERE username = '%s'" % username
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    cursor.execute(query)
"""
    state: SwarmState = {
        "repo_url": repo_url,
        "affected_file": affected_file,
        "patched_code": vulnerable_code,
        "trace_logs": []
    }
    state = auto_rescan_agent(state)
    print(f"Rescan Passed (0 findings): {state.get('rescan_passed')}\n")
    return state.get('rescan_passed') == False

if __name__ == "__main__":
    v1 = test_validation_rejects_bad_patch()
    v2 = test_test_execute_fails_bad_test()
    v3 = test_auto_rescan_catches_remaining_vuln()
    
    with open("break_test_results.txt", "w") as f:
        f.write("--- BREAK TEST RESULTS ---\n")
        f.write(f"Validation Agent rejected bad patch? {'YES' if v1 else 'NO'}\n")
        f.write(f"Test Execution Agent caught bad test? {'YES' if v2 else 'NO'}\n")
        f.write(f"Auto-Rescan Agent caught remaining vulnerability? {'YES' if v3 else 'NO'}\n")
    
    print("Done. Wrote results to break_test_results.txt")
