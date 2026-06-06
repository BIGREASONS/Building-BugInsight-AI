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
    assert state.get('is_patch_valid') == False

def test_execute_fails_bad_test():
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
    assert state.get('tests_passed') == False

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
    assert state.get('rescan_passed') == False
