import os
import tempfile
import subprocess
import shutil
import json
from typing import List, Dict, Any
from swarm.state import SwarmState

def scanner_agent(state: SwarmState) -> SwarmState:
    """Agent 1.5: Runs static analysis tools (Semgrep, Bandit) on the repository."""
    state["current_agent"] = "Scanner Agent"
    
    repo_url = state.get("repo_url", "")
    if not repo_url:
        state["scanner_findings"] = []
        state["trace_logs"].append({
            "agent": "Scanner Agent", 
            "log": "No repository URL provided. Skipping scan."
        })
        return state

    findings: List[Dict[str, Any]] = []
    
    # Create a temporary directory to clone the repo
    temp_dir = tempfile.mkdtemp(prefix="buginsight_scan_")
    
    try:
        # Clone the repository
        clone_result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            capture_output=True,
            text=True
        )
        if clone_result.returncode != 0:
            raise RuntimeError(f"Git clone failed: {clone_result.stderr}")
            
        # Run Semgrep
        semgrep_cmd = [
            "semgrep", 
            "scan", 
            "--config", "auto", 
            "--json", 
            temp_dir
        ]
        
        try:
            semgrep_result = subprocess.run(semgrep_cmd, capture_output=True, text=True)
            if semgrep_result.stdout:
                parsed_semgrep = json.loads(semgrep_result.stdout)
                for finding in parsed_semgrep.get("results", []):
                    findings.append({
                        "tool": "semgrep",
                        "rule": finding.get("check_id", "unknown"),
                        "severity": finding.get("extra", {}).get("severity", "UNKNOWN"),
                        "file": finding.get("path", "").replace(temp_dir + os.sep, "").replace(temp_dir + "/", ""),
                        "line": finding.get("start", {}).get("line", 0),
                        "description": finding.get("extra", {}).get("message", "")
                    })
        except Exception as e:
            state["trace_logs"].append({
                "agent": "Scanner Agent", 
                "log": f"Semgrep scan failed: {e}"
            })

        # Run Bandit (Python only)
        bandit_cmd = [
            "bandit",
            "-r", temp_dir,
            "-f", "json"
        ]
        
        try:
            bandit_result = subprocess.run(bandit_cmd, capture_output=True, text=True)
            if bandit_result.stdout:
                parsed_bandit = json.loads(bandit_result.stdout)
                for finding in parsed_bandit.get("results", []):
                    findings.append({
                        "tool": "bandit",
                        "rule": finding.get("test_id", "unknown"),
                        "severity": finding.get("issue_severity", "UNKNOWN"),
                        "file": finding.get("filename", "").replace(temp_dir + os.sep, "").replace(temp_dir + "/", ""),
                        "line": finding.get("line_number", 0),
                        "description": finding.get("issue_text", "")
                    })
        except Exception as e:
            state["trace_logs"].append({
                "agent": "Scanner Agent", 
                "log": f"Bandit scan failed: {e}"
            })

    except Exception as e:
        state["trace_logs"].append({
            "agent": "Scanner Agent", 
            "log": f"Scanner Agent error: {e}"
        })
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    state["scanner_findings"] = findings
    state["trace_logs"].append({
        "agent": "Scanner Agent", 
        "log": f"Found {len(findings)} issues via static analysis."
    })
    
    return state
