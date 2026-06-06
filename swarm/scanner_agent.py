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
            
        target_file = state.get("target_file")
        scan_path = os.path.join(temp_dir, target_file) if target_file else temp_dir
        
        # Run Semgrep
        semgrep_cmd = [
            "semgrep", 
            "scan", 
            "--config", "auto", 
            "--json", 
            scan_path
        ]
        
        try:
            semgrep_result = subprocess.run(semgrep_cmd, capture_output=True, text=True, timeout=120)
            if semgrep_result.returncode != 0 and semgrep_result.returncode != 1:
                raise RuntimeError(f"Semgrep crashed: {semgrep_result.stderr}")
            if semgrep_result.stdout:
                parsed_semgrep = json.loads(semgrep_result.stdout)
                for finding in parsed_semgrep.get("results", []):
                    # In file path, semgrep might output absolute or relative depending on scan_path
                    file_path = finding.get("path", "").replace(temp_dir + os.sep, "").replace(temp_dir + "/", "")
                    if target_file and target_file not in file_path:
                        continue
                    findings.append({
                        "tool": "semgrep",
                        "rule": finding.get("check_id", "unknown"),
                        "severity": finding.get("extra", {}).get("severity", "UNKNOWN"),
                        "file": file_path,
                        "line": finding.get("start", {}).get("line", 0),
                        "description": finding.get("extra", {}).get("message", "")
                    })
        except Exception as e:
            state["trace_logs"].append({
                "agent": "Scanner Agent", 
                "log": f"Semgrep scan failed: {e}"
            })
            raise RuntimeError(f"Scanner Agent failed: {e}")

        # Run Bandit (Python only)
        bandit_cmd = [
            "bandit",
            "-r", scan_path,
            "-f", "json"
        ]
        
        try:
            bandit_result = subprocess.run(bandit_cmd, capture_output=True, text=True, timeout=120)
            if bandit_result.returncode != 0 and bandit_result.returncode != 1:
                raise RuntimeError(f"Bandit crashed: {bandit_result.stderr}")
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
            raise RuntimeError(f"Scanner Agent failed: {e}")

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
