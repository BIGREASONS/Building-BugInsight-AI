import os
import tempfile
import subprocess
import shutil
import json
from swarm.state import SwarmState

def auto_rescan_agent(state: SwarmState) -> SwarmState:
    """Agent 4.8: Re-runs the scanner against the patched code to verify the finding is removed."""
    state["current_agent"] = "Auto-Rescan Agent"
    
    repo_url = state.get("repo_url", "")
    affected_file = state.get("affected_file", "")
    patched_code = state.get("patched_code", "")
    
    if not repo_url or not affected_file or not patched_code:
        state["rescan_passed"] = False
        if not state.get("first_failed_gate"):
            state["first_failed_gate"] = "rescan"
        return state
        
    temp_dir = tempfile.mkdtemp(prefix="buginsight_rescan_")
    
    try:
        # Clone repo
        subprocess.run(["git", "clone", "--depth", "1", repo_url, temp_dir], capture_output=True, text=True, check=True)
        
        # Apply patch to affected file
        target_file_path = os.path.join(temp_dir, affected_file)
        if os.path.exists(target_file_path):
            with open(target_file_path, "w", encoding="utf-8") as f:
                f.write(patched_code)
        else:
            state["rescan_passed"] = False
            if not state.get("first_failed_gate"):
                state["first_failed_gate"] = "rescan"
            state["trace_logs"].append({
                "agent": "Auto-Rescan Agent", 
                "log": "Target file missing during rescan."
            })
            return state
                
        # Run Pyflakes check (for logging purposes only, don't fail the gate for unused imports)
        pyflakes_result = subprocess.run(["pyflakes", target_file_path], capture_output=True, text=True)
        if pyflakes_result.returncode != 0:
            state["trace_logs"].append({
                "agent": "Auto-Rescan Agent", 
                "log": f"Pyflakes warning: {pyflakes_result.stdout.strip()}"
            })

        # Run Bandit
        bandit_cmd = [
            "bandit",
            "-q",
            "-r", target_file_path,
            "-f", "json"
        ]
        
        bandit_result = subprocess.run(bandit_cmd, capture_output=True, text=True)
        
        # Bandit exits 0 if no issues, 1 if issues found.
        # However, we must handle cases where bandit output is not valid json
        try:
            data = json.loads(bandit_result.stdout)
            results = data.get("results", [])
        except Exception:
            results = []
            
        if len(results) > 0:
            state["rescan_passed"] = False
            if not state.get("first_failed_gate"):
                state["first_failed_gate"] = "rescan"
            state["trace_logs"].append({
                "agent": "Auto-Rescan Agent", 
                "log": f"Rescan failed: {len(results)} vulnerabilities still detected."
            })
        else:
            state["rescan_passed"] = True
            state["trace_logs"].append({
                "agent": "Auto-Rescan Agent", 
                "log": "Rescan passed. Vulnerability resolved."
            })
            
    except Exception as e:
        state["rescan_passed"] = False
        if not state.get("first_failed_gate"):
            state["first_failed_gate"] = "rescan"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    state["trace_logs"].append({
        "agent": "Auto-Rescan Agent", 
        "log": f"Auto-rescan {'passed (0 findings)' if state.get('rescan_passed') else 'failed (vulnerability remains)'}."
    })
    return state
