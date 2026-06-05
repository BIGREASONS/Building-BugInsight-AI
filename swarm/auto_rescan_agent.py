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
            state["trace_logs"].append({
                "agent": "Auto-Rescan Agent", 
                "log": "Target file missing during rescan."
            })
            return state
                
        # Run Semgrep
        semgrep_cmd = [
            "semgrep", 
            "scan", 
            "--config", "auto", 
            "--json", 
            temp_dir
        ]
        
        semgrep_result = subprocess.run(semgrep_cmd, capture_output=True, text=True)
        
        remaining_findings = 0
        if semgrep_result.stdout:
            try:
                parsed = json.loads(semgrep_result.stdout)
                for finding in parsed.get("results", []):
                    file_path = finding.get("path", "").replace(temp_dir + os.sep, "").replace(temp_dir + "/", "")
                    if file_path == affected_file:
                        remaining_findings += 1
            except:
                pass
                
        state["rescan_passed"] = (remaining_findings == 0)
        
    except Exception as e:
        state["rescan_passed"] = False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        
    state["trace_logs"].append({
        "agent": "Auto-Rescan Agent", 
        "log": f"Auto-rescan {'passed (0 findings)' if state.get('rescan_passed') else 'failed (vulnerability remains)'}."
    })
    return state
