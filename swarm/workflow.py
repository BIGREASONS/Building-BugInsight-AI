import os
import requests
from langgraph.graph import StateGraph, END
from swarm.state import SwarmState
from swarm.repo_agent import clone_and_index_repo, retrieve_context
from swarm.agents import root_cause_agent, fix_agent, repair_agent, sprint_planning_agent, generate_severity_reasoning
from swarm.github_agent import github_action_agent
from swarm.scanner_agent import scanner_agent
from swarm.test_agent import test_agent
from swarm.validation_agent import validation_agent
from swarm.test_execute_agent import test_execute_agent
from swarm.auto_rescan_agent import auto_rescan_agent

def repo_node(state: SwarmState) -> SwarmState:
    """Agent 1: Dynamically clones and indexes the repo, then retrieves context."""
    state["current_agent"] = "Repository Agent"
    if "trace_logs" not in state:
        state["trace_logs"] = []
        
    repo_url = state.get("repo_url", "")
    job_id = state.get("job_id", "default_job")
    query = state.get("issue_text", "")
    
    try:
        # Clone and index the repository dynamically
        index_stats = clone_and_index_repo(repo_url, job_id)
        
        # Retrieve context from the newly created collection
        context = retrieve_context(query, job_id, n_results=2)
        
        state["repo_context"] = context
        state["index_stats"] = index_stats
        
        state["trace_logs"].append({
            "agent": "Repository Agent", 
            "log": f"Indexed {index_stats.get('files_indexed', 0)} files ({index_stats.get('size_mb', 0)} MB)."
        })
    except Exception as e:
        state["repo_context"] = f"Failed to retrieve context: {str(e)}"
        state["trace_logs"].append({
            "agent": "Repository Agent", 
            "log": f"Error indexing repository: {str(e)}"
        })
        
    return state


def severity_node(state: SwarmState) -> SwarmState:
    """Agent 2: Pings the local FastAPI CodeBERT endpoint for severity prediction."""
    state["current_agent"] = "Severity Agent"
    
    issue_text = state.get("issue_text", "")
    try:
        from swarm.api import _predict_severity
        severity, confidence = _predict_severity(issue_text)
        state["severity"] = severity
        state["confidence"] = confidence
    except Exception as e:
        # Fallback if model isn't loaded
        state["severity"] = "Critical (Fallback)"
        state["confidence"] = 0.99
        
    # Override severity if scanner found critical/high issues
    scanner_findings = state.get("scanner_findings", [])
    has_high_severity = any(
        f.get("severity", "").upper() in ["ERROR", "HIGH", "CRITICAL"] 
        for f in scanner_findings
    )
    if has_high_severity:
        state["severity"] = "Critical"
        state["confidence"] = max(state.get("confidence", 0.0), 0.92)
        
    # Generate 3 bullets explaining the severity
    if state["severity"] and state["severity"] not in ["Unknown", "Error"]:
        state["severity_reasoning"] = generate_severity_reasoning(issue_text, state["severity"])
    else:
        state["severity_reasoning"] = []
        
    state["trace_logs"].append({
        "agent": "Severity Agent", 
        "log": f"Predicted Severity: {state['severity']} ({state['confidence']*100:.1f}%)"
    })
    return state


# Build the LangGraph
workflow = StateGraph(SwarmState)

# Add nodes
workflow.add_node("repo_agent", repo_node)
workflow.add_node("scanner_agent", scanner_agent)
workflow.add_node("severity_agent", severity_node)
workflow.add_node("root_cause_agent", root_cause_agent)
workflow.add_node("fix_agent", fix_agent)
workflow.add_node("repair_agent", repair_agent)
workflow.add_node("validation_agent", validation_agent)
workflow.add_node("test_agent", test_agent)
workflow.add_node("test_execute_agent", test_execute_agent)
workflow.add_node("auto_rescan_agent", auto_rescan_agent)
workflow.add_node("github_agent", github_action_agent)
workflow.add_node("sprint_agent", sprint_planning_agent)

def route_validation(state: SwarmState) -> str:
    if state.get("is_patch_valid"):
        return "auto_rescan_agent"
    elif state.get("repair_attempts", 0) < 1:
        return "repair_agent"
    else:
        # Give up and go to github_agent to report failure
        return "github_agent"

# Define edges
workflow.set_entry_point("repo_agent")
workflow.add_edge("repo_agent", "scanner_agent")
workflow.add_edge("scanner_agent", "severity_agent")
workflow.add_edge("severity_agent", "root_cause_agent")
workflow.add_edge("root_cause_agent", "fix_agent")
workflow.add_edge("fix_agent", "validation_agent")
workflow.add_conditional_edges("validation_agent", route_validation)
workflow.add_edge("repair_agent", "validation_agent")
workflow.add_edge("test_agent", "test_execute_agent")
workflow.add_edge("test_execute_agent", "auto_rescan_agent")
workflow.add_edge("auto_rescan_agent", "github_agent")
workflow.add_edge("github_agent", "sprint_agent")
workflow.add_edge("sprint_agent", END)

# Compile graph
swarm_app = workflow.compile()

if __name__ == "__main__":
    # Test runner for Day 2 logic
    initial_state = {
        "issue_url": None,
        "repo_url": "https://github.com/BIGREASONS/buginsight-demo",
        "issue_text": "Users are reporting that they can log in without a valid password by passing special characters into the username field.",
        "trace_logs": []
    }
    
    print("Initializing BugInsight Swarm...")
    for output in swarm_app.stream(initial_state):
        # LangGraph stream yields {node_name: State} at each step
        for node_name, state in output.items():
            print(f"--- Completed: {node_name} ---")
            
    # Print final results
    final_state = state
    print("\n" + "="*50)
    print("FINAL SWARM STATE")
    print("="*50)
    print(f"Severity: {final_state.get('severity')} ({final_state.get('confidence')})")
    print(f"Root Cause: {final_state.get('root_cause')}")
    print(f"Suspected Files: {final_state.get('suspected_files')}")
    print(f"\nPatch:\n{final_state.get('patch')}")
    print(f"\nPR URL: {final_state.get('pr_url')}")
    print(f"Sprint Points: {final_state.get('story_points')} ({final_state.get('priority')})")
