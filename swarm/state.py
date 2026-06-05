from typing import TypedDict, List, Dict, Any, Optional

class SwarmState(TypedDict):
    """The state dictionary passed between all agents in the LangGraph."""
    
    # Input
    job_id: str
    issue_url: Optional[str]
    repo_url: Optional[str]
    issue_text: str
    index_stats: Optional[Dict[str, Any]]
    
    # Code Context (Agent 1)
    repo_context: str
    # Scanner Findings (Agent 1.5)
    scanner_findings: List[Dict[str, Any]]
    
    # Severity Prediction (Agent 2 - CodeBERT)
    severity: str
    confidence: float
    
    # Root Cause (Agent 3)
    root_cause: str
    suspected_files: List[str]
    suspected_functions: List[str]
    affected_file: str
    vulnerable_code: str
    exploit_example: str
    risk_if_unfixed: str
    
    # Fix Recommendation (Agent 4)
    patch: str
    risk_assessment: str
    fix_summary: str
    patched_code: str
    
    # Test Generation (Agent 4.5)
    regression_tests: str
    
    # Pull Request (Agent 5)
    pr_url: Optional[str]
    
    # Sprint Planning (Agent 6)
    story_points: int
    priority: str
    sprint_recommendation: str
    
    # Orchestration tracing
    current_agent: str
    trace_logs: List[Dict[str, Any]]
