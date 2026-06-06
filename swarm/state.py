from typing import TypedDict, List, Dict, Any, Optional

class SwarmState(TypedDict):
    """The state dictionary passed between all agents in the LangGraph."""
    
    # Input
    job_id: str
    issue_url: Optional[str]
    repo_url: Optional[str]
    target_file: Optional[str]
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
    files_modified: List[Dict[str, str]]
    
    # Validation (Agent 4.25)
    validation_score: int
    is_patch_valid: bool
    validation_reasoning: str
    repair_attempts: int
    repaired_successfully: bool

    # Test Generation (Agent 4.5)
    regression_tests: str
    
    # Diagnostic Fields (Phase A)
    raw_fix_output: str
    generated_patch_lines: int
    original_lines: int
    first_failed_gate: Optional[str]
    failure_category: Optional[str]
    failure_reason: Optional[str]

    # Test Execution (Agent 4.75)
    test_results: Dict[str, Any]
    tests_passed: bool
    
    # Auto-Rescan (Agent 4.8)
    rescan_passed: bool
    
    # Pull Request (Agent 5)
    pr_url: Optional[str]
    pr_mode: str
    github_error: str
    pr_number: Optional[int]
    
    # Sprint Planning (Agent 6)
    story_points: int
    priority: str
    sprint_recommendation: str
    
    # Orchestration tracing
    current_agent: str
    trace_logs: List[Dict[str, Any]]
