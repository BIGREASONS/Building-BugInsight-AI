import os
import uuid
import re
from github import Github
from swarm.state import SwarmState

def github_action_agent(state: SwarmState) -> SwarmState:
    """Agent 6: Uses PyGithub to create a branch/PR or comment on an existing PR."""
    state["current_agent"] = "GitHub Action Agent"
    
    token = os.environ.get("GITHUB_TOKEN")
    github_mode = os.environ.get("GITHUB_MODE", "SAFE").upper()
    pr_mode = os.environ.get("BUGINSIGHT_PR_MODE", "COMMENT").upper()
    
    repo_url = state.get("repo_url", "")
    pr_number = state.get("pr_number")
    files_modified = state.get("files_modified", [])
    
    # Gating checks
    is_valid = state.get("is_patch_valid", False)
    tests_passed = state.get("tests_passed", False)
    rescan_passed = state.get("rescan_passed", False)
    
    repo_name = repo_url.rstrip("/").split("github.com/")[-1] if "github.com/" in repo_url else ""
    
    # Load allowed repos from environment
    allowed_repos_env = os.environ.get("LIVE_MODE_ENABLED_REPOS", "")
    allowed_repos = [r.strip() for r in allowed_repos_env.split(",") if r.strip()]
    
    # Check if we should fallback to Mock PR
    print("TOKEN_PRESENT =", bool(token))
    print("GITHUB_MODE =", github_mode)
    print("REPO_URL =", repo_url)
    print("REPO_NAME =", repo_name)
    
    if not token or github_mode != "LIVE" or not repo_url or not repo_name:
        reason = "LIVE mode conditions not met (requires LIVE mode and valid token)."
        state["pr_url"] = f"https://github.com/mock-org/mock-repo/pull/{uuid.uuid4().hex[:6]}"
        state["pr_mode"] = "mock"
    elif allowed_repos and repo_name not in allowed_repos:
        reason = f"Repo '{repo_name}' is not in LIVE_MODE_ENABLED_REPOS list."
        state["pr_url"] = f"https://github.com/mock-org/mock-repo/pull/{uuid.uuid4().hex[:6]}"
        state["pr_mode"] = "mock"
    elif not is_valid or not rescan_passed or not files_modified:
        print(f"DEBUG GATE FAILURES: is_valid={is_valid}, rescan_passed={rescan_passed}, files_modified={bool(files_modified)}")
        print(f"DEBUG VALIDATION REASONING: {state.get('validation_reasoning')}")
        print(f"DEBUG PATCH: {state.get('patch')}")
        reason = "Security gates failed: Requires Validation=PASS, Rescan=PASS."
        state["pr_url"] = f"https://github.com/mock-org/mock-repo/pull/{uuid.uuid4().hex[:6]}"
        state["pr_mode"] = "mock"
    else:
        reason = None
        
    if reason:
        print(f"DEBUG MOCK REASON: {reason}")
        state["github_error"] = reason
        state["trace_logs"].append({
            "agent": "GitHub Action Agent", 
            "log": f"[MOCK] Created PR/Comment. Reason: {reason}"
        })
        return state
        
    # Format the Proof Report body
    finding = state.get("scanner_findings", [{}])[0] if state.get("scanner_findings") else {}
    rule = finding.get("rule", "Unknown Rule")
    severity = finding.get("severity", "Unknown Severity")
    
    body = f"""# BugInsight Security Remediation

## Scanner Findings
Rule: `{rule}`
Severity: **{severity.upper()}**
Affected File: `{files_modified[0].get('file', 'Unknown') if files_modified else 'Unknown'}`

## Root Cause
{state.get('root_cause', '')}

## Validation
Score: **{state.get('validation_score', 0)}/100**
Approved: **{"YES" if state.get('is_patch_valid') else "NO"}**

## Test Execution
Tests Passed: **{"YES" if state.get('tests_passed') else "NO"}**
Duration: {state.get('test_results', {}).get('duration', 0)}s

## Auto Rescan
Rescan Passed: **{"YES" if state.get('rescan_passed') else "NO"}**

## Proposed Patch
```diff
{state.get('patch', 'No patch generated.')}
```

## Fix Summary
{state.get('fix_summary', '')}

---
*Generated autonomously by [BugInsight Swarm](https://github.com/BIGREASONS/BugInsight-Swarm)*
"""

    try:
        g = Github(token)
        repo = g.get_repo(repo_name)
        
        # Check benchmark mode
        if os.environ.get("BENCHMARK_MODE") == "TRUE" or os.environ.get("BUGINSIGHT_PR_MODE") == "DISABLED":
            state["pr_mode"] = "benchmark"
            state["trace_logs"].append({
                "agent": "GitHub Action Agent", 
                "log": "Benchmark mode active. Suppressing GitHub writeback."
            })
            return state

        if pr_mode == "COMMENT":
            if not pr_number:
                raise ValueError("pr_number is required when BUGINSIGHT_PR_MODE=COMMENT")
            pull_request = repo.get_pull(int(pr_number))
            comment = pull_request.create_issue_comment(body)
            state["pr_url"] = comment.html_url
            state["pr_mode"] = "comment"
            state["trace_logs"].append({
                "agent": "GitHub Action Agent", 
                "log": f"Successfully commented on LIVE PR #{pr_number}: {state['pr_url']}"
            })
            return state

        # 1. Get default branch SHA
        default_branch = repo.default_branch
        ref = repo.get_git_ref(f"heads/{default_branch}")
        
        # 2. Create new branch
        short_id = uuid.uuid4().hex[:6]
        finding_rule = "security-fix"
        if state.get("scanner_findings") and len(state["scanner_findings"]) > 0:
            finding_rule = state["scanner_findings"][0].get("rule", "security-fix").split(".")[-1]
            
        # Sanitize rule name for branch
        finding_rule = re.sub(r'[^a-zA-Z0-9-]', '-', finding_rule).strip('-').lower()
        if not finding_rule:
            finding_rule = "security-fix"
            
        new_branch_name = f"buginsight/{finding_rule}-{short_id}"
        repo.create_git_ref(ref=f"refs/heads/{new_branch_name}", sha=ref.object.sha)
        
        # 3. Apply files
        for fm in files_modified:
            file_path = fm.get("file")
            content = fm.get("content")
            
            if not file_path or not content:
                continue
                
            file_path = file_path.replace("\\", "/")
            try:
                contents = repo.get_contents(file_path, ref=default_branch)
                file_sha = contents.sha
                repo.update_file(
                    path=file_path, 
                    message=f"fix: autogenerated fix for {finding_rule}", 
                    content=content, 
                    sha=file_sha, 
                    branch=new_branch_name
                )
            except Exception as e:
                raise Exception(f"Failed to update file {file_path}: {str(e)}")
                
        # 4. Open PR
        title = f"Fix: {rule} in {files_modified[0].get('file', 'codebase')}"
        pr = repo.create_pull(title=title, body=body, head=new_branch_name, base=default_branch)
        state["pr_url"] = pr.html_url
        state["pr_mode"] = "live"
        
        state["trace_logs"].append({
            "agent": "GitHub Action Agent", 
            "log": f"Successfully created LIVE Pull Request: {state['pr_url']}"
        })
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception("GitHub Agent failed")
        state["github_error"] = repr(e)
        # Fallback to mock mode on error
        mock_id = uuid.uuid4().hex[:6]
        state["pr_url"] = f"https://github.com/mock-org/mock-repo/pull/{mock_id}"
        state["pr_mode"] = "mock"
        state["trace_logs"].append({
            "agent": "GitHub Action Agent", 
            "log": f"[MOCK] LIVE mode failed ({type(e).__name__}: {str(e)}). Generated mock PR."
        })
        
    return state
