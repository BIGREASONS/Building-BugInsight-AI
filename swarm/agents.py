import os
import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from swarm.state import SwarmState
from swarm.utils import fetch_original_file, generate_unified_diff

import logging

logger = logging.getLogger(__name__)

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "qwen2.5-coder:3b")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama3.1:8b")

logger.info(f"Using PRIMARY_MODEL={PRIMARY_MODEL}")
logger.info(f"Using FALLBACK_MODEL={FALLBACK_MODEL}")

# Initialize Local LLMs via Ollama
primary_llm = ChatOllama(model=PRIMARY_MODEL, temperature=0.0, format="json")
fallback_llm = ChatOllama(model=FALLBACK_MODEL, temperature=0.0, format="json")

# Text-based LLMs for SWS
primary_llm_text = ChatOllama(model=PRIMARY_MODEL, temperature=0.0)
fallback_llm_text = ChatOllama(model=FALLBACK_MODEL, temperature=0.0)

# --- Pydantic Models for Structured Output ---

class RootCauseOutput(BaseModel):
    severity: str = Field(description="The severity level of the issue.")
    affected_file: str = Field(description="The exact file affected.")
    vulnerable_code: str = Field(description="The exact snippet of vulnerable code.")
    root_cause: str = Field(description="One-sentence root cause explanation.")
    exploit_example: str = Field(description="A realistic exploit or failure scenario.")
    risk_if_unfixed: str = Field(description="The risk if the vulnerability is not fixed.")

class FixOutput(BaseModel):
    vulnerability_type: str = Field(description="The type of vulnerability being fixed.")
    modified_functions: List[str] = Field(description="A list of functions being modified.")
    reasoning: str = Field(description="Reasoning behind the changes.")
    fix_summary: str = Field(description="A brief summary of the fix.")
    file_path: str = Field(description="The file path being modified.")
    full_file_content: str = Field(description="The COMPLETE modified file, preserving all original code except the fix.")

class RepairOutput(BaseModel):
    vulnerability_type: str = Field(description="The type of vulnerability being fixed.")
    modified_functions: List[str] = Field(description="A list of functions being modified.")
    reasoning: str = Field(description="Reasoning explaining how the patch addresses the validation failure.")
    full_file_content: str = Field(description="The COMPLETE modified file, preserving all original code except the fix.")

class SprintPlanningOutput(BaseModel):
    story_points: int = Field(description="Estimated engineering effort in Story Points (1, 2, 3, 5, 8).")
    priority: str = Field(description="Recommended priority level (e.g., P0, P1, P2).")
    recommendation: str = Field(description="General sprint recommendation.")
    time_saved_hours: float = Field(description="Estimated developer time saved by generating this fix automatically, in hours (e.g., 2.3).")
class SeverityReasoningOutput(BaseModel):
    reasoning_bullets: List[str] = Field(description="Exactly 3 short bullet points explaining why the issue is this severity.")

def generate_severity_reasoning(issue: str, severity: str) -> List[str]:
    """Helper to generate 3 bullet points explaining the CodeBERT severity prediction."""
    if not issue or severity == "Unknown":
        return []
        
    parser = JsonOutputParser(pydantic_object=SeverityReasoningOutput)
    prompt = PromptTemplate(
        template=(
            "An AI model predicted the severity of this software issue as {severity}.\n"
            "Issue: {issue}\n\n"
            "Explain exactly WHY it received this severity in exactly 3 short, technical bullet points.\n"
            "{format_instructions}"
        ),
        input_variables=["issue", "severity"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    chain = (prompt | primary_llm | parser).with_fallbacks([prompt | fallback_llm | parser])
    
    try:
        response = chain.invoke({"issue": issue, "severity": severity})
        if isinstance(response, dict):
            return response.get("reasoning_bullets", [])
        return response.reasoning_bullets
    except:
        return ["Impacts core functionality", "Potential security or stability risk", "Requires immediate attention"]

# --- Agents ---

def root_cause_agent(state: SwarmState) -> SwarmState:
    """Agent 3: Analyzes context and issue to find the root cause."""
    state["current_agent"] = "Root Cause Agent"
    
    parser = JsonOutputParser(pydantic_object=RootCauseOutput)
    
    prompt = PromptTemplate(
        template=(
            "You are a staff software engineer performing a production incident review.\n"
            "Issue: {issue}\n"
            "Severity: {severity}\n"
            "Scanner Findings (Semgrep/Bandit):\n{scanner_findings}\n"
            "Code Context:\n{repo_context}\n\n"
            "Return:\n"
            "1. Exact affected file\n"
            "2. Exact vulnerable code block\n"
            "3. One-sentence root cause\n"
            "4. One realistic exploit/failure scenario\n"
            "5. Severity justification\n\n"
            "Never describe the issue abstractly.\n"
            "Always quote the actual code responsible.\n\n"
            "{format_instructions}"
        ),
        input_variables=["issue", "severity", "scanner_findings", "repo_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = (prompt | primary_llm | parser).with_fallbacks([prompt | fallback_llm | parser])
    
    try:
        response: RootCauseOutput = chain.invoke({
            "issue": state.get("issue_text", ""),
            "severity": state.get("severity", "Unknown"),
            "scanner_findings": str(state.get("scanner_findings", [])),
            "repo_context": state.get("repo_context", "")
        })
        
        if isinstance(response, dict):
            state["affected_file"] = response.get("affected_file", "")
            state["vulnerable_code"] = response.get("vulnerable_code", "")
            state["root_cause"] = response.get("root_cause", "")
            state["exploit_example"] = response.get("exploit_example", "")
            state["risk_if_unfixed"] = response.get("risk_if_unfixed", "")
            state["suspected_files"] = [state["affected_file"]] if state["affected_file"] else []
        else:
            state["affected_file"] = response.affected_file
            state["vulnerable_code"] = response.vulnerable_code
            state["root_cause"] = response.root_cause
            state["exploit_example"] = response.exploit_example
            state["risk_if_unfixed"] = response.risk_if_unfixed
            state["suspected_files"] = [response.affected_file] if response.affected_file else []
            
        # FORCE TARGET FILE IF PROVIDED
        if state.get("target_file"):
            state["affected_file"] = state["target_file"]
            state["suspected_files"] = [state["target_file"]]
            
        state["suspected_functions"] = []
    except Exception as e:
        state["root_cause"] = f"Failed to generate root cause: {str(e)}"
        
    state["trace_logs"].append({"agent": "Root Cause Agent", "log": "Root cause identified."})
    return state


from langchain_core.output_parsers import StrOutputParser
from swarm.sws_extractor import extract_surgical_window, parse_and_splice
import ast

def fix_agent(state: SwarmState) -> SwarmState:
    """Agent 4: Generates a code patch to fix the bug using Surgical Window Splicing."""
    state["current_agent"] = "Fix Recommendation Agent"
    
    repo_url = state.get("repo_url", "")
    affected_file = state.get("target_file") or state.get("affected_file", "")
    
    # Load the ENTIRE file into context
    original_file_content = ""
    if repo_url and affected_file:
        try:
            original_file_content = fetch_original_file(repo_url, affected_file)
        except Exception as e:
            state["patch"] = f"Failed to generate patch: {str(e)}"
            state["trace_logs"].append({"agent": "Fix Recommendation Agent", "log": str(e)})
            return state
        
    state["original_file_content"] = original_file_content
    
    # Extract Surgical Window
    # Find the target line from scanner findings, default to 1 if none
    target_line = 1
    if state.get("scanner_findings"):
        target_line = state["scanner_findings"][0].get("line", 1)
        
    window_data = extract_surgical_window(original_file_content, target_line)
    
    # User requested logging to verify what is actually being fed to the LLM
    logger.info(f"--- SWS EXTRACTION VERIFICATION ---")
    logger.info(f"TARGET FILE: {affected_file}")
    logger.info(f"TARGET LINE: {target_line}")
    logger.info(f"EXTRACTED WINDOW:\n{window_data.get('window_code', '')}")
    logger.info(f"VULNERABILITY TYPE (Root Cause): {state.get('root_cause', '')}")
    logger.info(f"-----------------------------------")
    
    # Save window data to state so Repair Agent can reuse it
    state["sws_window_data"] = window_data
    
    # Phase 2b: Vulnerability-Specific Remediation Library
    import json
    import os
    rules_path = os.path.join(os.path.dirname(__file__), "remediation_rules.json")
    remediation_guidance = ""
    try:
        if os.path.exists(rules_path):
            with open(rules_path, "r") as f:
                rules = json.load(f)
            root_cause_lower = state.get("root_cause", "").lower()
            issue_lower = state.get("issue_text", "").lower()
            for key, data in rules.items():
                if any(kw in root_cause_lower or kw in issue_lower for kw in data["keywords"]):
                    remediation_guidance += f"\nSECURITY RULE FOR THIS VULNERABILITY:\n{data['rule']}\n"
    except Exception as e:
        logger.error(f"Error loading remediation rules: {e}")
    
    prompt_template = (
        "You are a security remediation specialist.\n"
        "Your task is to fix a security vulnerability found in the following localized code block.\n\n"
        "Function Name: {function_name}\n"
        "ORIGINAL CODE WINDOW (Lines {start_line} to {end_line}):\n"
        "```python\n"
        "{isolated_window}\n"
        "```\n\n"
        "VULNERABILITY DETAILS:\n"
        "- Root Cause: {root_cause}\n"
        "- Description: {issue}\n"
        "{remediation_guidance}\n"
        "INSTRUCTIONS:\n"
        "1. Rewrite the code window to fully resolve the vulnerability.\n"
        "2. Preserve original logic, variables, and function signatures. You may ONLY modify this function.\n"
        "3. Do not modify global variables or other classes.\n"
        "4. If your fix requires adding a new import (e.g., `import re`), do NOT write it in the code block. List it inside the <required_imports> tag.\n"
        "5. Return ONLY the new, corrected block of code inside <fixed_window> and <required_imports> tags. No explanations.\n\n"
        "Example response format:\n"
        "<required_imports>\n"
        "import hashlib\n"
        "</required_imports>\n"
        "<fixed_window>\n"
        "def verify_hash(user_input):\n"
        "    return hashlib.sha256(user_input.encode()).hexdigest()\n"
        "</fixed_window>\n"
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["function_name", "start_line", "end_line", "isolated_window", "root_cause", "issue", "remediation_guidance"]
    )
    chain = (prompt | primary_llm_text | StrOutputParser()).with_fallbacks([prompt | fallback_llm_text | StrOutputParser()])
    
    try:
        response_text = chain.invoke({
            "function_name": window_data["function_name"],
            "start_line": window_data["start_line"] + 1,
            "end_line": window_data["end_line"],
            "isolated_window": window_data["window_code"],
            "root_cause": state.get("root_cause", ""),
            "issue": state.get("issue_text", ""),
            "remediation_guidance": remediation_guidance
        })
        
        state["raw_fix_output"] = response_text
        state["fix_summary"] = f"Patched function {window_data['function_name']}"
        
        # SWS deterministic splice
        try:
            patched_code, fixed_window, new_imports = parse_and_splice(
                original_file_content, 
                response_text, 
                window_data["start_line"], 
                window_data["end_line"]
            )
            
            state["patched_code"] = patched_code
            state["generated_patch_lines"] = len(patched_code.splitlines())
            state["original_lines"] = len(original_file_content.splitlines())
            
            # Programmatically generate unified diff
            diff_str = generate_unified_diff(original_file_content, patched_code, affected_file)
            state["patch"] = diff_str
            
            state["files_modified"] = [
                {
                    "file": affected_file,
                    "content": patched_code
                }
            ]
            
            if len(state["patch"]) > 5000:
                state["patch"] = state["patch"][:5000] + "\n... [PATCH TRUNCATED FOR DISPLAY]"
                
        except SyntaxError as e:
            # SWS Splice failed AST check
            state["patch"] = f"AST Error: {e}"
            state["patched_code"] = original_file_content # Leave unchanged so Validation fails it safely
            state["is_patch_valid"] = False
            state["validation_reasoning"] = f"Functional Preservation Failed: SyntaxError in fixed window: {e}"
            if not state.get("first_failed_gate"):
                state["first_failed_gate"] = "functional_preservation"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        state["patch"] = f"Failed to generate patch: {str(e)}"
        state["files_modified"] = []
        
    state["trace_logs"].append({"agent": "Fix Recommendation Agent", "log": "Patch generated."})
    return state


def repair_agent(state: SwarmState) -> SwarmState:
    """Agent 4.1: Repairs a patch that failed validation using SWS."""
    state["current_agent"] = "Repair Agent"
    
    # Increment repair attempts
    current_attempts = state.get("repair_attempts", 0)
    state["repair_attempts"] = current_attempts + 1
    
    affected_file = state.get("affected_file", "")
    original_file_content = state.get("original_file_content", "")
    validation_reasoning = state.get("validation_reasoning", "")
    
    window_data = state.get("sws_window_data")
    if not window_data:
        # Fallback if somehow called directly
        target_line = state["scanner_findings"][0].get("line", 1) if state.get("scanner_findings") else 1
        window_data = extract_surgical_window(original_file_content, target_line)
        
    previous_response = state.get("raw_fix_output", "")
    
    # Phase 2b: Vulnerability-Specific Remediation Library
    import json
    import os
    rules_path = os.path.join(os.path.dirname(__file__), "remediation_rules.json")
    remediation_guidance = ""
    try:
        if os.path.exists(rules_path):
            with open(rules_path, "r") as f:
                rules = json.load(f)
            root_cause_lower = state.get("root_cause", "").lower()
            issue_lower = state.get("issue_text", "").lower()
            for key, data in rules.items():
                if any(kw in root_cause_lower or kw in issue_lower for kw in data["keywords"]):
                    remediation_guidance += f"\nSECURITY RULE FOR THIS VULNERABILITY:\n{data['rule']}\n"
    except Exception as e:
        logger.error(f"Error loading remediation rules: {e}")
    
    prompt_template = (
        "You are a security remediation specialist repairing a broken code patch.\n"
        "Your previous patch failed validation. Produce a corrected patch that fixes the vulnerability AND addresses the validation failure.\n\n"
        "Function Name: {function_name}\n"
        "ORIGINAL CODE WINDOW:\n"
        "```python\n"
        "{isolated_window}\n"
        "```\n\n"
        "YOUR PREVIOUS FAILED PATCH ATTEMPT:\n"
        "{previous_patch}\n\n"
        "VALIDATION FAILURE REASON (Fix this!):\n"
        "{validation_reasoning}\n"
        "{remediation_guidance}\n"
        "INSTRUCTIONS:\n"
        "1. Rewrite the code window to fully resolve the vulnerability AND the syntax/validation errors above.\n"
        "2. Preserve original logic, variables, and function signatures. You may ONLY modify this function.\n"
        "3. If your fix requires adding a new import, list it inside the <required_imports> tag.\n"
        "4. Return ONLY the new, corrected block of code inside <fixed_window> and <required_imports> tags. No explanations.\n"
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["function_name", "isolated_window", "previous_patch", "validation_reasoning", "remediation_guidance"]
    )
    chain = (prompt | primary_llm_text | StrOutputParser()).with_fallbacks([prompt | fallback_llm_text | StrOutputParser()])
    
    try:
        response_text = chain.invoke({
            "function_name": window_data["function_name"],
            "isolated_window": window_data["window_code"],
            "previous_patch": previous_response,
            "validation_reasoning": validation_reasoning,
            "remediation_guidance": remediation_guidance
        })
        
        state["raw_fix_output"] = response_text
        
        # SWS deterministic splice
        try:
            patched_code, fixed_window, new_imports = parse_and_splice(
                original_file_content, 
                response_text, 
                window_data["start_line"], 
                window_data["end_line"]
            )
            
            state["patched_code"] = patched_code
            state["generated_patch_lines"] = len(patched_code.splitlines())
            
            # Update patch unified diff
            diff_str = generate_unified_diff(original_file_content, patched_code, affected_file)
            state["patch"] = diff_str
            
            state["files_modified"] = [
                {
                    "file": affected_file,
                    "content": patched_code
                }
            ]
            
            if len(state["patch"]) > 5000:
                state["patch"] = state["patch"][:5000] + "\n... [PATCH TRUNCATED FOR DISPLAY]"
                
        except SyntaxError as e:
            state["patch"] = f"AST Error: {e}"
            state["patched_code"] = original_file_content # Leave unchanged
            state["is_patch_valid"] = False
            state["validation_reasoning"] = f"Functional Preservation Failed: SyntaxError in repaired window: {e}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        state["patch"] = f"Failed to generate repair patch: {str(e)}"
        
    state["trace_logs"].append({"agent": "Repair Agent", "log": f"Repair attempt {state['repair_attempts']} completed."})
    return state


def sprint_planning_agent(state: SwarmState) -> SwarmState:
    """Agent 5: Generates sprint points and priority, along with business impact."""
    state["current_agent"] = "Sprint Planning Agent"
    
    parser = JsonOutputParser(pydantic_object=SprintPlanningOutput)
    
    prompt = PromptTemplate(
        template=(
            "You are a Technical Product Manager.\n"
            "Issue: {issue}\n"
            "Severity: {severity}\n"
            "Proposed Fix (Unified Diff):\n{patch}\n\n"
            "Provide sprint planning recommendations. Estimate the engineering effort in Story Points.\n"
            "Also estimate the developer time saved by having the AI generate this fix automatically.\n\n"
            "{format_instructions}"
        ),
        input_variables=["issue", "severity", "patch"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = (prompt | primary_llm | parser).with_fallbacks([prompt | fallback_llm | parser])
    
    try:
        response = chain.invoke({
            "issue": state.get("issue_text", ""),
            "severity": state.get("severity", ""),
            "patch": state.get("patch", "")
        })
        
        if isinstance(response, dict):
            state["story_points"] = response.get("story_points", 0)
            state["priority"] = response.get("priority", "")
            recommendation = response.get("recommendation", "")
            time_saved = response.get("time_saved_hours", 0.0)
        else:
            state["story_points"] = response.story_points
            state["priority"] = response.priority
            recommendation = response.recommendation
            time_saved = response.time_saved_hours
            
        state["sprint_recommendation"] = (
            f"{recommendation}\n\n"
            f"Estimated developer time saved: {time_saved} hours"
        )
    except Exception as e:
        state["sprint_recommendation"] = f"Failed to generate sprint plan: {str(e)}"
        
    state["trace_logs"].append({"agent": "Sprint Planning Agent", "log": "Sprint plan generated."})
    return state
