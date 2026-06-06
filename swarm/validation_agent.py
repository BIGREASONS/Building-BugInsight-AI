from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from swarm.state import SwarmState
from swarm.agents import primary_llm, fallback_llm
from swarm.functional_preservation import check_functional_preservation
from swarm.utils import fetch_original_file
import json
import os

class ValidationAgentOutput(BaseModel):
    score: int = Field(description="A confidence score between 0 and 100 indicating how well the patch addresses the finding.")
    approved: bool = Field(description="True if the patch safely addresses the root cause, False otherwise.")
    reasoning: str = Field(description="A concise explanation of why the patch was approved or rejected.")

def validation_agent(state: SwarmState) -> SwarmState:
    """Agent 4.25: Validates if the generated patch actually addresses the scanner findings."""
    state["current_agent"] = "Validation Agent"
    
    patched_code = state.get("patched_code", "") or state.get("patch", "")
    affected_file = state.get("affected_file", "")
    repo_url = state.get("repo_url", "")
    vulnerable_code = state.get("vulnerable_code", "")
    
    # 0. Hard Validation Gates (Fast-Fail)
    if vulnerable_code and vulnerable_code.strip() and vulnerable_code.strip() in patched_code:
        state["validation_score"] = 0
        state["is_patch_valid"] = False
        state["validation_reasoning"] = "Original vulnerable code still present in the patched output."
        if not state.get("first_failed_gate"):
            state["first_failed_gate"] = "hard_validation"
        state["trace_logs"].append({
            "agent": "Validation Agent", 
            "log": "Patch rejected (Hard Gate: Original vulnerable code still present)"
        })
        return state

    if "pickle.loads(" in patched_code:
        state["validation_score"] = 0
        state["is_patch_valid"] = False
        state["validation_reasoning"] = "Unsafe deserialization (pickle.loads) still present."
        if not state.get("first_failed_gate"):
            state["first_failed_gate"] = "hard_validation"
        state["trace_logs"].append({
            "agent": "Validation Agent", 
            "log": "Patch rejected (Hard Gate: Unsafe deserialization still present)"
        })
        return state

    # 1. Functional Preservation Check (Heuristic Fast-Fail)
    if repo_url and affected_file:
        try:
            original_code = fetch_original_file(repo_url, affected_file)
            is_valid, reason_or_warning = check_functional_preservation(original_code, patched_code, vulnerable_code)
            
            if not is_valid:
                state["validation_score"] = 0
                state["is_patch_valid"] = False
                state["validation_reasoning"] = reason_or_warning
                if not state.get("first_failed_gate"):
                    state["first_failed_gate"] = "functional_preservation"
                state["trace_logs"].append({
                    "agent": "Validation Agent", 
                    "log": "Patch rejected (Functional Preservation Failed)"
                })
                return state
        except Exception as e:
            state["validation_score"] = 0
            state["is_patch_valid"] = False
            state["validation_reasoning"] = f"Failed to fetch original file: {e}"
            if not state.get("first_failed_gate"):
                state["first_failed_gate"] = "functional_preservation"
            state["trace_logs"].append({
                "agent": "Validation Agent", 
                "log": f"Patch rejected (Fetch Failed: {e})"
            })
            return state
                
            # If valid, we might still have warnings
            if reason_or_warning:
                # Prepend the warning to the LLM's reasoning later
                state["functional_warnings"] = reason_or_warning

    # 2. LLM Security Check
    parser = JsonOutputParser(pydantic_object=ValidationAgentOutput)
    
    prompt = PromptTemplate(
        template=(
            "You are a strict Principal Security Architect.\n"
            "Review the proposed code patch against the security scanner findings.\n\n"
            "Scanner Findings: {scanner_findings}\n"
            "Root Cause Analysis: {root_cause}\n"
            "Proposed Code Fix:\n{patch}\n\n"
            "Task: Determine if the patch actually fixes the vulnerability without introducing new issues.\n"
            "CRITICAL: If the finding is SQL Injection, the patch MUST use parameterized queries. Simple string replacements or variable renames are invalid and should be rejected.\n"
            "Provide a score out of 100, an approval boolean, and a reasoning string.\n\n"
            "{format_instructions}"
        ),
        input_variables=["scanner_findings", "root_cause", "patch"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = (prompt | primary_llm | parser).with_fallbacks([prompt | fallback_llm | parser])
    
    try:
        response = chain.invoke({
            "scanner_findings": json.dumps(state.get("scanner_findings", [])),
            "root_cause": state.get("root_cause", ""),
            "patch": patched_code
        })
        
        if isinstance(response, dict):
            state["validation_score"] = response.get("score", 0)
            state["is_patch_valid"] = response.get("approved", False)
            state["validation_reasoning"] = response.get("reasoning", "")
        else:
            state["validation_score"] = response.score
            state["is_patch_valid"] = response.approved
            state["validation_reasoning"] = response.reasoning
            
        if not state["is_patch_valid"] and not state.get("first_failed_gate"):
            state["first_failed_gate"] = "validation"
            
        if state.get("functional_warnings"):
            state["validation_reasoning"] = state["functional_warnings"] + "\n\n" + state["validation_reasoning"]
            
    except Exception as e:
        state["validation_score"] = 0
        state["is_patch_valid"] = False
        state["validation_reasoning"] = f"Validation failed due to error: {str(e)}"
        if not state.get("first_failed_gate"):
            state["first_failed_gate"] = "validation"
        
    state["trace_logs"].append({
        "agent": "Validation Agent", 
        "log": f"Patch validated: {'Approved' if state.get('is_patch_valid') else 'Rejected'} ({state.get('validation_score')}/100)"
    })
    return state
