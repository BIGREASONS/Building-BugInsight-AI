from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from swarm.state import SwarmState
from swarm.agents import primary_llm, fallback_llm
import json

class ValidationAgentOutput(BaseModel):
    score: int = Field(description="A confidence score between 0 and 100 indicating how well the patch addresses the finding.")
    approved: bool = Field(description="True if the patch safely addresses the root cause, False otherwise.")
    reasoning: str = Field(description="A concise explanation of why the patch was approved or rejected.")

def validation_agent(state: SwarmState) -> SwarmState:
    """Agent 4.25: Validates if the generated patch actually addresses the scanner findings."""
    state["current_agent"] = "Validation Agent"
    
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
            "patch": state.get("patched_code", "") or state.get("patch", "")
        })
        
        if isinstance(response, dict):
            state["validation_score"] = response.get("score", 0)
            state["is_patch_valid"] = response.get("approved", False)
            state["validation_reasoning"] = response.get("reasoning", "")
        else:
            state["validation_score"] = response.score
            state["is_patch_valid"] = response.approved
            state["validation_reasoning"] = response.reasoning
            
    except Exception as e:
        state["validation_score"] = 0
        state["is_patch_valid"] = False
        state["validation_reasoning"] = f"Validation failed due to error: {str(e)}"
        
    state["trace_logs"].append({
        "agent": "Validation Agent", 
        "log": f"Patch validated: {'Approved' if state.get('is_patch_valid') else 'Rejected'} ({state.get('validation_score')}/100)"
    })
    return state
