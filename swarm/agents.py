import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
from swarm.state import SwarmState

# Initialize Local LLMs via Ollama
primary_llm = ChatOllama(model="qwen2.5-coder:3b", temperature=0.0, format="json")
fallback_llm = ChatOllama(model="llama3.1:8b", temperature=0.0, format="json")

# --- Pydantic Models for Structured Output ---

class RootCauseOutput(BaseModel):
    severity: str = Field(description="The severity level of the issue.")
    affected_file: str = Field(description="The exact file affected.")
    vulnerable_code: str = Field(description="The exact snippet of vulnerable code.")
    root_cause: str = Field(description="One-sentence root cause explanation.")
    exploit_example: str = Field(description="A realistic exploit or failure scenario.")
    risk_if_unfixed: str = Field(description="The risk if the vulnerability is not fixed.")

class FixOutput(BaseModel):
    fix_summary: str = Field(description="A brief summary of the fix.")
    patched_code: str = Field(description="The exact snippet of the fixed code.")
    patch_diff: str = Field(description="A valid unified Git diff that can be applied directly to the repository.")

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
            
        state["suspected_functions"] = []
    except Exception as e:
        state["root_cause"] = f"Failed to generate root cause: {str(e)}"
        
    state["trace_logs"].append({"agent": "Root Cause Agent", "log": "Root cause identified."})
    return state


def fix_agent(state: SwarmState) -> SwarmState:
    """Agent 4: Generates a code patch to fix the bug."""
    state["current_agent"] = "Fix Recommendation Agent"
    
    parser = JsonOutputParser(pydantic_object=FixOutput)
    
    prompt = PromptTemplate(
        template=(
            "You are an elite Principal Software Engineer.\n"
            "Root Cause Analysis: {root_cause}\n"
            "Relevant Code Context:\n{repo_context}\n\n"
            "Generate a precise, production-ready code fix for this issue.\n"
            "CRITICAL: If fixing SQL injection, ALWAYS use parameterized queries (e.g., cursor.execute('... ?', (val,))) instead of simple type checking.\n"
            "You must output a valid unified Git diff that can be applied directly to the repository.\n\n"
            "{format_instructions}"
        ),
        input_variables=["root_cause", "repo_context"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = (prompt | primary_llm | parser).with_fallbacks([prompt | fallback_llm | parser])
    
    try:
        response = chain.invoke({
            "root_cause": state.get("root_cause", ""),
            "repo_context": state.get("repo_context", "")
        })
        
        if isinstance(response, dict):
            state["patch"] = response.get("patch_diff", "")
            state["fix_summary"] = response.get("fix_summary", "")
            state["patched_code"] = response.get("patched_code", "")
        else:
            state["patch"] = response.patch_diff
            state["fix_summary"] = response.fix_summary
            state["patched_code"] = response.patched_code
            
        if len(state["patch"]) > 5000:
            state["patch"] = state["patch"][:5000] + "\n... [PATCH TRUNCATED FOR DISPLAY]"
            
    except Exception as e:
        state["patch"] = f"Failed to generate patch: {str(e)}"
        
    state["trace_logs"].append({"agent": "Fix Recommendation Agent", "log": "Patch generated."})
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
