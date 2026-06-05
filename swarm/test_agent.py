from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from swarm.state import SwarmState
from swarm.agents import primary_llm, fallback_llm

class TestAgentOutput(BaseModel):
    regression_tests: str = Field(description="The complete regression test code to verify the fix.")

def test_agent(state: SwarmState) -> SwarmState:
    """Agent 4.5: Generates regression tests for the generated fix."""
    state["current_agent"] = "Test Agent"
    
    parser = JsonOutputParser(pydantic_object=TestAgentOutput)
    
    prompt = PromptTemplate(
        template=(
            "You are a Senior QA Automation Engineer.\n"
            "Root Cause Analysis: {root_cause}\n"
            "Proposed Code Fix:\n{patch}\n\n"
            "Generate executable regression tests to verify this fix.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Do not assume database contents exist.\n"
            "- Do not assume user IDs are hardcoded to 1.\n"
            "- Test the actual exploit payload (e.g., passing `' OR 1=1 --` into inputs).\n"
            "- The test should be designed to fail before the fix and pass after the fix.\n"
            "- Output the raw unit test code (e.g., using pytest, unittest, or appropriate framework).\n\n"
            "{format_instructions}"
        ),
        input_variables=["root_cause", "patch"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    chain = (prompt | primary_llm | parser).with_fallbacks([prompt | fallback_llm | parser])
    
    try:
        response = chain.invoke({
            "root_cause": state.get("root_cause", ""),
            "patch": state.get("patch", "")
        })
        
        if isinstance(response, dict):
            state["regression_tests"] = response.get("regression_tests", "")
        else:
            state["regression_tests"] = response.regression_tests
            
    except Exception as e:
        state["regression_tests"] = f"# Failed to generate tests: {str(e)}"
        
    state["trace_logs"].append({"agent": "Test Agent", "log": "Regression tests generated."})
    return state
