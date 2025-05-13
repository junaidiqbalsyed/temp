"""
feedback.py

Agentic, modular, and optimized financial analysis pipeline using Focused ReAct.
- Modular sub-agents for preprocessing, ReAct reasoning, and summarization.
- Focused ReAct: Early stop and reiteration to avoid loops and stay on user query [1].
- User feedback is a goal and a constraint, ensuring insights are richer than feedback.
- Designed for concurrent/scalable execution.
"""

import pandas as pd
import numpy as np
import logging
import time
import re
import json
from typing import Dict, Any, Optional, TypedDict, List

# LangChain/LangGraph imports (assumed installed)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# Optional dateparser for robust parsing
import dateparser

# --- Logging Setup ---
logger = logging.getLogger('focused_react_feedback')
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --- State Definition ---
class ResearchAgentState(TypedDict):
    raw_data_dict: Dict[str, Any]
    user_feedback_str: Optional[str]
    context_str: str
    llm: Optional[ChatOpenAI]
    cleaned_feedback: Optional[str]
    dataframe: Optional[pd.DataFrame]
    react_observations: Optional[List[str]]
    core_bullets: Optional[List[str]]
    final_report: Optional[str]
    error_message: Optional[str]
    total_tokens: int

# --- Preprocessing Sub-Agent ---
def clean_feedback(feedback: Optional[str]) -> str:
    return re.sub(r'\s+', ' ', feedback).strip() if feedback else ""


def parse_date_column(df: pd.DataFrame, llm: Optional[ChatOpenAI], tokens: int) -> (pd.DataFrame, int):
    """Parse 'date' column with algorithmic, dateparser, and LLM-assisted fallback."""
    if "date" not in df.columns:
        return df, tokens
    # Algorithmic parsing
    try:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if df['date'].notna().mean() > 0.5:
            logger.info("Algorithmic date parsing succeeded.")
            return df, tokens
    except Exception:
        pass
    # dateparser fallback
    try:
        df['date'] = df['date'].apply(lambda x: dateparser.parse(str(x)))
        if df['date'].notna().mean() > 0.5:
            logger.info("Dateparser fallback succeeded.")
            return df, tokens
    except Exception:
        pass
    # LLM fallback
    if llm:
        sample = [str(x) for x in df['date'].dropna().unique()[:10]]
        prompt = (
            "Given these sample values from a column named 'date', "
            "what is the pandas strftime format string to parse them? "
            "If ambiguous, reply 'NOT_A_DATE'.\n" + "\n".join(sample)
        )
        messages = [SystemMessage(content="You are a date format expert."), HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        fmt = response.content.strip()
        tokens += getattr(response, "response_metadata", {}).get("token_usage", {}).get("total_tokens", 0)
        if fmt and fmt != "NOT_A_DATE":
            try:
                df['date'] = pd.to_datetime(df['date'].astype(str), format=fmt, errors='coerce')
                logger.info(f"LLM date parsing with format '{fmt}' succeeded.")
            except Exception:
                logger.warning(f"LLM date parsing with format '{fmt}' failed.")
    return df, tokens


def preprocess_data_node(state: ResearchAgentState) -> ResearchAgentState:
    """Loads data, validates structure, cleans feedback, parses 'date' column robustly."""
    logger.info("Preprocessing: validating data and parsing date column.")
    raw = state.get('raw_data_dict')
    # Data validation
    if not isinstance(raw, dict) or not raw:
        return {**state, "error_message": "No valid data provided.", "dataframe": None}
    # Ensure columns all have equal lengths
    lengths = [len(v) for v in raw.values() if hasattr(v, '__len__')]
    if lengths and len(set(lengths)) != 1:
        return {**state, "error_message": "Inconsistent column lengths.", "dataframe": None}
    df = pd.DataFrame(raw)
    cleaned_feedback = clean_feedback(state.get("user_feedback_str"))
    tokens = state.get("total_tokens", 0)
    df, tokens = parse_date_column(df, state.get("llm"), tokens)
    if df.empty:
        return {**state, "error_message": "No insights as there is no data!", "dataframe": df, "cleaned_feedback": cleaned_feedback, "total_tokens": tokens}
    return {**state, "dataframe": df, "cleaned_feedback": cleaned_feedback, "total_tokens": tokens, "error_message": None}

# --- Python Executor Tool for ReAct ---
class PythonToolInput(BaseModel):
    code_snippet: str = Field(description="Python code using pandas DataFrame 'df'. Must print output.")

class PythonPandasExecutor(BaseTool):
    name: str = "PythonPandasExecutor"
    description: str = "Executes pandas code on 'df'. Only print output. No plotting or side effects."
    args_schema: type = PythonToolInput
    dataframe: Optional[pd.DataFrame] = None
    def _run(self, code_snippet: str) -> str:
        df = self.dataframe.copy() if self.dataframe is not None else None
        if df is None or df.empty:
            return "No DataFrame."
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(code_snippet, {"df": df, "pd": pd, "np": np})
            return mystdout.getvalue().strip() or "No output."
        except Exception as e:
            return f"Error: {e}"
        finally:
            sys.stdout = old_stdout

# --- Focused ReAct Insight Sub-Agent ---
def react_data_insight_node(state: ResearchAgentState) -> ResearchAgentState:
    """
    Focused ReAct: Iteratively generate insights using structured JSON responses,
    with early stop and reiteration to avoid loops and handle empty data.
    """
    logger.info("ReAct Insight Agent: Starting Focused ReAct loop.")
    llm = state.get("llm")
    df = state.get("dataframe")
    feedback = state.get("cleaned_feedback", "")
    tokens = state.get("total_tokens", 0)
    if df is None or df.empty:
        return {**state, "error_message": "No insights as there is no data!", "react_observations": []}
    if llm is None:
        return {**state, "error_message": "No LLM for ReAct.", "react_observations": []}

    tool = PythonPandasExecutor(dataframe=df)
    max_steps = 7
    observations = []
    focus = feedback if feedback else "Provide the most advanced and actionable financial insights possible."
    last_input = None
    for i in range(max_steps):
        prompt = {
            "thought": "What is the next best question or hypothesis to test?",
            "context": state.get("context_str", ""),
            "previous_observations": observations[-3:],
            "focus": focus
        }
        messages = [SystemMessage(content="Respond with a JSON object: {\"thought\":...,\"action\":...,\"action_input\":...}"),
                    HumanMessage(content=json.dumps(prompt))]
        response = llm.invoke(messages)
        tokens += getattr(response, "response_metadata", {}).get("token_usage", {}).get("total_tokens", 0)
        try:
            result = json.loads(response.content)
            thought = result.get("thought", "")
            action = result.get("action", "").strip()
            action_input = result.get("action_input", "").strip()
        except Exception:
            observations.append("Invalid JSON from LLM. Ending loop.")
            break
        logger.info(f"ReAct Step {i+1}: Thought: {thought}; Action: {action}")
        if action.lower() == "conclude":
            insights = result.get("action_input")
            if isinstance(insights, list):
                observations.extend(insights)
            else:
                observations.append(insights)
            break
        if action == "PythonPandasExecutor":
            obs = tool._run(code_snippet=action_input)
            observations.append(f"{thought}\nResult: {obs}")
        else:
            observations.append(f"{thought}\nUnrecognized action: {action}")
        # Prevent loops: refocusing if repeated input
        if action_input == last_input:
            focus = feedback or focus
        last_input = action_input
    return {**state, "react_observations": observations, "total_tokens": tokens, "error_message": None}

# --- Summarization Sub-Agent ---
def summarize_and_format_node(state: ResearchAgentState) -> ResearchAgentState:
    """LLM synthesizes ReAct observations into advanced, business-analyst-style bullets."""
    logger.info("Summarizer: Synthesizing advanced insights.")
    llm = state.get("llm")
    obs = state.get("react_observations") or []
    feedback = state.get("cleaned_feedback", "")
    tokens = state.get("total_tokens", 0)
    if not obs or not llm:
        return {**state, "core_bullets": obs, "error_message": "No insights or LLM for summarization."}
    prompt = (
        f"You are a principal business/data analyst. User feedback: '{feedback}'.\n"
        "Review these ReAct observations and synthesize 5-7 advanced, actionable, and original financial insights "
        "that go beyond the feedback. Use clear, formal business language, and focus on trends, anomalies, and implications.\n"
        "Return ONLY a Python list of bullet strings (starting with '- ').\n"
        f"Observations:\n{json.dumps(obs)}"
    )
    messages = [SystemMessage(content="Synthesize advanced insights."), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    tokens += getattr(response, "response_metadata", {}).get("token_usage", {}).get("total_tokens", 0)
    try:
        bullets = json.loads(response.content)
    except Exception:
        bullets = [line for line in response.content.splitlines() if line.startswith("- ")]
    return {**state, "core_bullets": bullets, "total_tokens": tokens, "error_message": None}

# --- Final Report Node ---
def finalize_report_node(state: ResearchAgentState) -> ResearchAgentState:
    """Formats the final report."""
    logger.info("Finalizing report.")
    bullets = state.get("core_bullets") or []
    df = state.get("dataframe")
    feedback = state.get("cleaned_feedback", "")
    def detect_time_period_label(df):
        if df is None or 'date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['date']):
            return ''
        sorted_dates = df['date'].dropna().sort_values().unique()
        if len(sorted_dates) < 2: return 'Snapshot'
        diffs = pd.Series(sorted_dates).diff().dropna()
        median = diffs.median().days if not diffs.empty else 0
        if 1 <= median <= 8: return 'Week over Week'
        if 25 <= median <= 35: return 'Month over Month'
        if 80 <= median <= 100: return 'Quarter over Quarter'
        if 350 <= median <= 380: return 'Year over Year'
        return 'Custom Period'
    title = "Financial Insights Summary"
    period = detect_time_period_label(df)
    if period: title += f" ({period})"
    report = [f"## {title}"] + bullets
    if feedback:
        report.append(f"\n- User Feedback Considered: \"{feedback}\"")
    return {**state, "final_report": "\n".join(report)}

# --- LangGraph Assembly ---
workflow = StateGraph(ResearchAgentState)
workflow.add_node("preprocess", preprocess_data_node)
workflow.add_node("react_insight", react_data_insight_node)
workflow.add_node("summarize", summarize_and_format_node)
workflow.add_node("finalize", finalize_report_node)
workflow.set_entry_point("preprocess")
workflow.add_edge("preprocess", "react_insight")
workflow.add_edge("react_insight", "summarize")
workflow.add_edge("summarize", "finalize")
workflow.add_edge("finalize", END)
app = workflow.compile()

def run_langgraph_analysis(input_data, input_context, user_feedback_str=None, llm_instance=None):
    """Orchestrates the full pipeline for a single dataset."""
    initial_state: ResearchAgentState = {
        "raw_data_dict": input_data,
        "user_feedback_str": user_feedback_str,
        "context_str": input_context,
        "llm": llm_instance,
        "cleaned_feedback": None,
        "dataframe": None,
        "react_observations": None,
        "core_bullets": None,
        "final_report": None,
        "error_message": None,
        "total_tokens": 0,
    }
    final_state = app.invoke(initial_state, {"recursion_limit": 15})
    return final_state.get("final_report", final_state.get("error_message", "No output."))

# --- Example Usage ---
if __name__ == '__main__':
    # Initialize LLM (ensure OPENAI_API_KEY is set)
    llm = None # Replace with actual ChatOpenAI initialization
    # Sample data
    data = {
        'date': [
            202306, 202307, 202308, 202309, 202310, 202311, 202312,
            202401, 202402, 202403, 202404, 202405, 202406, 202407,

        ],
        'categories': [
            'Large', 'Medium', 'Small', 'Large', 'Medium', 'Small', 'Large',
            'Medium', 'Small', 'Large', 'Medium', 'Small', 'Large', 'Medium',

        ],
        'values': [
            24.6872, 26.468696, 43.703522, 12.477786, 39.302211, 57.709185, 53.787661,
            68.05638, 37.911391, 46.911702, 52.49122, 51.27037, 19.774913, 69.594027,

        ]
    }
    context = "Trend analysis of value_metric by category, focusing on various date input styles."
    feedback = "Are there any outliers in value_metric for category A? How is category B trending in late 2023?"
    print(run_langgraph_analysis(data, context, feedback, llm))
