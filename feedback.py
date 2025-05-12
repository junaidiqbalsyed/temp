import pandas as pd
import numpy as np
import logging
import time
import re # For cleaning feedback
import json # For LLM output parsing
from typing import Dict, Any, Optional, TypedDict, Annotated, Sequence, List

# Optional LangChain/OpenAI imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None # type: ignore
    HumanMessage = None # type: ignore
    SystemMessage = None # type: ignore
    BaseMessage = None # type: ignore

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages # <--- ADD THIS LINE

# --- Logger Setup (INFO in Red) ---
class RedInfoFilter(logging.Filter):
    def filter(self, record):
        if record.levelno == logging.INFO:
            record.msg = str(record.msg) # Ensure msg is string
            record.msg = f"\033[91m{record.msg}\033[0m" # Red color for INFO
        return True

logger = logging.getLogger('langgraph_research_analyst_final') # Final version logger
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.addFilter(RedInfoFilter()) # Apply the red filter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# --- Helper Functions ---
def clean_feedback_str(feedback: Optional[str]) -> str:
    if not feedback:
        return ""
    cleaned = re.sub(r'\s+', ' ', feedback).strip()
    return cleaned

def get_updated_resampling_freq(df_ts_index: pd.DatetimeIndex) -> str:
    if len(df_ts_index) < 2:
        return 'ME' # Default to Month End if not enough data for other frequencies
    date_range_days = (df_ts_index.max() - df_ts_index.min()).days
    if date_range_days > 730: return 'YE' # Yearly (Year End) if data spans > ~2 years
    if date_range_days > 180: return 'QE' # Quarterly (Quarter End) if data spans > ~6 months
    return 'ME' # Default to Monthly (Month End)

def detect_time_period_label(df: Optional[pd.DataFrame]) -> str:
    if df is None or df.empty:
        return ''
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    if not datetime_cols:
        return ''
    
    date_col_name = datetime_cols[0]
    try:
        if not pd.api.types.is_datetime64_any_dtype(df[date_col_name]):
             return ''
        
        sorted_dates = df[date_col_name].dropna().sort_values().unique()
        if len(sorted_dates) < 2:
            return 'Snapshot'
        
        sorted_dates_ts = pd.to_datetime(sorted_dates)
        diffs = pd.Series(sorted_dates_ts).diff().dropna()
        if diffs.empty:
            return 'Irregular Periods'

        median_diff_days = diffs.median().days
        if 1 <= median_diff_days <= 8: return 'Week over Week'
        if 25 <= median_diff_days <= 35: return 'Month over Month'
        if 80 <= median_diff_days <= 100: return 'Quarter over Quarter'
        if 350 <= median_diff_days <= 380: return 'Year over Year'
        return 'Custom Time Periods'
    except Exception as e:
        logger.warning(f"Failed to detect time period label from column '{date_col_name}': {e}")
        return ''

# --- Core Python Analysis Functions ---
def parse_dates_dynamically_algorithmic(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    date_keywords = ['date', 'time', 'period', 'month', 'year', 'quarter', 'day', 'dt', 'ts', 'timestamp', 'datum']
    
    for col in df.columns: # Attempt numeric parsing first for YYYYMMDD/YYYYMM
        if df[col].dtype in [np.int64, np.float64]:
            try:
                temp_series = df[col].dropna().astype(int).astype(str)
                if not temp_series.empty:
                    if (temp_series.str.len() == 8).all():
                        parsed = pd.to_datetime(temp_series, format='%Y%m%d', errors='coerce')
                        if parsed.notna().sum() > 0.7 * len(parsed): df[col] = parsed; continue
                    if (temp_series.str.len() == 6).all():
                        parsed = pd.to_datetime(temp_series, format='%Y%m', errors='coerce')
                        if parsed.notna().sum() > 0.7 * len(parsed): df[col] = parsed; continue
            except Exception: pass

    for col in df.columns: # General string parsing
        if pd.api.types.is_datetime64_any_dtype(df[col]): continue
        is_potential_by_keyword = any(keyword in str(col).lower() for keyword in date_keywords)
        is_object_or_string = df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])

        if is_potential_by_keyword or is_object_or_string:
            original_column_data = df[col].copy()
            try:
                temp_dates = pd.to_datetime(df[col], errors='coerce') # No infer_datetime_format
                if temp_dates.notna().sum() > 0.5 * len(temp_dates):
                    df[col] = temp_dates
                    logger.info(f"Algorithmic parsing converted column '{col}' to datetime.")
                else: df[col] = original_column_data
            except Exception: df[col] = original_column_data
    return df

def convert_dict_to_dataframe_dynamically_v1(data: Dict[str, Any]) -> pd.DataFrame:
    if not data: return pd.DataFrame()
    try:
        df = pd.DataFrame(data)
        if df.empty: return df
        df = parse_dates_dynamically_algorithmic(df)
        for col in df.columns:
            if df[col].dtype == 'object':
                try: df[col] = pd.to_numeric(df[col])
                except ValueError:
                    if df[col].nunique() > 1 and df[col].nunique() / len(df[col]) < 0.5 and len(df[col]) > 20:
                        df[col] = df[col].astype('category')
        return df
    except Exception as e:
        raise ValueError(f"Input data could not be converted by v1 converter: {e}")

def perform_extended_dynamic_analysis(df: pd.DataFrame, context: str) -> Dict[str, Any]:
    analysis = {'context': context, 'dataframe_shape': df.shape, 'findings': {}}
    if df.empty:
        analysis['message'] = "DataFrame is empty, no analysis performed."
        return analysis
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    if numeric_cols:
        analysis['findings']['descriptive_stats'] = df[numeric_cols].describe().round(2).replace([np.inf, -np.inf], 'Infinity').fillna('N/A').to_dict()
        if len(numeric_cols) > 1:
            try:
                corr_cols = numeric_cols[:min(len(numeric_cols), 7)]
                analysis['findings']['correlation_matrix'] = df[corr_cols].corr().round(2).fillna('N/A').to_dict()
            except Exception as e: logger.warning(f"Could not compute correlation: {e}")
        analysis['findings']['outlier_counts'] = {}
        for col in numeric_cols[:min(len(numeric_cols), 3)]:
            if df[col].nunique() <= 1 or df[col].std() == 0 or pd.isna(df[col].std()): continue
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            analysis['findings']['outlier_counts'][col] = int((z_scores > 3).sum())

    if categorical_cols and numeric_cols:
        analysis['findings']['grouped_aggregations'] = {}
        if categorical_cols:
            cat_col_to_group = categorical_cols[0]
            if numeric_cols:
                num_col_to_agg = numeric_cols[0]
                if df[cat_col_to_group].nunique() < 50:
                    try:
                        grouped = df.groupby(cat_col_to_group)[num_col_to_agg].agg(['mean', 'sum', 'count']).round(2)
                        analysis['findings']['grouped_aggregations'][f'{num_col_to_agg}_by_{cat_col_to_group}'] = grouped.replace([np.inf, -np.inf], 'Infinity').fillna('N/A').to_dict()
                    except Exception as e: logger.warning(f"Grouping error for '{cat_col_to_group}' and '{num_col_to_agg}': {e}")

    if numeric_cols:
        num_col_for_top_n = numeric_cols[0]
        top_n_count = min(5, len(df))
        if num_col_for_top_n in df and pd.api.types.is_numeric_dtype(df[num_col_for_top_n]):
            analysis['findings'][f'top_{top_n_count}_{num_col_for_top_n}'] = df.nlargest(top_n_count, num_col_for_top_n).round(2).to_dict(orient='records')

    if datetime_cols:
        date_col = datetime_cols[0]
        analysis['findings']['time_series_analysis'] = {'date_column_used': date_col}
        df_ts_unsorted = df.set_index(date_col)
        df_ts = df_ts_unsorted.sort_index() if not (df_ts_unsorted.index.is_monotonic_increasing or df_ts_unsorted.index.is_monotonic_decreasing) else df_ts_unsorted
            
        if numeric_cols:
            num_col_for_ts = numeric_cols[0]
            if not df_ts.empty and hasattr(df_ts.index, 'nunique') and df_ts.index.nunique() > 1: # type: ignore
                freq = get_updated_resampling_freq(df_ts.index) # type: ignore
                analysis['findings']['time_series_analysis']['resampling_frequency_applied'] = freq
                try:
                    if num_col_for_ts in df_ts.columns and pd.api.types.is_numeric_dtype(df_ts[num_col_for_ts]):
                        resampled_sum = df_ts[num_col_for_ts].resample(freq).sum().round(2)
                        resampled_mean = df_ts[num_col_for_ts].resample(freq).mean().round(2)
                        analysis['findings']['time_series_analysis'][f'resampled_{num_col_for_ts}_sum'] = resampled_sum.replace([np.inf, -np.inf], 'Infinity').fillna('N/A').to_dict()
                        analysis['findings']['time_series_analysis'][f'resampled_{num_col_for_ts}_mean'] = resampled_mean.replace([np.inf, -np.inf], 'Infinity').fillna('N/A').to_dict()
                        if len(resampled_sum) > 1:
                            pop_change = resampled_sum.pct_change().fillna(0) * 100
                            analysis['findings']['time_series_analysis'][f'pop_change_{num_col_for_ts}_sum_percent'] = pop_change.round(2).replace([np.inf, -np.inf], 'Infinity').fillna('N/A').to_dict()
                        running_total = resampled_sum.cumsum()
                        analysis['findings']['time_series_analysis'][f'running_total_{num_col_for_ts}_sum'] = running_total.round(2).replace([np.inf, -np.inf], 'Infinity').fillna('N/A').to_dict()
                        if len(resampled_sum) >= 3:
                            moving_avg = resampled_sum.rolling(window=min(3, len(resampled_sum))).mean()
                            analysis['findings']['time_series_analysis'][f'moving_avg_3period_{num_col_for_ts}_sum'] = moving_avg.round(2).replace([np.inf, -np.inf], 'Infinity').fillna('N/A').to_dict()
                    else: logger.warning(f"Numeric column '{num_col_for_ts}' not found/numeric in TS DataFrame.")
                except Exception as e: logger.warning(f"Time series processing error for {num_col_for_ts} with freq {freq}: {e}")
            else: logger.info("Not enough unique date index values for TS resampling.")
    return analysis

# --- LangGraph State Definition ---
class ResearchAgentState(TypedDict):
    raw_data_dict: Dict[str, Any]
    user_feedback_str: Optional[str]
    cleaned_feedback_str: Optional[str]
    context_str: str
    dataframe: Optional[pd.DataFrame]
    structured_analysis: Optional[Dict[str, Any]]
    core_bullet_points: Optional[List[str]]
    final_report_str: Optional[str]
    error_message: Optional[str]
    llm_for_interpretation: Optional[ChatOpenAI] # type: ignore
    total_tokens_used_by_llm: int
    messages: Annotated[Sequence[BaseMessage], add_messages] # Now 'add_messages' 

# --- LangGraph Nodes ---
def _get_llm_date_format_suggestions(
    df: pd.DataFrame,
    columns_to_check: List[str],
    llm: ChatOpenAI # type: ignore
) -> Dict[str, str]:
    if not LANGCHAIN_AVAILABLE or not llm: return {}
    samples_for_llm = {}
    for col in columns_to_check:
        unique_vals = df[col].dropna().unique()
        samples_for_llm[col] = [str(val) for val in unique_vals[:5]]
    if not samples_for_llm: return {}

    prompt_template = """You are a data parsing expert. Given column names and samples, identify date columns and provide pandas strftime formats.
Rules:
1. Only consider columns not already datetime.
2. Date format: e.g., '%Y%m%d', '%m/%d/%Y', '%m-%Y', '%Y%m'.
3. If not a date or ambiguous, output "NOT_A_DATE".
4. Output ONLY valid JSON: {{ "col_A": "%Y%m", "col_B": "NOT_A_DATE" }}
Analyze:
{columns_and_samples_json}"""
    columns_and_samples_json_str = json.dumps(samples_for_llm, indent=2)
    full_prompt = prompt_template.format(columns_and_samples_json=columns_and_samples_json_str)
    logger.info(f"Prompting LLM for date formats with samples: {samples_for_llm}")
    messages = [SystemMessage(content="You are an expert in identifying date formats and providing pandas strftime strings."), HumanMessage(content=full_prompt)] # type: ignore
    llm_output_str = "Error: LLM did not respond"
    try:
        response = llm.invoke(messages)
        llm_output_str = response.content # type: ignore
        logger.info(f"LLM raw response for date formats: {llm_output_str}")
        match = re.search(r"``````", llm_output_str, re.DOTALL) # type: ignore
        json_str = match.group(1) if match else llm_output_str
        suggestions = json.loads(json_str) # type: ignore
        if isinstance(suggestions, dict):
            return {k: v for k, v in suggestions.items() if isinstance(k, str) and isinstance(v, str)}
        return {}
    except json.JSONDecodeError as e: logger.error(f"Failed to parse LLM JSON for date formats: {e}. Raw: {llm_output_str}")
    except Exception as e: logger.error(f"Error invoking LLM for date formats: {e}")
    return {}

def preprocess_data_node(state: ResearchAgentState) -> ResearchAgentState:
    logger.info("--- Executing Node: Preprocess Data (with LLM Date Assist) ---")
    llm_for_dates = state.get("llm_for_interpretation")
    total_tokens = state.get("total_tokens_used_by_llm", 0)
    try:
        df = convert_dict_to_dataframe_dynamically_v1(state["raw_data_dict"])
        cleaned_feedback = clean_feedback_str(state.get("user_feedback_str"))
        if df.empty:
            return {**state, "error_message": "Need data for analysis", "dataframe": None, "cleaned_feedback_str": cleaned_feedback, "total_tokens_used_by_llm": total_tokens}

        problematic_date_cols = []
        date_keywords = ['date', 'time', 'period', 'month', 'year', 'quarter', 'day', 'dt', 'ts', 'timestamp', 'datum']
        for col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                is_potential_by_keyword = any(keyword in str(col).lower() for keyword in date_keywords)
                is_object_or_string = df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col])
                if is_object_or_string and (is_potential_by_keyword or df[col].astype(str).str.match(r'.*\d{1,4}[-/.]?\d{1,2}[-/.]?\d{0,4}|\d{6,8}.*').any()): # type: ignore
                    sample_parses = pd.to_datetime(df[col].dropna().unique()[:5], errors='coerce')
                    if sample_parses.isna().all() or (sample_parses.notna().sum() / len(sample_parses) < 0.5 if len(sample_parses) > 0 else True):
                         problematic_date_cols.append(col)
        logger.info(f"Columns for potential LLM date format suggestion: {problematic_date_cols}")

        if problematic_date_cols and llm_for_dates and LANGCHAIN_AVAILABLE:
            llm_suggestions = _get_llm_date_format_suggestions(df, problematic_date_cols, llm_for_dates) # type: ignore
            logger.info(f"LLM suggested date formats: {llm_suggestions}")
            for col, fmt_str in llm_suggestions.items():
                if col in df.columns and fmt_str not in ["NOT_A_DATE", "AMBIGUOUS"] and fmt_str:
                    try:
                        logger.info(f"Attempting to parse column '{col}' with LLM suggested format: '{fmt_str}'")
                        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
                             parsed_dates = pd.to_datetime(df[col], format=fmt_str, errors='coerce')
                        else: 
                             parsed_dates = pd.to_datetime(df[col].astype(str), format=fmt_str, errors='coerce')

                        if parsed_dates.notna().sum() > 0.7 * len(parsed_dates):
                            df[col] = parsed_dates
                            logger.info(f"Successfully parsed column '{col}' using LLM format '{fmt_str}'.")
                        else: logger.warning(f"LLM format '{fmt_str}' for column '{col}' did not parse well.")
                    except Exception as e: logger.error(f"Error applying LLM format '{fmt_str}' to '{col}': {e}")
        
        if df.empty:
            return {**state, "error_message": "Need data for analysis", "dataframe": None, "cleaned_feedback_str": cleaned_feedback, "total_tokens_used_by_llm": total_tokens}
        return {**state, "dataframe": df, "error_message": None, "cleaned_feedback_str": cleaned_feedback, "total_tokens_used_by_llm": total_tokens}
    except ValueError as ve:
        logger.error(f"Data conversion error: {ve}")
        return {**state, "error_message": f"Data processing error: {ve}", "dataframe": None, "cleaned_feedback_str": clean_feedback_str(state.get("user_feedback_str")), "total_tokens_used_by_llm": total_tokens}
    except Exception as e:
        logger.exception("Unexpected error in preprocess_data_node.")
        return {**state, "error_message": f"Preprocessing failed: {e}", "dataframe": None, "cleaned_feedback_str": clean_feedback_str(state.get("user_feedback_str")), "total_tokens_used_by_llm": total_tokens}

def analyze_data_node(state: ResearchAgentState) -> ResearchAgentState:
    logger.info("--- Executing Node: Analyze Data ---")
    if state["dataframe"] is None or state["dataframe"].empty: # type: ignore
        return {**state, "structured_analysis": {"message": state.get("error_message", "Need data for analysis")}}
    try:
        analysis_results = perform_extended_dynamic_analysis(state["dataframe"], state["context_str"]) # type: ignore
        return {**state, "structured_analysis": analysis_results}
    except Exception as e:
        logger.exception("Error during analysis.")
        return {**state, "error_message": f"Analysis failed: {e}", "structured_analysis": None}

def format_analysis_to_llm_prompt_text_v2(structured_analysis: Dict[str, Any]) -> str:
    prompt_lines = [f"Context: {structured_analysis.get('context', 'N/A')}"]
    prompt_lines.append(f"Data Shape: Rows={structured_analysis.get('dataframe_shape', (0,0))[0]}, Cols={structured_analysis.get('dataframe_shape', (0,0))[1]}")
    if structured_analysis.get('message'):
        prompt_lines.append(f"Analysis Message: {structured_analysis['message']}")
        return "\n".join(prompt_lines)
    findings = structured_analysis.get('findings', {})
    def dict_to_str_for_prompt_v2(d, indent=0, title=""):
        s = ""
        if title: s += '  ' * indent + f"{title}:\n"
        for k, v_item in d.items():
            if isinstance(v_item, dict):
                s += '  ' * (indent + 1) + f"{k}:\n" + dict_to_str_for_prompt_v2(v_item, indent + 2)
            elif isinstance(v_item, list):
                 s += '  ' * (indent + 1) + f"{k}: [\n"
                 for item_idx, item_val in enumerate(v_item[:3]): s += '  ' * (indent + 2) + f"Item {item_idx+1}: {item_val}\n"
                 if len(v_item) > 3: s += '  ' * (indent + 2) + f"... and {len(v_item)-3} more items\n"
                 s += '  ' * (indent + 1) + "]\n"
            else: s += '  ' * (indent + 1) + f"{k}: {v_item}\n"
        return s
    if 'descriptive_stats' in findings:
        prompt_lines.append("\nKey Descriptive Statistics (sample):")
        for col, stats in list(findings['descriptive_stats'].items())[:2]: prompt_lines.append(f"  Column '{col}': Mean={stats.get('mean','N/A')}, Std={stats.get('std','N/A')}, Count={stats.get('count','N/A')}, Min={stats.get('min','N/A')}, Max={stats.get('max','N/A')}")
    if 'correlation_matrix' in findings: prompt_lines.append("\nCorrelation Matrix (sample):\n" + dict_to_str_for_prompt_v2(findings['correlation_matrix']))
    if 'outlier_counts' in findings and findings['outlier_counts']: prompt_lines.append("\nOutlier Counts (Z-score > 3):\n" + dict_to_str_for_prompt_v2(findings['outlier_counts']))
    if 'grouped_aggregations' in findings and findings['grouped_aggregations']: prompt_lines.append("\nGrouped Aggregations (sample):\n" + dict_to_str_for_prompt_v2(findings['grouped_aggregations']))
    if 'time_series_analysis' in findings:
        prompt_lines.append("\nTime Series Analysis Highlights (sample):")
        ts_findings = findings['time_series_analysis']
        prompt_lines.append(f"  Date Column Used: {ts_findings.get('date_column_used')}, Resampling Freq: {ts_findings.get('resampling_frequency_applied', 'N/A')}")
        for k, v_data in ts_findings.items():
            if k in ['date_column_used', 'resampling_frequency_applied'] or not isinstance(v_data, dict): continue
            prompt_lines.append(f"  {k.replace('_', ' ').title()}:")
            for period, value in list(v_data.items())[:3]:
                period_str = period.strftime('%Y-%m-%d') if isinstance(period, pd.Timestamp) else str(period)
                prompt_lines.append(f"    - {period_str}: {value}")
            if len(v_data) > 3: prompt_lines.append(f"    ... and {len(v_data)-3} more periods.")
    return "\n".join(prompt_lines)

def interpret_with_llm_node(state: ResearchAgentState) -> ResearchAgentState:
    logger.info("--- Executing Node: Interpret with LLM (Enhanced for Financial Analyst Tone) ---")
    llm = state.get("llm_for_interpretation")
    structured_analysis = state.get("structured_analysis")
    cleaned_feedback = state.get("cleaned_feedback_str", "")
    current_tokens = state.get("total_tokens_used_by_llm", 0)

    if not llm or not LANGCHAIN_AVAILABLE:
        return {**state, "error_message": "LLM interpret called without LLM.", "core_bullet_points": ["- LLM interpretation skipped due to missing LLM."]}
    if not structured_analysis or structured_analysis.get("message"): # type: ignore
        error_or_message = structured_analysis.get("message", "Analysis results are missing for LLM interpretation.") if structured_analysis else "Analysis results are missing for LLM interpretation." # type: ignore
        return {**state, "core_bullet_points": [error_or_message]}
    try:
        detailed_prompt_text = format_analysis_to_llm_prompt_text_v2(structured_analysis) # type: ignore
        system_prompt = f'''You are a seasoned business and data analyst specializing in financial data. Your task is to synthesize the provided structured data analysis findings and user context into a list of 5-7 comprehensive, insightful bullet points. These insights should be detailed, data-driven, and articulated with the professionalism and precision expected by financial analysts. Focus on key trends, significant changes, anomalies, and actionable insights relevant to financial decision-making. Use clear, formal business language, avoiding jargon unless necessary, and explain any technical terms briefly.
The findings provided are from an automated analysis; use your expertise to highlight what is most important and impactful for strategic financial discussions.
The user has provided the following feedback to guide your focus: "{cleaned_feedback if cleaned_feedback else "No specific feedback provided."}"
Return ONLY the Python list of strings, where each string is a bullet point. Example: ['- Point 1.', '- Point 2.']
'''
        human_prompt = f"User Context: \"{state['context_str']}\"\nAutomated Analysis Findings:\n\n{detailed_prompt_text}\n\nPlease generate the Python list of bullet point strings based on the analysis and feedback."
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)] # type: ignore
        logger.info("Invoking LLM with enhanced financial analyst prompt...")
        llm_response = llm.invoke(messages) # type: ignore
        raw_llm_output = llm_response.content # type: ignore
        core_bullets = []
        try:
            if raw_llm_output.strip().startswith('[') and raw_llm_output.strip().endswith(']'): # type: ignore
                potential_list = eval(raw_llm_output) # type: ignore
                if isinstance(potential_list, list) and all(isinstance(item, str) for item in potential_list):
                    core_bullets = [item if item.startswith("- ") else f"- {item}" for item in potential_list]
            if not core_bullets:
                core_bullets = [line.strip() if line.strip().startswith("- ") else f"- {line.strip()}" for line in raw_llm_output.splitlines() if line.strip() and line.strip() not in ["['", "']"]] # type: ignore
            if not core_bullets and raw_llm_output: core_bullets = [f"- {raw_llm_output}"] # type: ignore
            elif not core_bullets: core_bullets = ["- LLM generated an empty response."]
        except Exception as parse_err:
            logger.error(f"Could not parse LLM output as list: {parse_err}. Using raw output: {raw_llm_output}")
            core_bullets = [f"- LLM output (parsing failed): {raw_llm_output}"] # type: ignore
        new_tokens = 0
        if hasattr(llm_response, 'response_metadata') and 'token_usage' in llm_response.response_metadata: # type: ignore
            new_tokens = llm_response.response_metadata['token_usage'].get('total_tokens', 0) # type: ignore
        total_tokens = current_tokens + new_tokens
        logger.info(f"LLM interpretation successful. Tokens for this call: {new_tokens}. Cumulative tokens: {total_tokens}")
        return {**state, "core_bullet_points": core_bullets, "total_tokens_used_by_llm": total_tokens, "error_message": None}
    except Exception as e:
        logger.exception("Error during LLM interpretation.")
        error_bullets = [f"- LLM interpretation failed: {e}"]
        return {**state, "error_message": f"LLM interpretation failed: {e}", "core_bullet_points": state.get("core_bullet_points") or error_bullets, "total_tokens_used_by_llm": current_tokens}

def generate_direct_bullets_node(state: ResearchAgentState) -> ResearchAgentState:
    logger.info("--- Executing Node: Generate Direct Bullets ---")
    if state.get("error_message") and not state.get("structured_analysis"):
        return {**state, "core_bullet_points": [state["error_message"]]} # type: ignore
    structured_analysis = state.get("structured_analysis")
    if not structured_analysis or structured_analysis.get("message"): # type: ignore
        msg = structured_analysis.get("message", "Analysis results unavailable.") if structured_analysis else "Analysis results unavailable." # type: ignore
        return {**state, "core_bullet_points": [msg]}
    bullets = []
    findings = structured_analysis.get('findings', {}) # type: ignore
    if 'descriptive_stats' in findings:
        stats_dict = findings['descriptive_stats']
        all_means = [stats_dict[col].get('mean') for col in stats_dict if isinstance(stats_dict[col].get('mean'), (int, float))]
        all_stds = [stats_dict[col].get('std') for col in stats_dict if isinstance(stats_dict[col].get('std'), (int, float))]
        if all_means:
            overall_avg_val = np.mean(all_means) if all_means else 'N/A' # type: ignore
            overall_std_val = np.mean(all_stds) if all_stds else 'N/A' # type: ignore
            bullets.append(f"- Overall, the average value across key numeric metrics is {overall_avg_val:.2f}, with an average standard deviation of {overall_std_val:.2f}.") # type: ignore
    if 'grouped_aggregations' in findings and findings['grouped_aggregations']:
        for group_key, agg_data in findings['grouped_aggregations'].items():
            parts = group_key.split('_by_')
            if len(parts) == 2:
                num_col, cat_col = parts
                mean_agg = agg_data.get('mean') if isinstance(agg_data, dict) else None # Adapted based on expected structure
                sum_agg = agg_data.get('sum') if isinstance(agg_data, dict) else None # Adapted
                if isinstance(mean_agg, dict) and mean_agg: # Check if 'mean' itself is a dict of means
                    top_cat_by_mean = max(mean_agg, key=mean_agg.get) # type: ignore
                    bullets.append(f"- The category \"{top_cat_by_mean}\" has the highest average for '{num_col}' at {mean_agg[top_cat_by_mean]:.2f}.") # type: ignore
                if isinstance(sum_agg, dict) and sum_agg:  # Check if 'sum' itself is a dict of sums
                    top_cat_by_sum = max(sum_agg, key=sum_agg.get) # type: ignore
                    bullets.append(f"- In terms of total '{num_col}', \"{top_cat_by_sum}\" leads with {sum_agg[top_cat_by_sum]:.2f}.") # type: ignore
    if 'outlier_counts' in findings:
        total_outliers = sum(v for v in findings['outlier_counts'].values() if isinstance(v, int))
        if total_outliers == 0: bullets.append("- No significant outliers detected in key numeric columns based on Z-scores exceeding 3.")
        else: bullets.append(f"- A total of {total_outliers} potential outliers were detected in key numeric columns.")
    if 'time_series_analysis' in findings and findings['time_series_analysis'].get('resampling_frequency_applied'):
        tsa = findings['time_series_analysis']
        freq_label = tsa['resampling_frequency_applied']
        sum_key = next((k for k in tsa if 'resampled_' in k and '_sum' in k and not any(s in k for s in ['pop_change', 'running_total', 'moving_avg'])), None)
        mean_key = next((k for k in tsa if 'resampled_' in k and '_mean' in k), None)
        if sum_key and mean_key and isinstance(tsa.get(sum_key), dict) and isinstance(tsa.get(mean_key), dict):
            sum_values = [v for v in tsa[sum_key].values() if isinstance(v, (int,float))] # type: ignore
            mean_values = [v for v in tsa[mean_key].values() if isinstance(v, (int,float))] # type: ignore
            if sum_values and mean_values:
                avg_period_sum = np.mean(sum_values)
                avg_period_mean = np.mean(mean_values)
                bullets.append(f"- Time series analysis ({freq_label}) indicates an average periodic sum of {avg_period_sum:.2f} and an average periodic mean of {avg_period_mean:.2f} for the primary numeric metric.")
    if not bullets: bullets.append("- Automated analysis generated no specific summary bullet points.")
    return {**state, "core_bullet_points": bullets}

def finalize_report_node(state: ResearchAgentState) -> ResearchAgentState:
    logger.info("--- Executing Node: Finalize Report ---")
    core_bullets = state.get("core_bullet_points")
    cleaned_feedback = state.get("cleaned_feedback_str", "")
    df_for_time_detection = state.get("dataframe")

    if state.get("error_message") and not core_bullets:
        return {**state, "final_report_str": state["error_message"]} # type: ignore
    if not core_bullets:
        core_bullets = ["- No detailed insights could be generated."]

    time_period_label = detect_time_period_label(df_for_time_detection)
    title = f"Financial Insights Summary"
    if time_period_label:
        title += f" ({time_period_label})"
    
    report_parts = [f"## {title}"]
    report_parts.extend(core_bullets if isinstance(core_bullets, list) else [str(core_bullets)])
    if cleaned_feedback:
        report_parts.append(f"\n- User Feedback Considered: \"{cleaned_feedback}\"")

    final_str = "\n".join(report_parts)
    return {**state, "final_report_str": final_str}

# --- Conditional Edges ---
def should_use_llm_router(state: ResearchAgentState) -> str:
    if state.get("error_message") and not state.get("structured_analysis"):
        logger.info("Conditional Edge: Error before analysis, routing to finalize_report.")
        return "finalize_report"
    if state.get("llm_for_interpretation") and LANGCHAIN_AVAILABLE and \
       state.get("structured_analysis") and not state["structured_analysis"].get("message"): # type: ignore
        logger.info("Conditional Edge: Routing to LLM interpretation.")
        return "interpret_with_llm"
    logger.info("Conditional Edge: Routing to direct bullet generation.")
    return "generate_direct_bullets"

# --- Build the Graph ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

workflow = StateGraph(ResearchAgentState)
workflow.add_node("preprocess_data", preprocess_data_node)
workflow.add_node("analyze_data", analyze_data_node)
workflow.add_node("interpret_with_llm", interpret_with_llm_node)
workflow.add_node("generate_direct_bullets", generate_direct_bullets_node)
workflow.add_node("finalize_report", finalize_report_node)

workflow.set_entry_point("preprocess_data")
workflow.add_edge("preprocess_data", "analyze_data")
workflow.add_conditional_edges(
    "analyze_data",
    should_use_llm_router,
    {
        "interpret_with_llm": "interpret_with_llm",
        "generate_direct_bullets": "generate_direct_bullets",
        "finalize_report": "finalize_report", 
    }
)
workflow.add_edge("interpret_with_llm", "finalize_report")
workflow.add_edge("generate_direct_bullets", "finalize_report")
workflow.add_edge("finalize_report", END)

compiled_app = workflow.compile() # Renamed to avoid conflict with FastAPI app instance

# --- Main Function to Run LangGraph ---
def run_langgraph_analysis(
    input_data: Dict[str, Any],
    input_context: str,
    user_feedback_str: Optional[str] = None,
    llm_instance: Optional[ChatOpenAI] = None # type: ignore
) -> str:
    start_graph_time = time.time()
    request_id = pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')
    logger.info(f"LangGraph Request {request_id}: Started. Context: \"{input_context}\", Feedback: \"{user_feedback_str if user_feedback_str else 'None'}\"")
    
    # Ensure llm_instance is correctly typed for the state, or None
    current_llm_instance = llm_instance if (LANGCHAIN_AVAILABLE and isinstance(llm_instance, ChatOpenAI)) else None # type: ignore

    initial_state: ResearchAgentState = {
        "raw_data_dict": input_data, "user_feedback_str": user_feedback_str, "cleaned_feedback_str": "",
        "context_str": input_context, "dataframe": None, "structured_analysis": None,
        "core_bullet_points": None, "final_report_str": None, "error_message": None,
        "llm_for_interpretation": current_llm_instance, 
        "total_tokens_used_by_llm": 0, "messages": []
    }
    final_state: Dict[str, Any] = {} # Initialize as dict
    output_str = "Error: Processing did not complete."
    try:
        final_state_result = compiled_app.invoke(initial_state, {"recursion_limit": 15})
        final_state = final_state_result if final_state_result is not None else {}

        if final_state.get("error_message") and not final_state.get("final_report_str"):
            output_str = str(final_state.get("error_message", "An unspecified error occurred."))
        elif final_state.get("final_report_str"):
            output_str = str(final_state.get("final_report_str"))
        else:
            output_str = "Processing completed, but no final report string was generated. Check logs."
            
    except Exception as e:
        logger.exception(f"LangGraph Request {request_id}: Graph execution error: {e}")
        output_str = f"Critical error during graph execution: {e}"

    total_exec_time = time.time() - start_graph_time
    tokens_used = final_state.get("total_tokens_used_by_llm", 0) if isinstance(final_state, dict) else 0

    logger.info(f"LangGraph Request {request_id}: Total execution time: {total_exec_time:.4f} seconds. Total LLM tokens: {tokens_used}.")
    if total_exec_time > 25:
        logger.warning(f"LangGraph Request {request_id}: Execution time ({total_exec_time:.4f}s) EXCEEDED 25 second target!")
        
    return output_str if output_str else "No output generated."

