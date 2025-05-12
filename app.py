from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import asyncio
import os # For environment variables

# Import from feedback.py
from feedback import run_langgraph_analysis, LANGCHAIN_AVAILABLE, logger as feedback_logger
if LANGCHAIN_AVAILABLE:
    from feedback import ChatOpenAI # type: ignore
else:
    ChatOpenAI = None # type: ignore

# --- Pydantic Models for Request and Response ---
class AnalysisRequest(BaseModel):
    tableau_data: Dict[str, List[Any]] = Field(
        ..., 
        description="Columnar data. Keys are column names, values are lists of data for that column.",
        examples=[{
            "dates": ["202306", "202307", "2024-04"],
            "categories": ["Large", "Medium", "Small"],
            "values": [53.08, 26.42, 14.87]
        }]
    )
    input_context: str = Field(
        ...,
        description="Context or main question for the analysis.",
        examples=["Analysis of monthly category values."]
    )
    feedback_str: Optional[str] = Field(
        None, 
        description="Specific user feedback or focus points for the analysis.",
        examples=["Focus on the revenue trend for the Small category."]
    )

class AnalysisResponse(BaseModel):
    report: str

# --- FastAPI Application ---
app = FastAPI(
    title="LangGraph Research Analyst API",
    description="API for performing data analysis using a LangGraph workflow. \
                 Accepts columnar data under the 'tableau_data' key. \
                 OpenAI API key must be set as an environment variable (OPENAI_API_KEY).",
    version="1.2.0" # Updated version
)

@app.on_event("startup")
async def startup_event():
    feedback_logger.info("FastAPI application starting up.")
    if not LANGCHAIN_AVAILABLE:
        feedback_logger.warning("LangChain is not available. LLM functionalities will be disabled.")
    if not os.getenv("OPENAI_API_KEY") and LANGCHAIN_AVAILABLE:
        feedback_logger.warning("OPENAI_API_KEY environment variable not set. LLM features requiring it will be disabled unless LangChain is configured to find it elsewhere.")
    elif os.getenv("OPENAI_API_KEY") and LANGCHAIN_AVAILABLE:
        feedback_logger.info("OPENAI_API_KEY environment variable found.")


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data_endpoint(request: AnalysisRequest):
    """
    Analyzes the provided columnar data using the LangGraph workflow.
    
    - **tableau_data**: A dictionary where keys are column names (e.g., "dates", "categories", "values") 
      and values are lists representing the data for each column.
    - **input_context**: General context or question for the analysis.
    - **feedback_str** (optional): Specific feedback to guide the analysis narrative.
    
    **Note**: For LLM-based interpretation, the `OPENAI_API_KEY` environment variable 
    must be set on the server.
    """
    llm_instance_for_graph = None
    
    api_key_from_env = # Get API key ONLY from environment [1, 2]

    if api_key_from_env and LANGCHAIN_AVAILABLE and ChatOpenAI:
        try:
            llm_instance_for_graph = ChatOpenAI(
                openai_api_key=api_key_from_env, # Use environment variable
                temperature=0.8,
                request_timeout=30 
            )
            feedback_logger.info("ChatOpenAI instance created using environment API key.")
        except Exception as e:
            feedback_logger.error(f"Failed to initialize ChatOpenAI using environment API key: {e}")
    elif not LANGCHAIN_AVAILABLE and api_key_from_env:
        feedback_logger.warning("OPENAI_API_KEY environment variable found, but LangChain is not available. LLM features disabled.")
    elif LANGCHAIN_AVAILABLE and not api_key_from_env:
        feedback_logger.info("OPENAI_API_KEY environment variable not set. Proceeding without LLM features requiring it.")


    try:
        loop = asyncio.get_event_loop()
        report_str = await loop.run_in_executor(
            None,
            run_langgraph_analysis,
            request.tableau_data,
            request.input_context,
            request.feedback_str,
            llm_instance_for_graph
        )
        
        if "Error:" in report_str or "Critical error during graph execution:" in report_str:
             feedback_logger.error(f"Analysis completed with error: {report_str}")
        
        return AnalysisResponse(report=report_str)

    except Exception as e:
        feedback_logger.exception(f"Unhandled error in /analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# To run this FastAPI application:
# 1. Save your `feedback.py` (ensure it has the `add_messages` import fix).
# 2. Save this updated code block as `app.py` in the same directory.
# 3. Install necessary packages:
#    pip install fastapi uvicorn pandas numpy langchain langchain-openai openai pydantic # and other dependencies
# 4. **Crucially, set the OPENAI_API_KEY environment variable on your server:**
#    export OPENAI_API_KEY="your_actual_api_key" (macOS/Linux)
#    set OPENAI_API_KEY=your_actual_api_key (Windows CMD)
#    $env:OPENAI_API_KEY="your_actual_api_key" (Windows PowerShell)
# 5. Run Uvicorn server:
#    uvicorn app:app --reload
