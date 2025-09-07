# app.py
"""
GAIBA Campaign Builder — Streamlit front-end + LangChain agent using Groq API.

Features:
- LangChain agent (CONVERSATIONAL_REACT_DESCRIPTION) backed by a Groq LLM wrapper.
- Conversation memory (buffer or summary) stored in st.session_state.
- Tools wrap deterministic functions in marketing_analysis.py, email.py, utils.py.
- Uses Streamlit secrets for GROQ API key: st.secrets["GROQ_API_KEY"]
- Models (main/eval) are defined in-code and editable via the sidebar.

Drop this file into the same folder as:
- data_loading.py
- marketing_analysis.py
- email.py
- utils.py

Before deploying on Streamlit Cloud:
- Put your Groq API key into Streamlit Secrets under key: "GROQ_API_KEY"
"""
import os
import json
import requests
import time
import streamlit as st
import pandas as pd
from typing import Callable, Dict, Any, List, Optional

# LangChain imports
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory

# Local modules (make sure these files exist)
from marketing_analysis import run_full_marketing_analysis, apply_filters  # marketing_analysis.py
from email_utils import validate_email_df, get_sample_recipients, send_email_batch_stub  # email.py
from utils import budget_allocator, parse_clusters, format_currency  # utils.py

# -------------------------------
# Configurable defaults (edit in-code or through UI)
# -------------------------------
DEFAULT_MAIN_MODEL = "gpt/oss-120b"   # LLM A (campaign builder)
DEFAULT_EVAL_MODEL = "gpt/oss-120b"   # LLM B (evaluator)
DEFAULT_GROQ_API_URL = "https://api.groq.ai/v1/complete"
DEFAULT_MAX_TOKENS = 1024

# -------------------------------
# Streamlit secrets helper (preferred method on Streamlit Cloud)
# -------------------------------
def _get_groq_api_key() -> Optional[str]:
    try:
        # Preferred: use Streamlit secrets (set in Streamlit Cloud)
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.getenv("GROQ_API_KEY")
    return key

# -------------------------------
# Minimal Groq LLM wrapper compatible with LangChain's LLM base class
# -------------------------------
class GroqLLM(LLM):
    model_name: str
    api_key: str
    api_url: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = 0.0

    def __init__(self, model_name: str = None, api_key: str = None, api_url: str = None, **kwargs):
        self.model_name = model_name or DEFAULT_MAIN_MODEL
        self.api_key = api_key or _get_groq_api_key()
        self.api_url = api_url or os.getenv("GROQ_API_URL", DEFAULT_GROQ_API_URL)
        self.max_tokens = kwargs.get("max_tokens", DEFAULT_MAX_TOKENS)
        self.temperature = kwargs.get("temperature", 0.0)

        if not self.api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not found. Put your Groq API key in Streamlit secrets under 'GROQ_API_KEY' "
                "or set the environment variable GROQ_API_KEY for local testing."
            )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name, "max_tokens": self.max_tokens}

    def _call(self, prompt: str, stop: List[str] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Groq API request failed: {e} - body: {resp.text}")

        data = resp.json()
        # Adapt depending on your Groq response; common shapes:
        # {"text": "..."} or {"choices":[{"text":"..."}]}
        if isinstance(data, dict):
            if "text" in data:
                return data["text"]
            if "choices" in data and isinstance(data["choices"], list) and "text" in data["choices"][0]:
                return data["choices"][0]["text"]
        # fallback raw body
        return resp.text

# -------------------------------
# Evaluator helper (LLM B) — simple call + parse JSON output
# -------------------------------
def evaluator_score_campaign(campaign_spec: Dict[str, Any], historical_summary: Dict[str, Any], eval_model_name: Optional[str] = None) -> Dict[str, Any]:
    eval_model_name = eval_model_name or DEFAULT_EVAL_MODEL
    api_key = _get_groq_api_key()
    api_url = os.getenv("GROQ_API_URL", DEFAULT_GROQ_API_URL)
    if not api_key:
        return {"error": "GROQ_API_KEY not found in Streamlit secrets or environment."}

    prompt = f"""
You are CampaignEvaluator. Given the campaign spec and historical summary, evaluate:
- Data grounding: are expected metrics aligned with history?
- Budget feasibility: allocation realism.
- Targeting precision and safety (opt-ins).
- Suggested fixes.

Return a JSON object with keys:
  score (0-100),
  breakdown (dict),
  issues (list of strings),
  suggestions (list of strings).

campaign_spec: {json.dumps(campaign_spec)}
historical_summary: {json.dumps(historical_summary)}
"""

    payload = {"model": eval_model_name, "prompt": prompt, "max_tokens": 800, "temperature": 0.0}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(api_url, json=payload, headers=headers, timeout=60)
    try:
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Evaluator API request failed: {e}", "raw": resp.text}

    data = resp.json()
    text = None
    if isinstance(data, dict):
        if "text" in data:
            text = data["text"]
        elif "choices" in data and isinstance(data["choices"], list) and "text" in data["choices"][0]:
            text = data["choices"][0]["text"]
    text = text or resp.text

    try:
        parsed = json.loads(text.strip())
        return parsed
    except Exception:
        return {"raw_evaluator_output": text}

# -------------------------------
# Tool JSON wrapper helper
# -------------------------------
def make_json_tool(fn: Callable[[Any], Any]):
    """Wrap Python functions that accept kwargs into a string -> string tool for LangChain."""
    def tool_func(input_str: str) -> str:
        try:
            args = json.loads(input_str) if input_str else {}
        except Exception:
            args = {"q": input_str}
        try:
            res = fn(**args) if isinstance(args, dict) else fn(args)
            return json.dumps(res, default=str)
        except TypeError:
            res = fn(input_str)
            return json.dumps(res, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})
    return tool_func

# -------------------------------
# Streamlit UI & state setup
# -------------------------------
st.set_page_config(page_title="GAIBA Campaign Builder", layout="wide")
st.title("GAIBA — Campaign Builder (LangChain + Groq)")

# session_state defaults
if "email_df" not in st.session_state:
    st.session_state["email_df"] = None
if "last_campaign_spec" not in st.session_state:
    st.session_state["last_campaign_spec"] = None
if "agent" not in st.session_state:
    st.session_state["agent"] = None
if "agent_memory" not in st.session_state:
    st.session_state["agent_memory"] = None

# Sidebar: upload email CSV
st.sidebar.header("1) Upload email CSV (used for sends)")
email_file = st.sidebar.file_uploader("Upload email CSV", type=["csv"])
if email_file is not None:
    try:
        email_df = pd.read_csv(email_file)
        st.session_state["email_df"] = email_df
        v = validate_email_df(email_df)
        if not v.get("valid"):
            st.sidebar.error(f"Uploaded file missing columns: {v.get('missing_columns')}")
        else:
            st.sidebar.success(f"Email file valid: {v.get('n_valid_email')} emails detected")
            if st.sidebar.checkbox("Preview uploaded emails"):
                st.dataframe(email_df.head(10))
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

# Sidebar: agent settings
st.sidebar.header("2) Agent & model settings")
main_model = st.sidebar.text_input("Main model (LLM A) — model name", value=DEFAULT_MAIN_MODEL)
eval_model = st.sidebar.text_input("Evaluator model (LLM B) — model name", value=DEFAULT_EVAL_MODEL)
max_tokens = st.sidebar.number_input("Max tokens (agent)", value=DEFAULT_MAX_TOKENS)

# Sidebar: memory selection
st.sidebar.header("3) Memory")
memory_type = st.sidebar.selectbox("Memory type", ["buffer", "summary"], index=0,
                                   help="buffer keeps full chat history; summary keeps a rolling summary to save tokens")
show_memory = st.sidebar.checkbox("Show memory contents (debug)", value=False)

# Initialize LLM (Groq) once
if "llm_instance" not in st.session_state:
    try:
        st.session_state["llm_instance"] = GroqLLM(model_name=main_model, api_key=None, api_url=os.getenv("GROQ_API_URL", DEFAULT_GROQ_API_URL), max_tokens=max_tokens, temperature=0.0)
    except Exception as e:
        st.error(f"LLM init error: {e}")
        st.stop()

llm = st.session_state["llm_instance"]

# Initialize memory (persisted in session_state)
if st.session_state.get("agent_memory") is None:
    if memory_type == "buffer":
        st.session_state["agent_memory"] = ConversationBufferMemory(memory_key="chat_history", input_key="input", return_messages=True)
    else:
        # summary memory uses the LLM to create summaries of earlier history to save tokens
        st.session_state["agent_memory"] = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", input_key="input", return_messages=False, max_token_limit=1500)

memory = st.session_state["agent_memory"]

# -------------------------------
# Build tool closures that depend on current uploaded email_df
# -------------------------------
run_full_analysis_tool = make_json_tool(lambda filters=None: run_full_marketing_analysis(filters))
budget_allocator_tool = make_json_tool(lambda total_budget, expected_rois, min_caps=None, max_caps=None, method="proportional": budget_allocator(total_budget, expected_rois, method, min_caps, max_caps))

def get_sample_recipients_tool_fn(input_str: str) -> str:
    try:
        params = json.loads(input_str) if input_str else {}
    except Exception:
        params = {}
    email_df_local = st.session_state.get("email_df")
    if email_df_local is None:
        return json.dumps({"error": "no_email_file_uploaded"})
    targeting = params.get("targeting", {})
    n = int(params.get("n", 50))
    redact = bool(params.get("redact", True))
    res = get_sample_recipients(email_df_local, targeting=targeting, n=n, redact=redact)
    return json.dumps(res, default=str)

def send_email_tool_fn(input_str: str) -> str:
    try:
        params = json.loads(input_str) if input_str else {}
    except Exception:
        return json.dumps({"error": "invalid_json_input"})
    email_df_local = st.session_state.get("email_df")
    if email_df_local is None:
        return json.dumps({"error": "no_email_file_uploaded"})
    recipient_filter = params.get("recipient_filter", {})
    sample_n = int(params.get("sample_n", 20))
    require_confirmation = bool(params.get("require_confirmation", True))

    recs_df = email_df_local.copy()
    for k, v in (recipient_filter or {}).items():
        if k in recs_df.columns:
            recs_df = recs_df[recs_df[k] == v]
    recipients = recs_df.sample(n=min(sample_n, len(recs_df))).to_dict(orient="records") if len(recs_df) > 0 else []

    campaign_spec = params.get("campaign_spec", {})
    template = params.get("template", {})

    resp = send_email_batch_stub(campaign_spec=campaign_spec, template=template, recipients=recipients, require_confirmation=require_confirmation)
    return json.dumps(resp, default=str)

# Wrap as LangChain Tools
tools = [
    Tool(name="run_full_analysis", func=run_full_analysis_tool, description="Return a JSON summary of the marketing dataset. Input: JSON filters dict."),
    Tool(name="budget_allocator", func=budget_allocator_tool, description="Allocate budget across channels. Input: JSON with total_budget and expected_rois."),
    Tool(name="get_sample_recipients", func=get_sample_recipients_tool_fn, description="Return sample of recipients from uploaded email CSV. Input: JSON with targeting and n."),
    Tool(name="send_email", func=send_email_tool_fn, description="Prepare/send emails (gated). Input: JSON with campaign_spec, template, recipient_filter, sample_n, require_confirmation."),
]

# Initialize a conversational agent once and persist it in session_state
if st.session_state.get("agent") is None:
    try:
        agent_instance = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, memory=memory, verbose=True)
        st.session_state["agent"] = agent_instance
    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        st.stop()

agent = st.session_state["agent"]

# -------------------------------
# UI: Query / run agent
# -------------------------------
st.header("4) Ask the AI agent to analyze data or propose campaigns")
user_input = st.text_area("Ask the agent (examples: 'Analyze last 12 months ROI by channel', 'Propose 2 acquisition campaigns with INR 200000 budget for clusters 1 and 2')", height=160)
run_agent_btn = st.button("Run Agent")

if run_agent_btn:
    if not user_input.strip():
        st.warning("Please provide a question or instruction to the agent.")
    else:
        system_instructions = (
            "You are CampaignBuilder (LLM A). Use provided tools for any access to dataset-level numbers, "
            "aggregations, or budget allocation. Return final campaign specifications as JSON when requested. "
            "Do not invent exact numeric aggregates—always call run_full_analysis to retrieve historical numbers. "
            "Use the conversation history stored in memory to maintain context across turns."
        )
        prompt = system_instructions + "\n\nUser: " + user_input
        with st.spinner("Agent is thinking..."):
            try:
                agent_response = agent.run(prompt)
            except Exception as e:
                agent_response = f"Agent execution error: {e}"
            st.subheader("Agent output")
            st.text(agent_response)

            # Attempt to extract a campaign_spec from the agent response and store for convenience
            try:
                parsed = json.loads(agent_response)
                if isinstance(parsed, dict) and parsed.get("campaign_spec"):
                    st.session_state["last_campaign_spec"] = parsed["campaign_spec"]
            except Exception:
                # not JSON — ignore
                pass

# Optional: show memory contents if requested
if show_memory:
    try:
        mem_vars = memory.load_memory_variables({})  # returns dict with memory contents
        st.sidebar.subheader("Memory contents (debug)")
        st.sidebar.write(mem_vars.get("chat_history", mem_vars))
    except Exception as e:
        st.sidebar.write(f"Unable to show memory: {e}")

# -------------------------------
# UI: Evaluate a campaign (LLM B)
# -------------------------------
st.header("5) Evaluate a campaign (LLM B)")
st.markdown("Paste a campaign_spec JSON (from the agent) to get evaluator feedback.")
campaign_spec_text = st.text_area("campaign_spec JSON", height=180)
run_eval_btn = st.button("Run Evaluator")

if run_eval_btn:
    if not campaign_spec_text.strip():
        st.warning("Paste a campaign_spec JSON first.")
    else:
        try:
            campaign_spec = json.loads(campaign_spec_text)
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            campaign_spec = None

        if campaign_spec:
            # Derive a historical summary — e.g., for a target cluster or global
            target_clusters = campaign_spec.get("target_clusters", [])
            hist_summary = {}
            try:
                if target_clusters:
                    hist_summary_resp = run_full_marketing_analysis({"cluster": target_clusters[0]})
                    hist_summary = hist_summary_resp.get("analysis", hist_summary_resp)
                else:
                    hist_summary_resp = run_full_marketing_analysis(None)
                    hist_summary = hist_summary_resp.get("analysis", hist_summary_resp)
            except Exception as e:
                hist_summary = {"error_fetching_history": str(e)}

            with st.spinner("Running evaluator..."):
                eval_resp = evaluator_score_campaign(campaign_spec, hist_summary, eval_model_name=eval_model)
                st.subheader("Evaluator output")
                st.json(eval_resp)

# -------------------------------
# UI: Send sample (gated)
# -------------------------------
st.header("6) Send sample / preview (gated)")
if st.button("Prepare sample send (20 recipients) from last proposal in app"):
    if st.session_state.get("last_campaign_spec"):
        cs = st.session_state["last_campaign_spec"]
    else:
        st.warning("No stored campaign_spec in session. Copy agent output into the 'campaign_spec JSON' box.")
        cs = None
    if cs:
        payload = {"campaign_spec": cs, "template": cs.get("email_template_preview", {}), "sample_n": 20, "require_confirmation": True}
        resp = send_email_tool_fn(json.dumps(payload))
        st.write(json.loads(resp))

st.markdown("---")
st.caption("Notes: This demo wires your local deterministic tools as LangChain Tools that the agent can call. "
           "Customize GroqLLM._call() to match the exact Groq API response schema you have. "
           "Integrate a real mail provider inside send_email_batch_stub when you're ready to actually send emails.")
