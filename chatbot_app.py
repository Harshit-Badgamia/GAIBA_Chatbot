# smart_budget_allocator.py

import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import os
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# -----------------------------
# API Keys
# -----------------------------
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
NEWSAPI_KEY = st.secrets["NEWSAPI_KEY"]

# -----------------------------
# Placeholder Functions
# -----------------------------

def load_marketing_data(file):
    df = pd.read_csv(file)
    return df

def get_financial_forecast(company):
    stock = yf.Ticker(company)
    hist = stock.history(period="6mo")
    return hist

def analyze_sentiment(sector):
    url = f"https://newsapi.org/v2/everything?q={sector}&sortBy=publishedAt&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])[:5]
    titles = [article['title'] for article in articles]
    summary = f"Recent sentiment for sector '{sector}':\n" + "\n".join(titles)
    return summary

def optimize_budget(marketing_data, forecast, total_budget):
    marketing_data['Allocation'] = marketing_data['ROI'] / marketing_data['ROI'].sum()
    marketing_data['Suggested Budget'] = marketing_data['Allocation'] * total_budget
    return marketing_data[['Channel', 'Suggested Budget', 'ROI']]

def generate_summary(dataframe, company, sector_sentiment):
    summary = f"Based on financial trends of {company} and current market sentiment: \n{sector_sentiment}\n"
    summary += "the suggested marketing budget has been allocated proportionally to past ROI performance."
    return summary

# -----------------------------
# LangChain Agent Setup (with Gemini)
# -----------------------------

tools = [
    Tool(
        name="LoadMarketingData",
        func=load_marketing_data,
        description="Loads a marketing performance CSV file."
    ),
    Tool(
        name="GetFinancialForecast",
        func=get_financial_forecast,
        description="Fetches recent stock history for a given company."
    ),
    Tool(
        name="AnalyzeSentiment",
        func=analyze_sentiment,
        description="Analyzes sentiment for a given sector using NewsAPI."
    ),
    Tool(
        name="OptimizeBudget",
        func=optimize_budget,
        description="Optimizes the marketing budget based on ROI and forecasts."
    ),
    Tool(
        name="GenerateSummary",
        func=generate_summary,
        description="Generates a summary report based on insights."
    )
]

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY, temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Smart Budget Allocator", layout="wide")
st.title("💰 Smart Budget Allocator for Marketing Campaigns")

with st.sidebar:
    st.header("Input Parameters")
    uploaded_file = st.file_uploader("Upload Marketing CSV", type=["csv"])
    company = st.text_input("Target Company Ticker (e.g., TSLA)")
    budget = st.number_input("Total Budget (USD)", min_value=0.0, value=10000.0)
    sector = st.text_input("Sector of Interest (e.g., EV, Tech)")

if uploaded_file and company and sector:
    st.subheader("📊 Optimized Budget Allocation")
    df = load_marketing_data(uploaded_file)
    forecast = get_financial_forecast(company)
    sentiment = analyze_sentiment(sector)
    result = optimize_budget(df, forecast, budget)
    summary = generate_summary(result, company, sentiment)

    st.dataframe(result)
    st.markdown("---")
    st.subheader("📝 Summary")
    st.write(summary)

    st.markdown("---")
    st.subheader("💬 Ask the Agent")
    user_query = st.text_input("Ask a question about the campaign strategy:")
    if user_query:
        response = agent.run(user_query)
        st.write(response)
else:
    st.info("Please fill out all fields and upload a CSV to begin.")
