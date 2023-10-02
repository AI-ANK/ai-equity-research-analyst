import streamlit as st
import os
import pprint
import google.generativeai as palm
from llama_index.llms.palm import PaLM
import nest_asyncio
from llama_index import SimpleDirectoryReader, ServiceContext
from llama_index.response.pprint_utils import pprint_response
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
import time
from IPython.display import display, Markdown

nest_asyncio.apply()

# This assumes you've set up your secrets accordingly
palm_api_key = st.secrets["palm_api_key"]

palm.configure(api_key=palm_api_key)
model = PaLM(api_key=palm_api_key)
service_context = ServiceContext.from_defaults(llm=model)

def load_pdf(company):
    if company == "Apple":
        file_path = "https://github.com/AI-ANK/aiequityanalyst/raw/main/apple.pdf"
    elif company == "Tesla":
        file_path = "https://github.com/AI-ANK/aiequityanalyst/raw/main/tesla.pdf"
    
    return SimpleDirectoryReader(input_files=[file_path]).load_data()

def generate_report(data):
    index = VectorStoreIndex.from_documents(data, service_context=service_context)
    appleengine = index.as_query_engine(similarity_top_k=3)
    query_engine_tools = [
        QueryEngineTool(
            query_engine=appleengine,
            metadata=ToolMetadata(
                name="engine",
                description=f"Provides information about {company} annual 10-k financials",
            ),
        ),
    ]
    s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools, service_context=service_context)
    
    # Queries for the report
    query1 =  """
Perform a comprehensive analysis of the 10-k and generate the first half of an equity analysis report following this format in markdown:
    
## 1. Executive Summary
- A brief overview of the company, its industry, and the main conclusions of the report.

## 2. Company Overview
- Detailed background information about the company, including its history, products or services, and organizational structure.

## 3. Industry Analysis
- A review of the industry in which the company operates, including trends, competition, and growth prospects.

## 4. Financial Analysis
- An in-depth analysis of the company's financial statements, including ratio analysis, cash flows, and profitability.

### 4.1 Income Statement Analysis
- Examination of revenue, cost structures, and net income trends.

### 4.2 Balance Sheet Analysis
- A breakdown of assets, liabilities, and shareholder equity.

### 4.3 Cash Flow Statement Analysis
- Insights into the company's cash generation and expenditure patterns.
    """
    query2 = """
Perform a comprehensive analysis of the 10-k and generate the second half of an equity analysis report following this format in markdown:


## 5. Valuation
- Assessment of the company's current valuation using various methodologies like DCF, comparables, and precedent transactions.

## 6. SWOT Analysis
- A breakdown of the company's Strengths, Weaknesses, Opportunities, and Threats.

## 7. Investment Thesis
- A clear statement of the analyst's view on the stock (buy, hold, sell) and the rationale behind that view.

## 8. Risks & Challenges
- An outline of potential risks and challenges that could affect the company's future performance.

## 9. Conclusion & Recommendations
- Summarizing the main findings of the report and providing clear recommendations for potential investors.
"""
    
    response1 = s_engine.query(query1)
    response2 = s_engine.query(query2)
    
    return response1.response, response2.response

st.title("Equity Analysis Report Generator")

# Dropdown for company selection
company = st.selectbox("Select a company:", ["Apple", "Tesla"])

# Load button to trigger analysis
if st.button("Generate Report"):
    data = load_pdf(company)
    report1, report2 = generate_report(data)
    st.markdown(report1)
    st.markdown(report2)
