import langchain
import streamlit as st
import os
import google.generativeai as palm
from llama_index.llms.palm import PaLM
from llama_index import SimpleDirectoryReader, ServiceContext, VectorStoreIndex, StorageContext
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.callbacks.base_handler import BaseCallbackHandler
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.callbacks import CallbackManager
from time import sleep
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import load_index_from_storage
from typing import Optional, Dict, Any, List

import nest_asyncio
nest_asyncio.apply()

@st.cache_resource
def initialize_model(api_key):
    return PaLM(api_key=api_key)

@st.cache_data
def load_data(file_path):
    return SimpleDirectoryReader(input_files=[file_path]).load_data()

# Streamlit page configuration
st.set_page_config(page_title="AI Generated Equity Research Report", layout="wide")

# Import secret (You'll need to replace this with your own method of importing secrets)
palm_api_key = st.secrets['palm_api_key']
palm.configure(api_key=palm_api_key)

# Initialize model
model = initialize_model(palm_api_key)

# Set up Callback Manager

# Revised Callback Handler
class SubQuestionHandler(BaseCallbackHandler):
    def __init__(self, event_starts_to_ignore=[], event_ends_to_ignore=[]):
        super().__init__(event_starts_to_ignore, event_ends_to_ignore)
        
    def on_event_start(self, event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: str = "", parent_id: str = "", **kwargs: Any) -> str:
        if event_type == CBEventType.SUB_QUESTION:
            sub_question_data = payload.get(EventPayload.SUB_QUESTION.value)
            if sub_question_data:
                sub_question = sub_question_data.sub_q.sub_question
                answer = sub_question_data.answer
               
        return event_id  # Return the event ID (unchanged)
    
    def on_event_end(self, event_type: CBEventType, payload: Optional[Dict[str, Any]] = None, event_id: str = "", **kwargs: Any) -> None:
        if event_type == CBEventType.SUB_QUESTION:
            sub_question_data = payload.get(EventPayload.SUB_QUESTION.value)
            if sub_question_data:
                sub_question = sub_question_data.sub_q.sub_question
                answer = sub_question_data.answer
                # Print the sub-question and its answer
                st.write(f"Sub-Question: {sub_question} \n")
                st.write(f"Answer: {answer}\n")
                st.markdown("---") 
        return event_id  # Return the event ID (unchanged)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(self, trace_id: Optional[str] = None, trace_map: Optional[Dict[str, List[str]]] = None) -> None:
        pass


callback_manager = CallbackManager(handlers=[SubQuestionHandler()])

# Optionally set specific embed model
embed_model = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en")

# Set service context
service_context = ServiceContext.from_defaults(llm=model, embed_model=embed_model, callback_manager = callback_manager)

# Company Data
company_data = {
    'Apple': {
        'url': "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/b4266e40-1de6-4a34-9dfb-8632b8bd57e0.pdf",
        'financial_year': 'For the fiscal year ended September 24, 2022'
    },
    'Microsoft': {
        'url': "https://microsoft.gcs-web.com/static-files/e2931fdb-9823-4130-b2a8-f6b8db0b15a9",
        'financial_year': 'For the Fiscal Year Ended June 30, 2023'
    },
    'Amazon': {
        'url': "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001018724/d2fde7ee-05f7-419d-9ce8-186de4c96e25.pdf",
        'financial_year': 'For the fiscal year ended December 31, 2022'
    },
    'Nvidia': {
        'url': "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/4e9abe7b-fdc7-4cd2-8487-dc3a99f30e98.pdf",
        'financial_year': 'For the fiscal year ended January 29, 2023'
    },
    'Alphabet': {
        'url': "https://abc.xyz/assets/9a/bd/838c917c4b4ab21f94e84c3c2c65/goog-10-k-q4-2022.pdf",
        'financial_year': 'For the fiscal year ended December 31, 2022'
    },
    'Meta': {
        'url': "https://d18rn0p25nwr6d.cloudfront.net/CIK-0001326801/e574646c-c642-42d9-9229-3892b13aabfb.pdf",
        'financial_year': 'For the fiscal year ended December 31, 2022'
    },
    'Tesla': {
        'url': "https://ir.tesla.com/_flysystem/s3/sec/000095017023001409/tsla-20221231-gen.pdf",
        'financial_year': 'For the fiscal year ended December 31, 2022'
    },
    # ...
}

# Sidebar
st.sidebar.markdown('## Created By')
st.sidebar.markdown("""
[Harshad Suryawanshi](https://www.linkedin.com/in/harshadsuryawanshi/)
""")

st.sidebar.markdown('## Other Projects')
st.sidebar.markdown("""
- [Recasting "The Office" Scene](https://blackmirroroffice.streamlit.app/)
- [Story Generator](https://appstorycombined-agaf9j4ceit.streamlit.app/)
""")

st.sidebar.markdown('## Disclaimer')
st.sidebar.markdown("""
This application is a conceptual prototype created to demonstrate the potential of Large Language Models (LLMs) in generating equity research reports. The contents generated by this application are purely illustrative and should not be construed as financial advice, endorsements, or recommendations. The author and the application do not provide any guarantee regarding the accuracy, completeness, or timeliness of the information provided.
""")

# Streamlit UI: Company Selector
st.title("Select a Company to Generate Equity Research Report")
company = st.selectbox(" ", list(company_data.keys()))

if st.button("Generate Report"):
    with st.spinner("Processing..."):    
        # Load pdf 
        pdf_file_path = f"./tenk/10k_{company}.pdf"
        tenk_company = load_data(pdf_file_path)

        # If you want to build a fresh vector index, uncomment the following lines:
        # storage_context = StorageContext.from_defaults()
        # index = VectorStoreIndex.from_documents(tenk_company, service_context=service_context, use_async=True, storage_context = storage_context)
        # index.set_index_id("index_"+company)
        # index.storage_context.persist(persist_dir="storage")
        
        # Load vector indexes from folder
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        index = load_index_from_storage(storage_context, index_id=f"index_{company}", service_context=service_context)
        
        # Build query engine
        engine = index.as_query_engine(similarity_top_k=3)
        query_engine_tools = [
            QueryEngineTool(
                query_engine=engine,
                metadata=ToolMetadata(
                    name="10k engine",
                    description=f"Provides information about {company} annual 10-k financials {company_data[company]['financial_year']}",
                ),
            ),
        ]
        
        s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_engine_tools, service_context=service_context, use_async=True)
    
    # Query Response
    if company:
        with st.status("Generating sub-questions and fetching reponses...", expanded = True) as status:
            response = s_engine.query(
            """
            Perform a comprehensive analysis of the 10-k and generate the first half of an equity analysis report while strictly following this format in markdown:
                
            ## 1. Executive Summary
            - A brief overview of the company, its industry, and the main conclusions of the report.
            
            ## 2. Industry Analysis
            - A review of the industry in which the company operates, including trends, competition, and growth prospects.
            
            ## 3. Financial Analysis
            - An in-depth analysis of the company's financial statements, including ratio analysis, cash flows, and profitability.
            
            ## 3. Financial Analysis
            - A narrative analysis of the company's financial statements, focusing on the trends, anomalies, or notable items.
            
            ### 3.1 Income Statement Analysis
            - **Profitability Trends**: Discuss the trends in net income and operating income. Identify any irregularities or notable events.
            
            ### 3.2 Balance Sheet Analysis
            - **Liquidity and Solvency**: Comment on the company's short-term and long-term financial health, referencing relevant items from the balance sheet.
            
            ### 3.3 Cash Flow Statement Analysis
            - **Free Cash Flow Trends**: Comment on the trend in free cash flow and discuss any factors affecting it.
            
            DO NOT INCLUDE ANY CONCLUSION HERE
                """
            )
            
            response2 = s_engine.query(
            
            """
            Perform a comprehensive analysis of the 10-k and generate the second half of an equity analysis report while strictly following this format in markdown:
            
            ## 4. Valuation
            
            - A qualitative assessment of the company’s valuation based on the information available in the 10-K report.
            
            ### 4.1 Market Perception
            - **Investor Sentiment**: Discuss any commentary on investor sentiment or market perception provided in the 10-K.
            - **Historical Stock Performance**: Reflect on the historical stock performance and any factors mentioned in the 10-K that might have influenced it.
            
            ### 4.2 Forward-Looking Statements
            - **Outlook and Projections**: Summarize any forward-looking statements or financial outlook provided in the 10-K.
            - **Strategic Initiatives**: Highlight any strategic initiatives or plans discussed in the 10-K that are expected to impact future valuation.
            
            
            ## 5. SWOT Analysis
            - A breakdown of the company's Strengths, Weaknesses, Opportunities, and Threats.
            
            ## 6.  Risk Factors
            - **Identified Risks**: Summarize the key risk factors identified in the 10-K that could affect the company’s valuation.
            - **Mitigation Strategies**: Discuss any mitigation strategies mentioned to address these risks.
            
            
            ## 7. Conclusion & Recommendations
            - A clear statement of the analyst's view on the stock (buy, hold, sell) and the rationale behind that view.
            """
            )
            status.update(state="complete", expanded=False)
    
        # Display Results
        st.markdown(f"# {company} Equity Research Report (Simplified Version)")
        st.markdown(response.response)
        st.markdown(response2.response)
