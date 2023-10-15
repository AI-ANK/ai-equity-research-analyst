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

# Streamlit page configuration
st.set_page_config(page_title="AI Generated Equity Research Report", layout="wide")

# Import secret (You'll need to replace this with your own method of importing secrets)
palm_api_key = st.secrets['palm_api_key']
palm.configure(api_key=palm_api_key)

# Initialize model
model = PaLM(api_key=palm_api_key)

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
                print(f"Sub-Question: {sub_question}")
                sleep(2)
                print(f"Answer: {answer}\n")
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
    # ... add more companies as needed
}

# Streamlit UI: Company Selector
st.header("Select a Company")
company = st.selectbox("Choose a company", list(company_data.keys()))


# Load pdf from HuggingFace or another source
pdf_file_path = f"./tenk/10k_{company}.pdf"
tenk_company = SimpleDirectoryReader(input_files=[pdf_file_path]).load_data()

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
with st.spinner("Fetching data..."):
    response = s_engine.query("What is the company name")

# Display Results
st.markdown(f"# {company} Basic Equity Research Report")
st.markdown(response.response)
