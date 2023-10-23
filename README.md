
# AI Equity Research Analyst with LlamaIndex

Welcome to the AI Equity Research Analyst project! This application showcases the power of Large Language Models (LLMs) in generating equity research reports.

## Overview

This Streamlit app demonstrates the use of LLMs, specifically using the Google PaLM API, to analyze and extract data from 10-K filings of renowned NYSE listed companies. The LlamaIndex orchestration framework, along with its SubQuestionQuery Engine, breaks down complex prompts into sub-questions, retrieves answers, and assembles the final equity research report.

### Demo Features:
* Select from 7 renowned NYSE listed companies.
* App reads, analyzes the latest 10-K filings, and extracts data based on complex prompts.
* LlamaIndex's SubQuestionQuery Engine decomposes complex prompts into sub-questions, fetches answers, and assembles the final report.

### Potential Enhancements:
* API-driven financial data access.
* Dedicated Python REPL for accurate calculations.

[Try the demo here!](https://lnkd.in/dYwEXFYy)

## Tools and Technologies Used:
- **LLM**: Google PaLM API
- **LLM Orchestration Framework**: LlamaIndex
- **Embedding Model**: Hugging Face bge-small-en
- **Vector Store**: Llamaindex's in-memory SimpleVectorStore
- **Query Engine**: SubQuestionQueryEngine
- **UI**: Streamlit

## Project Structure:
- `app.py`: The main application script for the Streamlit app.
- `requirements.txt`: Lists the Python packages required for this project.
- `storage/`: Directory containing resources or data for the project.
- `tenk/`: Directory holding data or resources, likely related to 10-K filings.

## Setup and Usage:

1. **Clone the Repository**:
    ```
    git clone https://github.com/AI-ANK/ai-equity-research-analyst
    ```

2. **Install Required Packages**:
    ```
    pip install -r requirements.txt
    ```

3. **Run the Streamlit App**:
    ```
    streamlit run app.py
    ```

## Feedback and Contributions:
Feel free to raise issues or submit pull requests if you think something can be improved or added. Your feedback is highly appreciated!

---

Developed by Harshad Suryawanshi. If you find this project useful, consider giving it a ⭐ on GitHub!
