import os
import operator
import PyPDF2
import yfinance
import streamlit as st
from typing import TypedDict, Iterable, Optional, Annotated
from requests.exceptions import HTTPError, ReadTimeout
from urllib3.exceptions import ConnectionError

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain.agents import AgentType, initialize_agent
from dotenv import load_dotenv
memory = MemorySaver()
load_dotenv()

st.title("Finance Insight Analyzer")
# Get API keys from the configuration
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Check if API keys are present
if not TAVILY_API_KEY or not OPENAI_API_KEY:
    st.error("API keys not found in the environment variables. Please check your .env file.")
    st.stop()

# Set environment variables for downstream use
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Define the state with annotated lists
class State(TypedDict):
    input_content: Annotated[list, operator.add]
    input_type: Annotated[list, operator.add]
    extracted_text: Annotated[list, operator.add]
    search_results: Annotated[list, operator.add]
    insights: Annotated[list, operator.add]
    sources: Annotated[list, operator.add]
    pdf: Annotated[list, operator.add]
    base_text: Annotated[list, operator.add]

# Initialize LLM and tools
llm = ChatOpenAI(model="gpt-4o", temperature=0)
tavily_search = TavilySearchResults(max_results=4)

class YahooFinanceNewsTool(BaseTool):
    """
    Custom tool to retrieve news articles about a given company from Yahoo Finance.
    """
    name: str = "yahoo_finance_news"
    description: str = (
        "Useful for when you need to find financial news "
        "about a public company. "
        "Input should be a company ticker. "
        "For example, AAPL for Apple, MSFT for Microsoft."
    )
    top_k: int = 10

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Retrieve news articles about a given company ticker.
        """
        # Retrieve ticker data
        company = yfinance.Ticker(query)
        
        # Check if the company ticker is valid
        try:
            if company.isin is None:
                return f"Company ticker {query} not found."
        except (HTTPError, ReadTimeout, ConnectionError):
            return f"Company ticker {query} not found."

        # Get news links from Yahoo Finance
        links = []
        try:
            links = [n["link"] for n in company.news if n["type"] == "STORY"]
        except (HTTPError, ReadTimeout, ConnectionError):
            if not links:
                return f"No news found for company with {query} ticker."
        if not links:
            return f"No news found for company with {query} ticker."

        # Load news content using WebBaseLoader
        loader = WebBaseLoader(
            web_paths=links,
            header_template={
                "Cookie": "EuConsent:#AddYourNavigatorCookieHere"
            },
        )
        docs = loader.load()
        
        # Format the retrieved documents
        result = self._format_results(docs, query)
        if not result:
            return f"No news found for company with {query} ticker."
        return result

    @staticmethod
    def _format_results(docs: Iterable[Document], query: str) -> str:
        """
        Format the documents for presentation.
        """
        doc_strings = [
            "\n".join([doc.metadata["title"], doc.metadata["description"]])
            for doc in docs
            if "description" in doc.metadata
            and (query in doc.metadata["description"] or query in doc.metadata["title"])
        ]
        return "\n\n".join(doc_strings)

# Define workflow states and transitions
builder = StateGraph(State)

def input_collection_state(context):
    """
    Collect input content and/or a PDF file from the user.
    """
    # Collect text input
    input_text = st.text_area("Enter text about the stock:")
    
    # Collect file input
    uploaded_file = st.file_uploader("Upload a PDF file:", type="pdf")
    
    # Update context based on the input
    if st.button("Submit"):
        
        if uploaded_file is None:
            context['input_type'] = ["Text"]
            context['input_content'] = [input_text]
        else:
            context['input_type'] = ["Upload File"]
            context['input_content'] = [input_text]
            context['pdf'] = [uploaded_file.read()]
        return context
    else:
        st.stop()

def text_extraction_state(context):
    """
    Extract text from the provided input or uploaded PDF file.
    """
    input_type = context.get('input_type', [None])[0]
    input_content = context.get('input_content', [None])[0]

    context['extracted_text'] = [input_content]

    if input_type == 'Upload File':
        pdf_content = context['pdf'][0]  # Access the PDF content bytes
        try:
            # Use PyPDF2 to process the PDF content
            from io import BytesIO
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
            text = ''.join(page.extract_text() for page in pdf_reader.pages)
            context['base_text'] = [text] if text else []
        except Exception as e:
            st.error(f"Error extracting text from file: {e}")

    if context['extracted_text']:
        return context
    else:
        st.error("No text could be extracted. Please check your input.")
        return context

def search_web(context):
    """
    Search for related content on the web using Tavily Search.
    """
    st.write("Analyzing insights...")
    query = context['extracted_text'][0]
    
    # Invoke Tavily Search and format results
    search_docs = tavily_search.invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'{doc.get("content", "Content not available")}'
            for doc in search_docs
        ]
    )   
    sources_ = [f'{doc.get("url", "Content not available")}'
                for doc in search_docs]
    
    # Update context with search results
    context['search_results'] = [formatted_search_docs]
    context['sources'] = [sources_]
    return context

def search_yahoo_finance(context):
    """
    Search for related financial news using YahooFinanceNewsTool.
    """
    query = context['extracted_text']
    tools = [YahooFinanceNewsTool()]
    
    # Initialize and run the agent chain
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )
    processed_results = agent_chain.invoke(query)
    context['search_results'] = [processed_results]
    return context

def analysis_state(context):
    """
    Analyze and generate insights from the gathered data.
    """
    st.write("Analyzing insights...")
    search_results = context['search_results']
    extracted_text = context['extracted_text']
    input_type = context.get('input_type', [None])[0]
    input_content = context.get('input_content', [None])[0]
    sources_ = context['sources']

    # Prepare instructions for LLM based on input type
    if input_type == "Upload File":
        base_text = context.get('base_text', [None])[0]
    else:
        base_text = ''
    answer_template = (
        "You are an expert financial analyst specializing in multilingual stock insights. "
        "Your task is to analyze the provided information, including text and search results, to generate comprehensive, actionable insights. "
        "Ensure your output is clear, concise, and professionally formatted.\n\n"
        "User Input Instructions:\n"
        "Given the following context:\n\n"
        "Base Text(file): {base_text} (if provided; otherwise, focus on the query)\n"
        "Query: {input_content}\n"
        "Search Results: {search_results}\n"
        "Sources: {sources_}\n\n"
        "1. Provide a detailed summary of insights regarding the query or stock-related topics.\n"
        "2. Use search results and base text for context, if available, but do not explicitly reference them in the insights.\n"
        "3. Include actionable points or trends related to the stock (e.g., performance, market news, or relevant risks).\n"
        "4. Conclude with a neatly formatted section showing credible sources, if posiible show links."
        )

    answer_instructions = answer_template.format(
        base_text=base_text, input_content=input_content, sources_=sources_, search_results=search_results
    )

    # Invoke LLM to generate insights
    answer = llm.invoke([
        SystemMessage(content=answer_instructions),
        HumanMessage(content="Generate insights")
    ])
    
    # Update context with generated insights
    context['insights'] = [{'title': extracted_text, 'insight': answer.content}]
    return context

def output_state(context):
    """
    Output the generated insights to the user.
    """
    insights = context.get('insights', [])
    st.write("-----------------------------------------------------")
    for summary in insights:
        st.write(f"{summary['insight']}")
        st.markdown("---")

# Add states to the workflow
builder.add_node('InputCollectionState', input_collection_state)
builder.add_node('TextExtractionState', text_extraction_state)
builder.add_node('SearchWeb', search_web)
builder.add_node('SearchYahooFinance', search_yahoo_finance)
builder.add_node('AnalysisState', analysis_state)
builder.add_node('OutputState', output_state)

# Define transitions
builder.add_edge(START, 'InputCollectionState')
builder.add_edge('InputCollectionState', 'TextExtractionState')
builder.add_edge('TextExtractionState', 'SearchWeb')
builder.add_edge('TextExtractionState', 'SearchYahooFinance')
builder.add_edge('SearchWeb', 'AnalysisState')
builder.add_edge('SearchYahooFinance', 'AnalysisState')
builder.add_edge('AnalysisState', 'OutputState')
builder.add_edge('OutputState', END)

# Compile the graph
graph = builder.compile(checkpointer=memory)

# Streamlit interface
if __name__ == "__main__":

    initial_context = {
        "input_content": [],
        "input_type": [],
        "extracted_text": [],
        "search_results": [],
        "insights": [],
    }
    config = {"configurable":{"thread_id":"1"}}
    result_context = graph.invoke(initial_context,config)
