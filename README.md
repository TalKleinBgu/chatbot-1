# Finance Insight Analyzer

This Streamlit-based application is designed to provide insightful analysis of financial and stock-related data using text inputs, uploaded PDF files, and online resources. The app leverages powerful AI tools like OpenAI's GPT-4 and Tavily Search to generate multilingual, actionable insights.
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-template.streamlit.app/)

## Key Features

- **Text and PDF Input**: Analyze user-provided text or uploaded PDF files.
- **Web Search**: Use Tavily Search to retrieve relevant content from the web.
- **Financial News Analysis**: Fetch and analyze financial news articles from Yahoo Finance using company tickers.
- **AI-Powered Insights**: Generate professional-grade, multilingual stock insights using GPT-4.
- **Stateful Workflow**: A step-by-step workflow enables efficient data collection, processing, and analysis.

## Prerequisites

- **Python 3.11 or later**: [Python Installation Guide](https://www.tutorialsteacher.com/python/install-python)
- **API Keys**:
  - **Tavily**: [Sign Up](https://tavily.com/)
  - **OpenAI**: [Sign Up](https://openai.com/)
  - **Anthropic**: [Sign Up](https://console.anthropic.com/settings/keys)

## Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/danielleyahalom/company-researcher.git
cd company-researcher
```

### 2. Create a Virtual Environment

To avoid dependency conflicts, create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

### 3. Configure API Keys

Set up your Tavily, OpenAI API keys. You can export them as environment variables or store them in a .env file:

```bash
export TAVILY_API_KEY={Your Tavily API Key}
export OPENAI_API_KEY={Your OpenAI API Key}
```

### 4. Install Dependencies

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 5. Run the Application

Start the application:

```bash
python app.py
```

### 6. Open the App

Navigate to http://localhost:8051 in your browser.

## Design Decisions

### Architecture

The application employs a Retrieval-Augmented Generation (RAG) architecture to combine the strengths of retrieval and generative models:

- **Retrieval**: Tavily Search and Yahoo Finance are used to gather relevant documents and financial news.
- **Augmented Generation**: GPT-4 processes the retrieved content and user inputs to generate actionable insights.

### Workflow Design

A stateful workflow is implemented using StateGraph to ensure clarity and modularity:

1. **InputCollectionState**: Collect user inputs (text or PDF files).
2. **TextExtractionState**: Extract text from inputs for analysis.
3. **SearchWeb**: Use Tavily Search for web-based content retrieval.
4. **SearchYahooFinance**: Fetch financial news using Yahoo Finance.
5. **AnalysisState**: Generate insights based on retrieved data and inputs.
6. **OutputState**: Display the insights to the user.


## Code Overview

### Key Components

#### Custom Yahoo Finance Tool

The YahooFinanceNewsTool retrieves and processes financial news for a given company ticker using yfinance and WebBaseLoader.

#### Tavily Search Integration

The app integrates Tavily Search to retrieve relevant online resources, ensuring multilingual support and robust insights.

#### StateGraph Workflow

The StateGraph ensures modular and logical progression through each stage of the analysis pipeline, from input collection to insights generation.
