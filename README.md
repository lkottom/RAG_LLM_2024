# RAG-Based Biblical Sermon Chatbot

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) chatbot that provides answers based on Biblical sermons, featuring a web scraper, core chatbot logic, Streamlit UI, and Flask API backend.

## Components

### Core Chatbot (`main_rag_bot.py`)
- Handles RAG logic with Pinecone and HuggingFace
- Processes sermons and manages vector storage
- Run with: `python main_rag_bot.py`

### Streamlit UI (`mainUI.py`) 
- Provides chat interface with sermon source links
- Run with: `streamlit run mainUI.py`

### Flask API (`server.py`)
- Serves chatbot via REST endpoints
- Supports CORS and JSON responses
- Run with: `python server.py`

### Sermon Scraper (`scraping.ipynb`)
- Jupyter notebook for sermon data collection
- Formats data into JSON files

## Setup

### Dependencies
```bash
pip install -r requirements.txt
```

## Setup Environment Variables

### Create .env:

```bash 
PINECONE_API_KEY=<your-pinecone-api-key>
HUGGINGFACE_ACCESS_TOKEN=<your-huggingface-access-token>
```

## API Endpoints
- GET /: Welcome message
- POST /chat:
    - Input: {"message": "your question"}
    - Output: Chatbot response with sources

## Quick Start 
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Scrape the webside 
- Open scraping.ipynb in Jupyter Notebook/Lab
- Execute cells sequentially to gather sermon data

3. Start backend:
```bash
python server.py
```

4. Test API/frontend on developmental server:
```bash
curl -X POST http://127.0.0.1:5001/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is faith?"}'
```

## Running Streamlit Chatbot UI
Launch the user interface with:
```bash
streamlit run mainUI.py
```


