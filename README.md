# FaceBase Chatbot: RAG-Enhanced Data Discovery

A chatbot system that leverages Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) to enhance the discoverability of specialized craniofacial biomedical data from the FaceBase repository, aligning with FAIR (Findable, Accessible, Interoperable, Reusable) data principles.

## Overview

This project addresses the challenge of discovering specialized biomedical data within complex repositories. By combining OpenAI's GPT-3.5-turbo with RAG techniques and FAISS vector search, the chatbot enables researchers to query the FaceBase dataset using natural language and receive relevant, contextual responses.

## Features

- **Natural Language Querying**: Interpret complex research queries using LLMs
- **Semantic Search**: FAISS-powered vector similarity search for relevant data retrieval
- **Interactive Interface**: User-friendly Streamlit web application
- **Context-Aware Responses**: RAG-enhanced generation for accurate, informative answers
- **FAIR Alignment**: Improves data findability within the FaceBase repository

## Technology Stack

- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: OpenAI Embeddings via LangChain
- **Vector Search**: FAISS (Facebook AI Similarity Search)
- **Frontend**: Streamlit
- **Language**: Python

## Installation
```bash
pip install streamlit openai langchain-openai faiss-cpu numpy
```

## Usage

1. Set up your OpenAI API key in Streamlit secrets:
```toml
# .streamlit/secrets.toml
[openai]
api_key = "your-api-key-here"
```

2. Ensure `isa_dataset (1).json` and `FaceBase.png` are in the project directory

3. Run the application:
```bash
streamlit run app.py
```

4. Enter your query in the text input and click "send" to receive responses

## Architecture

1. **Query Processing**: User input is preprocessed and embedded
2. **Retrieval**: FAISS searches for top-k most relevant dataset descriptions
3. **Generation**: Retrieved context is combined with the query and sent to GPT-3.5-turbo
4. **Response**: Generated answer is displayed along with relevant data snippets

## Research Context

This chatbot was developed as part of directed research at USC's Information Sciences Institute under Prof. Carl Kesselman. The project demonstrates how AI techniques can revolutionize data discovery in specialized biomedical domains while promoting FAIR data principles.

## Future Improvements

- Metadata standardization and enrichment
- Scalability enhancements for larger datasets
- Integration with additional FAIR principles
- Comprehensive qualitative assessment
- User feedback incorporation

## Contact

Minoo Ahmadi - minooahm@usc.edu
