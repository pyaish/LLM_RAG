# LLM_RAG
RAG-style AI Assistant demo
"""
mini_rag_assistant.py

A tiny Retrieval-Augmented Generation (RAG) demo using LangChain.

What it does:
- Loads text files from the ./docs directory
- Splits them into chunks and builds a FAISS vector index
- Uses an LLM (OpenAI) + retriever to answer questions grounded in your docs

How to run:
1. pip install langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu python-dotenv
2. Create a .env file with: OPENAI_API_KEY="your_api_key_here"
3. Put some .txt files into a folder called "docs" next to this script.
4. python mini_rag_assistant.py
"""
