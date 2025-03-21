# LangChain Ecosystem: Practical Guide

This repository contains the code examples from the article "Demystifying the LangChain Ecosystem for LLM-Powered Application Development." It provides practical examples of working with the LangChain framework to build LLM-powered applications.

## 🌟 Overview

The code in this repo demonstrates different aspects of the LangChain ecosystem, from setting up your environment to implementing a complete Retrieval Augmented Generation (RAG) system. It's designed to be beginner-friendly while covering advanced topics.

## 🛠️ Setting Up

### Prerequisites

- Python 3.11 or higher
- Poetry (for dependency management)
- API keys for LLM providers (OpenAI, Google Gemini, Anthropic, etc.)

### Environment Setup

```bash
# Install Pyenv (if not already installed)
curl https://pyenv.run | bash

# Add Pyenv to your shell
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

# Install Python 3.11
pyenv install 3.11

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Create and set up project
mkdir langchain-project
cd langchain-project
pyenv local 3.11
poetry init
poetry config virtualenvs.in-project true
poetry env use python

# Install dependencies
poetry add langchain langchain-core langchain-community
poetry add langchain-openai langchain-google-genai langchain-anthropic
poetry add python-dotenv
poetry add ipykernel jupyter notebook
```

### API Keys

Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY="your-openai-api-key"
GEMINI_API_KEY="your-gemini-api-key"
ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## 📚 Code Examples

The repository includes examples for various LangChain components:

### 1. Direct LLM Provider Usage

```python
# Example with OpenAI
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
print(llm.invoke("What is the capital of France?"))

# Example with Anthropic
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229")
print(llm.invoke("What is the capital of Germany?"))
```

### 2. Prompt Templates

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Simple template
template = PromptTemplate.from_template("Tell me about {topic}")
formatted = template.format(topic="LangChain")

# Chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("user", "Tell me about {topic}")
])
messages = chat_template.format_messages(topic="vector databases")
```

### 3. Document Loaders

```python
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("./data/document.pdf")
documents = loader.load()
```

### 4. Text Splitters

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)
```

### 5. Vector Databases

Multiple examples of vector database integration including FAISS, Pinecone, Weaviate, Milvus, and Chroma.

```python
# FAISS example
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = FAISS.from_documents(chunks, embeddings)
results = vector_store.similarity_search("What is LangChain?", k=4)
```

## 🚀 Complete RAG Implementation

The repository includes a fully functional implementation of a Retrieval Augmented Generation system:

```python
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# See full implementation in rag_system.py
```

## 📊 Diagrams

The repository includes sequence diagrams (in Mermaid format) that visualize the RAG system's component interactions.

## 🌐 Vector Database Connection Instructions

Detailed instructions for connecting to different vector databases:

- FAISS (local)
- Pinecone
- Weaviate
- Milvus/Zilliz
- Chroma

## 📋 Requirements

See `pyproject.toml` for the full list of dependencies.


## 🔗 Related Resources

- [Official LangChain Documentation](https://python.langchain.com/docs/)
- [LangSmith Platform](https://smith.langchain.com/)
- [Vector Database Documentation Links]
