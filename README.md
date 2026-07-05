# LLM Research Assistant

An end-to-end Retrieval-Augmented Generation (RAG) system that answers questions over academic research papers using semantic search, vector retrieval, reranking and a local Large Language Model.

---

## Overview

Large Language Models are powerful at generating natural language, but they cannot reliably answer questions about documents they have never seen. This project addresses that problem by implementing a complete Retrieval-Augmented Generation (RAG) pipeline for academic papers.

The system allows a user to upload research papers, automatically indexes their contents, retrieves the most relevant sections for a question, and generates grounded answers using only the retrieved evidence. Every response includes citations back to the original paper pages to improve transparency and reduce hallucinations.

The project was built to explore modern LLM engineering techniques rather than simply calling an API. Much of the work focuses on improving retrieval quality before the prompt reaches the language model.

---

## Example

### Question

> How does the paper explain the relationship between military expenditure and GDP growth?

↓

### Retrieval

The system:

- identifies the most relevant document
- retrieves the highest-scoring text chunks
- reranks retrieved passages
- extracts atomic evidence

↓

### Generated Answer

Military expenditure was associated with short-term economic growth only under specific conditions. The authors argue that increased defence spending stimulated industrial production during periods of conflict, but this effect diminished once military expenditure exceeded productive capacity.

### Supporting Evidence

- Military expenditure can temporarily increase aggregate demand. *(Page 18)*
- Long-term GDP growth depends more heavily on productivity improvements. *(Page 21)*
- Excessive defence expenditure may crowd out productive investment. *(Page 23)*

---

## Features

- Upload and index PDF research papers
- Automatic document parsing
- Intelligent text chunking
- Semantic embedding generation
- Vector database indexing
- Adaptive document retrieval
- Query planning
- Passage reranking
- Local LLM integration using Ollama
- Citation-aware answer generation
- Streamlit web interface
- Automated unit tests

---

# System Architecture

```text
                  PDF Research Papers
                           │
                           ▼
                  PDF Loading Pipeline
                           │
                           ▼
                    Text Extraction
                           │
                           ▼
                     Intelligent Chunking
                           │
                           ▼
                  Embedding Generation
                           │
                           ▼
                    Vector Index Store
                           │
                           ▼
                  Adaptive Retrieval
                           │
                           ▼
                     Passage Reranking
                           │
                           ▼
                   Prompt Construction
                           │
                           ▼
                    Local LLM (Ollama)
                           │
                           ▼
              Answer + Evidence + Citations
```

---

## Pipeline

The application follows a multi-stage Retrieval-Augmented Generation workflow.

### 1. PDF Processing

Research papers are loaded and converted into plain text while preserving page information.

The loader stores document metadata so retrieved answers can later reference the original pages.

---

### 2. Chunking

Instead of embedding entire papers, documents are divided into smaller overlapping chunks.

Chunking improves retrieval accuracy by allowing the vector database to retrieve highly specific sections rather than entire documents.

---

### 3. Embedding Generation

Each chunk is converted into a dense semantic vector.

These embeddings capture meaning rather than exact keywords, allowing semantically similar passages to be retrieved even when different wording is used.

---

### 4. Vector Storage

Embeddings are stored inside a vector index.

At query time the system performs similarity search to retrieve the most relevant document chunks.

---

### 5. Adaptive Retrieval

Rather than returning a fixed number of passages every time, the retrieval stage dynamically selects evidence depending on the question and retrieval confidence.

This reduces unnecessary context while improving answer quality.

---

### 6. Query Planning

Incoming questions are analysed before retrieval.

The system attempts to identify the underlying information need and retrieve evidence that directly answers the user's question instead of relying purely on embedding similarity.

---

### 7. Passage Reranking

Retrieved chunks are scored again before being sent to the language model.

This additional ranking stage helps ensure that the highest-quality evidence is used when constructing the final prompt.

---

### 8. Prompt Construction

Instead of passing raw retrieved text directly to the LLM, the application builds a structured prompt containing:

- retrieved evidence
- atomic facts
- citation information
- answer formatting instructions
- hallucination constraints

This produces more consistent and more trustworthy answers.

---

### 9. Answer Generation

The final prompt is sent to a locally running language model through Ollama.

The model generates:

- a concise answer
- supporting evidence
- page citations
- limitations when appropriate

---

# Project Structure

```text
LLM-Research-Assistant/

├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── indexes/
│   ├── metadata/
│   └── papers/
│
├── src/
│   ├── adaptive_retrieval.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── index_builder.py
│   ├── ollama_client.py
│   ├── pdf_loader.py
│   ├── qa_engine.py
│   ├── query_planner.py
│   ├── reranker.py
│   ├── retriever.py
│   └── vector_store.py
│
├── tests/
│   ├── test_chunker.py
│   └── test_retriever.py
│
├── requirements.txt
├── Makefile
└── README.md
```

---

# Technologies

| Category | Technology |
|-----------|------------|
| Language | Python |
| Interface | Streamlit |
| LLM | Ollama |
| Retrieval | Vector Search |
| NLP | Sentence Embeddings |
| Document Processing | PDF Parsing |
| Testing | Pytest |

---



# Installation

These commands should be run in your terminal, not inside a Python file.

## 1. Clone the repository

```bash
git clone https://github.com/Brxckfuller/LLM-Research-Assistant.git
cd LLM-Research-Assistant
```

## 2. Create and activate a virtual environment

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Install and start Ollama

This project uses Ollama to run the local language model.

Install Ollama from:

```text
https://ollama.com
```

Start Ollama:

```bash
ollama serve
```

In a separate terminal window, pull the model used by the project:

```bash
ollama pull llama3
```

## 5. Add research papers

Place PDF research papers inside:

```text
data/papers/
```

If the folder does not exist, create it:

```bash
mkdir -p data/papers
```

## 6. Build the document index

After adding PDFs to `data/papers/`, build the index:

```bash
python src/index_builder.py
```

This processes the PDFs, extracts text, chunks the documents, creates embeddings and stores the searchable index.

## 7. Run the application

```bash
streamlit run app/streamlit_app.py
```

The Streamlit app should open in your browser. You can then ask questions about the indexed papers.

---

# Using the Application

## 1. Add PDF papers

Place one or more academic PDF papers inside:

```text
data/papers/
```
You can either place PDFs in data/papers/ and build the index manually, or upload a PDF through the Streamlit interface.

## 2. Build or rebuild the index

Run this command from the project root:

```bash
python src/index_builder.py
```

You should rebuild the index whenever you add, remove or change papers in `data/papers/`.

## 3. Start Ollama

Make sure Ollama is running before asking questions:

```bash
ollama serve
```

If you have not already downloaded the model, run:

```bash
ollama pull llama3
```

## 4. Launch the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## 5. Ask a question

Use the web interface to ask questions in natural language, such as:

```text
What is the main argument of this paper?
```

```text
What evidence does the author provide?
```

```text
What are the limitations of the study?
```

## 6. Review the answer and citations

The system retrieves relevant passages from the indexed papers and generates an answer grounded in those passages.

Where available, answers include page citations so the user can check the response against the original source document.

---
# Technical Design

## Retrieval Strategy

The application uses retrieval before generation rather than asking the language model to answer from memory.

This is important because academic papers often contain dense, specific claims that must be answered from the source text rather than from general knowledge.

The retrieval pipeline is designed to:

- find the most relevant passages
- reduce irrelevant context
- preserve source page numbers
- keep the final answer grounded in retrieved evidence

---

## Query Planning

Before retrieval, the question is analysed to improve the search process.

For example, a broad question such as:

```text
What is the author's central argument?
```

requires a different retrieval strategy from a narrow factual question such as:

```text
How many participants were included in the study?
```

The query planning step helps decide how broadly the system should search and which passages are likely to be useful.

---

## Evidence Extraction

The system uses an intermediate evidence extraction step before generating the final answer.

Instead of immediately asking the LLM to answer the question, the model first extracts atomic facts from the retrieved passages.

This helps reduce hallucination because the final response is based on a smaller set of directly supported claims.

---

## Citation-Aware Answers

Each retrieved passage keeps its page number.

The final answer is instructed to cite claims using the page where the supporting evidence was found.

Example format:

```text
- The author argues that physicalism requires all concrete phenomena to be physical phenomena. (Page 3)
```

This makes the system easier to audit because users can check the answer against the original paper.

---

## Local LLM Support

The project uses Ollama so the system can run with a local language model.

This avoids relying entirely on hosted APIs and makes the project easier to experiment with locally.

---

# Example Questions

The assistant can answer questions such as:

```text
What is the main argument of this paper?
```

```text
What evidence does the author give for their position?
```

```text
What are the limitations of the study?
```

```text
How does this paper define physicalism?
```

```text
What does the paper say about North Korean support for Russia?
```


---

# Example Output Format

```markdown
# Answer

The paper argues that physicalism should be understood as the view that every real, concrete phenomenon is physical, while also rejecting overly simplistic versions of materialism.

# Key points

- The author defines physicalism as the view that every real, concrete phenomenon is physical. (Page 1)
- The paper distinguishes physicalism from older forms of materialism. (Page 2)
- The author argues that mental phenomena must be explained without treating them as non-physical substances. (Page 6)

# Limitations

The retrieved evidence does not fully establish how the author responds to every objection discussed later in the paper.
```

---

# Testing

The project includes automated tests for core retrieval and document processing components.

Run the tests with:

```bash
pytest
```

The test suite currently covers:

- document chunking
- retrieval behaviour
- basic pipeline reliability

---

# Design Decisions

## Why not just send the whole PDF to the model?

Large academic papers often exceed context limits, and sending the entire document can introduce irrelevant information.

Retrieval allows the system to send only the most relevant passages to the model.

---

## Why use chunking?

Chunking makes retrieval more precise.

If a whole paper is embedded as one document, a search result may be too broad. Smaller chunks allow the system to retrieve the exact section that answers the question.

---

## Why use citations?

Citations make the output easier to verify.

For academic documents, it is not enough for an answer to sound plausible. The system should show where the answer came from.

---

## Why use a local model?

Using Ollama makes the project easier to run privately and gives more control over experimentation.

It also demonstrates that the system architecture is not tied to one hosted LLM provider.

---

# Current Limitations

This project is still a prototype.

Current limitations include:

- PDF extraction quality depends on the structure of the uploaded document
- scanned PDFs may require OCR before they can be processed
- citation accuracy depends on reliable page extraction
- answer quality depends heavily on retrieval quality
- local models may be slower than hosted models
- very long or complex papers may require more careful chunking

These limitations are typical of RAG systems and were useful in understanding the trade-offs involved in building document-grounded LLM applications.

---


# Repository Status

This project is actively being improved.

The current version focuses on building a working end-to-end RAG pipeline with local LLM generation, citation-aware responses and a usable Streamlit interface.

---

# About

Built by Brock Fuller as part of a machine learning and AI engineering portfolio.

The project was designed to demonstrate practical skills in:

- LLM application development
- Retrieval-Augmented Generation
- semantic search
- document processing
- prompt engineering
- Streamlit application design
- Python project organisation
- testing and debugging
