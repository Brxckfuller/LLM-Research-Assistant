# 📄 LLM Research Paper Assistant

> A local-first Retrieval-Augmented Generation (RAG) system that answers questions about research papers using adaptive retrieval, dense vector search, passage reranking, evidence extraction, and optional live web search.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-black)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

Large Language Models cannot reliably answer questions about documents they have never seen. This project implements a complete Retrieval-Augmented Generation (RAG) pipeline that grounds every answer in evidence retrieved directly from uploaded research papers.

Unlike basic RAG demonstrations that simply retrieve the nearest text chunks, this system introduces several engineering improvements:

- Adaptive retrieval routing
- Dense vector search
- Cross-encoder passage reranking
- Atomic evidence extraction
- Citation-aware prompting
- Optional live web search
- Fully local answer generation using Ollama

The result is a research assistant capable of answering detailed questions while remaining grounded in source material and providing supporting evidence for its responses.

---

## Key Engineering Highlights

- **Local-first RAG pipeline** using Ollama for private, offline-capable answer generation.
- **Hybrid retrieval** combining dense semantic search with BM25 keyword search for stronger retrieval coverage.
- **FAISS vector indexing** for fast local similarity search over uploaded academic papers.
- **Cross-encoder reranking** to improve passage relevance before context is sent to the language model.
- **Adaptive routing** between uploaded document retrieval and live web search using Tavily.
- **Two-stage prompting** with evidence extraction before final answer generation to reduce hallucination.
- **Citation-aware responses** that preserve page numbers and source references.
- **Streamlit interface** with upload, indexing, retrieval diagnostics, source inspection, and progress visualisation.
- **Modular Python architecture** designed so retrieval, reranking, prompting, and web search can be improved independently.

---


# Demo

The following screenshots show the complete workflow of the application.

---

## Upload Research Paper


![Upload Research Paper](docs/images/upload-paper.png)

---

## Ask Questions


![Question Interface](docs/images/query-interface.png)

---

## Retrieval Progress


![Retrieval Progress](docs/images/retrieval-progress1.png)

---

## Generated Answer


![Generated Answer](docs/images/generated-answer.png)

---

### Generated Answer with Citations

![Citation](docs/images/citation.png)

---

# Features

## 📄 PDF Processing

Research papers are parsed using **PyMuPDF**, preserving page ordering while extracting clean textual content for indexing.

---

## ✂ Intelligent Chunking

Rather than embedding an entire paper as one document, papers are divided into overlapping chunks suitable for semantic retrieval.

The chunking strategy uses:

- Recursive text splitting
- Configurable chunk size
- Configurable overlap
- Metadata preservation
- Page number tracking

This improves retrieval recall while ensuring enough surrounding context is retained for accurate answer generation.

---

## 🧠 Dense Semantic Retrieval

Every chunk is embedded using a SentenceTransformer embedding model and indexed in **FAISS**.

During retrieval:

1. The user question is embedded.
2. FAISS performs approximate nearest-neighbour search.
3. The most semantically similar chunks are returned.

Unlike keyword search, semantic retrieval can locate relevant passages even when different terminology is used.

---

## 🎯 Cross-Encoder Reranking

Nearest-neighbour retrieval is fast but imperfect.

To improve relevance, retrieved passages are reranked using a cross-encoder model that jointly evaluates the question and each candidate passage.

Pipeline:

```
User Question
      │
      ▼
 FAISS Retrieval
      │
Top K Chunks
      │
      ▼
Cross Encoder
      │
      ▼
Ranked Passages
```

Only the highest scoring passages are forwarded to the language model.

This significantly reduces irrelevant context and improves answer quality.

---

## 🔎 Adaptive Retrieval

The system automatically determines how a question should be answered.

Questions are first classified into one of two categories:

- Document-specific
- General knowledge

Document questions are answered using the uploaded paper.

General knowledge questions are automatically routed to live web search.

This prevents irrelevant document retrieval while allowing the assistant to answer broader research questions.

Example:

```
"What methodology does this paper use?"
```

↓

Uses document retrieval

---

```
"What is Retrieval-Augmented Generation?"
```

↓

Uses web search

---

## 🌍 Live Web Search

When a question cannot be answered from the uploaded paper, the system automatically switches to live web search.

The web pipeline:

```
Question
      │
      ▼
Query Planning
      │
      ▼
Tavily Search API
      │
      ▼
Result Processing
      │
      ▼
Evidence Extraction
      │
      ▼
Local LLM
```

This enables the assistant to answer questions beyond the scope of uploaded documents while still grounding responses in retrieved evidence.

---

## 📑 Evidence Extraction

Rather than passing every retrieved passage directly to the LLM, the system extracts the most relevant evidence before prompt construction.

Benefits include:

- Reduced prompt size
- Improved factual grounding
- Less irrelevant context
- Lower hallucination risk

Only the highest-value evidence is supplied to the language model.

---

## 🤖 Local LLM Generation

All answers are generated locally using **Ollama**.

Advantages include:

- No cloud inference costs
- Offline capability
- Data privacy
- Low latency after model loading

Because retrieval occurs before generation, the LLM answers using retrieved evidence rather than relying solely on its internal knowledge.

---

# Architecture

The complete retrieval pipeline is shown below.

```text
                    User Question
                          │
                          ▼
               Adaptive Route Selection
                    ┌─────────────┐
          Document? │             │ Web?
                    ▼             ▼
          PDF Retrieval      Tavily Search
                    │             │
                    ▼             ▼
            Dense Retrieval   Search Results
                    │             │
                    ▼             ▼
          Cross Encoder      Evidence Extraction
              Reranking             │
                    └──────┬────────┘
                           ▼
                  Prompt Construction
                           │
                           ▼
                     Ollama (Local)
                           │
                           ▼
                  Citation-aware Answer
```

---

# Technology Stack

| Component | Technology |
|-----------|------------|
| Interface | Streamlit |
| PDF Parsing | PyMuPDF |
| Embeddings | SentenceTransformers |
| Vector Database | FAISS |
| Reranker | Cross Encoder |
| Local LLM | Ollama |
| Web Search | Tavily API |
| Programming Language | Python 3.12 |

---

# Why this project?

Many RAG demonstrations stop after embedding documents into a vector database.

This project focuses on the engineering components that make modern retrieval systems more reliable:

- adaptive routing
- passage reranking
- evidence extraction
- citation-aware prompting
- retrieval visualisation
- local inference
- optional live web augmentation

These components more closely resemble the architecture used in production retrieval systems than a minimal vector-search example.

# System Pipeline

The assistant follows a multi-stage Retrieval-Augmented Generation (RAG) pipeline that separates document retrieval from answer generation.

```
                   User Question
                         │
                         ▼
              Adaptive Route Selection
                ┌───────────────┐
        Document Question?      Web Question?
                │                     │
                ▼                     ▼
         PDF Retrieval          Tavily Search
                │                     │
                ▼                     ▼
        Semantic Retrieval     Search Results
                │                     │
                ▼                     ▼
      Cross Encoder Reranking  Evidence Extraction
                │                     │
                └──────────┬──────────┘
                           ▼
                 Prompt Construction
                           ▼
                  Ollama Local LLM
                           ▼
                   Citation Generation
                           ▼
                    Final Response
```

---

# Project Structure

```
LLM-Research-Assistant/

├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── papers/
│   ├── faiss/
│   └── metadata/
│
├── src/
│   ├── adaptive_retrieval.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── evidence_extraction.py
│   ├── index_builder.py
│   ├── parser.py
│   ├── prompt_builder.py
│   ├── qa_engine.py
│   ├── reranker.py
│   ├── retriever.py
│   └── web_search.py
│
├── tests/
│
├── requirements.txt
└── README.md
```

The project is intentionally modular so that each retrieval stage can be improved independently without affecting the remainder of the pipeline.

---

# 1. PDF Ingestion

When a research paper is uploaded, it first passes through the ingestion pipeline.

```
PDF
 │
 ▼
PyMuPDF
 │
 ▼
Raw Text
 │
 ▼
Chunking
 │
 ▼
Embeddings
 │
 ▼
FAISS Index
```

PyMuPDF was selected because it provides:

- fast extraction
- page-aware parsing
- reliable Unicode handling
- minimal external dependencies

Each paper is indexed only once.

Subsequent questions reuse the existing vector index without rebuilding embeddings.

---

# 2. Intelligent Chunking

Research papers are too large to embed as single documents.

Instead, the document is divided into overlapping chunks.

The implementation uses a recursive text splitter that attempts to preserve semantic structure while respecting the model's context window.

Each chunk stores:

- document name
- page number
- chunk text
- chunk ID

The overlap between chunks helps preserve context across section boundaries while reducing the chance that important information is split apart.

Example

```
Chunk 18

"...Transformer models achieve state-of-the-art
performance on multiple benchmarks..."

---------------- overlap ----------------

Chunk 19

"...performance on multiple benchmarks.
The authors evaluate..."
```

This significantly improves retrieval continuity.

---

# 3. Embedding Generation

Each chunk is converted into a dense vector representation using a SentenceTransformer embedding model.

```
Chunk
 │
 ▼
SentenceTransformer
 │
 ▼
768-dimensional embedding
```

Dense embeddings allow semantically similar passages to be retrieved even when they contain different wording.

For example,

Question

```
How is hallucination reduced?
```

can successfully retrieve a paragraph discussing

```
grounded factual generation
```

despite neither phrase containing the exact keywords.

---

# 4. Vector Search

All embeddings are stored inside a FAISS index.

During question answering:

```
Question

↓

Embedding Model

↓

Question Embedding

↓

FAISS Similarity Search

↓

Top-k Candidate Passages
```

The number of retrieved passages can be configured directly from the Streamlit interface.

Increasing the retrieval depth generally improves recall but also increases prompt size.

---

# 5. Adaptive Routing

Before retrieval begins, the question is classified to determine its source.

```
User Question
      │
      ▼
Route Classifier
      │
 ┌────┴────┐
 │         │
 ▼         ▼
Document   Web
```

Document-specific questions are answered using the uploaded paper.

Examples

```
What dataset did the authors use?

Explain Figure 5.

What loss function is proposed?
```

General knowledge questions automatically trigger live web search.

Examples

```
Explain Retrieval-Augmented Generation.

Who invented transformers?

What are vector databases?
```

This prevents unnecessary document retrieval for questions that are unrelated to the uploaded paper.

---

# 6. Passage Reranking

Nearest-neighbour retrieval is efficient but not always accurate.

The top retrieved passages are therefore reranked using a cross-encoder.

Pipeline

```
Top 20 Retrieved Chunks

↓

Cross Encoder

↓

Relevance Scores

↓

Sorted Passages

↓

Top Evidence
```

Unlike embedding similarity, a cross-encoder jointly evaluates both the question and passage, producing substantially more accurate rankings.

This stage improves retrieval precision by removing semantically similar but irrelevant chunks before prompt construction.

---

# 7. Evidence Extraction

Instead of sending every retrieved passage to the language model, the system extracts only the highest-value evidence.

```
Retrieved Chunks

↓

Evidence Selection

↓

Relevant Facts

↓

Prompt Builder
```

This reduces:

- prompt length
- duplicated context
- token usage
- hallucination risk

Only evidence relevant to the user's question is forwarded to the LLM.

---

# 8. Prompt Construction

The retrieved evidence is inserted into a structured prompt before being passed to Ollama.

The prompt instructs the model to:

- answer only using supplied evidence
- avoid unsupported claims
- produce clear explanations
- reference retrieved sources where appropriate

Separating retrieval from prompt construction keeps the generation stage deterministic and easier to modify.

---

# 9. Local Answer Generation

Answers are generated using a locally hosted Ollama model.

```
Evidence

↓

Prompt

↓

Ollama

↓

Answer
```

Running the model locally provides several advantages:

- private inference
- offline operation
- zero API costs
- reproducible responses

The retrieval pipeline performs the factual grounding, while the LLM focuses on explanation and synthesis.

---

# 10. Progressive User Interface

Rather than appearing frozen while retrieval occurs, the interface displays each retrieval stage in real time.

For document retrieval, the interface shows:

- Scanning paper
- Reranking passages
- Extracting evidence
- Building answer
- Generating response

For web search, the interface displays:

- Expanding query
- Searching web
- Collecting results
- Filtering and reranking
- Extracting evidence
- Building answer
- Generating response

This provides users with visibility into the retrieval process and improves the responsiveness of the application during longer operations.

# Engineering Decisions

Modern Retrieval-Augmented Generation systems are composed of many interchangeable components. Rather than selecting technologies arbitrarily, each component in this project was chosen based on the specific requirements of document-grounded question answering.

---

# Why FAISS?

Several vector databases were considered during development, including ChromaDB, Pinecone and Weaviate.

FAISS was selected because it offers:

- Extremely fast similarity search
- Lightweight local deployment
- No external services
- Excellent Python integration
- Simple persistence of indexes

For a local research assistant where documents are indexed once and queried many times, FAISS provides an excellent balance between performance and simplicity.

Trade-offs:

Advantages

- Fast retrieval
- Small memory footprint
- Easy local deployment

Disadvantages

- Limited metadata filtering
- No distributed scaling
- Manual index management

---

# Why Sentence Transformers?

Embedding quality determines retrieval quality.

Rather than relying on keyword matching, dense embeddings allow semantically related passages to be retrieved even when they share few common words.

For example,

Question

```
How does the paper reduce hallucinations?
```

may retrieve a passage discussing

```
grounded factual generation
```

despite neither phrase containing identical vocabulary.

This semantic capability significantly improves recall compared with traditional keyword search.

---

# Why Hybrid Retrieval?

The system combines multiple retrieval strategies rather than relying solely on dense vectors.

Current retrieval pipeline:

```
Question

↓

Semantic Retrieval (FAISS)

+

BM25 Keyword Search

↓

Combined Candidate Passages

↓

Cross Encoder Reranking

↓

Final Context
```

Dense retrieval captures semantic similarity.

BM25 improves retrieval when exact terminology, names, equations or abbreviations appear.

Combining both methods increases robustness across a wider range of academic writing styles.

---

# Why Cross-Encoder Reranking?

Vector search retrieves passages independently.

A Cross Encoder evaluates:

```
(question, passage)
```

jointly.

This allows the reranker to understand whether the retrieved passage actually answers the question rather than merely discussing a related topic.

Although reranking increases latency slightly, the improvement in passage relevance was considered worthwhile for an academic assistant where answer quality is more important than absolute speed.

---

# Why Evidence Extraction?

Many RAG implementations simply concatenate retrieved passages into a single prompt.

This project instead performs an intermediate evidence extraction stage.

Pipeline

```
Retrieved Passages

↓

Evidence Extraction

↓

Prompt Builder

↓

Local LLM
```

Advantages

- Smaller prompts
- Less duplicated information
- Better factual grounding
- Reduced hallucination
- Easier citation generation

This separation also makes the pipeline easier to extend with future retrieval methods.

---

# Why Local Inference?

Instead of relying on hosted APIs, answer generation is performed locally using Ollama.

Advantages

- Privacy
- Offline capability
- No inference costs
- Easy experimentation with different models
- No dependence on third-party APIs

Trade-offs

Advantages

- User data remains local
- Predictable running costs
- Model selection is flexible

Disadvantages

- Slower than commercial hosted models
- Higher hardware requirements
- Model quality depends on local hardware

For a research assistant designed to analyse private documents, these trade-offs were considered acceptable.

---

# Why Adaptive Routing?

The project supports two independent retrieval pipelines.

Document Retrieval

```
Question

↓

Uploaded Paper

↓

Answer
```

Web Retrieval

```
Question

↓

Tavily Search

↓

Answer
```

A lightweight routing stage determines which retrieval strategy should be used.

This prevents unnecessary searches of uploaded documents while still allowing the assistant to answer broader questions about machine learning, AI, or recent research.

Future versions may replace the current rule-based router with an LLM-powered routing model capable of reasoning about ambiguous questions.

---

# Evaluation

The retrieval pipeline has been designed to support quantitative evaluation rather than relying solely on qualitative examples.

Planned evaluation metrics include:

| Metric | Purpose |
|---------|----------|
| Recall@k | Percentage of questions where the relevant passage appears in the retrieved set |
| Mean Reciprocal Rank (MRR) | Measures how highly relevant passages are ranked |
| Retrieval Latency | Time required to retrieve candidate passages |
| Reranking Latency | Time spent by the Cross Encoder |
| Total Response Time | End-to-end question answering latency |
| Index Build Time | Time required to process uploaded PDFs |
| Memory Usage | Storage required for embeddings and indexes |

These benchmarks will allow retrieval improvements to be measured objectively rather than relying on anecdotal examples.

---

# Planned Ablation Study

One criticism of many Retrieval-Augmented Generation projects is that additional components are introduced without demonstrating their contribution.

To evaluate the effectiveness of each stage, future work will compare:

| Configuration | Expected Purpose |
|--------------|------------------|
| Baseline Vector Search | Reference implementation |
| + Hybrid Retrieval | Improve recall |
| + Cross Encoder | Improve ranking precision |
| + Query Planning | Improve retrieval intent |
| + Evidence Extraction | Reduce hallucination |
| + Adaptive Routing | Improve retrieval selection |

The goal is to quantify the contribution of each component rather than assuming additional complexity automatically improves performance.

---

# Testing

Automated tests currently cover core components including:

- document chunking
- retrieval behaviour
- reranking pipeline
- prompt generation
- indexing
- parser reliability

Future work will expand testing to include:

- retrieval regression tests
- citation verification
- prompt consistency
- latency benchmarking
- end-to-end integration tests

This will make future changes easier to validate while reducing the risk of retrieval regressions.

---

# Example Workflow

```
Upload PDF

↓

Parse document

↓

Chunk text

↓

Generate embeddings

↓

Build FAISS index

↓

Ask question

↓

Adaptive routing

↓

Retrieve candidates

↓

Cross Encoder reranking

↓

Evidence extraction

↓

Prompt construction

↓

Ollama

↓

Grounded answer with citations
```
# Installation

## Prerequisites

Before running the project, install the following software:

- Python 3.11+
- Ollama
- Git

---

## 1. Clone the repository

```bash
git clone https://github.com/Brxckfuller/LLM-Research-Assistant.git
cd LLM-Research-Assistant
```

---

## 2. Create a virtual environment

macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install Ollama

Download Ollama from

https://ollama.com

Start the Ollama server

```bash
ollama serve
```

Download the model used by the project

```bash
ollama pull llama3
```

You may substitute another supported Ollama model if desired.

---

## 5. Configure Web Search (Optional)

Live web retrieval uses the Tavily Search API.

Create a `.env` file in the project root.

```text
TAVILY_API_KEY=your_api_key_here
```

Create a free API key at

https://tavily.com

If no API key is provided, the application will continue to function using document retrieval only.

---

## 6. Launch the application

```bash
streamlit run app/streamlit_app.py
```

The Streamlit interface will automatically open in your browser.

---

# Using the Application

The assistant supports two complementary retrieval modes.

---

## Document Mode

Upload a research paper using the sidebar.

The application automatically:

- parses the PDF
- extracts text
- generates embeddings
- builds the FAISS index
- stores metadata

No manual indexing commands are required.

Example questions

```
What is the main argument of this paper?

How was the experiment conducted?

What evidence supports the author's conclusion?

Summarise the discussion section.

What limitations does the paper identify?
```

---

## Web Mode

Questions requiring general knowledge automatically trigger live web search.

Examples

```
What is Retrieval-Augmented Generation?

Who introduced the Transformer architecture?

What is the latest research on GraphRAG?

Explain Agentic AI.

What are vector databases?
```

The assistant automatically determines whether the uploaded paper or the web is the most appropriate information source.

---

## Adaptive Retrieval

Some questions require both sources.

Example

```
Compare this paper's retrieval pipeline with modern GraphRAG systems.
```

In these cases the retrieval planner combines:

- uploaded document evidence
- live web search

before generating a final response.

---

# Example Workflow

```
Upload Research Paper

↓

Automatic Index Construction

↓

Ask Question

↓

Adaptive Route Selection

↓

Retrieve Evidence

↓

Cross Encoder Reranking

↓

Evidence Extraction

↓

Prompt Construction

↓

Local LLM

↓

Grounded Answer
```

---

# Streamlit Interface

The interface has been designed to make retrieval behaviour transparent rather than treating the language model as a black box.

Current interface components include

- PDF upload
- automatic indexing
- adaptive retrieval
- progress visualisation
- citation-aware answers
- retrieval diagnostics
- previous question history
- downloadable responses

---

## Document Retrieval

When analysing uploaded papers, the interface displays the current retrieval stage.

```
Scanning paper

↓

Reranking passages

↓

Extracting evidence

↓

Building answer

↓

Generating response
```

This allows users to understand where computation time is being spent.

---

## Web Retrieval

When web search is selected, a dedicated progress interface displays:

```
Expanding query

↓

Searching the web

↓

Collecting results

↓

Filtering and reranking

↓

Extracting evidence

↓

Generating response
```

This provides feedback during external retrieval while making it clear that the assistant is consulting external information rather than relying solely on the uploaded document.

---

# Screenshots

*(Replace placeholders with actual screenshots.)*

---

## Uploading a Paper

```text
images/upload-paper.png
```

---

## Document Retrieval

```text
images/document-loader.png
```

---

## Web Search

```text
images/web-loader.png
```

---

## Generated Answer

```text
images/generated-answer.png
```

---

## Retrieved Evidence

```text
images/retrieved-passages.png
```

---

## Retrieval Details

```text
images/retrieval-details.png
```

---

# Performance

Performance depends on

- document length
- embedding model
- reranking depth
- local hardware
- Ollama model
- web search latency

Typical workflow

```
PDF Upload

↓

Index Creation

↓

Question

↓

Retrieval

↓

Reranking

↓

LLM Generation

↓

Answer
```

Future releases will include quantitative benchmarks for

- Recall@k
- Mean Reciprocal Rank
- Retrieval latency
- Indexing time
- Memory usage
- Total response latency

These metrics will allow retrieval improvements to be measured objectively across future versions.

---

# Reproducibility

The project has been designed so that another developer can reproduce the complete pipeline using only

- Python
- Ollama
- the supplied requirements.txt
- a Tavily API key (optional)

No proprietary software or hosted inference services are required.

All document retrieval, indexing and answer generation can be performed locally.

# Current Limitations

Although the system performs well for research paper question answering, it remains a prototype intended for experimentation rather than production deployment.

Current limitations include:

### OCR Support

The system assumes uploaded PDFs contain machine-readable text.

Scanned documents and image-based PDFs cannot currently be processed without an OCR pipeline.

Potential improvement:

- Tesseract OCR
- PaddleOCR
- Nougat (academic PDF OCR)

---

### Metadata Filtering

Retrieval currently relies primarily on semantic similarity.

Future versions could allow retrieval constrained by metadata such as

- author
- publication year
- conference
- journal
- section
- document type

This would improve retrieval precision for larger document collections.

---

### Multi-document Reasoning

Although multiple papers can be indexed, the assistant primarily retrieves evidence independently rather than reasoning jointly across multiple documents.

Future work could include

- cross-paper synthesis
- contradiction detection
- literature review generation
- automatic consensus extraction

---

### Benchmarking

The current implementation has not yet undergone large-scale quantitative evaluation.

Future benchmarking will measure

- Recall@k
- Mean Reciprocal Rank
- Retrieval Precision
- Citation Accuracy
- Response Latency
- Memory Usage

to better understand the contribution of each retrieval component.

---

### LLM Routing

The adaptive router currently uses rule-based routing between uploaded documents and live web search.

A more sophisticated implementation could use a lightweight LLM classifier capable of reasoning about ambiguous questions rather than relying on keyword detection.

---

### Web Retrieval

Web search currently retrieves relevant pages before generating an answer.

Future improvements could include

- automatic source credibility estimation
- duplicate source removal
- query rewriting using an LLM
- multi-stage web retrieval
- retrieval across academic APIs such as Semantic Scholar

---

# Future Work

Several extensions are planned.

## GraphRAG

Replace flat chunk retrieval with graph-based retrieval capable of reasoning over relationships between concepts.

Potential technologies

- Neo4j
- NetworkX
- GraphRAG

---

## Agentic Retrieval

Instead of retrieving once, future versions may allow an LLM agent to perform iterative retrieval.

Example

Question

↓

Retrieve evidence

↓

Recognise missing information

↓

Search again

↓

Generate improved answer

This approach more closely resembles modern Deep Research systems.

---

## Automatic Evaluation

Develop a benchmark suite capable of automatically measuring retrieval quality against annotated question-answer datasets.

This would enable regression testing as retrieval components evolve.

---

## Multimodal Documents

Support figures, tables and equations rather than relying solely on extracted text.

Possible approaches

- OCR
- Vision-Language Models
- Table extraction
- Figure caption grounding

---

## Citation Verification

Automatically verify that every generated claim is supported by retrieved evidence.

Potential additions include

- citation confidence
- unsupported claim detection
- evidence highlighting

---

## Model Flexibility

Support multiple local and hosted models.

Examples

- Llama 3
- Mistral
- Gemma
- Qwen
- OpenAI
- Anthropic

without requiring changes to the retrieval pipeline.

---

# Lessons Learned

This project reinforced several observations about modern Retrieval-Augmented Generation systems.

The largest gains in answer quality often came from improving retrieval rather than changing the language model itself.

Adding

- reranking
- evidence extraction
- adaptive retrieval
- query planning

generally produced larger improvements than replacing the underlying LLM.

The project also demonstrated the importance of treating retrieval and generation as separate engineering problems.

A stronger language model cannot compensate for poor retrieval, while high-quality retrieval significantly improves grounded responses even when using relatively small local models.

Finally, building the system highlighted that modern LLM applications involve considerably more than prompt engineering.

Document parsing, indexing, retrieval, ranking, evidence selection and user experience all contribute meaningfully to overall system performance.

---

# Repository Roadmap

Future releases will focus on the following areas.

- [ ] Retrieval benchmarking
- [ ] GraphRAG
- [ ] OCR support
- [ ] Agentic retrieval
- [ ] Citation verification
- [ ] Integration testing
- [ ] Multimodal document support
- [ ] Automatic evaluation suite
- [ ] Academic search connectors
- [ ] Retrieval visualisation dashboard

---

# Contributing

Suggestions, bug reports and pull requests are welcome.

If you discover a bug or have an idea for improving retrieval quality, please open an issue describing

- the problem
- reproduction steps
- expected behaviour
- proposed improvement

---

# License

This project is released under the MIT License.

---

# About

This project was developed as part of my AI engineering portfolio.

Its primary purpose is to explore the engineering challenges involved in building modern Retrieval-Augmented Generation systems rather than simply consuming hosted LLM APIs.

Particular emphasis was placed on

- semantic retrieval
- adaptive routing
- local language models
- evidence-grounded prompting
- retrieval quality
- software architecture
- reproducible experimentation

As the project evolves, new retrieval methods, evaluation pipelines and reasoning strategies will continue to be added.

---

# Contact

**Brock Fuller**

GitHub

https://github.com/Brxckfuller

LinkedIn

www.linkedin.com/in/brock-fuller-8497593a0 

---

## Acknowledgements

This project builds upon the work of the open-source machine learning community.

Particular thanks to the developers of

- Streamlit
- Ollama
- FAISS
- Sentence Transformers
- PyMuPDF
- Hugging Face
- Tavily

whose tools made this project possible.
