# 🚀 SEC 10-Q Analyst

> A modular, agentic Python service that fetches the latest 10-Q for a given ticker from SEC EDGAR, parses and chunks it, stores chunks in a vector store, and uses Pydantic-AI agents to produce a structured equity-research style report and **Buy / Hold / Sell** recommendation.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/downloads/)
[![OpenAPI Spec](https://img.shields.io/badge/API-Swagger_UI-orange)](http://localhost:8000/docs)

---

## ⚠️ Disclaimer

This project is for informational/educational use only. **It does not provide investment advice.** Always do your own research and consult a qualified professional.

---

## ✨ What it does (TL;DR)

Given a ticker like **AAPL**, the system executes a sophisticated retrieval-augmented generation (RAG) pipeline to analyze the latest 10-Q filing.

### The Pipeline

1.  **Resolve** ticker → CIK using SEC `company_tickers.json`.
2.  **Fetch** company submissions from `data.sec.gov` and **locate** the latest 10-Q (or a specified period).
3.  **Download** the filing HTML from SEC Archives.
4.  **Parse** into sections and **chunk** into smaller text blocks.
5.  **Embed** chunks and store them in a **vector store** (e.g., PostgreSQL/pgvector).
6.  **Retrieve** the most relevant chunks based on the analyst prompt.
7.  **Run an elite equity research analyst prompt** via GPT-5, structured using Pydantic.
8.  **Output a structured report** covering:
    * Fundamental analysis
    * Thesis validation
    * Sector & macro view
    * Catalyst watch
    * Investment summary (Buy / Hold / Sell)

---

## 🏗️ How it works (High Level)

The system is built on two main components: the **Ingestion Pipeline** and the **Agentic Analysis** layer, gated by a **Caching Mechanism**. 

### 1) Ingestion Pipeline

| Component | Function |
| :--- | :--- |
| **CIK Resolution** | `CikResolver` downloads and caches SEC `company_tickers.json`. |
| **Submissions Fetch** | `SubmissionsService` pulls `CIK{cik}.json` and selects the latest 10-Q. |
| **Download** | `FilingDownloader` builds SEC Archives URLs and stores HTML locally. |
| **Parse + Chunk** | `TenQParser` extracts text and segments sections. `chunking.simple_paragraph_chunker` creates \~2k-char chunks with overlap. |
| **Embed + Upsert** | `EmbeddingService` generates embeddings. `VectorStore.upsert_chunks` stores text + metadata + vector. |

### 2) Caching Gate (Network Minimization)

The system caches latest 10-Q metadata per ticker on disk: `data/cache/latest_tenq.json`.

* It only re-calls SEC if the latest filing filed date or period end date changes.
* If cached metadata exists **and** chunks for that accession are already in the vector store, it skips the entire SEC ingestion process.

### 3) Agentic Analysis

* **Insights Agent (GPT-5):** Uses a retrieval tool (capped top-k and truncated) to pull relevant chunks from the vector store. It produces a structured equity-research report per the elite-analyst framework.
* **Decision Agent (GPT-5):** Consumes the structured insights JSON from the first agent and outputs the final **Buy / Hold / Sell** recommendation, confidence score, and rationale.

---

## 🛠️ Dependencies & Setup

### 1. Install Python dependencies

Bash

```
pip install -e ".[dev]"
```

### 2. Update the .env file
Create a .env file at the repository root (or copy from .env.example) and fill in your credentials and settings:

Code snippet

```
# SEC EDGAR SETTINGS
SEC_USER_AGENT="Your Name Contact Info / sec-10q-analyst"
SEC_MAX_RPS=8

# OPENAI / LLM SETTINGS
OPENAI_API_KEY="sk-..."
LLM_MODEL="openai:gpt-5"
EMBEDDING_MODEL="text-embedding-3-large"

# VECTOR DATABASE SETTINGS (e.g., pgvector)
VECTOR_DB_URL="postgresql://user:pass@localhost:5432/sec_vectors"
```


### 3. Run the API Locally (No Docker)
Bash

```
uvicorn app.api.main:app --reload
```

### 4. Running with Docker
Ensure your .env file exists at the repo root with your API key and SEC user agent.

From the infra/ directory:

Bash

```
docker compose build
docker compose up
```

### 5. Test the Endpoint:
You can test the core functionality via a curl command:

Bash

```
curl -X POST "http://localhost:8000/summaries/10q" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"AAPL"}'
```


### 6 . Sample Output (Abridged)
The output is a structured JSON object containing both the detailed insights and the final investment decision.

JSON
```
{
  "insights": {
    "company_profile": {
      "name": "Apple Inc.",
      "ticker": "AAPL"
      // ... metadata ...
    },
    "high_level_summary": "Apple reported steady Services growth offsetting softer hardware demand...",
    "financial_summary": {
      "key_metrics": {
        "revenue_yoy_pct": 4.2,
        "gross_margin_pct": 45.1,
        "free_cash_flow": 22000000000
      },
      "narrative": "Revenue expansion was driven by Services and iPhone upgrades..."
    },
    "risk_summary": [
      {
        "title": "China demand volatility",
        "description": "Management notes continued macro uncertainty affecting regional sales.",
        "changed_since_prior": true
      }
    ],
    // ... other sections ...
  },
  "decision": {
    "decision": "hold",
    "confidence": 0.64,
    "time_horizon": "6–12 months",
    "positives": [
      "Services momentum and margin resilience",
      "Strong FCF supporting shareholder returns"
    ],
    "negatives": [
      "Hardware cyclicality and China exposure",
      "Regulatory headwinds in EU/US"
    ],
    "risk_profile": "Moderate",
    "disclaimer": "This is an automated, heuristic assessment based on the latest 10-Q filing and does not constitute investment advice. Do your own research."
  }
}
```
