# rag-ai

A standalone NestJS service for **RAG + a Planner agent** over project data.

- **Generation** (chat answers, planner reasoning) → **OpenRouter** (OpenAI-compatible; default model `openai/gpt-4o-mini`)
- **Embeddings** → **Google Gemini** (`gemini-embedding-001`, 3072 dims)
- **Vector store** → **Qdrant**
- **No text-to-SQL** — aggregates (totals/top-N/filtered sums) are computed in code over the vector store.

## Architecture

```
POST /rag/openai/chat   → routes: aggregate Q → analytics tool ; lookup Q → RAG ; streams answer
POST /rag/planner       → LLM tool-calling loop (searchProjects + projectAnalytics), returns {answer, trace}
POST /rag/projects/analytics → exact aggregates directly (no LLM)
POST /rag/openai/qdrant/ingest → chunk + embed + store
```

Tools the Planner can call:
- `searchProjects` — semantic RAG lookup (Gemini embeddings → Qdrant search)
- `projectAnalytics` — exact totals / top-N / averages / filtered sums (reads all points from Qdrant, computes in code)

## Setup

### 1. Install
```bash
npm install
```

### 2. Configure
```bash
cp .env.example .env
# then fill in:
#   OPEN_ROUTER_API_KEY  (https://openrouter.ai/keys)
#   GEMINI_API_KEY       (https://aistudio.google.com/apikey — used for embeddings)
#   QDRANT_URL / QDRANT_API_KEY  (local Docker or Qdrant Cloud)
```

### 3. Start Qdrant (local) — or use Qdrant Cloud and set QDRANT_URL/API_KEY
```bash
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

### 4. Run the app
```bash
npm run start:dev
# → http://localhost:3000  (Swagger at /docs)
```

### 5. Ingest sample data (also creates the customerId index)
```bash
chmod +x rag-data/ingest-projects.sh
./rag-data/ingest-projects.sh
```

## Try it

```bash
# Aggregate (analytics tool)
curl -sN -X POST http://localhost:3000/rag/openai/chat \
  -H "Content-Type: application/json" \
  -d '{ "question": "total estimated value of all projects", "customerId": "cust_123" }'

# Lookup (RAG)
curl -sN -X POST http://localhost:3000/rag/openai/chat \
  -H "Content-Type: application/json" \
  -d '{ "question": "who is the design manager on yatra 87?", "customerId": "cust_123" }'

# Planner with trace (see which tools it called)
curl -s -X POST http://localhost:3000/rag/planner \
  -H "Content-Type: application/json" \
  -d '{ "question": "top 5 projects by value where Nitish is a team member", "customerId": "cust_123" }'
```

## Notes
- Change the LLM in one place: `OPENROUTER_MODEL` in `src/rag/openai.service.ts` (e.g. `google/gemini-2.5-flash`).
- Ingest more data: add objects to `rag-data/projects.json` and re-run the ingest script (re-ingest is idempotent — point ids derive from `projectId`).
- Embedding dimension is fixed at 3072 (`gemini-embedding-001`) in `src/rag/qdrant.service.ts`.
