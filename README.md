# RegIntel — Regulatory Intelligence Agent

Agente de compliance para EU AI Act y GDPR.
Clasifica sistemas de IA, lista obligaciones, genera checklists y alerta sobre cambios regulatorios.

## Stack

| Capa | Herramienta | Costo |
|---|---|---|
| LLM | Groq + Llama 3.1 70B | **Gratis** |
| Embeddings | BGE-M3 (local) | **Gratis** |
| Re-ranker | CrossEncoder ms-marco | **Gratis** |
| Vector store | Supabase pgvector | **Gratis** |
| Backend | FastAPI en Render | **Gratis** |
| Frontend | Next.js en Vercel | **Gratis** |

**Costo total para portfolio demo: €0/mes**

---

## Paso 1: Clonar y configurar entorno

```bash
git clone https://github.com/tu-usuario/regintel
cd regintel

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Paso 2: Variables de entorno

```bash
cp .env.example .env
```

Edita `.env` con:
- `GROQ_API_KEY` → regístrate gratis en [console.groq.com](https://console.groq.com)
- `SUPABASE_URL` y `SUPABASE_KEY` → crea un proyecto gratis en [supabase.com](https://supabase.com)

## Paso 3: Configurar Supabase (una sola vez)

En el dashboard de Supabase → SQL Editor → pega y ejecuta:

```sql
create extension if not exists vector;

create table if not exists documents (
    id        bigserial primary key,
    content   text not null,
    metadata  jsonb,
    embedding vector(1024)
);

create index if not exists documents_embedding_idx
    on documents using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

create or replace function match_documents (
    query_embedding vector(1024),
    match_count     int   default 10,
    filter          jsonb default '{}'
)
returns table (id bigint, content text, metadata jsonb, similarity float)
language plpgsql as $$
begin
    return query
    select d.id, d.content, d.metadata,
           1 - (d.embedding <=> query_embedding) as similarity
    from documents d
    where d.metadata @> filter
    order by d.embedding <=> query_embedding
    limit match_count;
end;
$$;
```

## Paso 4: Ingestar el EU AI Act

```bash
# Descarga, chunkea y sube a Supabase (~10-15 min, descarga el modelo BGE-M3 la primera vez)
python -m ingestion.ingest --source eu_ai_act

# Opcional: añade GDPR también
python -m ingestion.ingest --source gdpr
```

Deberías ver:
```
✓ 847 chunks generados
✓ Subiendo a Supabase... [████████████] 100%
✓ Ingesta completa: 847 chunks en Supabase
```

## Paso 5: Probar el agente en terminal

```bash
python -m agent.agent
```

```
Tú: Vamos a lanzar un sistema de scoring de crédito con ML. ¿Qué necesitamos?

RegIntel: [Thought] El usuario describe un sistema, necesito clasificarlo...
          [Action] check_obligations("sistema de scoring de crédito")
          [Observation] Clasificado como Alto Riesgo · Annex III §5b

Tu sistema cae en la categoría de ALTO RIESGO (Annex III, punto 5b)...
```

## Paso 6: Levantar la API

```bash
uvicorn api.main:app --reload --port 8000
```

Prueba en `http://localhost:8000/docs` (Swagger UI automático de FastAPI).

```bash
# Test rápido
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "¿Qué es el EU AI Act?", "session_id": "test"}'
```

## Paso 7: Evaluación RAGAS

```bash
python -m evaluation.evaluate
```

```
✓ faithfulness          0.872  (objetivo: >0.85)  [PASS]
✓ answer_relevancy      0.841  (objetivo: >0.80)  [PASS]
✓ context_precision     0.783  (objetivo: >0.75)  [PASS]
✗ context_recall        0.698  (objetivo: >0.70)  [FAIL]

Consejo: context_recall bajo → aumenta k_dense en el retriever
```

## Paso 8: Deploy gratuito

### Backend en Render

1. Push a GitHub
2. [render.com](https://render.com) → New Web Service → conecta tu repo
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Añade las variables de entorno en el dashboard de Render

### Frontend en Vercel

```bash
cd frontend
npx create-next-app@latest .   # si aún no existe
vercel deploy
```

---

## Estructura del proyecto

```
regintel/
├── ingestion/
│   └── ingest.py          ← descarga, chunking, embeddings, upload
├── rag/
│   └── retriever.py       ← dense + BM25 + CrossEncoder reranker
├── agent/
│   └── agent.py           ← 5 tools + AgentExecutor ReAct
├── api/
│   └── main.py            ← FastAPI con SSE streaming
├── evaluation/
│   ├── evaluate.py        ← métricas RAGAS
│   └── results/           ← reportes JSON por timestamp
├── requirements.txt
└── .env.example
```

## Para entrevistas

Puntos clave que puedes defender con evidencia:

- **RAG evaluado cuantitativamente** → muestra `evaluation/results/ragas_report_*.json`
- **Decisión de retrieval híbrido** → "el re-ranker mejoró context_precision de 0.61 a 0.78"
- **LLM intercambiable** → "cambié de Groq a GPT-4o mini en una línea de código"
- **Chunking jerárquico** → "preserva la estructura Article > Paragraph, no chunking por tokens"
- **Costo demostrable** → "la demo corre a €0/mes con free tiers"
