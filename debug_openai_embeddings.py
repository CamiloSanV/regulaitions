"""
debug_openai_embeddings.py
──────────────────────────
Checks if OpenAI embeddings are working correctly with the
new 1536-dim Supabase schema after the switch from BGE-M3.

Usage:
  python debug_openai_embeddings.py
"""

import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("STEP 1 — How many chunks are in Supabase?")
print("=" * 60)

client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
result = client.table("documents").select("id, metadata", count="exact").execute()
print(f"Total chunks: {result.count}")

# Check embedding dimensions
sample = client.table("documents").select("embedding").limit(1).execute()
if sample.data:
    emb = sample.data[0].get("embedding")
    if emb:
        # embedding is stored as a string like "[0.1, 0.2, ...]"
        dims = len(emb.split(",")) if isinstance(emb, str) else len(emb)
        print(f"Embedding dimensions: {dims}")
    else:
        print("WARNING: embedding field is NULL — chunks may not have been embedded!")
else:
    print("No chunks found at all!")

print("\n" + "=" * 60)
print("STEP 2 — Test direct similarity search")
print("=" * 60)

from rag.retriever import get_embeddings
from langchain_community.vectorstores import SupabaseVectorStore

embeddings = get_embeddings()
vectorstore = SupabaseVectorStore(
    client=client,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

test_query = "Article 9 risk management system"
print(f"Query: '{test_query}'")

docs = vectorstore.similarity_search(test_query, k=5)
print(f"Results: {len(docs)}")
for d in docs:
    print(f"  - {d.metadata.get('article_label', '?')} ({d.metadata.get('source', '?')})")

print("\n" + "=" * 60)
print("STEP 3 — Test match_documents function directly")
print("=" * 60)

# Generate a test embedding and call match_documents directly
test_embedding = embeddings.embed_query("risk management system")
print(f"Query embedding dimensions: {len(test_embedding)}")

rpc_result = client.rpc("match_documents", {
    "query_embedding": test_embedding,
    "match_count": 5,
    "filter": {}
}).execute()

print(f"match_documents returned: {len(rpc_result.data)} results")
for row in rpc_result.data[:3]:
    meta = row.get("metadata", {})
    print(f"  - {meta.get('article_label', '?')} (similarity: {row.get('similarity', '?'):.4f})")
