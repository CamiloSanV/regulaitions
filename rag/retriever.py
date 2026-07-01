"""
rag/retriever.py
────────────────
Retrieval híbrido en tres etapas:
  1. Dense retrieval  → pgvector cosine similarity (top-20)
  2. Sparse retrieval → BM25 sobre el mismo corpus (top-20)
  3. Re-ranking       → FlashRank cross-encoder (no torch needed)

Uses OpenAI text-embedding-3-small for consistency between local
development and production (Render). FlashRank replaces the original
sentence-transformers CrossEncoder to eliminate the torch dependency
while keeping reranking quality.
"""

import os
from typing import Optional

from langchain.schema import BaseRetriever, Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.retrievers import BM25Retriever
from supabase import create_client
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


# ── Singleton de embeddings ────────────────────────────────────────────────────
_embeddings_instance = None

def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = OpenAIEmbeddings(
            model="text-embedding-3-small",
            dimensions=1536,
        )
    return _embeddings_instance


# ── Singleton de re-ranker (FlashRank — no torch required) ────────────────────
_reranker_instance = None

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        from flashrank import Ranker
        # ms-marco-MiniLM-L-12-v2 is a proven cross-encoder for passage ranking
        # ~90MB ONNX model, downloads once and caches locally
        _reranker_instance = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return _reranker_instance

# ── Retriever híbrido ──────────────────────────────────────────────────────────
class HybridRegulatoryRetriever(BaseRetriever):
    """
    Retriever que combina:
    - Dense:  pgvector (semántico, encuentra conceptos similares)
    - Sparse: BM25     (léxico, encuentra términos exactos como "Article 9")
    - Rerank: CrossEncoder (reordena los 40 candidatos → top k)

    Parámetros:
      k_dense:   cuántos docs traer del vector store
      k_sparse:  cuántos docs traer de BM25
      k_final:   cuántos devolver tras la fusión y boost
      filter:    filtro de metadatos (e.g. {"source": "EU_AI_Act"})
    """

    k_dense: int = Field(default=20)
    k_sparse: int = Field(default=20)
    k_final: int = Field(default=5)
    filter: Optional[dict] = Field(default=None)

    _vectorstore: Optional[SupabaseVectorStore] = None
    _bm25: Optional[BM25Retriever] = None
    _reranker: Optional[object] = None
    _all_docs: list[Document] = []

    class Config:
        arbitrary_types_allowed = True

    def setup(self, all_docs: list[Document]) -> "HybridRegulatoryRetriever":
        supabase_client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY"),
        )
        self._vectorstore = SupabaseVectorStore(
            client=supabase_client,
            embedding=get_embeddings(),
            table_name="documents",
            query_name="match_documents",
        )
        self._bm25 = BM25Retriever.from_documents(all_docs)
        self._bm25.k = self.k_sparse
        self._reranker = get_reranker()
        self._all_docs = all_docs
        return self

    def _get_relevant_documents(self, query: str) -> list[Document]:
        """
        Pipeline completo de retrieval para una query.
        Llamado automáticamente por LangChain.

        Para preguntas largas en lenguaje natural, primero genera una
        query expandida más densa en terminología legal — esto mejora
        sustancialmente el recall para preguntas conversacionales tipo
        "What are the obligations for providers under Article 9?"
        que de otra forma compiten semánticamente con artículos
        relacionados pero distintos.

        IMPORTANTE: cuando la query menciona explícitamente un número
        de artículo (e.g. "Article 9", "under Article 13"), ese artículo
        se prioriza directamente, sin pasar por el juicio del reranker.
        Esto es necesario porque el CrossEncoder puede preferir artículos
        con frases introductorias genéricas (alto solapamiento léxico)
        sobre el artículo específico que el usuario pidió — un sesgo
        conocido de los CrossEncoders en texto legal/estructurado.
        """
        # ── Detección de artículo explícito en la query ───────────
        explicit_article = self._extract_article_number(query)

        search_query = self._expand_query(query) if len(query.split()) > 8 else query

        # ── Etapa 1: Dense retrieval ──────────────────────────────
        metadata_filter = self.filter or {}
        dense_docs = self._vectorstore.similarity_search(
            search_query,
            k=self.k_dense,
            filter=metadata_filter,
        )

        # Si la búsqueda densa devuelve muy pocos resultados (puede pasar
        # con queries muy específicas o cortas), reintenta con la query
        # original sin filtro de metadata como fallback
        if len(dense_docs) < 5:
            fallback_docs = self._vectorstore.similarity_search(
                query, k=self.k_dense, filter=metadata_filter,
            )
            dense_docs = dense_docs + fallback_docs

        # ── Etapa 2: Sparse retrieval (BM25) — usa AMBAS queries ──
        sparse_docs = self._bm25.get_relevant_documents(search_query)
        if search_query != query:
            sparse_docs += self._bm25.get_relevant_documents(query)

        # ── Fusión y deduplicación ────────────────────────────────
        seen_ids = set()
        candidates = []
        for doc in dense_docs + sparse_docs:
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                candidates.append(doc)

        # ── Etapa 3: Re-ranking con CrossEncoder ──────────────────
        if not candidates:
            return []

        # El reranker usa la query EXPANDIDA (terminología densa) ya
        # que el CrossEncoder responde mejor a queries cortas y
        # ── Etapa 3: Re-ranking con FlashRank ────────────────────
        # FlashRank uses an ONNX cross-encoder — no torch required,
        # runs on CPU, scores (query, passage) pairs for relevance.
        try:
            from flashrank import RerankRequest
            reranker = get_reranker()
            passages = [
                {"id": i, "text": doc.page_content}
                for i, doc in enumerate(candidates)
            ]
            rerank_req = RerankRequest(query=search_query, passages=passages)
            rerank_results = reranker.rerank(rerank_req)
            # rerank_results is sorted by score descending
            id_to_score = {r["id"]: r["score"] for r in rerank_results}
            ranked = sorted(
                candidates,
                key=lambda d: id_to_score.get(candidates.index(d), 0),
                reverse=True,
            )
        except Exception:
            # Fallback: position ordering if reranker fails
            ranked = candidates

        # ── Boost: artículo explícito siempre va primero ─────────
        if explicit_article:
            explicit_docs = [d for d in ranked if d.metadata.get("article") == explicit_article]
            other_docs    = [d for d in ranked if d.metadata.get("article") != explicit_article]
            ranked = explicit_docs + other_docs

        top_docs = ranked[: self.k_final]
        for i, doc in enumerate(top_docs):
            top_docs[i].metadata["rank"] = i + 1
            if explicit_article and doc.metadata.get("article") == explicit_article:
                top_docs[i].metadata["boosted"] = True

        return top_docs

    @staticmethod
    def _extract_article_number(query: str) -> Optional[str]:
        """
        Extracts an explicit article number from a query, if present.
        E.g. "What does Article 9 say?" -> "9"
             "obligations under Article 13" -> "13"
        Returns None if no article number is mentioned.
        """
        import re
        match = re.search(r"\bArticle\s+(\d+)\b", query, re.IGNORECASE)
        return match.group(1) if match else None

    def _expand_query(self, query: str) -> str:
        """
        Rephrase a long natural-language question into a short,
        terminology-dense search query, closer to how the regulation
        itself is worded. This is a lightweight form of HyDE.

        Falls back silently to the original query if the LLM call fails
        or if no LLM is configured.
        """
        try:
            from agent.agent import get_llm
            llm = get_llm()
            prompt = (
                "Rewrite this question as a short search query (5-8 words) "
                "using precise legal/regulatory terminology, dropping filler "
                "words like 'what are', 'under', 'for'. "
                "Only output the rewritten query, nothing else.\n\n"
                f"Question: {query}\n"
                "Search query:"
            )
            expanded = llm.invoke(prompt).content.strip().strip('"')
            return expanded if expanded else query
        except Exception:
            return query

    async def _aget_relevant_documents(self, query: str) -> list[Document]:
        """Versión async (delega a la síncrona por ahora)."""
        return self._get_relevant_documents(query)


# ── Función de conveniencia para cargar docs desde Supabase ───────────────────
def load_docs_from_supabase(
    source_filter: Optional[str] = None,
    limit: int = 5000,
) -> list[Document]:
    """
    Carga documentos desde Supabase para inicializar BM25.
    Filtra opcionalmente por fuente ("EU_AI_Act", "GDPR").
    """
    from supabase import create_client

    client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY"),
    )

    query = client.table("documents").select("content, metadata").limit(limit)
    if source_filter:
        query = query.eq("metadata->>source", source_filter)

    result = query.execute()

    docs = []
    for row in result.data:
        docs.append(Document(
            page_content=row["content"],
            metadata=row["metadata"] or {},
        ))
    return docs


# ── Factory: crea el retriever listo para usar ────────────────────────────────
def build_retriever(
    source_filter: Optional[str] = None,
    k_final: int = 5,
) -> HybridRegulatoryRetriever:
    """
    Crea y configura el retriever híbrido completo.

    Uso en la app:
        retriever = build_retriever(source_filter="EU_AI_Act")

    Uso en el agente:
        tool = create_retriever_tool(retriever, "search_regulation", "...")
    """
    print("Cargando documentos desde Supabase para BM25...")
    all_docs = load_docs_from_supabase(source_filter=source_filter)
    print(f"  {len(all_docs)} documentos cargados")

    retriever = HybridRegulatoryRetriever(
        k_dense=20,
        k_sparse=20,
        k_final=k_final,
        filter={"source": source_filter} if source_filter else None,
    )
    retriever.setup(all_docs)
    print("  Retriever híbrido listo (dense + BM25 + reranker)")
    return retriever
