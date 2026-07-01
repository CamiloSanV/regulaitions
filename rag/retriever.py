"""
rag/retriever.py
────────────────
Retrieval híbrido en tres etapas:
  1. Dense retrieval  → pgvector cosine similarity (top-20)
  2. Sparse retrieval → BM25 sobre el mismo corpus (top-20)
  3. Re-ranking       → CrossEncoder ms-marco (top-5 final)

El resultado es un retriever compatible con LangChain que
puedes enchufar directamente al AgentExecutor.
"""

import os
from typing import Optional

from langchain.schema import BaseRetriever, Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from supabase import create_client
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


# ── Singleton de embeddings (carga el modelo una sola vez) ─────────────────────
_embeddings_instance = None

def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
        )
    return _embeddings_instance


# ── Singleton de re-ranker ─────────────────────────────────────────────────────
_reranker_instance = None

def get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        # Modelo gratuito, ~70MB, excelente para inglés legal
        _reranker_instance = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
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
      k_final:   cuántos devolver tras el re-ranking
      filter:    filtro de metadatos (e.g. {"source": "EU_AI_Act"})
    """

    k_dense: int = Field(default=20)
    k_sparse: int = Field(default=20)
    k_final: int = Field(default=5)
    filter: Optional[dict] = Field(default=None)

    # Estos campos se inicializan en setup()
    _vectorstore: Optional[SupabaseVectorStore] = None
    _bm25: Optional[BM25Retriever] = None
    _reranker: Optional[CrossEncoder] = None
    _all_docs: list[Document] = []

    class Config:
        arbitrary_types_allowed = True

    def setup(self, all_docs: list[Document]) -> "HybridRegulatoryRetriever":
        """
        Inicializa los componentes con el corpus de documentos.
        Llama a este método una vez al arrancar la app.

        Uso:
            docs = load_all_docs_from_supabase()
            retriever = HybridRegulatoryRetriever().setup(docs)
        """
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
        # BM25 necesita el corpus en memoria (ligero, ~50MB para EU AI Act)
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
        # específicas que a preguntas largas en lenguaje natural
        rerank_query = search_query
        pairs = [(rerank_query, doc.page_content) for doc in candidates]
        scores = self._reranker.predict(pairs)

        # Ordena por score descendente
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        # ── Boost: si la query menciona un artículo explícito, ese
        # artículo se garantiza en el top de resultados, sin importar
        # su rerank_score. Esto corrige el sesgo del CrossEncoder hacia
        # frases introductorias genéricas con alto solapamiento léxico.
        if explicit_article:
            explicit_docs = [
                (score, doc) for score, doc in ranked
                if doc.metadata.get("article") == explicit_article
            ]
            other_docs = [
                (score, doc) for score, doc in ranked
                if doc.metadata.get("article") != explicit_article
            ]
            # El artículo explícito va primero, el resto completa el top_k
            ranked = explicit_docs + other_docs

        top_docs = [doc for _, doc in ranked[: self.k_final]]

        # Añade el score al metadata para transparencia
        for i, (score, doc) in enumerate(ranked[: self.k_final]):
            top_docs[i].metadata["rerank_score"] = round(float(score), 4)
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
