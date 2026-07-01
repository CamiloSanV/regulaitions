"""
evaluation/evaluate.py
───────────────────────
Evalúa el pipeline RAG con métricas RAGAS:
  - Faithfulness      → ¿la respuesta está soportada por los docs?
  - Answer relevancy  → ¿la respuesta responde la pregunta?
  - Context precision → ¿los mejores chunks aparecen primero?
  - Context recall    → ¿los chunks tienen toda la info necesaria?

Uso:
  python -m evaluation.evaluate

Genera: evaluation/results/ragas_report_{timestamp}.json
"""

import json
from datetime import datetime
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset de evaluación ──────────────────────────────────────────────────────
# Preguntas representativas que cubren los casos de uso principales
EVAL_QUESTIONS = [
    {
        "question": "What are the obligations for providers of high-risk AI systems under Article 9?",
        "ground_truth": "Providers of high-risk AI systems must establish a risk management system that includes identification and analysis of known and foreseeable risks, estimation and evaluation of risks, and adoption of risk management measures. This system must be a continuous iterative process throughout the lifecycle of the high-risk AI system.",
    },
    {
        "question": "What AI systems are considered prohibited under the EU AI Act?",
        "ground_truth": "Prohibited AI systems include those using subliminal techniques, exploiting vulnerabilities of persons, social scoring by public authorities, real-time remote biometric identification in public spaces (with exceptions), and AI used to infer emotions in workplace or educational institutions.",
    },
    {
        "question": "What does Article 13 require regarding transparency of high-risk AI systems?",
        "ground_truth": "Article 13 requires that high-risk AI systems be designed with sufficient transparency to enable deployers to interpret the system's output and use it appropriately. Technical documentation must be provided along with instructions for use, including information about the system's capabilities and limitations.",
    },
    {
        "question": "What is the definition of a general-purpose AI model under the EU AI Act?",
        "ground_truth": "A general-purpose AI model is an AI model that is trained with a large amount of data using self-supervision at scale, that displays significant generality and is capable of performing a wide range of distinct tasks, and that can be integrated into various downstream systems or applications.",
    },
    {
        "question": "What are the requirements for human oversight of high-risk AI systems under Article 14?",
        "ground_truth": "Article 14 requires that high-risk AI systems be designed and developed to enable effective human oversight. This includes measures allowing individuals to understand the capabilities and limitations, be aware of automation bias, interpret the output correctly, and intervene or override the system when necessary.",
    },
    {
        "question": "When does the EU AI Act fully apply and what are the key transition dates?",
        "ground_truth": "The EU AI Act entered into force in August 2024. Prohibited AI practices apply from February 2025. GPAI model obligations apply from August 2025. High-risk AI system obligations apply from August 2026 for new systems, with some existing systems having until 2027.",
    },
]


def run_evaluation():
    print("=" * 60)
    print("regulAItions — RAGAS Evaluation")
    print("=" * 60)

    # Inicializa el retriever
    print("\n1. Inicializando retriever híbrido...")
    from rag.retriever import build_retriever, get_embeddings
    retriever = build_retriever(source_filter="EU_AI_Act", k_final=5)

    # Inicializa el LLM
    print("2. Inicializando LLM...")
    from agent.agent import get_llm
    llm = get_llm()

    # Prepara el dataset
    print("3. Generando respuestas para el dataset de evaluación...")
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for i, item in enumerate(EVAL_QUESTIONS):
        print(f"   [{i+1}/{len(EVAL_QUESTIONS)}] {item['question'][:60]}...")

        # Retrieval
        docs = retriever._get_relevant_documents(item["question"])
        contexts = [doc.page_content for doc in docs]

        # Generación
        context_text = "\n\n".join(contexts[:3])
        prompt = f"""Answer this question about EU AI Act based ONLY on the provided context.
Be precise and cite article numbers when relevant.

Context:
{context_text}

Question: {item['question']}

Answer:"""
        answer = llm.invoke(prompt).content

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["ground_truth"])

    # Ejecuta RAGAS — usa el mismo LLM (Groq) y embeddings (BGE-M3) que el resto
    # del proyecto, en lugar del default de RAGAS que requiere OpenAI
    print("\n4. Calculando métricas RAGAS...")
    dataset = Dataset.from_dict(eval_data)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    # Muestra resultados
    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    scores = results.to_pandas()[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()

    thresholds = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.80,
        "context_precision": 0.75,
        "context_recall": 0.70,
    }

    report = {"timestamp": datetime.now().isoformat(), "metrics": {}}

    for metric, score in scores.items():
        threshold = thresholds.get(metric, 0.75)
        status = "PASS" if score >= threshold else "FAIL"
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {metric:<25} {score:.3f}  (objetivo: >{threshold})  [{status}]")
        report["metrics"][metric] = {
            "score": round(float(score), 4),
            "threshold": threshold,
            "status": status,
        }

    overall = all(v["status"] == "PASS" for v in report["metrics"].values())
    report["overall"] = "PASS" if overall else "FAIL"

    print(f"\n  Overall: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)

    # Guarda el reporte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"ragas_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReporte guardado: {report_path}")
    print("\nConsejos si alguna métrica falla:")
    print("  context_precision bajo → reduce k_final o mejora el re-ranker")
    print("  context_recall bajo   → aumenta k_dense o mejora el chunking")
    print("  faithfulness bajo     → el LLM está alucinando → añade más contexto")
    print("  answer_relevancy bajo → ajusta el prompt del agente")

    return report


if __name__ == "__main__":
    run_evaluation()
