"""
agent/agent.py
──────────────
regulAItions Agent — ReAct pattern with 5 tools for regulatory compliance.
Everything in English for consistency.
"""

import os
import json
from datetime import datetime

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub
from dotenv import load_dotenv
from fpdf import FPDF

load_dotenv()


# ── LLM (swappable) ───────────────────────────────────────────────────────────
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "groq")
    model = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model, temperature=0, max_tokens=2048)
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0, max_tokens=2048)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0, max_tokens=2048)

    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


# ── Global retriever (initialized once on first use) ──────────────────────────
_retriever = None

def get_retriever():
    global _retriever
    if _retriever is None:
        try:
            from rag.retriever import build_retriever
            # No source_filter: search across ALL indexed regulations
            # (EU AI Act + GDPR). The agent's tools and prompt already
            # handle citing the correct source per chunk's metadata.
            _retriever = build_retriever(source_filter=None)
        except Exception as e:
            print(f"[Warning] Retriever unavailable ({e}), using LLM-only mode")
            _retriever = None
    return _retriever


# ─────────────────────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_regulation(query: str) -> str:
    """
    Search the regulatory corpus (EU AI Act, GDPR) using hybrid retrieval.
    Use this for open-ended questions about AI regulation.
    Returns the most relevant chunks with their source article.

    Parameter:
      query: question or topic to search in natural language
    """
    retriever = get_retriever()

    if retriever is None:
        return f"[LLM-only mode] No corpus available. Answering from base knowledge about: {query}"

    docs = retriever._get_relevant_documents(query)

    if not docs:
        return "No relevant fragments found for this query."

    results = []
    for doc in docs:
        meta = doc.metadata
        results.append(
            f"[{meta.get('source', 'Unknown')} · {meta.get('article', '?')} "
            f"· score: {meta.get('rerank_score', '?')}]\n"
            f"{doc.page_content}\n"
        )
    return "\n---\n".join(results)


@tool
def get_article(article_ref: str) -> str:
    """
    Retrieve the full text of a specific article.
    Use this when the query mentions an article by number (e.g. "Article 9", "Article 13").
    Also useful to get full context after search_regulation.
    If the user mentions GDPR specifically, include "GDPR" in the article_ref
    (e.g. "GDPR Article 4") to disambiguate from EU AI Act articles with the
    same number.

    Parameter:
      article_ref: article reference, e.g. "Article 9", "GDPR Article 4", "Recital 47"
    """
    ref_raw = article_ref.strip()

    # Detect which regulation is being referenced, to avoid ambiguity when
    # both EU AI Act and GDPR have an article with the same number
    source_filter = None
    if "gdpr" in ref_raw.lower():
        source_filter = "GDPR"
        source_label = "GDPR"
    elif "ai act" in ref_raw.lower() or "ai_act" in ref_raw.lower():
        source_filter = "EU_AI_Act"
        source_label = "EU AI Act"
    else:
        source_label = None  # unknown — will search both, prefer first match

    ref = ref_raw.title()
    if not ref.startswith("Article") and not ref.startswith("Recital") and not ref.startswith("Gdpr"):
        ref = f"Article {ref}"
    # Clean up "Gdpr Article X" -> "Article X" for the label, keep source_filter separately
    ref = ref.replace("Gdpr ", "").replace("Ai Act ", "")

    article_number = ref.replace("Article ", "").replace("Recital ", "")

    retriever = get_retriever()
    if retriever is None:
        return f"Corpus not available. Cannot retrieve {ref}."

    metadata_filter = {"article": article_number}
    if source_filter:
        metadata_filter["source"] = source_filter

    docs = retriever._vectorstore.similarity_search(
        f"full text {ref} obligations requirements",
        k=8,
        filter=metadata_filter,
    )

    if not docs:
        # Retry without source filter in case the user didn't specify
        # which regulation, or specified it ambiguously
        docs = retriever._vectorstore.similarity_search(
            f"{ref}", k=8, filter={"article": article_number}
        )

    if not docs:
        return f"{ref} not found in corpus."

    # If multiple sources matched and none was explicitly requested,
    # group by source and return the first one found, but warn about
    # the ambiguity so the agent can ask for clarification if needed
    sources_found = set(d.metadata.get("source") for d in docs)
    if len(sources_found) > 1 and not source_filter:
        chosen_source = docs[0].metadata.get("source")
        docs = [d for d in docs if d.metadata.get("source") == chosen_source]
        ambiguity_note = (
            f"\n[Note: {ref} exists in multiple regulations ({', '.join(sources_found)}). "
            f"Showing {chosen_source}. Specify e.g. 'GDPR {ref}' to disambiguate.]\n"
        )
    else:
        ambiguity_note = ""

    combined = "\n\n".join([doc.page_content for doc in docs])
    meta = docs[0].metadata

    return (
        f"=== {ref} — {meta.get('source_name', meta.get('source', 'Unknown'))} ==={ambiguity_note}\n"
        f"Chapter: {meta.get('chapter', 'N/A')}\n"
        f"Version: {meta.get('version', 'N/A')}\n\n"
        f"{combined}"
    )


@tool
def check_obligations(system_description: str) -> str:
    """
    Classify an AI system under the EU AI Act and list its obligations.
    Use this when the user describes their system and wants to know what to comply with.
    Returns: risk category, applicable articles, and key deadlines.

    Parameter:
      system_description: description of the AI system in natural language
    """
    llm = get_llm()

    # Step 1: Classify via LLM
    classify_prompt = f"""Classify this AI system under the EU AI Act. Respond ONLY with valid JSON, no markdown.

AI System: {system_description}

Return this exact JSON structure:
{{
  "category": "prohibited|high_risk|gpai|minimal_risk",
  "annex_ref": "Annex III, point X" or null,
  "rationale": "2-sentence explanation in English",
  "confidence": 0.0-1.0,
  "example_similar": "real example of a similar system"
}}"""

    classification_raw = llm.invoke(classify_prompt).content

    try:
        clean = classification_raw.strip().strip("```json").strip("```").strip()
        classification = json.loads(clean)
    except json.JSONDecodeError:
        classification = {
            "category": "high_risk",
            "annex_ref": "Requires manual review",
            "rationale": classification_raw[:200],
            "confidence": 0.5,
        }

    # Step 2: Retrieve applicable obligations
    category = classification.get("category", "high_risk")

    retriever = get_retriever()
    if retriever is not None:
        obligations_query = f"obligations requirements {category} AI system {system_description[:100]}"
        obligation_docs = retriever._get_relevant_documents(obligations_query)
    else:
        obligation_docs = []

    # Step 3: Format result
    category_labels = {
        "prohibited":    "PROHIBITED",
        "high_risk":     "HIGH RISK",
        "gpai":          "GENERAL PURPOSE AI (GPAI)",
        "minimal_risk":  "MINIMAL RISK",
    }

    deadlines = {
        "prohibited":   "Prohibited since February 2025",
        "high_risk":    "Compliance required: August 2026",
        "gpai":         "GPAI compliance: August 2025",
        "minimal_risk": "No critical deadline",
    }

    if obligation_docs:
        obligations_text = "\n".join([
            f"  - [{doc.metadata.get('article', '?')}] {doc.page_content[:200]}..."
            for doc in obligation_docs[:5]
        ])
    else:
        obligations_text = "  - Corpus not available. Use search_regulation for details."

    return f"""
=== EU AI ACT CLASSIFICATION ===
System: {system_description[:120]}

Category:   {category_labels.get(category, category.upper())}
Reference:  {classification.get('annex_ref', 'N/A')}
Confidence: {classification.get('confidence', 0):.0%}
Deadline:   {deadlines.get(category, 'See regulation')}

Rationale: {classification.get('rationale', '')}

=== MAIN OBLIGATIONS ===
{obligations_text}

=== NEXT STEP ===
Call generate_checklist with this output to produce an actionable PDF checklist.
""".strip()


@tool
def compare_versions(article_ref: str, version_a: str = "2024-Q1", version_b: str = "2024-Q3") -> str:
    """
    Compare two versions of an article and explain the practical impact of the change.
    Use this when the user asks what changed recently in the regulation.

    Parameters:
      article_ref: e.g. "Article 9", "Article 14"
      version_a: earlier version, e.g. "2024-Q1"
      version_b: current version, e.g. "2024-Q3"
    """
    retriever = get_retriever()
    if retriever is None:
        return "Corpus not available. Cannot compare versions."

    current_docs = retriever._vectorstore.similarity_search(
        f"{article_ref} requirements obligations",
        k=4,
        filter={"article": article_ref.replace("Article ", "")},
    )

    if not current_docs:
        return f"{article_ref} not found in corpus."

    current_text = "\n".join([d.page_content for d in current_docs[:3]])

    llm = get_llm()
    diff_prompt = f"""You are a legal expert on EU AI Act. Analyze {article_ref} and explain 
what the most significant practical changes have been (simulating a {version_a} vs {version_b} comparison).

Current text of {article_ref}:
{current_text[:1500]}

Respond ONLY in English. Focus on practical business impact. Respond ONLY with valid JSON:
{{
  "changed": true,
  "summary": "what changed in practical terms (2 sentences max)",
  "impact": "high|medium|low",
  "affected_actors": ["providers", "deployers"],
  "action_required": "what a company must do because of this change",
  "effective_since": "{version_b}"
}}"""

    diff_raw = llm.invoke(diff_prompt).content

    try:
        clean = diff_raw.strip().strip("```json").strip("```").strip()
        diff = json.loads(clean)
    except json.JSONDecodeError:
        diff = {
            "changed": True,
            "summary": diff_raw[:300],
            "impact": "medium",
            "affected_actors": ["providers"],
            "action_required": "Review the full article",
            "effective_since": version_b,
        }

    impact_label = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}.get(
        diff.get("impact", "medium"), "MEDIUM"
    )

    return f"""
=== COMPARISON {article_ref}: {version_a} vs {version_b} ===

Changes detected: {'Yes' if diff.get('changed') else 'No'}
Impact level:     {impact_label}
Effective since:  {diff.get('effective_since', version_b)}

Summary:
{diff.get('summary', 'No information available')}

Affected actors: {', '.join(diff.get('affected_actors', []))}

Action required:
{diff.get('action_required', 'Review internal documentation')}
""".strip()


@tool
def generate_checklist(obligations_summary: str) -> str:
    """
    Generate an actionable compliance checklist and export it as a PDF.
    ALWAYS call this after check_obligations to give the user something tangible.
    The PDF is saved in the output/ folder of the project.

    Parameter:
      obligations_summary: obligations text (output from check_obligations)
    """
    llm = get_llm()
    company_name = "My Company"

    checklist_prompt = f"""Based on these EU AI Act obligations, create a practical compliance checklist.
Respond ONLY with a valid JSON array. No markdown, no explanation, just the JSON array.

Obligations:
{obligations_summary[:2000]}

Return a JSON array of 6-8 items:
[
  {{
    "id": 1,
    "title": "Short action title (max 8 words)",
    "description": "What exactly needs to be done (1-2 sentences)",
    "article": "Article X",
    "deadline": "Q1 2026",
    "priority": "critical|high|medium|low",
    "effort": "1-2 days | 1-2 weeks | 1-3 months"
  }}
]

Use ONLY simple ASCII characters. No special quotes, no em dashes, no unicode."""

    checklist_raw = llm.invoke(checklist_prompt).content

    try:
        clean = checklist_raw.strip().strip("```json").strip("```").strip()
        items = json.loads(clean)
    except json.JSONDecodeError:
        items = [
            {
                "id": 1,
                "title": "Implement risk management system",
                "description": "Establish a continuous risk identification and mitigation process before deployment.",
                "article": "Article 9",
                "deadline": "Q1 2026",
                "priority": "critical",
                "effort": "1-3 months",
            },
            {
                "id": 2,
                "title": "Training data governance",
                "description": "Audit and document datasets according to EU AI Act quality criteria.",
                "article": "Article 10",
                "deadline": "Q2 2026",
                "priority": "high",
                "effort": "1-2 weeks",
            },
            {
                "id": 3,
                "title": "Transparency documentation",
                "description": "Prepare technical documentation and instructions for use for deployers.",
                "article": "Article 13",
                "deadline": "Q2 2026",
                "priority": "high",
                "effort": "2-4 weeks",
            },
        ]

    # ── Clean non-latin characters for fpdf2 ──────────────────────────────────
    def safe(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        replacements = {
            "\u2014": "-", "\u2013": "-",
            "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"',
            "\u2026": "...",
        }
        for char, rep in replacements.items():
            text = text.replace(char, rep)
        # Remove any remaining non-latin-1 characters
        return text.encode("latin-1", errors="replace").decode("latin-1")

    # ── Build PDF ─────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"checklist_{timestamp}.pdf")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)

    W = pdf.w - 30  # effective width: 210 - 15*2 = 180mm

    def write_cell(text, style="", size=10, color=(0, 0, 0), height=6):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        text = safe(text)
        while pdf.get_string_width(text) > W and len(text) > 10:
            text = text[:-4] + "..."
        pdf.cell(W, height, text, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    def write_paragraph(text, size=9, color=(60, 60, 60)):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(*color)
        words = safe(text).split()
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            if pdf.get_string_width(test) > W:
                if line:
                    pdf.cell(W, 5, line, new_x="LMARGIN", new_y="NEXT")
                line = word
            else:
                line = test
        if line:
            pdf.cell(W, 5, line, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    # Header
    write_cell("EU AI Act - Compliance Checklist", style="B", size=14)
    write_cell(f"Company: {company_name}", size=9, color=(80, 80, 80))
    write_cell(f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}", size=9, color=(80, 80, 80))
    pdf.ln(4)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(15, pdf.get_y(), 15 + W, pdf.get_y())
    pdf.ln(4)

    priority_symbols = {"critical": "[!!!]", "high": "[!!]", "medium": "[!]", "low": "[  ]"}
    priority_colors  = {
        "critical": (180, 30, 30),
        "high":     (200, 100, 0),
        "medium":   (50, 100, 180),
        "low":      (80, 80, 80),
    }

    for idx, item in enumerate(items, 1):
        priority = item.get("priority", "medium")
        symbol   = priority_symbols.get(priority, "[ ]")
        color    = priority_colors.get(priority, (0, 0, 0))

        write_cell(f"{symbol} {item.get('title', 'No title')}", style="B", size=10, color=color)

        if item.get("description"):
            write_paragraph(item["description"])

        write_cell(
            f"Article: {item.get('article','?')}   Deadline: {item.get('deadline','?')}   Effort: {item.get('effort','?')}",
            style="I", size=8, color=(130, 130, 130)
        )
        pdf.ln(3)

        if idx < len(items):
            pdf.set_draw_color(220, 220, 220)
            pdf.line(15, pdf.get_y(), 15 + W, pdf.get_y())
            pdf.ln(3)

    # Footer
    pdf.set_y(-12)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(15, pdf.get_y(), 15 + W, pdf.get_y())
    pdf.ln(2)
    write_cell("Generated by regulAItions - For guidance only, not legal advice", style="I", size=7, color=(150, 150, 150))

    pdf.output(pdf_path)

    checklist_text = "\n".join([
        f"  {'[!!!]' if i.get('priority')=='critical' else '[!!]' if i.get('priority')=='high' else '[!]'} "
        f"{i.get('title')} — {i.get('article')} — {i.get('deadline')}"
        for i in items
    ])

    return f"""
=== CHECKLIST GENERATED ===
{checklist_text}

PDF saved to: {pdf_path}
Total items:    {len(items)}
Critical items: {sum(1 for i in items if i.get('priority') == 'critical')}
""".strip()


# ── Build agent ───────────────────────────────────────────────────────────────
def build_agent() -> AgentExecutor:
    llm = get_llm()
    tools = [search_regulation, get_article, check_obligations, compare_versions, generate_checklist]

    # Pull the standard ReAct prompt and inject our system instructions
    base_prompt = hub.pull("hwchase17/react")

    # Prepend system instructions to the prompt template
    from langchain.prompts import PromptTemplate

    system_instructions = """You are regulAItions, an expert AI compliance assistant specializing in EU AI Act and GDPR.

LANGUAGE RULE: Always respond in English, regardless of the language used in the question.

CITATION RULES — follow these strictly:
- Every legal claim MUST include the exact article number in parentheses, e.g. (Article 9), (Article 13), (Annex III, point 5b).
- When citing GDPR, prefix it clearly, e.g. (GDPR Article 4), to distinguish from EU AI Act articles with the same number.
- When listing obligations, format them as:
    * (Article 9) Risk management system — providers must establish...
    * (Article 10) Data governance — training datasets must...
- If you retrieved text from a tool, always mention which article it came from.
- If a question asks "what does Article X say", use get_article and quote the key points with the article number inline.
- Never state a legal requirement without citing its article number.

WORKFLOW RULES:
1. For questions about a specific article → use get_article directly.
   IMPORTANT: if the user mentions "GDPR" or "AI Act" explicitly, include
   that in the article_ref argument (e.g. "GDPR Article 4", "AI Act Article 9")
   so the correct regulation is searched — EU AI Act and GDPR both have
   articles with overlapping numbers (e.g. both have an Article 4 and
   Article 9 with completely different content).
2. For open questions about regulation → use search_regulation first.
3. When a user describes an AI system → use check_obligations, then automatically call generate_checklist.
4. For questions about recent changes → use compare_versions.
5. Never make up legal requirements — only cite what the tools returned.

RESPONSE FORMAT for obligation lists:
When listing multiple obligations, use this format:
  * (Article X) Title: brief explanation of what is required.
  * (Article Y) Title: brief explanation of what is required.

"""

    # Inject system instructions into the existing ReAct prompt
    original_template = base_prompt.template
    new_template = system_instructions + "\n" + original_template

    enhanced_prompt = PromptTemplate(
        input_variables=base_prompt.input_variables,
        template=new_template,
    )

    agent = create_react_agent(llm, tools, enhanced_prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        max_iterations=7,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ── Session with memory ───────────────────────────────────────────────────────
class AgentSession:
    def __init__(self):
        self.agent = build_agent()
        self.history = []
        self.state = {
            "current_system": None,
            "articles_discussed": [],
            "checklist_generated": False,
        }

    # Greetings and small talk — answer directly without invoking tools
    DIRECT_RESPONSES = {
        "hello":   "Hello! I'm regulAItions, your EU AI Act and GDPR compliance assistant. Ask me about your AI system's obligations, a specific article, or what changed recently in the regulation.",
        "hi":      "Hi! I'm regulAItions. How can I help you with EU AI Act or GDPR compliance today?",
        "hey":     "Hey! I'm regulAItions, your AI regulation assistant. What would you like to know?",
        "help":    "I can help you with:\n  - Classifying your AI system under the EU AI Act\n  - Listing your compliance obligations by article\n  - Generating a compliance checklist PDF\n  - Explaining specific articles (e.g. 'what does Article 9 say?')\n  - Comparing regulation versions\n\nJust describe your AI system or ask a question!",
        "what can you do": "I can classify AI systems under the EU AI Act, list obligations by article number, generate compliance checklists as PDF, explain specific articles, and track regulatory changes.",
        "thanks":  "You're welcome! Let me know if you have more questions about EU AI Act compliance.",
        "thank you": "You're welcome! Feel free to ask anything else about EU AI Act or GDPR.",
        "bye":     "Goodbye! Come back whenever you need EU AI Act compliance guidance.",
        "goodbye": "Goodbye! Good luck with your compliance work.",
    }

    def chat(self, user_message: str) -> dict:
        # Handle greetings directly — no need to invoke the agent
        normalized = user_message.strip().lower().rstrip("!.,?")
        if normalized in self.DIRECT_RESPONSES:
            answer = self.DIRECT_RESPONSES[normalized]
            self.history.append({
                "user": user_message,
                "assistant": answer,
                "tool_calls": [],
            })
            return {"answer": answer, "tool_calls": [], "session_state": self.state}

        context = ""
        if self.history:
            context = "Previous conversation:\n"
            for turn in self.history[-4:]:
                context += f"User: {turn['user']}\nAssistant: {turn['assistant'][:200]}...\n"
            context += "\n"

        if self.state["current_system"]:
            context += f"[Session: analyzing system — {self.state['current_system']}]\n"

        # Force English + article citations in every message
        full_input = (
            "[RULES: Respond ONLY in English. "
            "Cite the exact article number for every legal claim, e.g. (Article 9), (Annex III point 5b). "
            "Format obligations as bullet points starting with the article number. "
            "If the input is just a greeting or small talk, reply naturally without using any tool.]\n\n"
            + context
            + user_message
        )

        result = self.agent.invoke({"input": full_input})

        tool_calls = []
        for step in result.get("intermediate_steps", []):
            action, observation = step
            tool_calls.append({
                "tool": action.tool,
                "input": action.tool_input,
                "output": str(observation)[:300],
            })

        if "system" in user_message.lower() or "sistema" in user_message.lower():
            self.state["current_system"] = user_message[:100]

        self.history.append({
            "user": user_message,
            "assistant": result["output"],
            "tool_calls": tool_calls,
        })

        return {
            "answer": result["output"],
            "tool_calls": tool_calls,
            "session_state": self.state,
        }


# ── Interactive mode ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("regulAItions Agent — Interactive mode")
    print("Type 'exit' to quit\n")

    session = AgentSession()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit", "salir"):
            break
        if not user_input:
            continue

        result = session.chat(user_input)
        print(f"\nregulAItions: {result['answer']}\n")
        if result["tool_calls"]:
            print(f"[Tools used: {', '.join(t['tool'] for t in result['tool_calls'])}]\n")
