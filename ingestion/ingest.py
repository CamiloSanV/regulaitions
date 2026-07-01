"""
ingestion/ingest.py
───────────────────
Robust ingestion pipeline for EU AI Act and GDPR.

Handles all structural edge cases:
  - Articles with numbered sub-clauses (1. ... (a) (b) (c))
  - Annexes (Annex I, Annex III, Annex IV) — different format from articles
  - Definitions (Article 3) — 60+ items, chunked individually
  - Cross-reference extraction — which articles does each chunk mention
  - Header/footer contamination from PDF pages
  - Recitals (numbered considerations before the articles)

Steps:
  1. Download PDF from EUR-Lex
  2. Clean page text (remove headers/footers)
  3. Segment into structural blocks (recital / article / annex / definition)
  4. Chunk each block respecting sub-clause boundaries
  5. Embed with BGE-M3 (free, local)
  6. Upload to Supabase pgvector

Usage:
  python -m ingestion.ingest --source eu_ai_act
  python -m ingestion.ingest --source gdpr
  python -m ingestion.ingest --source all
"""

import re
import json
from typing import Optional
import argparse
from pathlib import Path
from datetime import datetime

import httpx
import fitz  # PyMuPDF
from tqdm import tqdm
from rich.console import Console
from dotenv import load_dotenv
import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.schema import Document
from supabase import create_client

load_dotenv()
console = Console()

# ── Sources ────────────────────────────────────────────────────────────────────
SOURCES = {
    "eu_ai_act": {
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
        "name": "EU AI Act",
        "short": "EU_AI_Act",
        "effective_date": "2024-08-01",
        "version": "2024-Q3",
        "definitions_article": 3,
    },
    "gdpr": {
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
        "name": "GDPR",
        "short": "GDPR",
        "effective_date": "2018-05-25",
        "version": "2018-Q2",
        "definitions_article": 4,
    },
}

DATA_DIR = Path("data/pdfs")
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Patterns ───────────────────────────────────────────────────────────────────
# Article heading: "Article 9" on its own line (with optional title on next line)
RE_ARTICLE      = re.compile(r"^Article\s+(\d+)\s*$", re.IGNORECASE)
# Chapter heading: "CHAPTER I", "CHAPTER II", etc.
RE_CHAPTER      = re.compile(r"^CHAPTER\s+([IVX]+)\s*$")
# Section heading: "SECTION 1", "SECTION 2"
RE_SECTION      = re.compile(r"^SECTION\s+(\d+)\s*$")
# Annex heading: "ANNEX I", "ANNEX III", "Annex IV"
RE_ANNEX        = re.compile(r"^ANNEX\s+([IVX\d]+)\s*$", re.IGNORECASE)
# Numbered paragraph: "1.", "2.", "3." at start of line
RE_NUMBERED_PAR = re.compile(r"^(\d+)\.\s+(.+)")
# Lettered sub-clause: "(a)", "(b)", "(c)" at start of line
RE_LETTERED_SUB = re.compile(r"^\(([a-z])\)\s+(.+)")
# Roman numeral sub-clause: "(i)", "(ii)", "(iii)"
RE_ROMAN_SUB    = re.compile(r"^\((i{1,3}|iv|v|vi{0,3}|ix|x)\)\s+(.+)", re.IGNORECASE)
# Recital: "(1) Text..." OR "(1)" alone on a line (PDF often splits number from text)
RE_RECITAL      = re.compile(r"^\((\d+)\)\s+(.+)")
RE_RECITAL_NUM_ONLY = re.compile(r"^\((\d+)\)\s*$")
# Cross-reference to another article
RE_CROSS_REF    = re.compile(r"Article\s+(\d+)", re.IGNORECASE)
# Definition entry: "'term' means..." OR isolated "(1)" before a definition line
RE_DEFINITION   = re.compile(r"^'([^']+)'\s+means\s+", re.IGNORECASE)
RE_DEF_NUM_ONLY = re.compile(r"^\((\d+)\)\s*$")

# Lines to discard as PDF noise (headers/footers)
NOISE_PATTERNS = [
    re.compile(r"^L\s+\d+/\d+"),           # "L 1689/12" page references
    re.compile(r"^Official Journal"),        # "Official Journal of the EU"
    re.compile(r"^\d+\.\s*\d+\.\s*\d{4}"), # dates like "12. 7. 2024"
    re.compile(r"^EN$"),                     # language marker
    re.compile(r"^EUR-Lex"),
    re.compile(r"^ELI:"),
]


# ── 1. Download ────────────────────────────────────────────────────────────────
def download_pdf(source_key: str) -> Path:
    source = SOURCES[source_key]
    out_path = DATA_DIR / f"{source_key}.pdf"

    if out_path.exists():
        console.print(f"[green]✓[/green] PDF already exists: {out_path}")
        return out_path

    console.print(f"[blue]↓[/blue] Downloading {source['name']}...")
    with httpx.stream("GET", source["url"], follow_redirects=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk_bytes in r.iter_bytes(chunk_size=8192):
                f.write(chunk_bytes)

    console.print(f"[green]✓[/green] Downloaded: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


# ── 2. PDF text extraction with noise removal ──────────────────────────────────
def extract_clean_text(pdf_path: Path) -> list[str]:
    """
    Extract all lines from the PDF, removing header/footer noise.
    Returns a flat list of clean lines across all pages.
    """
    doc = fitz.open(str(pdf_path))
    all_lines = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        lines = page_text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip noise lines
            if any(p.match(line) for p in NOISE_PATTERNS):
                continue
            # Skip very short isolated lines that are likely page numbers
            if re.match(r"^\d+$", line) and len(line) <= 4:
                continue
            all_lines.append(line)

    doc.close()
    console.print(f"  Extracted {len(all_lines)} clean lines from {page_num + 1} pages")
    return all_lines


# ── 3. Structural segmentation ─────────────────────────────────────────────────
def segment_document(lines: list[str], definitions_article: Optional[int] = None) -> list[dict]:
    """
    Segment a flat list of lines into structural blocks:
      - recital:    numbered considerations (1), (2), ...
      - article:    Article N with its sub-clauses
      - annex:      Annex I, III, IV etc.
      - definition: the article containing defined terms

    Each block is a dict with:
      type, number, title, lines, chapter, section

    Parameters:
      definitions_article: explicit article number that holds the
        definitions (e.g. 3 for EU AI Act, 4 for GDPR). If None,
        auto-detected by checking whether the article's title contains
        "Definitions" — works for any regulation without hardcoding.
    """
    blocks = []
    current_chapter = None
    current_section = None
    current_block = None

    def flush():
        nonlocal current_block
        if current_block and current_block["lines"]:
            blocks.append(current_block)
        current_block = None

    i = 0
    in_recitals = True  # recitals come before the articles

    while i < len(lines):
        line = lines[i]

        # ── Chapter ──────────────────────────────────────────────────────────
        if RE_CHAPTER.match(line):
            current_chapter = line.strip()
            in_recitals = False
            i += 1
            continue

        # ── Section ──────────────────────────────────────────────────────────
        if RE_SECTION.match(line):
            current_section = line.strip()
            i += 1
            continue

        # ── Annex ────────────────────────────────────────────────────────────
        if RE_ANNEX.match(line):
            flush()
            annex_match = RE_ANNEX.match(line)
            # Peek next line for annex title
            title = lines[i + 1].strip() if i + 1 < len(lines) else ""
            current_block = {
                "type": "annex",
                "number": annex_match.group(1),
                "title": title,
                "chapter": current_chapter,
                "section": current_section,
                "lines": [],
            }
            in_recitals = False
            i += 1
            continue

        # ── Article ──────────────────────────────────────────────────────────
        if RE_ARTICLE.match(line):
            flush()
            art_match = RE_ARTICLE.match(line)
            art_num = int(art_match.group(1))
            # Peek next line — often the article title
            title = ""
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Title lines are usually Title Case and don't start with a number
                if next_line and not RE_NUMBERED_PAR.match(next_line) and not RE_ARTICLE.match(next_line):
                    title = next_line
                    i += 1  # consume the title line

            # Detect the definitions article either by explicit parameter
            # or by checking if the title contains "Definitions" — this
            # works across different regulations without hardcoding the
            # article number (EU AI Act: Article 3, GDPR: Article 4, etc.)
            is_definitions = (
                (definitions_article is not None and art_num == definitions_article)
                or "definition" in title.lower()
            )
            block_type = "definition" if is_definitions else "article"
            current_block = {
                "type": block_type,
                "number": art_num,
                "title": title,
                "chapter": current_chapter,
                "section": current_section,
                "lines": [],
            }
            in_recitals = False
            i += 1
            continue

        # ── Recital ──────────────────────────────────────────────────────────
        if in_recitals:
            # Case 1: "(1) Text on same line"
            rec_match = RE_RECITAL.match(line)
            if rec_match:
                flush()
                current_block = {
                    "type": "recital",
                    "number": int(rec_match.group(1)),
                    "title": "",
                    "chapter": "Recitals",
                    "section": None,
                    "lines": [rec_match.group(2)],
                }
                i += 1
                continue

            # Case 2: "(1)" alone on its own line, text follows on next line(s)
            num_only_match = RE_RECITAL_NUM_ONLY.match(line)
            if num_only_match:
                flush()
                current_block = {
                    "type": "recital",
                    "number": int(num_only_match.group(1)),
                    "title": "",
                    "chapter": "Recitals",
                    "section": None,
                    "lines": [],
                }
                i += 1
                continue

        # ── Accumulate into current block ─────────────────────────────────
        if current_block is not None:
            current_block["lines"].append(line)

        i += 1

    flush()
    return blocks


# ── 4. Sub-clause aware chunking ───────────────────────────────────────────────
def parse_subclauses(lines: list[str]) -> list[dict]:
    """
    Parse article lines into a structured tree of paragraphs and sub-clauses.

    Returns list of paragraph dicts:
      {
        "num": "1",           # paragraph number (or None for intro text)
        "text": "...",        # paragraph text
        "subclauses": [
          {"key": "a", "text": "..."},
          {"key": "b", "text": "..."},
        ]
      }
    """
    paragraphs = []
    current_para = None
    current_sub = None
    intro_lines = []

    for line in lines:
        # Skip the article label we inserted
        if line.startswith("[Article"):
            continue

        num_match = RE_NUMBERED_PAR.match(line)
        letter_match = RE_LETTERED_SUB.match(line)
        roman_match = RE_ROMAN_SUB.match(line)

        if num_match:
            # Start new numbered paragraph
            if current_para:
                if current_sub:
                    current_para["subclauses"].append(current_sub)
                    current_sub = None
                paragraphs.append(current_para)
            current_para = {
                "num": num_match.group(1),
                "text": num_match.group(2),
                "subclauses": [],
            }
        elif letter_match and current_para:
            # Sub-clause (a), (b), (c) under current paragraph
            if current_sub:
                current_para["subclauses"].append(current_sub)
            current_sub = {"key": letter_match.group(1), "text": letter_match.group(2)}
        elif roman_match and current_sub:
            # Sub-sub-clause (i), (ii) under current sub-clause
            current_sub["text"] += f" ({roman_match.group(1)}) {roman_match.group(2)}"
        else:
            # Continuation text
            if current_sub:
                current_sub["text"] += " " + line
            elif current_para:
                current_para["text"] += " " + line
            else:
                intro_lines.append(line)

    # Flush
    if current_sub and current_para:
        current_para["subclauses"].append(current_sub)
    if current_para:
        paragraphs.append(current_para)

    # Add intro text as paragraph 0 if exists
    if intro_lines:
        paragraphs.insert(0, {"num": None, "text": " ".join(intro_lines), "subclauses": []})

    return paragraphs


def paragraphs_to_chunks(
    paragraphs: list[dict],
    article_title: str,
    max_words: int = 350,
) -> list[str]:
    """
    Convert parsed paragraphs into text chunks, keeping sub-clauses with
    their parent paragraph for context.

    Strategy:
      - Each numbered paragraph + its sub-clauses = one unit
      - Group units into chunks until max_words is reached
      - Overlap: carry last paragraph into next chunk
    """
    units = []

    for para in paragraphs:
        lines_out = []
        if para["num"]:
            lines_out.append(f"{para['num']}. {para['text']}")
        else:
            lines_out.append(para["text"])

        for sub in para.get("subclauses", []):
            lines_out.append(f"  ({sub['key']}) {sub['text']}")

        unit_text = "\n".join(lines_out)
        units.append(unit_text)

    if not units:
        return []

    chunks = []
    current = []
    current_words = 0

    for unit in units:
        unit_words = len(unit.split())
        if current_words + unit_words > max_words and current:
            chunk_text = f"[{article_title}]\n" + "\n\n".join(current)
            chunks.append(chunk_text)
            # Overlap: keep last unit for context
            current = [current[-1], unit]
            current_words = len(current[-1].split()) + unit_words
        else:
            current.append(unit)
            current_words += unit_words

    if current:
        chunk_text = f"[{article_title}]\n" + "\n\n".join(current)
        chunks.append(chunk_text)

    return chunks


def chunk_definitions(lines: list[str], article_title: str) -> list[str]:
    """
    Special chunking for Article 3 (Definitions).
    Each definition gets its own chunk to maximize retrieval precision.
    Groups definitions into small chunks of ~3 definitions each.

    Strategy: join all lines into one continuous string, strip the bare
    "(N)" numbering markers (which PyMuPDF often splits onto their own
    line), then split on every occurrence of "'term' means" — this is
    far more robust than trying to track state line by line, since the
    PDF layout for numbers vs. text is inconsistent.
    """
    # Join everything, removing bare number markers like "(1)" "(23)" etc.
    full_text = " ".join(lines)
    full_text = re.sub(r"\(\d+\)\s*", " ", full_text)
    full_text = re.sub(r"\s+", " ", full_text).strip()

    # Split right before every "'term' means" occurrence
    # Using a lookahead so the delimiter itself stays with the following text
    pattern = re.compile(r"(?=['\u2018]([^'\u2018\u2019]{1,80})['\u2019]\s+means\s)")
    parts = pattern.split(full_text)

    # First part (before the first definition) is usually just intro text — drop if short
    definitions = [p.strip() for p in parts if p.strip() and "means" in p.lower()]

    # Group into chunks of 3 definitions
    chunks = []
    for i in range(0, len(definitions), 3):
        group = definitions[i:i+3]
        chunk = f"[{article_title} — Definitions]\n" + "\n\n".join(group)
        chunks.append(chunk)

    return chunks if chunks else [f"[{article_title}]\n" + full_text]



def chunk_annex(lines: list[str], annex_number: str, annex_title: str) -> list[str]:
    """
    Special chunking for Annexes (I, III, IV, etc.).
    Annexes are lists — split on numbered or lettered top-level items.
    """
    label = f"Annex {annex_number}"
    if annex_title:
        label += f" — {annex_title}"

    # Split on numbered top-level items: "1.", "2.", etc.
    chunks = []
    current = []

    for line in lines:
        if RE_NUMBERED_PAR.match(line) and current:
            chunk = f"[{label}]\n" + "\n".join(current)
            if len(chunk.split()) > 20:
                chunks.append(chunk)
            current = [line]
        else:
            current.append(line)

    if current:
        chunk = f"[{label}]\n" + "\n".join(current)
        if len(chunk.split()) > 20:
            chunks.append(chunk)

    # If annex had no numbered items, return as single chunk
    if not chunks:
        chunks = [f"[{label}]\n" + " ".join(lines)]

    return chunks


# ── 5. Extract cross-references ────────────────────────────────────────────────
def extract_cross_refs(text: str) -> list[str]:
    """Extract all article numbers referenced in a chunk of text."""
    refs = RE_CROSS_REF.findall(text)
    return list(set(refs))  # deduplicate


# ── 6. Infer which actors a chunk applies to ───────────────────────────────────
def infer_applies_to(text: str) -> list[str]:
    text_lower = text.lower()
    actors = []
    if any(w in text_lower for w in ["provider", "developer", "places on the market", "makes available"]):
        actors.append("providers")
    if any(w in text_lower for w in ["deployer", "user of ai", "puts into service"]):
        actors.append("deployers")
    if any(w in text_lower for w in ["importer"]):
        actors.append("importers")
    if any(w in text_lower for w in ["distributor"]):
        actors.append("distributors")
    if any(w in text_lower for w in ["high-risk", "high risk", "annex iii"]):
        actors.append("high_risk")
    if any(w in text_lower for w in ["general-purpose", "gpai", "general purpose ai"]):
        actors.append("gpai")
    if any(w in text_lower for w in ["prohibited", "shall not be placed", "unacceptable risk"]):
        actors.append("prohibited")
    if not actors:
        actors.append("general")
    return actors


# ── 7. Convert blocks to LangChain Documents ───────────────────────────────────
def blocks_to_documents(blocks: list[dict], source_key: str) -> list[Document]:
    """
    Convert structural blocks into LangChain Documents with rich metadata.
    Dispatches to the right chunking strategy per block type.
    """
    source = SOURCES[source_key]
    documents = []

    for block in blocks:
        btype  = block["type"]
        bnum   = block["number"]
        btitle = block.get("title", "")
        lines  = block["lines"]

        # Build human-readable label
        if btype in ("article", "definition"):
            label = f"Article {bnum}"
            if btitle:
                label += f" — {btitle}"
        elif btype == "annex":
            label = f"Annex {bnum}"
            if btitle:
                label += f" — {btitle}"
        elif btype == "recital":
            label = f"Recital {bnum}"
        else:
            label = f"{btype} {bnum}"

        # Choose chunking strategy
        if btype == "recital":
            # Recitals are already one block — use as-is
            raw_chunks = [" ".join(lines)]

        elif btype == "definition":
            # Article 3: chunk each definition individually
            raw_chunks = chunk_definitions(lines, label)

        elif btype == "annex":
            # Annexes: split on numbered items
            raw_chunks = chunk_annex(lines, str(bnum), btitle)

        else:
            # Regular articles: parse sub-clauses then chunk
            paragraphs = parse_subclauses(lines)
            raw_chunks = paragraphs_to_chunks(paragraphs, label, max_words=350)

        # Build Documents
        for idx, chunk_text in enumerate(raw_chunks):
            if len(chunk_text.strip()) < 40:
                continue

            cross_refs = extract_cross_refs(chunk_text)
            applies_to = infer_applies_to(chunk_text)

            metadata = {
                # Identity
                "source":        source["short"],
                "source_name":   source["name"],
                # "article" is ONLY set for actual articles/definitions —
                # recitals and annexes use a different number space and
                # would otherwise collide (e.g. Recital 9 vs Article 9)
                "article":       str(bnum) if btype in ("article", "definition") else "",
                "block_number":  str(bnum),  # raw number regardless of type, for debugging
                "article_label": label,
                "chapter":       block.get("chapter") or "Unknown",
                "section":       block.get("section") or "",
                "chunk_type":    btype,
                "chunk_index":   idx,
                "chunk_id":      f"{source['short']}:{btype}_{bnum}:chunk_{idx}",
                # Versioning
                "version":       source["version"],
                "effective_date": source["effective_date"],
                "ingested_at":   datetime.now().isoformat(),
                # Semantic metadata
                "applies_to":    applies_to,
                "cross_refs":    cross_refs,
                "has_subclauses": any(RE_LETTERED_SUB.search(l) for l in lines),
                "word_count":    len(chunk_text.split()),
            }

            documents.append(Document(page_content=chunk_text.strip(), metadata=metadata))

    return documents


# ── 8. Supabase setup ─────────────────────────────────────────────────────────
def setup_vectorstore(embeddings) -> SupabaseVectorStore:
    client = create_client(
        os.getenv("SUPABASE_URL"),
        os.getenv("SUPABASE_KEY"),
    )
    return SupabaseVectorStore(
        client=client,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents",
    )


# ── 9. Main pipeline ───────────────────────────────────────────────────────────
def ingest(source_key: str) -> int:
    console.rule(f"[bold]Ingesting: {SOURCES[source_key]['name']}[/bold]")

    # Step 1: Download
    pdf_path = download_pdf(source_key)

    # Step 2: Extract clean text
    console.print("[blue]→[/blue] Extracting and cleaning text...")
    lines = extract_clean_text(pdf_path)

    # Step 3: Segment into structural blocks
    console.print("[blue]→[/blue] Segmenting document structure...")
    def_article = SOURCES[source_key].get("definitions_article")
    blocks = segment_document(lines, definitions_article=def_article)

    # Stats per block type
    type_counts = {}
    for b in blocks:
        type_counts[b["type"]] = type_counts.get(b["type"], 0) + 1
    console.print(f"  [green]✓[/green] {len(blocks)} blocks: {type_counts}")

    # Step 4: Convert to Documents
    console.print("[blue]→[/blue] Chunking with sub-clause awareness...")
    all_documents = blocks_to_documents(blocks, source_key)

    # Stats
    chunk_type_counts = {}
    for d in all_documents:
        ct = d.metadata["chunk_type"]
        chunk_type_counts[ct] = chunk_type_counts.get(ct, 0) + 1

    console.print(f"  [green]✓[/green] {len(all_documents)} chunks generated")
    console.print(f"  Breakdown: {chunk_type_counts}")

    # Show example
    if all_documents:
        ex = all_documents[0]
        console.print(f"\n  [dim]Example chunk:[/dim]")
        console.print(f"  label:      {ex.metadata['article_label']}")
        console.print(f"  type:       {ex.metadata['chunk_type']}")
        console.print(f"  cross_refs: {ex.metadata['cross_refs']}")
        console.print(f"  applies_to: {ex.metadata['applies_to']}")
        console.print(f"  text:       {ex.page_content[:120]}...")

    # Step 5: Load embeddings
    console.print("\n[blue]→[/blue] Loading BGE-M3 embeddings (first run downloads ~1.5GB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 16},
    )

    # Step 6: Upload to Supabase in batches (with retry on transient network errors)
    console.print("\n[blue]→[/blue] Uploading to Supabase pgvector...")
    vectorstore = setup_vectorstore(embeddings)

    batch_size = 30  # smaller batches reduce chance of mid-upload SSL drops
    batches = [all_documents[i:i+batch_size] for i in range(0, len(all_documents), batch_size)]

    import time

    failed_batches = []
    for i, batch in enumerate(tqdm(batches, desc="Uploading")):
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                vectorstore.add_documents(batch)
                break
            except Exception as e:
                if attempt == max_retries:
                    console.print(f"\n[red]✗[/red] Batch {i+1} failed after {max_retries} attempts: {e}")
                    failed_batches.append(i)
                else:
                    wait = 2 ** attempt  # 2s, 4s, 8s
                    console.print(f"\n[yellow]⚠[/yellow] Batch {i+1} failed (attempt {attempt}/{max_retries}), retrying in {wait}s...")
                    time.sleep(wait)

    if failed_batches:
        console.print(f"\n[yellow]⚠ Warning:[/yellow] {len(failed_batches)} batch(es) failed permanently: {failed_batches}")
        console.print("  Re-run the ingestion command — already-uploaded chunks will upsert cleanly.")
    else:
        console.print(f"\n[bold green]✓ Done:[/bold green] {len(all_documents)} chunks in Supabase")

    console.print(f"  Source:  {SOURCES[source_key]['name']}")
    console.print(f"  Version: {SOURCES[source_key]['version']}")

    return len(all_documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regulatory document ingestion")
    parser.add_argument(
        "--source",
        choices=["eu_ai_act", "gdpr", "all"],
        default="eu_ai_act",
    )
    args = parser.parse_args()

    sources = list(SOURCES.keys()) if args.source == "all" else [args.source]
    total = sum(ingest(src) for src in sources)
    console.rule(f"[bold green]Total: {total} chunks indexed[/bold green]")
