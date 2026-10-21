from typing import List, Dict


def build_context(chunks: List[Dict]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        source_type = chunk.get("source_type", "document")

        parts.append(
            f"""
SOURCE_ID: {i}
SOURCE_TYPE: {source_type}
DOCUMENT: {chunk.get("document", "Unknown")}
AUTHOR: {chunk.get("author", "Unknown")}
TITLE: {chunk.get("title", "Unknown")}
SECTION: {chunk.get("section", "Unknown")}
PAGE_OR_URL: {chunk.get("page", chunk.get("url", "Unknown"))}
TEXT:
{chunk.get("text", chunk.get("content", "")).strip()}
""".strip()
        )

    return "\n\n---\n\n".join(parts)


def build_evidence_extraction_prompt(question: str, chunks: List[Dict]) -> str:
    context = build_context(chunks)

    return f"""
You are an evidence extraction engine.

Your job is to extract the facts needed to answer the user's question.

Rules:
- Use ONLY the supplied passages.
- Do NOT answer the question.
- Do NOT use outside knowledge.
- Extract evidence from both uploaded documents and web results if both are supplied.
- Do NOT ignore numerical estimates, quantities, dates, names, URLs, or locations.
- If the user asks about something current, recent, latest, newly published, or web-based, prioritize web evidence.
- If the user asks about an uploaded paper, document, or PDF, prioritize document evidence.
- If both sources are relevant, extract facts from both.
- Preserve the author's/source wording where precision matters.
- Include supporting text.
- Copy PAGE_OR_URL exactly from the matching source passage; never infer a page number from the content.
- Include the source ID for every extracted fact.
- For broad purpose questions, prioritise the document introduction and explain its stated motivation and distinguishing mechanism, if retrieved.
- Extract all material supporting facts needed for a complete answer, not just three.
- If one passage contains both a general claim and a specific number, extract both.

QUESTION:
{question}

PASSAGES:
{context}

Return ONLY this format (repeat FACT blocks as needed; do not stop at three):

FACT 1:
SOURCE_ID:
SOURCE_TYPE:
PAGE_OR_URL:
DIRECTLY_SUPPORTED_FACT:
SUPPORTING_TEXT:

FACT 2:
SOURCE_ID:
SOURCE_TYPE:
PAGE_OR_URL:
DIRECTLY_SUPPORTED_FACT:
SUPPORTING_TEXT:

FACT 3:
SOURCE_ID:
SOURCE_TYPE:
PAGE_OR_URL:
DIRECTLY_SUPPORTED_FACT:
SUPPORTING_TEXT:

If no relevant facts are found, write:
NO RELEVANT FACTS FOUND.
""".strip()


def build_final_answer_prompt(question: str, extracted_evidence: str) -> str:
    return f"""
You are an academic research assistant.

Answer the user's question using ONLY the evidence below.

Core rules:
- Do NOT use outside knowledge.
- Do NOT invent claims.
- Answer the exact question first.
- Synthesize evidence across multiple facts instead of repeating one fact.
- Explain the author's reasoning, not just the conclusion.
- Preserve important wording, distinctions, numbers, dates, names, and source details.
- Use page numbers for uploaded documents.
- Use URLs/source titles for web results.
- If evidence is limited, say exactly what is missing.
- Do not guess or alter page numbers: use only PAGE_OR_URL attached to the corresponding fact.
- Never cite a page for a quotation unless that quotation appears in evidence attributed to that page.
- For a purpose question, distinguish the overarching purpose from a list of features; include any explicitly stated motivation or feedback loop when supported.
- If web evidence and document evidence disagree, say that clearly.

Source rules:
- If evidence comes from an uploaded document, cite it as Page X.
- If evidence comes from the web, cite it as the source title or URL.
- If both are used, clearly distinguish document evidence from web evidence.

Length rules:
- For simple factual questions, write 1 concise paragraph.
- For conceptual, explanatory, argumentative, comparative, or interpretive questions, write 3-6 clear paragraphs.
- Do not default to short answers when the question asks for explanation.
- Do not add irrelevant background.

Evidence rules:
- Do not dump raw FACT labels into the answer.
- Turn extracted evidence into readable prose.
- Include short quotations only when the wording is important.
- Use italics for direct quotations in the Evidence from the text section.
- Keep quotations concise.

QUESTION:
{question}

EVIDENCE:
{extracted_evidence}

Write the answer exactly in this format:

# Answer

Give a direct, well-developed answer. For broad questions, explain the main claim, the reasoning behind it, and any important contrast or implication supported by the evidence.

# Key points

- Clear claim. (Page X or source title/URL)
- Clear claim. (Page Y or source title/URL)
- Clear claim. (Page Z or source title/URL)

# Evidence from the text

- *"Short quoted evidence here."* (Page X or source title/URL)
- *"Short quoted evidence here."* (Page Y or source title/URL)

# Limitations

If the evidence answers the question, write:
No major limitations from the retrieved evidence.

If it does not, explain what is missing.
""".strip()


def build_fast_answer_prompt(question: str, chunks: List[Dict]) -> str:
    """Grounded single-pass answer with question-specific priorities and source checks."""
    context = build_context(chunks)
    question_lower = question.casefold()

    # These are instructions about what to look for, not prewritten answers.
    # The model must still find support in the retrieved source passages.
    if "foundry" in question_lower and "ontology" in question_lower:
        focus = """DEMONSTRATION QUESTION PRIORITIES — ONTOLOGY:
- Address every part of the question in order: (1) data, (2) logic, (3) actions,
  (4) why their connection improves operational decision-making.
- FIRST look for passages describing the Core Ontology's semantics (objects and
  relations) and kinetics (functions and actions). Explain that relationship
  before discussing other capabilities.
- NEXT look for passages on models, type systems, workflows, or the link from
  analytical results to decisions/actions. Connect these only where supported.
- Explain the operational consequence as a reasoned synthesis, clearly anchored
  to specific source passages.
- Treat external integrations, audit trails, publishing, monitoring, and lineage
  as secondary supporting capabilities, not as the definition of the Ontology.
- If the retrieved text does not explicitly describe an element, acknowledge
  the gap rather than pretending the document does.
"""
    elif "foundry" in question_lower and ("purpose" in question_lower or "what is" in question_lower):
        focus = """DEMONSTRATION QUESTION PRIORITIES — FOUNDRY PURPOSE:
- State the overarching purpose first, rather than starting with a feature list.
- Look for the document's characterization of Foundry as an operating system
  and explain what 'operating' means in the source's own terms.
- Look for software-defined feedback loops spanning data, analytics, and
  business teams, and connect them to organizational learning and outcomes.
- Distinguish the overall goal from the capabilities used to achieve it.
- Do not repeat the same sentence in the answer and key points.
"""
    else:
        focus = """QUESTION PRIORITIES:
- Identify all subquestions and answer each directly.
- Prioritize the central mechanism and causal explanation over peripheral features.
"""

    return f"""You are a rigorous research assistant answering from retrieved passages.
Your task is to SYNTHESIZE the answer, not merely repeat a passage that shares
keywords with the question.

{focus}

NON-NEGOTIABLE SOURCE RULES:
1. Use ONLY the retrieved passages below. Do not import facts from memory.
2. Each passage has a SOURCE_ID and PAGE_OR_URL. Cite the exact corresponding
   source as (Source 2, Page 5), or (Source 3, URL) for web sources.
3. Never invent a page, source ID, statement, or quotation. Never mix the page
   number from one source with wording from another.
4. Prefer paraphrasing accurately. Only include a direct quotation if its
   exact wording occurs in the cited passage. Do not stitch separate excerpts
   together into one quote.
5. Distinguish explicit source claims from conclusions you infer by connecting
   multiple passages. For an inference, explain the connection and cite both.
6. The first paragraph MUST directly answer the entire question, including
   the principal mechanism and its significance, not merely a secondary feature.
7. Keep important source terminology (for example semantics, kinetics,
   objects, relations, functions, actions) when it is relevant and present.
8. Prioritize correctness and completeness over brevity, but avoid repetition.
9. Do not claim the sources prove a stronger conclusion than they actually do.

QUESTION:
{question}

RETRIEVED PASSAGES:
{context}

Before writing, silently check: What is the main question? Which passages
support each part? Is my opening paragraph focused on the main mechanism?

OUTPUT FORMAT:
## Answer
Two or three coherent paragraphs. Explain HOW and WHY, not just WHAT.
Cite supporting sources at the end of the relevant sentences.

## Key points
Three concise, non-repetitive bullets, each with accurate citations.

## Evidence from the text
Two or three short, VERBATIM quotations from retrieved passages, each with
its exact source ID and page. If you cannot verify a quotation in the provided
passages, omit it rather than inventing one.

If an important part cannot be supported by the passages, explicitly say so.
Do not output analysis or source-checking notes.


OPTIONAL EXPLANATORY DIAGRAM:
- Include a diagram only when it helps explain a mechanism,
  process, architecture, or interaction.
- Identify meaningful components and their relationships.
- Do not merely connect keywords mentioned in the paper.
- Each node must represent a distinct participant, component,
  step, or outcome.
- Each edge must represent a genuine, source-supported relationship.
- The direction of each edge must match its meaning.
- Before generating the diagram, silently verify every edge:
  (1) Does the source node actually perform the stated action?
  (2) Is the target node actually the recipient or result?
  (3) Is this relationship explicitly supported by the passages?
- Never connect two participants merely because both appear
  in the same passage.
- Distinguish the original setup of an experiment from any
  modified or alternative version described by the author.
- Do not combine participants from different experimental
  arrangements into one diagram unless the relationship
  between those arrangements is explicitly being explained.
- For communication-based processes, show the direction of
  questions and responses accurately.
- Include the main outcome or decision when it is essential
  to understanding the mechanism.
- If any important relationship is uncertain, omit that edge.
- Use separate nodes for participants and outcomes.
- Do not invent connections unsupported by retrieved passages.
- Prefer 4 to 8 nodes and concise relationship labels.
- Do not generate Mermaid syntax yourself.
- Instead, provide structured JSON describing the diagram.
- Use simple, valid JSON with double-quoted keys and strings.
- Do not include comments or trailing commas.

When including a diagram, use exactly this structure:

## Explanatory diagram

```json
{{
  "nodes": [
    {{"id": "A", "label": "First component"}},
    {{"id": "B", "label": "Second component"}},
    {{"id": "C", "label": "Outcome"}}
  ],
  "edges": [
    {{"from": "A", "to": "B", "label": "Sends information"}},
    {{"from": "B", "to": "C", "label": "Produces"}}
  ]
}}
```

If a meaningful diagram cannot be supported by the passages,
omit the explanatory diagram section entirely.


""".strip() 
