from typing import List, Dict


def build_context(chunks: List[Dict]) -> str:
    parts = []

    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"""
SOURCE_ID: {i}
DOCUMENT: {chunk.get("document", "Unknown")}
AUTHOR: {chunk.get("author", "Unknown")}
TITLE: {chunk.get("title", "Unknown")}
SECTION: {chunk.get("section", "Unknown")}
PAGE: {chunk.get("page", "Unknown")}
TEXT:
{chunk.get("text", "").strip()}
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
- Do NOT ignore numerical estimates, quantities, dates, names, or locations.
- If the user asks about an "estimated amount", "number", "how many", or "quantity",
  extract every directly relevant number or estimate.
- If the user's wording is slightly imprecise, extract the closest directly supported facts.
- Preserve the author's wording where precision matters.
- Include supporting text.
- Preserve page numbers.
- If one passage contains both a general claim and a specific number, extract both.

QUESTION:
{question}

PASSAGES:
{context}

Return ONLY this format:

FACT 1:
SOURCE_ID:
PAGE:
DIRECTLY_SUPPORTED_FACT:
SUPPORTING_TEXT:

FACT 2:
SOURCE_ID:
PAGE:
DIRECTLY_SUPPORTED_FACT:
SUPPORTING_TEXT:

FACT 3:
SOURCE_ID:
PAGE:
DIRECTLY_SUPPORTED_FACT:
SUPPORTING_TEXT:

If no relevant facts are found, write:
NO RELEVANT FACTS FOUND.
""".strip()


def build_final_answer_prompt(question: str, extracted_evidence: str) -> str:
    return f"""
You are an academic research assistant.

Answer the question using ONLY the atomic facts below.

Core rules:
- Do NOT use outside knowledge.
- Do NOT invent claims.
- Use page numbers.
- Preserve precise wording, numbers, dates, names, and distinctions where they matter.
- Answer the user's exact question first.
- If the evidence directly answers the question, do not say the evidence is insufficient.
- If evidence is limited, clearly say what can and cannot be concluded.

Length rules:
- If the question asks for a simple fact, date, number, definition, author, title, or yes/no answer, give a concise answer.
- If the question asks "why", "how", "explain", "compare", "what is the argument", "what is the evidence", "what are the limitations", or asks about a broad concept, give a detailed answer.
- For detailed answers, use multiple paragraphs and explain the reasoning clearly.
- Do not make every answer short by default.
- Do not make every answer long by default.
- Let the complexity of the question determine the length.

QUESTION:
{question}

ATOMIC FACTS:
{extracted_evidence}

Write the answer in this format:

# Answer

Answer the question directly.

For simple factual questions, write 1 concise paragraph.

For broader conceptual or explanatory questions, write 3-6 paragraphs that:
- define the key concept
- explain the author's argument
- distinguish important terms or contrasts
- connect the retrieved evidence into a coherent explanation
- cite relevant pages

# Key points

- Claim here. (Page X)
- Claim here. (Page Y)
- Claim here. (Page Z)

# Evidence from the text

Include 2-5 pieces of supporting evidence if the question is broad or explanatory.
For simple factual questions, include only 1-2 pieces of evidence.

# Limitations

Only mention limitations if the retrieved evidence genuinely cannot answer the user's intended question.
If the evidence answers the question, write:
No major limitations from the retrieved evidence.
""".strip()