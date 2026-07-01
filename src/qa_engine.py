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

Rules:
- Do NOT use outside knowledge.
- Do NOT invent claims.
- Do NOT ignore numerical estimates if they appear in the evidence.
- Do NOT say the evidence is insufficient if it contains a close supported answer.
- If the user's wording is slightly imprecise, answer using the closest supported evidence.
- Preserve the paper's wording when precision matters.
- If the evidence gives an estimate, say it is an estimate.
- If the evidence distinguishes between "sent", "deployed", "training", or "on the ground",
  preserve that distinction.
- Use page numbers.
- Key points must be Markdown dot points, not a table.
- Do not use inline bullet symbols.

Additional answering rules:
- Answer the user's exact question before providing background or context.
- If the question asks for numerical values, percentages, dates, quantities, or comparisons, begin the answer with those values.
- Prefer the paper's own wording whenever it directly answers the question.
- Do not replace precise figures with vague summaries such as "significantly increased" or "many".
- Do not introduce related statistics unless they directly answer the user's question.
- Keep different measurements separate, such as government spending, military expenditure, GDP, defence budgets, and security spending, unless the paper explicitly compares them.
- If multiple pieces of retrieved evidence refer to different metrics, clearly distinguish them instead of combining them.
- If the paper contains a sentence that directly answers the question, closely follow that sentence before adding explanation.

QUESTION:
{question}

ATOMIC FACTS:
{extracted_evidence}

Write the answer exactly like this:

# Answer

One short paragraph answering the question directly. If the evidence supports a more detailed answer, write 2-3 concise paragraphs.

# Key points

- Claim here. (Page X)
- Claim here. (Page Y)
- Claim here. (Page Z)

# Limitations

Only mention limitations if the retrieved evidence genuinely cannot answer the user's intended question.
If the evidence answers the question, write:
No major limitations from the retrieved evidence.
""".strip()