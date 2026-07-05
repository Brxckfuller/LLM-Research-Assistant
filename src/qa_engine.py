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

Answer the user's question using ONLY the evidence below.

Core rules:
- Do NOT use outside knowledge.
- Do NOT invent claims.
- Answer the exact question first.
- Synthesize evidence across multiple facts instead of repeating one fact.
- Explain the author's reasoning, not just the conclusion.
- Preserve important wording, distinctions, numbers, dates, and names.
- Use page numbers naturally.
- If evidence is limited, say exactly what is missing.

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

- Clear claim. (Page X)
- Clear claim. (Page Y)
- Clear claim. (Page Z)

# Evidence from the text

- *"Short quoted evidence here."* (Page X)
- *"Short quoted evidence here."* (Page Y)

# Limitations

If the evidence answers the question, write:
No major limitations from the retrieved evidence.

If it does not, explain what is missing.
""".strip() 