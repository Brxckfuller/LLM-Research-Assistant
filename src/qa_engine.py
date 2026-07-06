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
- Preserve page numbers or URLs.
- If one passage contains both a general claim and a specific number, extract both.

QUESTION:
{question}

PASSAGES:
{context}

Return ONLY this format:

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