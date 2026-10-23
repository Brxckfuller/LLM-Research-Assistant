import re
from enum import Enum


class AnswerClassification(str, Enum):
    ANSWERED = "ANSWERED"
    ABSTAINED = "ABSTAINED"
    MIXED = "MIXED"


ABSTENTION_PATTERNS = (
    r"\bdo(?:es)? not provide information\b",
    r"\bno (?:direct )?evidence\b",
    r"\bnot (?:mentioned|provided|specified)\b",
    r"\bcannot (?:answer|determine)\b",
    r"\bnot (?:directly )?supported\b",
    r"\binsufficient information\b",
    r"\b(?:do|does) not contain enough information\b",
    r"\bnot in the (?:provided )?(?:document|text|passages)\b",
)

ANSWER_AFTER_ABSTENTION = re.compile(
    r"\b(?:but|however|nevertheless)\b"
    r"[^.!?\n]{0,200}"
    r"\b(?:the answer is|it is|it's|the capital is)\b",
    re.IGNORECASE,
)


def classify_answer(answer: str) -> AnswerClassification:
    if not isinstance(answer, str):
        raise TypeError("answer must be a string")

    text = " ".join(answer.split())

    has_abstention = any(
        re.search(pattern, text, re.IGNORECASE)
        for pattern in ABSTENTION_PATTERNS
    )

    if not has_abstention:
        return AnswerClassification.ANSWERED

    if ANSWER_AFTER_ABSTENTION.search(text):
        return AnswerClassification.MIXED

    return AnswerClassification.ABSTAINED
