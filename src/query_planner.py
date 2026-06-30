from typing import Dict, List


COMPARISON_TRIGGERS = [
    "compare",
    "contrast",
    "difference",
    "differences",
    "similarity",
    "similarities",
    "versus",
    " vs ",
]

QUOTE_TRIGGERS = [
    "quote",
    "quotes",
    "quotation",
    "exact words",
    "what does the author say",
    "passage",
]

LIST_TRIGGERS = [
    "what experiments",
    "which experiments",
    "what studies",
    "which studies",
    "what examples",
    "which examples",
    "what methods",
    "which methods",
    "what evidence",
    "which evidence",
    "what criticisms",
    "which criticisms",
    "what objections",
    "which objections",
    "what reasons",
    "which reasons",
    "what does it reference",
    "what does the paper reference",
    "list",
    "mentioned",
    "mentions",
    "identify",
    "name",
]

DEFINITION_TRIGGERS = [
    "define",
    "definition",
    "meaning of",
    "what is",
    "what are",
]

POSITION_TRIGGERS = [
    "approach",
    "view",
    "opinion",
    "position",
    "argument",
    "central argument",
    "main argument",
    "thesis",
    "claim",
    "claims",
    "what does",
    "how does",
    "why does",
    "explain",
    "describe",
    "summarise",
    "summarize",
    "account of",
    "theory",
    "believe",
    "believes",
    "think",
    "thinks",
    "according to",
]


def contains_any(text: str, triggers: List[str]) -> bool:
    return any(trigger in text for trigger in triggers)


def classify_question(question: str) -> Dict:
    q = f" {question.lower().strip()} "

    if contains_any(q, COMPARISON_TRIGGERS):
        question_type = "comparison"
    elif contains_any(q, QUOTE_TRIGGERS):
        question_type = "quote"
    elif contains_any(q, LIST_TRIGGERS):
        question_type = "list"
    elif contains_any(q, DEFINITION_TRIGGERS):
        question_type = "definition"
    elif contains_any(q, POSITION_TRIGGERS):
        question_type = "paper_level_argument"
    else:
        question_type = "standard"

    return {
        "question_type": question_type,
        "needs_broad_retrieval": question_type in {
            "paper_level_argument",
            "list",
            "comparison",
        },
    }


def build_retrieval_queries(question: str) -> List[str]:
    plan = classify_question(question)
    question_type = plan["question_type"]

    if question_type == "paper_level_argument":
        return [
            question,
            f"{question} thesis central argument main claim conclusion",
            f"{question} author argues author claims author rejects author criticises author concludes",
            f"{question} objection response problem solution view position",
            f"{question} Chalmers hard problem chimera illusion easy problems",
        ]

    if question_type == "list":
        return [
            question,
            f"{question} experiment experiments study studies",
            f"{question} researchers subjects participants task tested results findings",
            f"{question} example examples evidence references mentioned",
            "experiment study subjects participants tested found results",
        ]

    if question_type == "definition":
        return [
            question,
            f"{question} definition means refers to called known as",
        ]

    if question_type == "quote":
        return [
            question,
            f"{question} quote passage exact words says writes states",
        ]

    if question_type == "comparison":
        return [
            question,
            f"{question} compare contrast difference similarity",
            f"{question} position view argument objection response",
        ]

    return [
        question,
        f"{question} main point relevant passage evidence",
    ]