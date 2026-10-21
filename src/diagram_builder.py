
import json
import re


def extract_diagram_data(answer: str):
    """Separate structured diagram data from the written answer."""

    pattern = (
        r"(?:^|\n)#{1,3}\s*Explanatory diagram\s*\n"
        r"\s*```json\s*\n"
        r"(.*?)"
        r"\n\s*```"
    )

    match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)

    if not match:
        return answer, None

    clean_answer = (
        answer[:match.start()] + answer[match.end():]
    ).strip()

    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return clean_answer, None

    if not isinstance(data, dict):
        return clean_answer, None

    return clean_answer, data


def build_mermaid(diagram_data: dict):
    """Build a Mermaid flowchart from validated diagram data."""

    if not isinstance(diagram_data, dict):
        return None

    nodes = diagram_data.get("nodes")
    edges = diagram_data.get("edges")

    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None

    if not (3 <= len(nodes) <= 10):
        return None

    if not (1 <= len(edges) <= 16):
        return None

    valid_nodes = {}
    lines = ["flowchart LR"]

    for node in nodes:
        if not isinstance(node, dict):
            return None

        node_id = node.get("id")
        label = node.get("label")

        if not isinstance(node_id, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", node_id
        ):
            return None

        if (
            node_id.lower() in {"end", "graph", "flowchart"}
            or node_id in valid_nodes
            or not isinstance(label, str)
            or not label.strip()
            or len(label) > 75
        ):
            return None

        valid_nodes[node_id] = label

        safe_label = (
            label.replace("&", "and")
            .replace('"', "'")
            .replace("<", "")
            .replace(">", "")
            .replace("\n", " ")
        )

        lines.append(f'    {node_id}["{safe_label}"]')

    for edge in edges:
        if not isinstance(edge, dict):
            return None

        source = edge.get("from")
        target = edge.get("to")
        label = edge.get("label", "")

        if (
            source not in valid_nodes
            or target not in valid_nodes
            or not isinstance(label, str)
            or len(label) > 60
        ):
            return None

        safe_label = re.sub(
            r"[|<>\n\r]",
            " ",
            label.replace('"', "'"),
        ).strip()

        if safe_label:
            lines.append(
                f"    {source} -->|{safe_label}| {target}"
            )
        else:
            lines.append(f"    {source} --> {target}")

    return "\n".join(lines)
