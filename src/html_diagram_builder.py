import html


def build_html_diagram(diagram_data: dict):
    """Build a simple diagram from structured nodes and edges."""

    if not isinstance(diagram_data, dict):
        return None

    nodes = diagram_data.get("nodes", [])
    edges = diagram_data.get("edges", [])

    if not isinstance(nodes, list) or not nodes:
        return None

    if not isinstance(edges, list):
        return None

    labels = {}

    for node in nodes:
        if not isinstance(node, dict):
            return None

        node_id = node.get("id")
        label = node.get("label")

        if (
            not isinstance(node_id, str)
            or not node_id
            or not isinstance(label, str)
            or not label.strip()
            or node_id in labels
        ):
            return None

        labels[node_id] = html.escape(label)

    parts = [
        '<div style="font-family:Arial,sans-serif;'
        'display:flex;flex-direction:column;gap:16px;padding:12px;">'
    ]

    for edge in edges:
        if not isinstance(edge, dict):
            return None

        source = edge.get("from")
        target = edge.get("to")
        relation = edge.get("label", "")

        if source not in labels or target not in labels:
            return None

        if not isinstance(relation, str):
            return None

        parts.append(
            '<div style="display:flex;align-items:center;'
            'justify-content:center;gap:12px;flex-wrap:wrap;">'
            f'<div style="padding:14px;border:1px solid #94a3b8;'
            f'border-radius:8px;text-align:center;">{labels[source]}</div>'
            '<div style="text-align:center;">'
            f'<div style="font-size:12px;">{html.escape(relation)}</div>'
            '<div style="font-size:24px;">→</div>'
            '</div>'
            f'<div style="padding:14px;border:1px solid #94a3b8;'
            f'border-radius:8px;text-align:center;">{labels[target]}</div>'
            '</div>'
        )

    parts.append("</div>")

    return "\n".join(parts)