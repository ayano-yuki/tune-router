from __future__ import annotations

import re
from typing import Any

from constants import LABELS
from graphs import GraphDefinition, NodeDefinition
from models import Budget


GENERATED_GRAPH_ID = "bounded_graph"
ALLOWED_NODE_ROLES = {"specialist", "verifier", "repair", "synthesizer", "clarifier", "policy"}
LOCAL_NODE_ROLES = {"clarifier", "policy"}
MAX_GENERATED_NODES = 8
NODE_ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")


def validate_bounded_graph_plan(
    plan: dict[str, Any],
    *,
    allowed_models: set[str],
    catalog_by_alias: dict[str, dict[str, Any]],
    risk_level: str,
    budget: Budget,
) -> dict[str, Any]:
    if plan.get("plan_type") != GENERATED_GRAPH_ID:
        raise ValueError("bounded graph plan must set plan_type to bounded_graph")
    raw_nodes = plan.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("bounded graph plan requires a non-empty nodes array")
    max_nodes = min(MAX_GENERATED_NODES, max(1, budget.max_steps))
    if len(raw_nodes) > max_nodes:
        raise ValueError(f"bounded graph has too many nodes: {len(raw_nodes)} > {max_nodes}")

    nodes = [_normalize_node(raw, allowed_models, catalog_by_alias) for raw in raw_nodes]
    ids = [node["id"] for node in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError("bounded graph node ids must be unique")
    id_set = set(ids)
    for node in nodes:
        unknown = set(node["dependencies"]) - id_set
        if unknown:
            raise ValueError(f"node {node['id']} has unknown dependencies: {sorted(unknown)}")
    _validate_acyclic(nodes)

    final_node = str(plan.get("final_node") or ids[-1])
    if final_node not in id_set:
        raise ValueError(f"bounded graph final_node is not defined: {final_node}")
    if risk_level == "high":
        verifier_ids = {node["id"] for node in nodes if node["role"] == "verifier"}
        if not verifier_ids:
            raise ValueError("high-risk bounded graph requires a verifier node")
        final_ancestors = _ancestors(final_node, nodes)
        if not (verifier_ids & (final_ancestors | {final_node})):
            raise ValueError("high-risk bounded graph final output must depend on verifier evidence")

    return {
        "plan_type": GENERATED_GRAPH_ID,
        "version": "generated-v1",
        "max_steps": max(1, min(int(plan.get("max_steps", len(nodes))), budget.max_steps, MAX_GENERATED_NODES)),
        "nodes": nodes,
        "final_node": final_node,
    }


def graph_from_bounded_plan(plan: dict[str, Any]) -> GraphDefinition:
    return GraphDefinition(
        id=GENERATED_GRAPH_ID,
        version=str(plan.get("version", "generated-v1")),
        nodes=tuple(
            NodeDefinition(
                id=str(node["id"]),
                role=str(node["role"]),
                dependencies=tuple(node.get("dependencies", ())),
                model=node.get("model"),
                input_mapping=tuple(node.get("input_mapping", ("query", "dependencies"))),
                system_prompt=_system_prompt(str(node["role"]), str(node.get("objective", ""))),
                max_tokens=int(node.get("max_tokens", 2048)),
                timeout_seconds=float(node.get("timeout_seconds", 30.0)),
                retries=int(node.get("retries", 0)),
                fallback_models=tuple(node.get("fallback_models", ())),
                output_schema=dict(node.get("output_schema", {})),
            )
            for node in plan["nodes"]
        ),
        final_node=str(plan["final_node"]),
        max_steps=int(plan.get("max_steps", len(plan["nodes"]))),
    )


def _normalize_node(
    raw: Any,
    allowed_models: set[str],
    catalog_by_alias: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("bounded graph nodes must be objects")
    node_id = str(raw.get("id", "")).strip()
    if not NODE_ID_RE.fullmatch(node_id):
        raise ValueError(f"invalid bounded graph node id: {node_id}")
    role = str(raw.get("role", "")).strip()
    if role not in ALLOWED_NODE_ROLES:
        raise ValueError(f"node {node_id} has unsupported role: {role}")
    dependencies = _string_list(raw.get("dependencies", ()), f"node {node_id} dependencies")
    label = str(raw.get("label", "")).strip()
    if label and label not in LABELS:
        raise ValueError(f"node {node_id} has unknown label: {label}")

    model = raw.get("model")
    if role in LOCAL_NODE_ROLES:
        model = None
    else:
        model = str(model or "").strip()
        if not model:
            raise ValueError(f"node {node_id} requires model")
        if model not in allowed_models:
            raise ValueError(f"node {node_id} model is not allowed: {model}")
        if label and not _supports_label(catalog_by_alias.get(model), label):
            raise ValueError(f"node {node_id} model {model} does not support label {label}")

    output_schema = dict(raw.get("output_schema", {}))
    if role == "verifier" and not output_schema:
        output_schema = {"type": "object", "required": ["passed", "issues"]}
    return {
        "id": node_id,
        "role": role,
        "label": label or None,
        "model": model,
        "dependencies": dependencies,
        "objective": str(raw.get("objective", "")).strip(),
        "input_mapping": tuple(raw.get("input_mapping", ("query", "dependencies"))),
        "max_tokens": int(raw.get("max_tokens", 1024 if role == "verifier" else 2048)),
        "timeout_seconds": float(raw.get("timeout_seconds", 20.0 if role == "verifier" else 30.0)),
        "retries": int(raw.get("retries", 0)),
        "fallback_models": _string_list(raw.get("fallback_models", ()), f"node {node_id} fallback_models"),
        "output_schema": output_schema,
    }


def _validate_acyclic(nodes: list[dict[str, Any]]) -> None:
    pending = {node["id"]: set(node["dependencies"]) for node in nodes}
    resolved: set[str] = set()
    while pending:
        ready = [node_id for node_id, deps in pending.items() if deps <= resolved]
        if not ready:
            raise ValueError("bounded graph contains a dependency cycle")
        resolved.update(ready)
        for node_id in ready:
            pending.pop(node_id)


def _ancestors(node_id: str, nodes: list[dict[str, Any]]) -> set[str]:
    by_id = {node["id"]: node for node in nodes}
    seen: set[str] = set()
    stack = list(by_id[node_id]["dependencies"])
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(by_id[current]["dependencies"])
    return seen


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list | tuple):
        raise ValueError(f"{field} must be an array")
    return [str(item) for item in value]


def _supports_label(catalog_item: dict[str, Any] | None, label: str) -> bool:
    if catalog_item is None:
        return True
    domains = catalog_item.get("domains")
    if not domains:
        return True
    return "*" in domains or label in domains


def _system_prompt(role: str, objective: str) -> str:
    defaults = {
        "specialist": "You are a domain specialist. Give a correct, actionable, and safe answer.",
        "verifier": "You are a verifier. Return only JSON: {\"passed\": boolean, \"issues\": [string]}.",
        "repair": "You are a repair agent. Correct the candidate using all feedback.",
        "synthesizer": "You are a synthesizer. Reconcile expert outputs into a prioritized answer.",
    }
    base = defaults.get(role, "Answer safely.")
    return f"{base}\n\nDelegated objective: {objective}" if objective else base
