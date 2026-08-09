from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class NodeDefinition:
    id: str
    role: str
    dependencies: tuple[str, ...] = ()
    model: str | None = None
    model_selector: str | None = None
    fan_out: bool = False
    input_mapping: tuple[str, ...] = ("query", "dependencies")
    system_prompt: str = ""
    max_tokens: int = 2048
    timeout_seconds: float = 30.0
    retries: int = 0
    fallback_models: tuple[str, ...] = ()
    estimated_cost_usd: float = 0.0
    output_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopDefinition:
    verifier_node: str
    max_iterations: int = 0
    repair_model_selector: str = "primary_label"


@dataclass(frozen=True)
class GraphDefinition:
    id: str
    version: str
    nodes: tuple[NodeDefinition, ...]
    final_node: str
    max_steps: int = 12
    loop: LoopDefinition | None = None


def load_graphs(path: Path) -> dict[str, GraphDefinition]:
    if path.is_dir():
        files = sorted([*path.glob("*.json"), *path.glob("*.yaml"), *path.glob("*.yml")])
    else:
        files = [path]
    if not files:
        raise ValueError(f"no graph definitions found in {path}")

    graphs: dict[str, GraphDefinition] = {}
    for file_path in files:
        raw = _load_document(file_path)
        graph = _parse_graph(raw, file_path)
        if graph.id in graphs:
            raise ValueError(f"duplicate graph id: {graph.id}")
        graphs[graph.id] = graph
    return graphs


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    raw = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"graph definition must be an object: {path}")
    return raw


def _parse_graph(raw: dict[str, Any], path: Path) -> GraphDefinition:
    node_values = raw.get("nodes")
    if isinstance(node_values, dict):
        node_values = [{"id": node_id, **value} for node_id, value in node_values.items()]
    if not isinstance(node_values, list) or not node_values:
        raise ValueError(f"graph nodes must be a non-empty array or object: {path}")

    nodes = tuple(
        NodeDefinition(
            id=str(value["id"]),
            role=str(value["role"]),
            dependencies=tuple(value.get("dependencies", ())),
            model=value.get("model"),
            model_selector=value.get("model_selector"),
            fan_out=bool(value.get("fan_out", False)),
            input_mapping=tuple(value.get("input_mapping", ("query", "dependencies"))),
            system_prompt=str(value.get("system_prompt", "")),
            max_tokens=int(value.get("max_tokens", 2048)),
            timeout_seconds=float(value.get("timeout_seconds", 30.0)),
            retries=int(value.get("retries", 0)),
            fallback_models=tuple(value.get("fallback_models", ())),
            estimated_cost_usd=float(value.get("estimated_cost_usd", 0.0)),
            output_schema=dict(value.get("output_schema", {})),
        )
        for value in node_values
    )
    ids = {node.id for node in nodes}
    if len(ids) != len(nodes):
        raise ValueError(f"node ids must be unique: {path}")
    for node in nodes:
        unknown = set(node.dependencies) - ids
        if unknown:
            raise ValueError(f"node {node.id} has unknown dependencies: {sorted(unknown)}")
        if node.model and node.model_selector:
            raise ValueError(f"node {node.id} cannot set both model and model_selector")
        if node.model_selector not in {None, "primary_label", "selected_labels"}:
            raise ValueError(f"node {node.id} has unknown model_selector: {node.model_selector}")
        if node.role not in {"clarifier", "policy"} and not (node.model or node.model_selector):
            raise ValueError(f"node {node.id} requires model or model_selector")
        unknown_inputs = set(node.input_mapping) - {"query", "dependencies"}
        if unknown_inputs:
            raise ValueError(f"node {node.id} has unknown input mappings: {sorted(unknown_inputs)}")

    final_node = str(raw.get("final_node", nodes[-1].id))
    if final_node not in ids:
        raise ValueError(f"final_node is not defined: {final_node}")
    _validate_acyclic(nodes)

    loop_value = raw.get("loop")
    loop = None
    if loop_value:
        loop = LoopDefinition(
            verifier_node=str(loop_value["verifier_node"]),
            max_iterations=int(loop_value.get("max_iterations", 0)),
            repair_model_selector=str(loop_value.get("repair_model_selector", "primary_label")),
        )
        if loop.verifier_node not in ids:
            raise ValueError(f"loop verifier_node is not defined: {loop.verifier_node}")
    return GraphDefinition(
        id=str(raw["id"]),
        version=str(raw.get("version", "0.1.0")),
        nodes=nodes,
        final_node=final_node,
        max_steps=int(raw.get("max_steps", 12)),
        loop=loop,
    )


def _validate_acyclic(nodes: tuple[NodeDefinition, ...]) -> None:
    pending = {node.id: set(node.dependencies) for node in nodes}
    resolved: set[str] = set()
    while pending:
        ready = [node_id for node_id, deps in pending.items() if deps <= resolved]
        if not ready:
            raise ValueError("graph contains a dependency cycle")
        resolved.update(ready)
        for node_id in ready:
            pending.pop(node_id)
