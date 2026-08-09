from __future__ import annotations

import json
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from clients import ModelClient
from constants import LABEL_TO_MODEL
from graphs import GraphDefinition, NodeDefinition
from models import Budget, ExecutionResult, ModelResponse, NodeTrace, RouteDecision, RouterSignal


class BudgetExceeded(RuntimeError):
    pass


class _BudgetTracker:
    def __init__(self, budget: Budget, graph_max_steps: int) -> None:
        self.budget = replace(budget, max_steps=min(budget.max_steps, graph_max_steps))
        self.started = time.perf_counter()
        self.steps = 0
        self.cost_usd = 0.0
        self.reserved_cost_usd = 0.0
        self._lock = threading.Lock()

    def reserve(self, estimated_cost_usd: float) -> None:
        with self._lock:
            if self.elapsed_ms >= self.budget.max_latency_ms:
                raise BudgetExceeded("latency budget exceeded")
            if self.steps + 1 > self.budget.max_steps:
                raise BudgetExceeded("step budget exceeded")
            projected = self.cost_usd + self.reserved_cost_usd + estimated_cost_usd
            if projected > self.budget.max_cost_usd:
                raise BudgetExceeded("cost budget exceeded")
            self.steps += 1
            self.reserved_cost_usd += estimated_cost_usd

    def complete(self, estimated_cost_usd: float, actual_cost_usd: float) -> None:
        with self._lock:
            self.reserved_cost_usd = max(0.0, self.reserved_cost_usd - estimated_cost_usd)
            self.cost_usd += actual_cost_usd

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def stop_reason(self) -> str | None:
        if self.elapsed_ms >= self.budget.max_latency_ms:
            return "latency_budget_exceeded"
        if self.cost_usd > self.budget.max_cost_usd:
            return "cost_budget_exceeded"
        return None


class GraphExecutor:
    def __init__(self, model_client: ModelClient, max_workers: int = 4) -> None:
        self.model_client = model_client
        self.max_workers = max_workers

    def execute(
        self,
        text: str,
        signal: RouterSignal,
        decision: RouteDecision,
        graph: GraphDefinition,
        budget: Budget | None = None,
    ) -> ExecutionResult:
        if graph.id != decision.graph_id:
            raise ValueError(f"selected graph {decision.graph_id} does not match definition {graph.id}")
        tracker = _BudgetTracker(budget or Budget(), graph.max_steps)
        materialized, expanded_ids = self._materialize(graph, decision)
        outputs: dict[str, str] = {}
        traces: list[NodeTrace] = []
        pending = {node.id: node for node in materialized}
        failure_type: str | None = None
        stop_reason = "completed"

        while pending:
            ready = [
                node for node in pending.values()
                if all(dependency in outputs for dependency in node.dependencies)
            ]
            if not ready:
                failure_type = "graph_execution"
                stop_reason = "unresolved_dependencies"
                break
            ready.sort(key=lambda node: node.id)
            group = self._execute_group(ready, text, outputs, tracker)
            for node, node_trace, response in group:
                traces.append(node_trace)
                pending.pop(node.id, None)
                if response is not None:
                    outputs[node.id] = response.content
                else:
                    failure_type = "budget" if node_trace.error and "budget" in node_trace.error else "model_execution"
                    stop_reason = "budget_exceeded" if failure_type == "budget" else "node_failed"
            post_call_budget_stop = tracker.stop_reason()
            if post_call_budget_stop:
                failure_type = "budget"
                stop_reason = post_call_budget_stop
            if failure_type:
                break

        final_node_ids = expanded_ids.get(graph.final_node, [graph.final_node])
        final_answer = next((outputs[node_id] for node_id in final_node_ids if node_id in outputs), "")

        if graph.loop and graph.loop.verifier_node in expanded_ids:
            verifier_id = expanded_ids[graph.loop.verifier_node][0]
            verifier_output = outputs.get(verifier_id)
            if verifier_output is not None:
                final_answer, loop_traces, loop_stop = self._run_repair_loop(
                    text=text,
                    decision=decision,
                    graph=graph,
                    candidate=final_answer,
                    verifier_output=verifier_output,
                    tracker=tracker,
                )
                traces.extend(loop_traces)
                stop_reason = loop_stop
                if loop_stop == "budget_exceeded":
                    failure_type = "budget"

        if graph.id == "clarify_first":
            stop_reason = "clarification_requested"
        elif graph.id == "safe_refusal_or_handoff":
            stop_reason = "policy_stop"

        failure_detail = _failure_detail(traces)
        if not final_answer:
            final_answer = self._degraded_answer(outputs, decision, stop_reason, failure_detail)

        trace = {
            "trace_id": f"orch-{datetime.now(UTC).strftime('%Y%m%d')}-{uuid.uuid4().hex[:12]}",
            "created_at": _utc_now(),
            "input": {"text": redact_secrets(text)},
            "router": {
                "model": signal.model,
                "latency_ms": round(signal.latency_ms, 3),
                "scores": {label: round(score, 8) for label, score in signal.scores.items()},
                "primary_labels": list(decision.primary_labels),
                "secondary_labels": list(decision.secondary_labels),
                "confidence": round(decision.confidence, 8),
                "margin": round(decision.margin, 8),
            },
            "policy": {
                "risk_level": decision.policy.risk_level,
                "action": decision.policy.action,
                "reason": decision.policy.reason,
            },
            "graph": {
                "id": graph.id,
                "version": graph.version,
                "selection_reason": decision.reason,
                "selector_type": decision.selector_type,
                "fallback_reason": decision.fallback_reason,
                "delegations": [
                    {"label": item.label, "model": item.model, "objective": item.objective}
                    for item in decision.delegations
                ],
                "synthesis_strategy": decision.synthesis_strategy,
                "generated_graph": decision.generated_graph,
                "selection_metadata": decision.selection_metadata,
                "stop_reason": stop_reason,
            },
            "nodes": [item.to_dict() for item in traces],
            "usage": {
                "steps": tracker.steps,
                "cost_usd": round(tracker.cost_usd, 8),
                "latency_ms": round(tracker.elapsed_ms + signal.latency_ms, 3),
            },
            "evaluation": {
                "user_rating": None,
                "review_label": None,
                "failure_type": failure_type,
                "failure_detail": failure_detail,
            },
            "final_answer": redact_secrets(final_answer),
        }
        return ExecutionResult(trace=trace, final_answer=final_answer)

    def _execute_group(
        self,
        nodes: list[NodeDefinition],
        text: str,
        outputs: dict[str, str],
        tracker: _BudgetTracker,
    ) -> list[tuple[NodeDefinition, NodeTrace, ModelResponse | None]]:
        if len(nodes) == 1:
            node = nodes[0]
            try:
                trace, response = self._execute_node(node, text, outputs, tracker)
            except Exception as exc:
                now = _utc_now()
                trace = NodeTrace(node.id, node.role, node.model, "failed", now, now, 0.0, error=str(exc))
                response = None
            return [(node, trace, response)]

        results: dict[str, tuple[NodeTrace, ModelResponse | None]] = {}
        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(nodes))) as pool:
            futures = {
                pool.submit(self._execute_node, node, text, dict(outputs), tracker): node
                for node in nodes
            }
            for future in as_completed(futures):
                node = futures[future]
                try:
                    results[node.id] = future.result()
                except Exception as exc:  # Defensive: preserve a trace even for executor bugs.
                    now = _utc_now()
                    results[node.id] = (
                        NodeTrace(node.id, node.role, node.model, "failed", now, now, 0.0, error=str(exc)),
                        None,
                    )
        return [(node, *results[node.id]) for node in nodes]

    def _execute_node(
        self,
        node: NodeDefinition,
        text: str,
        outputs: dict[str, str],
        tracker: _BudgetTracker,
    ) -> tuple[NodeTrace, ModelResponse | None]:
        started_iso = _utc_now()
        started = time.perf_counter()
        if node.role in {"clarifier", "policy"}:
            content = self._local_response(node.role, text)
            response = ModelResponse(content=content, model="local-policy")
            return (
                NodeTrace(
                    id=node.id,
                    role=node.role,
                    model=response.model,
                    status="completed",
                    started_at=started_iso,
                    completed_at=_utc_now(),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    output=redact_secrets(content),
                ),
                response,
            )

        if not node.model:
            raise ValueError(f"materialized node has no model: {node.id}")
        aliases = [node.model] * (node.retries + 1) + list(node.fallback_models)
        errors: list[str] = []
        response: ModelResponse | None = None
        attempted_model: str | None = None
        attempts = 0
        for alias in aliases:
            attempted_model = alias
            attempts += 1
            try:
                tracker.reserve(node.estimated_cost_usd)
            except BudgetExceeded as exc:
                errors.append(str(exc))
                break
            try:
                response = self.model_client.complete(
                    alias,
                    self._messages(node, text, outputs),
                    node.max_tokens,
                    node.timeout_seconds,
                )
            except Exception as exc:
                errors.append(f"{alias}: {exc}")
                tracker.complete(node.estimated_cost_usd, 0.0)
                continue
            tracker.complete(node.estimated_cost_usd, response.cost_usd)
            try:
                _validate_output_schema(response.content, node.output_schema)
            except ValueError as exc:
                errors.append(f"{alias}: invalid output: {exc}")
                response = None
                continue
            break

        latency = (time.perf_counter() - started) * 1000
        if response is None:
            return (
                NodeTrace(
                    id=node.id,
                    role=node.role,
                    model=attempted_model,
                    status="failed",
                    started_at=started_iso,
                    completed_at=_utc_now(),
                    latency_ms=latency,
                    attempts=attempts,
                    error="; ".join(errors),
                ),
                None,
            )
        return (
            NodeTrace(
                id=node.id,
                role=node.role,
                model=response.model,
                status="completed",
                started_at=started_iso,
                completed_at=_utc_now(),
                latency_ms=latency,
                attempts=attempts,
                cost_usd=response.cost_usd,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                output=redact_secrets(response.content),
                error="; ".join(errors) or None,
            ),
            response,
        )

    def _run_repair_loop(
        self,
        text: str,
        decision: RouteDecision,
        graph: GraphDefinition,
        candidate: str,
        verifier_output: str,
        tracker: _BudgetTracker,
    ) -> tuple[str, list[NodeTrace], str]:
        if _verifier_passed(verifier_output):
            return candidate, [], "verifier_passed"
        traces: list[NodeTrace] = []
        verifier_definition = next(node for node in graph.nodes if node.id == graph.loop.verifier_node)
        for iteration in range(1, graph.loop.max_iterations + 1):
            repair = NodeDefinition(
                id=f"repair_{iteration}",
                role="repair",
                model=LABEL_TO_MODEL[decision.primary_labels[0]],
                system_prompt="You are a repair agent. Correct the answer using the verifier feedback.",
                max_tokens=2048,
                dependencies=("candidate", "verifier_feedback"),
            )
            values = {"candidate": candidate, "verifier_feedback": verifier_output}
            repair_trace, repair_response = self._execute_node(repair, text, values, tracker)
            traces.append(repair_trace)
            if repair_response is None:
                reason = "budget_exceeded" if repair_trace.error and "budget" in repair_trace.error else "repair_failed"
                return candidate, traces, reason
            candidate = repair_response.content

            verify = replace(
                verifier_definition,
                id=f"verifier_{iteration + 1}",
                dependencies=("candidate",),
            )
            verify_trace, verify_response = self._execute_node(verify, text, {"candidate": candidate}, tracker)
            traces.append(verify_trace)
            if verify_response is None:
                reason = "budget_exceeded" if verify_trace.error and "budget" in verify_trace.error else "verification_failed"
                return candidate, traces, reason
            verifier_output = verify_response.content
            if _verifier_passed(verifier_output):
                return candidate, traces, "repaired_and_verified"
        return candidate, traces, "max_repair_iterations"

    @staticmethod
    def _messages(node: NodeDefinition, text: str, outputs: dict[str, str]) -> list[dict[str, str]]:
        defaults = {
            "specialist": "You are a domain specialist. Give a correct, actionable, and safe answer.",
            "verifier": (
                "You are a verifier. Check correctness, evidence, actionability, and safety. "
                "Return only JSON: {\"passed\": boolean, \"issues\": [string]}."
            ),
            "synthesizer": (
                "You are a synthesizer. Merge expert outputs, remove duplication, identify conflicts, "
                "and provide a prioritized answer with risks and follow-up questions."
            ),
            "repair": "You are a repair agent. Correct the candidate using all feedback.",
        }
        messages = [{"role": "system", "content": node.system_prompt or defaults.get(node.role, "Answer safely.")}]
        context = "\n\n".join(f"[{key}]\n{value}" for key, value in outputs.items() if key in node.dependencies)
        user = f"User request:\n{text}" if "query" in node.input_mapping else ""
        if context and "dependencies" in node.input_mapping:
            user += f"\n\nDependency outputs:\n{context}"
        messages.append({"role": "user", "content": user})
        return messages

    @staticmethod
    def _materialize(
        graph: GraphDefinition,
        decision: RouteDecision,
    ) -> tuple[list[NodeDefinition], dict[str, list[str]]]:
        expanded: dict[str, list[NodeDefinition]] = {}
        objectives = {item.label: item.objective for item in decision.delegations}
        delegated_models = {item.label: item.model for item in decision.delegations}
        for node in graph.nodes:
            if node.fan_out:
                if node.model_selector != "selected_labels":
                    raise ValueError(f"fan-out node requires selected_labels selector: {node.id}")
                expanded[node.id] = [
                    replace(
                        node,
                        id=f"{node.id}_{label.lower()}",
                        model=delegated_models.get(label, LABEL_TO_MODEL[label]),
                        model_selector=None,
                        fan_out=False,
                        system_prompt=_with_objective(node.system_prompt, objectives.get(label), node.role),
                    )
                    for label in decision.selected_labels
                ]
            else:
                model = node.model
                if node.model_selector == "primary_label":
                    primary = decision.primary_labels[0]
                    model = delegated_models.get(primary, LABEL_TO_MODEL[primary])
                prompt = node.system_prompt
                if node.role == "specialist" and node.model_selector == "primary_label":
                    prompt = _with_objective(prompt, objectives.get(decision.primary_labels[0]), node.role)
                elif node.role == "synthesizer":
                    prompt = _with_objective(prompt, decision.synthesis_strategy, node.role)
                expanded[node.id] = [replace(node, model=model, model_selector=None, system_prompt=prompt)]

        materialized: list[NodeDefinition] = []
        expanded_ids = {base: [item.id for item in values] for base, values in expanded.items()}
        for values in expanded.values():
            for node in values:
                dependencies = tuple(
                    expanded_id
                    for dependency in node.dependencies
                    for expanded_id in expanded_ids[dependency]
                )
                materialized.append(replace(node, dependencies=dependencies))
        return materialized, expanded_ids

    @staticmethod
    def _local_response(role: str, text: str) -> str:
        del text
        if role == "clarifier":
            return "適切な専門家を選ぶため、対象システム、発生している症状、直前の変更、期待する結果を教えてください。"
        return "この依頼は安全ポリシーにより自動実行できません。防御目的の範囲、対象環境への権限、必要な支援内容を明確にして担当者へ確認してください。"

    @staticmethod
    def _degraded_answer(
        outputs: dict[str, str],
        decision: RouteDecision,
        reason: str,
        failure_detail: str | None = None,
    ) -> str:
        if outputs:
            partial = next(reversed(outputs.values()))
            return f"一部の処理のみ完了しました（{reason}）。\n\n{partial}"
        labels = ", ".join(decision.selected_labels)
        if failure_detail == "model_endpoint_unreachable":
            return f"処理を完了できませんでした（{reason}）。実モデルのOpenAI互換endpointに接続できません。候補領域: {labels}。"
        if failure_detail == "model_credentials_missing":
            return f"処理を完了できませんでした（{reason}）。実モデルendpointの認証環境変数が未設定です。候補領域: {labels}。"
        if failure_detail == "model_response_invalid":
            return f"処理を完了できませんでした（{reason}）。実モデルendpointの応答形式を確認してください。候補領域: {labels}。"
        return f"処理を完了できませんでした（{reason}）。候補領域: {labels}。追加情報を添えて再試行してください。"


def _verifier_passed(content: str) -> bool:
    try:
        value = json.loads(content)
        if isinstance(value, dict) and isinstance(value.get("passed"), bool):
            return value["passed"]
    except json.JSONDecodeError:
        pass
    return bool(re.match(r"^\s*pass(?:ed)?\b", content, re.I))


def _with_objective(system_prompt: str, objective: str | None, role: str | None = None) -> str:
    if not objective:
        return system_prompt
    defaults = {
        "specialist": "You are a domain specialist. Give a correct, actionable, and safe answer.",
        "synthesizer": "You are a synthesizer. Reconcile expert outputs into a prioritized answer.",
    }
    base = system_prompt.strip() or defaults.get(role, "")
    return f"{base}\n\nDelegated objective: {objective}" if base else f"Delegated objective: {objective}"


def _validate_output_schema(content: str, schema: dict[str, Any]) -> None:
    if not schema:
        return
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("response is not valid JSON") from exc
    expected_type = schema.get("type")
    type_map = {"object": dict, "array": list, "string": str, "number": (int, float), "boolean": bool}
    if expected_type in type_map and not isinstance(value, type_map[expected_type]):
        raise ValueError(f"expected JSON {expected_type}")
    if isinstance(value, dict):
        missing = [name for name in schema.get("required", ()) if name not in value]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")


def _failure_detail(traces: list[NodeTrace]) -> str | None:
    errors = "\n".join(node.error for node in traces if node.error)
    if not errors:
        return None
    if "Connection refused" in errors or "Errno 111" in errors:
        return "model_endpoint_unreachable"
    if "environment variable is not set" in errors:
        return "model_credentials_missing"
    if "choices[0].message.content" in errors or "invalid output" in errors or "response is not valid JSON" in errors:
        return "model_response_invalid"
    if "HTTP 404" in errors:
        return "model_endpoint_not_found"
    if "model alias is not configured" in errors:
        return "model_alias_missing"
    return "model_execution_failed"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[=:]\s*)[^\s,;]+"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b"),
)


def redact_secrets(value: str) -> str:
    redacted = value
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted
