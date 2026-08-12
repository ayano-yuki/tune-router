"""Shared constants for the TuneRouter orchestrator."""

from __future__ import annotations

LABEL_TO_MODEL = {
    "Storage": "storage-specialist",
    "Network": "network-specialist",
    "Coding": "coding-specialist",
    "Security": "security-specialist",
    "Database": "database-specialist",
    "General": "general-fallback",
}

LABELS = tuple(LABEL_TO_MODEL)

MULTI_AGENT_GRAPHS = {
    "parallel_experts",
    "specialist_with_verifier",
    "plan_execute_review",
}
