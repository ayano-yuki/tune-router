from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from tune_artifacts import read_registry
from tune_bandit import (
    BanditMonitorConfig,
    BanditPolicyConfig,
    BanditPromotionConfig,
    BanditReplayConfig,
    BanditRolloutPlanConfig,
    append_bandit_release_registry,
    apply_bandit_policy,
    artifact_digest,
    build_bandit_current_release,
    build_bandit_release_manifest,
    build_bandit_rollback_current_release,
    build_bandit_runtime_bundle,
    build_bandit_state,
    context_key_from_signal,
    evaluate_bandit_promotion,
    monitor_bandit_rollout,
    plan_bandit_rollout,
    replay_bandit_policy,
    select_bandit_rollback_release,
    update_bandit_release_registry,
    validate_bandit_rollout_binding,
    validate_bandit_rollout_artifacts,
    validate_bandit_release_manifest,
    validate_bandit_current_artifacts,
    validate_bandit_current_release,
    validate_bandit_rollback_candidate,
    validate_bandit_runtime_bundle,
)
from tune_models import Delegation, PolicyDecision, RouteDecision, RouterSignal
from tune_selector import GraphSelector


class BanditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.signal = RouterSignal(
            scores={"Database": 0.46, "Storage": 0.36, "Network": 0.1, "General": 0.01, "Coding": 0.04, "Security": 0.03}
        )

    def test_builds_state_from_reviewed_traces(self) -> None:
        state = build_bandit_state([self._trace("t1", "parallel_experts", 5), self._trace("t2", "single_specialist", 2)])
        context = context_key_from_signal(self.signal, "normal")
        self.assertIn(context, state["contexts"])
        self.assertEqual(2, state["summary"]["observations"])
        self.assertEqual(2, len(state["contexts"][context]["arms"]))

    def test_builds_state_from_reviewed_shadow_traces(self) -> None:
        parent = self._trace("served", "single_specialist", 2)
        parent["shadow_executions"] = [
            {
                "reason": "forced_parallel_experts",
                "status": "completed",
                "trace": self._trace("shadow", "parallel_experts", 5),
            }
        ]
        state = build_bandit_state([parent])
        context = context_key_from_signal(self.signal, "normal")
        self.assertEqual(2, state["summary"]["observations"])
        self.assertEqual(2, len(state["contexts"][context]["arms"]))

    def test_policy_selects_observed_better_alternative(self) -> None:
        state = build_bandit_state(
            [
                self._trace("p1", "parallel_experts", 5),
                self._trace("p2", "parallel_experts", 5),
                self._trace("s1", "single_specialist", 2),
            ]
        )
        served = RouteDecision(
            graph_id="single_specialist",
            primary_labels=("Database",),
            secondary_labels=(),
            selected_labels=("Database",),
            confidence=0.46,
            margin=0.1,
            reason="served candidate",
            policy=PolicyDecision(risk_level="normal", action="allow"),
            delegations=(Delegation("Database", "database-specialist", "Check waits."),),
        )
        selected = apply_bandit_policy(
            text="PostgreSQL on NFS is slow",
            signal=self.signal,
            served_decision=served,
            risk_level="auto",
            selector=GraphSelector(),
            state=state,
            config=BanditPolicyConfig(
                min_observations=1,
                exploration_weight=0.0,
                rollout_digest="rollout-digest",
                release_digest="release-digest",
                release_id="bandit-release-test",
                release_current_digest="current-digest",
            ),
        )
        self.assertEqual("bandit_policy", selected.selector_type)
        self.assertEqual("parallel_experts", selected.graph_id)
        self.assertEqual("single_specialist", selected.selection_metadata["bandit"]["served_graph_id"])
        self.assertEqual("rollout-digest", selected.selection_metadata["bandit"]["rollout_config_digest"])
        self.assertEqual("release-digest", selected.selection_metadata["bandit"]["release_manifest_digest"])
        self.assertEqual("bandit-release-test", selected.selection_metadata["bandit"]["release_id"])
        self.assertEqual("current-digest", selected.selection_metadata["bandit"]["release_current_digest"])

    def test_policy_canary_keeps_served_decision_outside_rollout_bucket(self) -> None:
        state = build_bandit_state(
            [
                self._trace("p1", "parallel_experts", 5),
                self._trace("p2", "parallel_experts", 5),
                self._trace("s1", "single_specialist", 2),
            ]
        )
        served = RouteDecision(
            graph_id="single_specialist",
            primary_labels=("Database",),
            secondary_labels=(),
            selected_labels=("Database",),
            confidence=0.46,
            margin=0.1,
            reason="served candidate",
            policy=PolicyDecision(risk_level="normal", action="allow"),
            delegations=(Delegation("Database", "database-specialist", "Check waits."),),
        )
        selected = apply_bandit_policy(
            text="PostgreSQL on NFS is slow",
            signal=self.signal,
            served_decision=served,
            risk_level="auto",
            selector=GraphSelector(),
            state=state,
            config=BanditPolicyConfig(
                min_observations=1,
                exploration_weight=0.0,
                traffic_percent=0.0,
                rollout_digest="rollout-digest",
            ),
        )
        self.assertEqual("single_specialist", selected.graph_id)
        self.assertIn("bandit_canary", selected.selection_metadata)
        self.assertFalse(selected.selection_metadata["bandit_canary"]["sampled"])
        self.assertEqual("parallel_experts", selected.selection_metadata["bandit_canary"]["candidate_graph_id"])
        self.assertEqual("rollout-digest", selected.selection_metadata["bandit_canary"]["rollout_config_digest"])

    def test_policy_skips_high_risk_by_default(self) -> None:
        signal = RouterSignal(
            scores={"Security": 0.8, "General": 0.05, "Network": 0.05, "Coding": 0.04, "Storage": 0.03, "Database": 0.03}
        )
        served = GraphSelector().select("Review defensive controls", signal)
        selected = apply_bandit_policy(
            text="Review defensive controls",
            signal=signal,
            served_decision=served,
            risk_level="auto",
            selector=GraphSelector(),
            state=build_bandit_state([]),
            config=BanditPolicyConfig(min_observations=1),
        )
        self.assertIs(served, selected)

    def test_replay_reports_reward_lift_from_reviewed_shadow_candidate(self) -> None:
        parent = self._trace("served", "single_specialist", 2)
        parent["shadow_executions"] = [
            {
                "reason": "forced_parallel_experts",
                "status": "completed",
                "trace": self._trace("shadow", "parallel_experts", 5),
            }
        ]
        state = build_bandit_state(
            [
                self._trace("p1", "parallel_experts", 5),
                self._trace("p2", "parallel_experts", 5),
                self._trace("s1", "single_specialist", 2),
                parent,
            ]
        )
        replay = replay_bandit_policy(
            [parent],
            state,
            BanditReplayConfig(min_observations=1, exploration_weight=0.0),
        )
        self.assertEqual(1, replay["summary"]["evaluated_requests"])
        self.assertEqual(1, replay["summary"]["switches"])
        self.assertGreater(replay["summary"]["mean_reward_delta"], 0.0)
        self.assertEqual("shadow:forced_parallel_experts", replay["details"][0]["selected_source"])

    def test_promotion_gate_checks_replay_thresholds(self) -> None:
        replay = {
            "summary": {
                "requests": 10,
                "evaluated_requests": 8,
                "mean_reward_delta": 0.05,
                "loss_rate": 0.0,
                "switch_rate": 0.25,
                "skipped": {"insufficient_observations": 2},
            }
        }
        promotion = evaluate_bandit_promotion(
            replay,
            BanditPromotionConfig(
                min_evaluated_requests=5,
                min_mean_reward_delta=0.0,
                max_loss_rate=0.05,
                max_switch_rate=0.5,
                max_skip_rate=0.3,
            ),
        )
        self.assertEqual("pass", promotion["status"])
        failed = evaluate_bandit_promotion(replay, BanditPromotionConfig(min_evaluated_requests=9))
        self.assertEqual("fail", failed["status"])

    def test_monitor_bandit_rollout_passes_healthy_canary(self) -> None:
        traces = [
            self._runtime_trace("b1", "parallel_experts", selector_type="bandit_policy", cost=0.02, latency=1200),
            self._runtime_trace("b2", "parallel_experts", selector_type="bandit_policy", cost=0.03, latency=1400),
            self._runtime_trace("s1", "single_specialist", selector_type="deterministic", cost=0.01, latency=900),
        ]
        monitor = monitor_bandit_rollout(
            traces,
            BanditMonitorConfig(
                min_bandit_traces=2,
                max_bandit_failure_rate=0.0,
                max_relative_failure_rate=0.1,
                max_bandit_p95_latency_ms=2000,
                max_bandit_mean_cost_usd=0.05,
            ),
        )
        self.assertEqual("pass", monitor["status"])
        self.assertEqual(2, monitor["summary"]["bandit"]["traces"])
        self.assertEqual(2, monitor["summary"]["canary_sampled"])

    def test_monitor_bandit_rollout_fails_unhealthy_canary(self) -> None:
        traces = [
            self._runtime_trace("b1", "parallel_experts", selector_type="bandit_policy", stop_reason="node_failed"),
            self._runtime_trace("b2", "parallel_experts", selector_type="bandit_policy", stop_reason="completed"),
            self._runtime_trace("s1", "single_specialist", selector_type="deterministic", stop_reason="completed"),
        ]
        monitor = monitor_bandit_rollout(
            traces,
            BanditMonitorConfig(min_bandit_traces=2, max_bandit_failure_rate=0.1),
        )
        self.assertEqual("fail", monitor["status"])
        self.assertFalse(monitor["checks"]["max_bandit_failure_rate"]["passed"])

    def test_rollout_plan_advances_after_passing_gates(self) -> None:
        promotion = {"status": "pass"}
        monitor = {
            "status": "pass",
            "summary": {"bandit": {"traces": 20}},
        }
        plan = plan_bandit_rollout(
            promotion=promotion,
            monitor=monitor,
            config=BanditRolloutPlanConfig(
                current_traffic_percent=10,
                step_percent=10,
                max_traffic_percent=50,
                rollout_salt="prod",
                min_monitor_bandit_traces=10,
            ),
        )
        self.assertEqual("advance", plan["action"])
        self.assertEqual(20, plan["target_traffic_percent"])
        self.assertTrue(plan["runtime"]["enabled"])
        self.assertEqual("prod", plan["runtime"]["rollout_salt"])

    def test_rollout_plan_binds_to_bandit_state_digest(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        plan = plan_bandit_rollout(
            promotion={"status": "pass"},
            monitor=None,
            state=state,
            config=BanditRolloutPlanConfig(current_traffic_percent=0, step_percent=5),
        )
        self.assertEqual(artifact_digest(state), plan["artifacts"]["bandit_state_digest"])
        self.assertIn("expires_at", plan)
        binding = validate_bandit_rollout_binding(state=state, rollout=plan)
        self.assertEqual("pass", binding["status"])
        other_state = build_bandit_state([self._trace("s1", "single_specialist", 2)])
        mismatch = validate_bandit_rollout_binding(state=other_state, rollout=plan)
        self.assertEqual("fail", mismatch["status"])
        self.assertFalse(mismatch["checks"]["state_digest_matches"]["passed"])

    def test_rollout_binding_rejects_expired_plan_by_default(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        plan = plan_bandit_rollout(
            promotion={"status": "pass"},
            monitor=None,
            state=state,
            config=BanditRolloutPlanConfig(current_traffic_percent=0, step_percent=5),
        )
        expired = {**plan, "expires_at": "2000-01-01T00:00:00.000Z"}
        binding = validate_bandit_rollout_binding(state=state, rollout=expired)
        self.assertEqual("fail", binding["status"])
        self.assertFalse(binding["checks"]["not_expired"]["passed"])
        allowed = validate_bandit_rollout_binding(state=state, rollout=expired, require_not_expired=False)
        self.assertEqual("pass", allowed["status"])

    def test_rollout_artifact_verification_checks_promotion_and_monitor_digests(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        promotion = {"status": "pass", "summary": {"evaluated_requests": 30}}
        monitor = {"status": "pass", "summary": {"bandit": {"traces": 20}}}
        plan = plan_bandit_rollout(
            promotion=promotion,
            monitor=monitor,
            state=state,
            config=BanditRolloutPlanConfig(current_traffic_percent=10, step_percent=10),
        )
        verification = validate_bandit_rollout_artifacts(
            rollout=plan,
            state=state,
            promotion=promotion,
            monitor=monitor,
            require_monitor_binding=True,
        )
        self.assertEqual("pass", verification["status"])
        mismatch = validate_bandit_rollout_artifacts(
            rollout=plan,
            state=state,
            promotion={"status": "pass", "summary": {"evaluated_requests": 31}},
            monitor=monitor,
            require_monitor_binding=True,
        )
        self.assertEqual("fail", mismatch["status"])
        self.assertFalse(mismatch["checks"]["promotion_digest_matches"]["passed"])

    def test_release_manifest_bundles_verified_rollout_artifacts(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        promotion = {"status": "pass", "summary": {"evaluated_requests": 30}}
        monitor = {"status": "pass", "summary": {"bandit": {"traces": 20}}}
        plan = plan_bandit_rollout(
            promotion=promotion,
            monitor=monitor,
            state=state,
            config=BanditRolloutPlanConfig(current_traffic_percent=10, step_percent=10),
        )
        verification = validate_bandit_rollout_artifacts(
            rollout=plan,
            state=state,
            promotion=promotion,
            monitor=monitor,
            require_monitor_binding=True,
        )
        manifest = build_bandit_release_manifest(
            state=state,
            rollout=plan,
            promotion=promotion,
            monitor=monitor,
            verification=verification,
            require_monitor=True,
        )
        self.assertEqual("pass", manifest["status"])
        self.assertEqual(20, manifest["runtime"]["traffic_percent"])
        self.assertEqual(artifact_digest(plan), manifest["artifacts"]["rollout_digest"])
        release_validation = validate_bandit_release_manifest(
            manifest=manifest,
            state=state,
            rollout=plan,
            verification=verification,
        )
        self.assertEqual("pass", release_validation["status"])
        other_state = build_bandit_state([self._trace("s1", "single_specialist", 2)])
        mismatch = validate_bandit_release_manifest(manifest=manifest, state=other_state)
        self.assertEqual("fail", mismatch["status"])
        self.assertFalse(mismatch["checks"]["state_digest_matches"]["passed"])

    def test_current_release_pointer_binds_to_manifest_digest(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        promotion = {"status": "pass", "summary": {"evaluated_requests": 30}}
        plan = plan_bandit_rollout(
            promotion=promotion,
            monitor=None,
            state=state,
            config=BanditRolloutPlanConfig(current_traffic_percent=0, step_percent=5),
        )
        manifest = build_bandit_release_manifest(state=state, rollout=plan, promotion=promotion)
        current = build_bandit_current_release(
            manifest=manifest,
            manifest_path="bandit-release.json",
            channel="production",
        )
        validation = validate_bandit_current_release(current=current, manifest=manifest)
        self.assertEqual("pass", validation["status"])
        tampered = {**manifest, "release_id": "bandit-release-other"}
        mismatch = validate_bandit_current_release(current=current, manifest=tampered)
        self.assertEqual("fail", mismatch["status"])
        self.assertFalse(mismatch["checks"]["manifest_digest_matches"]["passed"])

    def test_release_registry_records_current_release_once(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        manifest = build_bandit_release_manifest(
            state=state,
            rollout=plan_bandit_rollout(promotion={"status": "pass"}, monitor=None, state=state),
            promotion={"status": "pass"},
        )
        current = build_bandit_current_release(
            manifest=manifest,
            manifest_path="bandit-release.json",
            channel="production",
        )
        registry = append_bandit_release_registry(registry=None, current=current, manifest=manifest)
        registry = append_bandit_release_registry(registry=registry, current=current, manifest=manifest)
        self.assertEqual("tune-orchestrator-bandit-release-registry-v1", registry["format"])
        self.assertEqual(1, registry["summary"]["entries"])
        self.assertEqual(manifest["release_id"], registry["entries"][0]["release_id"])
        self.assertEqual(artifact_digest(manifest), registry["entries"][0]["manifest_digest"])

    def test_release_registry_writer_is_idempotent_and_revisioned(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        manifest = build_bandit_release_manifest(
            state=state,
            rollout=plan_bandit_rollout(promotion={"status": "pass"}, monitor=None, state=state),
            promotion={"status": "pass"},
        )
        current = build_bandit_current_release(
            manifest=manifest,
            manifest_path="bandit-release.json",
            channel="production",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "registry.json"
            first = update_bandit_release_registry(path, current=current, manifest=manifest)
            second = update_bandit_release_registry(path, current=current, manifest=manifest)
            stored = read_registry(path, expected_format="tune-orchestrator-bandit-release-registry-v1")
        self.assertEqual(1, first["revision"])
        self.assertEqual(2, second["revision"])
        self.assertEqual(1, second["summary"]["entries"])
        self.assertEqual(second, stored)

    def test_current_artifact_verification_checks_state_and_registry(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        manifest = build_bandit_release_manifest(
            state=state,
            rollout=plan_bandit_rollout(promotion={"status": "pass"}, monitor=None, state=state),
            promotion={"status": "pass"},
        )
        current = build_bandit_current_release(
            manifest=manifest,
            manifest_path="bandit-release.json",
            channel="production",
        )
        registry = append_bandit_release_registry(registry=None, current=current, manifest=manifest)
        verification = validate_bandit_current_artifacts(
            current=current,
            manifest=manifest,
            state=state,
            registry=registry,
            require_registry_entry=True,
        )
        self.assertEqual("pass", verification["status"])
        wrong_state = build_bandit_state([self._trace("s1", "single_specialist", 2)])
        mismatch = validate_bandit_current_artifacts(
            current=current,
            manifest=manifest,
            state=wrong_state,
            registry=registry,
            require_registry_entry=True,
        )
        self.assertEqual("fail", mismatch["status"])
        self.assertFalse(mismatch["checks"]["state_digest_matches"]["passed"])
        missing_registry = validate_bandit_current_artifacts(
            current=current,
            manifest=manifest,
            state=state,
            registry=None,
            require_registry_entry=True,
        )
        self.assertEqual("fail", missing_registry["status"])
        self.assertFalse(missing_registry["checks"]["registry_entry_present"]["passed"])

    def test_runtime_bundle_binds_current_artifacts(self) -> None:
        state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        manifest = build_bandit_release_manifest(
            state=state,
            rollout=plan_bandit_rollout(promotion={"status": "pass"}, monitor=None, state=state),
            promotion={"status": "pass"},
        )
        current = build_bandit_current_release(
            manifest=manifest,
            manifest_path="bandit-release.json",
            channel="production",
        )
        registry = append_bandit_release_registry(registry=None, current=current, manifest=manifest)
        current_verification = validate_bandit_current_artifacts(
            current=current,
            manifest=manifest,
            state=state,
            registry=registry,
            require_registry_entry=True,
        )
        bundle = build_bandit_runtime_bundle(
            current=current,
            manifest=manifest,
            state=state,
            current_verification=current_verification,
            registry=registry,
            graphs_digest="graphs-digest",
            model_config_digest="models-digest",
        )
        self.assertEqual("pass", bundle["status"])
        verification = validate_bandit_runtime_bundle(
            bundle=bundle,
            current=current,
            manifest=manifest,
            state=state,
            current_verification=current_verification,
            registry=registry,
            graphs_digest="graphs-digest",
            model_config_digest="models-digest",
        )
        self.assertEqual("pass", verification["status"])
        wrong_state = build_bandit_state([self._trace("s1", "single_specialist", 2)])
        mismatch = validate_bandit_runtime_bundle(
            bundle=bundle,
            current=current,
            manifest=manifest,
            state=wrong_state,
            current_verification=current_verification,
            registry=registry,
            graphs_digest="graphs-digest",
            model_config_digest="models-digest",
        )
        self.assertEqual("fail", mismatch["status"])
        self.assertFalse(mismatch["checks"]["state_digest_matches"]["passed"])

    def test_selects_latest_healthy_prior_release_for_rollback(self) -> None:
        previous_state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        current_state = build_bandit_state([self._trace("p2", "single_specialist", 5)])
        previous_manifest = build_bandit_release_manifest(
            state=previous_state,
            rollout=plan_bandit_rollout(
                promotion={"status": "pass"},
                monitor=None,
                state=previous_state,
                config=BanditRolloutPlanConfig(current_traffic_percent=0, step_percent=5),
            ),
            promotion={"status": "pass"},
        )
        current_manifest = build_bandit_release_manifest(
            state=current_state,
            rollout=plan_bandit_rollout(
                promotion={"status": "pass"},
                monitor=None,
                state=current_state,
                config=BanditRolloutPlanConfig(current_traffic_percent=5, step_percent=5),
            ),
            promotion={"status": "pass"},
        )
        previous = build_bandit_current_release(
            manifest=previous_manifest,
            manifest_path="bandit-release-previous.json",
            channel="production",
        )
        current = build_bandit_current_release(
            manifest=current_manifest,
            manifest_path="bandit-release-current.json",
            channel="production",
        )
        registry = append_bandit_release_registry(registry=None, current=previous, manifest=previous_manifest)
        registry = append_bandit_release_registry(registry=registry, current=current, manifest=current_manifest)
        rollback = select_bandit_rollback_release(
            registry=registry,
            current_release_id=current_manifest["release_id"],
            channel="production",
        )
        self.assertEqual("pass", rollback["status"])
        self.assertEqual(previous_manifest["release_id"], rollback["selected_release_id"])
        self.assertEqual("bandit-release-previous.json", rollback["candidate"]["manifest_path"])

    def test_rollback_candidate_validation_binds_manifest_and_state(self) -> None:
        previous_state = build_bandit_state([self._trace("p1", "parallel_experts", 5)])
        current_state = build_bandit_state([self._trace("p2", "single_specialist", 5)])
        previous_manifest = build_bandit_release_manifest(
            state=previous_state,
            rollout=plan_bandit_rollout(
                promotion={"status": "pass"},
                monitor=None,
                state=previous_state,
                config=BanditRolloutPlanConfig(current_traffic_percent=0, step_percent=5),
            ),
            promotion={"status": "pass"},
        )
        current_manifest = build_bandit_release_manifest(
            state=current_state,
            rollout=plan_bandit_rollout(
                promotion={"status": "pass"},
                monitor=None,
                state=current_state,
                config=BanditRolloutPlanConfig(current_traffic_percent=5, step_percent=5),
            ),
            promotion={"status": "pass"},
        )
        previous = build_bandit_current_release(
            manifest=previous_manifest,
            manifest_path="bandit-release-previous.json",
            channel="production",
        )
        current = build_bandit_current_release(
            manifest=current_manifest,
            manifest_path="bandit-release-current.json",
            channel="production",
        )
        registry = append_bandit_release_registry(registry=None, current=previous, manifest=previous_manifest)
        registry = append_bandit_release_registry(registry=registry, current=current, manifest=current_manifest)
        rollback = select_bandit_rollback_release(
            registry=registry,
            current_release_id=current_manifest["release_id"],
            channel="production",
        )
        verification = validate_bandit_rollback_candidate(
            rollback=rollback,
            manifest=previous_manifest,
            state=previous_state,
        )
        self.assertEqual("pass", verification["status"])
        rollback_current = build_bandit_rollback_current_release(
            rollback=rollback,
            manifest=previous_manifest,
            manifest_path="bandit-release-previous.json",
        )
        self.assertEqual(previous_manifest["release_id"], rollback_current["release_id"])
        self.assertEqual(current_manifest["release_id"], rollback_current["rollback"]["from_release_id"])
        tampered = {**previous_manifest, "release_id": "bandit-release-tampered"}
        mismatch = validate_bandit_rollback_candidate(
            rollback=rollback,
            manifest=tampered,
            state=previous_state,
        )
        self.assertEqual("fail", mismatch["status"])
        self.assertFalse(mismatch["checks"]["release_id_matches"]["passed"])
        self.assertFalse(mismatch["checks"]["manifest_digest_matches"]["passed"])

    def test_rollout_plan_starts_initial_canary_without_monitor(self) -> None:
        plan = plan_bandit_rollout(
            promotion={"status": "pass"},
            monitor=None,
            config=BanditRolloutPlanConfig(current_traffic_percent=0, step_percent=5),
        )
        self.assertEqual("advance", plan["action"])
        self.assertEqual(5, plan["target_traffic_percent"])

    def test_rollout_plan_rolls_back_on_monitor_failure(self) -> None:
        plan = plan_bandit_rollout(
            promotion={"status": "pass"},
            monitor={"status": "fail", "summary": {"bandit": {"traces": 30}}},
            config=BanditRolloutPlanConfig(current_traffic_percent=20),
        )
        self.assertEqual("rollback", plan["action"])
        self.assertEqual(0.0, plan["target_traffic_percent"])
        self.assertFalse(plan["runtime"]["enabled"])

    def test_rollout_plan_holds_until_monitor_has_enough_traces(self) -> None:
        plan = plan_bandit_rollout(
            promotion={"status": "pass"},
            monitor={"status": "pass", "summary": {"bandit": {"traces": 3}}},
            config=BanditRolloutPlanConfig(current_traffic_percent=10, min_monitor_bandit_traces=10),
        )
        self.assertEqual("hold", plan["action"])
        self.assertEqual(10, plan["target_traffic_percent"])

    def _trace(self, trace_id: str, graph_id: str, rating: int) -> dict:
        if graph_id == "parallel_experts":
            delegations = [
                {"label": "Database", "model": "database-specialist", "objective": "Check waits."},
                {"label": "Storage", "model": "storage-specialist", "objective": "Check NFS."},
            ]
            primary = ["Database"]
            secondary = ["Storage"]
        else:
            delegations = [{"label": "Database", "model": "database-specialist", "objective": "Check waits."}]
            primary = ["Database"]
            secondary = []
        return {
            "trace_id": trace_id,
            "router": {
                "scores": self.signal.scores,
                "primary_labels": primary,
                "secondary_labels": secondary,
            },
            "policy": {"risk_level": "normal", "action": "allow"},
            "graph": {
                "id": graph_id,
                "selector_type": "deterministic",
                "delegations": delegations,
                "stop_reason": "completed",
            },
            "usage": {"cost_usd": 0.01, "latency_ms": 1000},
            "evaluation": {"user_rating": rating},
        }

    def _runtime_trace(
        self,
        trace_id: str,
        graph_id: str,
        *,
        selector_type: str,
        stop_reason: str = "completed",
        cost: float = 0.01,
        latency: float = 1000,
    ) -> dict:
        metadata = {}
        if selector_type == "bandit_policy":
            metadata["bandit"] = {
                "context_key": context_key_from_signal(self.signal, "normal"),
                "arm_key": f"graph={graph_id}",
                "canary_sampled": True,
            }
        return {
            "trace_id": trace_id,
            "router": {"scores": self.signal.scores},
            "policy": {"risk_level": "normal", "action": "allow"},
            "graph": {
                "id": graph_id,
                "selector_type": selector_type,
                "selection_metadata": metadata,
                "stop_reason": stop_reason,
            },
            "usage": {"cost_usd": cost, "latency_ms": latency},
            "nodes": [{"role": "specialist", "status": "completed" if stop_reason == "completed" else "failed"}],
        }


if __name__ == "__main__":
    unittest.main()
