"""Unit tests for k_calibration/k_optimize.py."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import optuna
import pytest
import torch

# ensure k_calibration is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from k_calibration.k_optimize import (
    LocalOptunaTrialRecorder,
    _ensure_optuna_storage_writable,
    _optimize_with_storage_guard,
    _resolve_optimize_data_dir,
    make_objective,
    per_pixel_probs,
    spatial_tv_distance,
    temporal_dt_probs,
    temporal_tv_distance,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class TestResolveOptimizeDataDir:
    def test_keeps_direct_raw_calibration_dir(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data" / "k_calib_20250509"
        _touch(data_dir / "dv_output_20250509_223721.aedat4")
        _touch(data_dir / "metadata_k_calib_20250509_223725.dat")
        _touch(data_dir / "raw_frames_k_calib_20250509_223725.dat")
        _touch(data_dir / "rgb_frames_k_calib_20250509_223725.dat")

        resolved = _resolve_optimize_data_dir(str(data_dir))

        assert resolved == str(data_dir)

    def test_redirects_stage1_analysis_dir_to_matching_raw_dir(
        self, tmp_path: Path
    ) -> None:
        repo_root = tmp_path / "repo"
        stage1_dir = repo_root / "k_calib_20250509"
        raw_dir = repo_root / "data" / "k_calib_20250509"

        _touch(
            stage1_dir
            / "calib_1"
            / "frames_analysis_full"
            / "events_with_luminance_raw.pt"
        )
        _touch(raw_dir / "dv_output_20250509_223721.aedat4")
        _touch(raw_dir / "metadata_k_calib_20250509_223725.dat")
        _touch(raw_dir / "raw_frames_k_calib_20250509_223725.dat")
        _touch(raw_dir / "rgb_frames_k_calib_20250509_223725.dat")

        resolved = _resolve_optimize_data_dir(str(stage1_dir))

        assert resolved == str(raw_dir)

    def test_explains_stage1_dir_without_matching_raw_dir(self, tmp_path: Path) -> None:
        stage1_dir = tmp_path / "k_calib_20250509"
        _touch(
            stage1_dir
            / "calib_1"
            / "frames_analysis_full"
            / "events_with_luminance_raw.pt"
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            _resolve_optimize_data_dir(str(stage1_dir))

        message = str(exc_info.value)
        assert "Stage 1 analysis directory" in message
        assert "data/k_calib_20250509" in message


class TestOptunaStorageGuard:
    def test_rejects_readonly_sqlite_before_trials_start(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "readonly_output"
        output_dir.mkdir()
        db_path = output_dir / "k_optimize_raw.db"

        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE probe (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        db_path.chmod(0o444)
        output_dir.chmod(0o555)

        try:
            with pytest.raises(RuntimeError) as exc_info:
                _ensure_optuna_storage_writable(db_path)
        finally:
            output_dir.chmod(0o755)
            db_path.chmod(0o644)

        message = str(exc_info.value)
        assert "Optuna SQLite storage is not writable" in message
        assert str(db_path) in message


class TestLocalOptunaTrialRecorder:
    def test_records_last_trial_best_snapshot_and_trial_csv(
        self, tmp_path: Path
    ) -> None:
        recorder = LocalOptunaTrialRecorder(
            tmp_path, "k_optimize_raw", "Raw2DVS346", "raw"
        )

        recorder.record_trial(
            trial_number=0,
            status="completed",
            k_values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            n_gen=123,
            n_dv=200,
            emd=0.4,
            count_pen=0.2,
            loss=0.6,
            trial_seconds=1.25,
            sinkhorn_seconds=0.75,
        )
        recorder.record_trial(
            trial_number=1,
            status="completed",
            k_values=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            n_gen=456,
            n_dv=200,
            emd=0.1,
            count_pen=0.05,
            loss=0.15,
            trial_seconds=0.8,
            sinkhorn_seconds=0.3,
        )

        last_payload = json.loads(recorder.last_trial_path.read_text())
        best_payload = json.loads(recorder.best_trial_path.read_text())
        csv_rows = list(
            csv.DictReader(recorder.trial_csv_path.read_text().splitlines())
        )

        assert last_payload["trial_number"] == 1
        assert last_payload["loss"] == pytest.approx(0.15)
        assert best_payload["trial_number"] == 1
        assert best_payload["K"] == [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        assert recorder.best_payload is not None
        assert recorder.best_payload["loss"] == pytest.approx(0.15)
        assert len(csv_rows) == 2
        assert csv_rows[0]["status"] == "completed"
        assert csv_rows[0]["trial_number"] == "0"
        assert csv_rows[0]["k1"] == "1.0"
        assert csv_rows[0]["trial_seconds"] == "1.25"
        assert csv_rows[0]["sinkhorn_seconds"] == "0.75"
        assert csv_rows[0]["loss_delta_from_prev"] == ""
        assert csv_rows[0]["best_loss_so_far"] == "0.6"
        assert csv_rows[1]["trial_number"] == "1"
        assert csv_rows[1]["loss"] == "0.15"
        assert csv_rows[1]["loss_delta_from_prev"] == "-0.44999999999999996"
        assert csv_rows[1]["best_loss_so_far"] == "0.15"
        assert csv_rows[1]["best_improvement"] == "0.44999999999999996"
        assert csv_rows[1]["count_ratio"] == "2.28"


class TestOptimizeWithStorageGuard:
    def test_converts_nested_readonly_storage_failure_to_clear_runtime_error(
        self, tmp_path: Path
    ) -> None:
        storage_exc = optuna.exceptions.StorageInternalError(
            "commit failed: attempt to write a readonly database"
        )
        try:
            raise AssertionError("Should not reach.") from storage_exc
        except AssertionError as exc:
            nested_exc = exc

        class FakeStudy:
            def optimize(self, objective, n_trials, timeout) -> None:
                raise nested_exc

        recovery_path = tmp_path / "Raw2DVS346_K_recovered.json"

        def recovery_writer() -> Path:
            recovery_path.write_text("{}")
            return recovery_path

        args = argparse.Namespace(n_trials=10, timeout=None)
        with pytest.raises(RuntimeError) as exc_info:
            _optimize_with_storage_guard(
                study=FakeStudy(),
                objective=lambda _: 0.0,
                args=args,
                db_path=tmp_path / "k_optimize_raw.db",
                recovery_writer=recovery_writer,
            )

        message = str(exc_info.value)
        assert "readonly database" in message
        assert str(recovery_path) in message
        assert recovery_path.exists()


class TestPerPixelProbsAndSpatialTV:
    def test_empty_events_fall_back_to_uniform_distribution(self) -> None:
        probs = per_pixel_probs(
            torch.zeros((0, 4), dtype=torch.float32), height=4, width=5
        )
        assert probs.shape == (20,)
        assert torch.allclose(probs, torch.full((20,), 1.0 / 20, dtype=torch.float64))

    def test_bincount_normalizes_and_clamps_out_of_bounds(self) -> None:
        events = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 2.0, 1.0, 1.0],
                [0.0, 2.0, 1.0, 1.0],
                [0.0, 999.0, 999.0, 1.0],  # clamped to bottom-right
            ],
            dtype=torch.float32,
        )
        probs = per_pixel_probs(events, height=2, width=3)
        assert probs.shape == (6,)
        assert probs.sum().item() == pytest.approx(1.0)
        # (y=0,x=0) -> 1 hit, (y=1,x=2) -> 2 + 1 (clamped) = 3 hits
        assert probs[0].item() == pytest.approx(1.0 / 4.0)
        assert probs[5].item() == pytest.approx(3.0 / 4.0)

    def test_tv_distance_zero_for_identical_distributions(self) -> None:
        p = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
        assert spatial_tv_distance(p, p) == pytest.approx(0.0)

    def test_tv_distance_bounded_one_for_disjoint_supports(self) -> None:
        p = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        q = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
        assert spatial_tv_distance(p, q) == pytest.approx(1.0)

    def test_tv_distance_captures_uniform_vs_concentrated_gap(self) -> None:
        n = 100
        uniform = torch.full((n,), 1.0 / n, dtype=torch.float64)
        concentrated = torch.zeros(n, dtype=torch.float64)
        concentrated[0] = 0.5
        concentrated[1] = 0.5
        # (1 - 2/n) of the mass in `uniform` lies on pixels where `concentrated` is 0.
        assert spatial_tv_distance(uniform, concentrated) == pytest.approx(
            1.0 - 2.0 / n
        )


class TestTemporalDtProbsAndTemporalTV:
    def test_temporal_dt_probs_numpy_case_matches_expected_bins(self) -> None:
        events = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1000.0, 0.0, 0.0, 1.0],
                [3000.0, 0.0, 0.0, 1.0],
                [6000.0, 0.0, 0.0, 1.0],
                [10000.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        probs = temporal_dt_probs(
            events, min_events_per_pixel=5, max_dt_us=10_000.0, num_bins=10
        )

        assert probs.shape == (10,)
        assert probs.sum().item() == pytest.approx(1.0)
        assert probs[1].item() == pytest.approx(0.25)
        assert probs[2].item() == pytest.approx(0.25)
        assert probs[3].item() == pytest.approx(0.25)
        assert probs[4].item() == pytest.approx(0.25)
        assert torch.count_nonzero(probs).item() == 4

    def test_temporal_dt_probs_respects_min_events_per_pixel_filter(self) -> None:
        events = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1000.0, 0.0, 0.0, 1.0],
                [3000.0, 0.0, 0.0, 1.0],
                [6000.0, 0.0, 0.0, 1.0],
                [10000.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1000.0, 1.0, 0.0, 1.0],
                [2000.0, 1.0, 0.0, 1.0],
                [3000.0, 1.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        probs = temporal_dt_probs(
            events, min_events_per_pixel=5, max_dt_us=10_000.0, num_bins=10
        )

        assert probs.sum().item() == pytest.approx(1.0)
        assert probs[1].item() == pytest.approx(0.25)
        assert probs[2].item() == pytest.approx(0.25)
        assert probs[3].item() == pytest.approx(0.25)
        assert probs[4].item() == pytest.approx(0.25)
        assert torch.count_nonzero(probs).item() == 4

    def test_temporal_dt_probs_clips_large_intervals_to_last_bin(self) -> None:
        events = np.asarray(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1000.0, 0.0, 0.0, 1.0],
                [3000.0, 0.0, 0.0, 1.0],
                [6000.0, 0.0, 0.0, 1.0],
                [50000.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        probs = temporal_dt_probs(
            events, min_events_per_pixel=5, max_dt_us=10_000.0, num_bins=10
        )

        assert probs.sum().item() == pytest.approx(1.0)
        assert probs[1].item() == pytest.approx(0.25)
        assert probs[2].item() == pytest.approx(0.25)
        assert probs[3].item() == pytest.approx(0.25)
        assert probs[-1].item() == pytest.approx(0.25)

    def test_temporal_tv_distance_bounds_match_tv_definition(self) -> None:
        p = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
        q = torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64)
        r = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

        assert temporal_tv_distance(p, q) == pytest.approx(0.0)
        assert temporal_tv_distance(p, r) == pytest.approx(1.0)

    def test_temporal_dt_probs_empty_or_ineligible_events_return_zero_histogram(
        self,
    ) -> None:
        empty = temporal_dt_probs(np.zeros((0, 4), dtype=np.float64), num_bins=8)
        ineligible = temporal_dt_probs(
            np.asarray(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 1.0],
                    [2.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            min_events_per_pixel=5,
            num_bins=8,
        )

        assert torch.equal(empty, torch.zeros(8, dtype=torch.float64))
        assert torch.equal(ineligible, torch.zeros(8, dtype=torch.float64))
        assert temporal_tv_distance(empty, ineligible) == pytest.approx(0.0)


class TestObjectiveBadTrialHandling:
    def test_returns_inf_and_records_invalid_trial_when_event_generation_hits_recursion_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class FakeTrial:
            number = 3

            def suggest_float(
                self, name: str, low: float, high: float, log: bool = False
            ) -> float:
                return low if not log else max(low, 1e-6)

        frames = torch.zeros((2, 2, 2), dtype=torch.float32)
        pi_timestamps = torch.tensor([0.0, 10.0], dtype=torch.float64)
        dv_events = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [5.0, 1.0, 1.0, 1.0],
                [10.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        ranges = {
            "k1": (1.0, 2.0, False),
            "k2": (1.0, 2.0, False),
            "k3": (1e-6, 1e-5, True),
            "k4": (1.0, 2.0, False),
            "k5": (1.0, 2.0, False),
            "k6": (1e-6, 1e-5, True),
        }

        def fake_generate_events_tensor(*args, **kwargs):
            raise RecursionError("maximum recursion depth exceeded")

        monkeypatch.setattr(
            "k_calibration.k_optimize.dvs_generate.generate_events_tensor",
            fake_generate_events_tensor,
        )
        recorder = LocalOptunaTrialRecorder(
            tmp_path, "k_optimize_raw", "Raw2DVS346", "raw"
        )

        objective = make_objective(
            frames=frames,
            pi_timestamps=pi_timestamps,
            dv_events=dv_events,
            is_rgb=False,
            sinkhorn_fn=lambda *args, **kwargs: torch.tensor(0.0),
            ranges=ranges,
            max_pts=100,
            base_seed=42,
            count_penalty_weight=0.1,
            trial_recorder=recorder,
        )

        assert objective(FakeTrial()) == float("inf")

        csv_rows = list(
            csv.DictReader(recorder.trial_csv_path.read_text().splitlines())
        )
        assert len(csv_rows) == 1
        assert csv_rows[0]["trial_number"] == "3"
        assert csv_rows[0]["status"] == "invalid_recursion_error"
        assert csv_rows[0]["loss"] == "inf"
        assert "maximum recursion depth exceeded" in csv_rows[0]["error"]


class TestObjectiveTemporalPenalty:
    def test_temporal_penalty_weight_adds_loss_and_records_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        class FakeTrial:
            number = 7

            def __init__(self) -> None:
                self.user_attrs: dict[str, float] = {}

            def suggest_float(
                self, name: str, low: float, high: float, log: bool = False
            ) -> float:
                return low if not log else max(low, 1e-6)

            def set_user_attr(self, key: str, value: float) -> None:
                self.user_attrs[key] = value

        frames = torch.zeros((2, 2, 2), dtype=torch.float32)
        pi_timestamps = torch.tensor([0.0, 10_000.0], dtype=torch.float64)
        dv_events = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1000.0, 0.0, 0.0, 1.0],
                [3000.0, 0.0, 0.0, 1.0],
                [6000.0, 0.0, 0.0, 1.0],
                [10000.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        gen_events = torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [1000.0, 0.0, 0.0, 1.0],
                [2000.0, 0.0, 0.0, 1.0],
                [3000.0, 0.0, 0.0, 1.0],
                [4000.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )
        ranges = {
            "k1": (1.0, 2.0, False),
            "k2": (1.0, 2.0, False),
            "k3": (1e-6, 1e-5, True),
            "k4": (1.0, 2.0, False),
            "k5": (1.0, 2.0, False),
            "k6": (1e-6, 1e-5, True),
        }

        monkeypatch.setattr(
            "k_calibration.k_optimize.dvs_generate.generate_events_tensor",
            lambda *args, **kwargs: gen_events.clone(),
        )
        recorder = LocalOptunaTrialRecorder(
            tmp_path, "k_optimize_raw", "Raw2DVS346", "raw"
        )

        objective = make_objective(
            frames=frames,
            pi_timestamps=pi_timestamps,
            dv_events=dv_events,
            is_rgb=False,
            sinkhorn_fn=lambda *args, **kwargs: torch.tensor(0.0),
            ranges=ranges,
            max_pts=100,
            base_seed=42,
            count_penalty_weight=0.0,
            temporal_penalty_weight=2.0,
            trial_recorder=recorder,
        )

        trial = FakeTrial()
        expected_temporal_pen = temporal_tv_distance(
            temporal_dt_probs(gen_events),
            temporal_dt_probs(dv_events),
        )

        assert objective(trial) == pytest.approx(2.0 * expected_temporal_pen)
        assert trial.user_attrs["temporal_penalty"] == pytest.approx(
            expected_temporal_pen
        )

        csv_rows = list(
            csv.DictReader(recorder.trial_csv_path.read_text().splitlines())
        )
        assert float(csv_rows[0]["temporal_penalty"]) == pytest.approx(
            expected_temporal_pen
        )
