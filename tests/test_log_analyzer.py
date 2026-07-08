from pathlib import Path

from wings_control.log_analyzer import log_analyzer


class _StubPlugin:
    def get_stage_definitions(self):
        return {"init": {"name": "init"}}

    def get_log_patterns(self):
        return [{"pattern": r"engine ready", "phase_code": "ready", "progress_calc": 100}]

    def get_error_patterns(self):
        return [{"pattern": r"RuntimeError:", "error_type": "RuntimeError"}]

    def get_accel_patterns(self):
        return [{"pattern": r"\[wings-accel\] ok", "status": "success", "progress": 100}]


def test_log_analyzer_initializes_accel_patterns_before_compile(monkeypatch, tmp_path):
    monkeypatch.setattr(
        log_analyzer.PatternPluginManager,
        "_load_plugin",
        lambda self: _StubPlugin(),
    )

    analyzer = log_analyzer.LogAnalyzer(
        config={},
        log_file=str(tmp_path / "engine.log"),
        progress_file=str(tmp_path / "progress.jsonl"),
    )

    assert analyzer.accel_patterns
    assert all("compiled" in pattern for pattern in analyzer.accel_patterns)
    assert Path(tmp_path / "progress.jsonl").exists()
