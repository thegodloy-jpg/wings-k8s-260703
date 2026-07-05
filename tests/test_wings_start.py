import os
import subprocess
import sys
from pathlib import Path


def test_start_script_clears_application_pycache_before_launcher_imports():
    script = Path("wings_control/wings_start.sh").read_text(encoding="utf-8")

    pycache_cleanup = (
        'find "${APP_WORKDIR:-/opt/wings-control}" '
        '"${WINGS_PACKAGE_DIR:-/opt/wings_control}" '
        "-type d -name '__pycache__' -prune -exec rm -rf {} +"
    )

    assert pycache_cleanup in script
    assert script.index(pycache_cleanup) < script.index('exec "${PYTHON_BIN}" -m wings_control')


def test_start_script_prioritizes_opt_and_app_workdir_before_inherited_pythonpath():
    script = Path("wings_control/wings_start.sh").read_text(encoding="utf-8")

    expected = 'export PYTHONPATH="/opt:${APP_WORKDIR:-/opt/wings-control}${PYTHONPATH:+:${PYTHONPATH}}"'

    assert expected in script
    assert script.index(expected) < script.index('exec "${PYTHON_BIN}" -m wings_control')


def test_package_main_runs_launcher_entrypoint():
    main_py = Path("wings_control/__main__.py").read_text(encoding="utf-8")

    assert "from wings_control.wings_control import run" in main_py
    assert "raise SystemExit(run())" in main_py


def test_package_main_bootstraps_application_module_path_without_pythonpath():
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-m", "wings_control", "--help"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "usage: wings-launcher-v4" in result.stdout
