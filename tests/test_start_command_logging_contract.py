import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wings_control"))

from engines import vllm_adapter  # noqa: E402


def test_wings_cmd_echo_keeps_long_command_complete():
    long_value = "x" * 1200
    command = (
        "exec python3 -m vllm.entrypoints.openai.api_server "
        f"--model /weights --served-model-name {long_value}\n"
    )

    script = vllm_adapter._inject_env_echo(command)

    assert "...<truncated>" not in script
    assert command.strip() in script


def test_wings_cmd_echo_includes_accel_install_command():
    script = (
        "export WINGS_ENGINE_PATCH_OPTIONS='{\"vllm\":{\"features\":[\"sparse\"]}}'\n"
        'python3 /accel-volume/install.py --features "$WINGS_ENGINE_PATCH_OPTIONS"\n'
    )

    rendered = vllm_adapter._inject_env_echo(script)

    assert (
        'echo "[wings-env] export WINGS_ENGINE_PATCH_OPTIONS='
        '${WINGS_ENGINE_PATCH_OPTIONS:-}"'
    ) in rendered
    assert (
        "echo '[wings-cmd] >>> python3 /accel-volume/install.py "
        "--features \"$WINGS_ENGINE_PATCH_OPTIONS\"'"
    ) in rendered


def test_wings_cmd_echo_includes_cwd_accel_install_command():
    script = "(cd \"/accel-volume\" && python install.py --config '/tmp/lmcache.json')\n"

    rendered = vllm_adapter._inject_env_echo(script)

    assert "[wings-cmd] >>> (cd" in rendered
    assert "/tmp/lmcache.json" in rendered
