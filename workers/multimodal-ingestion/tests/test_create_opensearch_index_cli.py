from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_help_does_not_start_opensearch() -> None:
    root = Path(__file__).resolve().parents[3]

    script = (
        root
        / "workers"
        / "multimodal-ingestion"
        / "scripts"
        / "create_opensearch_index.py"
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(
        root
        / "workers"
        / "multimodal-ingestion"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--help",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout

    assert (
        "OPENSEARCH INDEX CREATION"
        not in result.stdout
    )

    assert (
        "Existing compatible index found"
        not in result.stdout
    )

    assert (
        "Creating hybrid vector index"
        not in result.stdout
    )
