from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

from scripts.verify_release import validate_release


def issue_codes(root: Path) -> set[str]:
    return {issue.code for issue in validate_release(root)}


def test_real_release_passes() -> None:
    assert validate_release(Path(__file__).resolve().parents[1]) == []


def test_forbidden_root_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "runtime").mkdir()

    assert "forbidden-path" in issue_codes(tmp_path)


def test_excluded_script_is_rejected(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "repair_p1_judge_fragment_failure.py"
    script.parent.mkdir()
    script.write_text("", encoding="utf-8")

    assert "excluded-script" in issue_codes(tmp_path)


def test_private_path_and_email_are_rejected(tmp_path: Path) -> None:
    private_path = "/" + "Users" + "/" + "private"
    email = "person" + "@" + "example.com"
    (tmp_path / "unsafe.txt").write_text(f"{private_path}\n{email}\n", encoding="utf-8")

    assert {"private-path", "email-address"} <= issue_codes(tmp_path)


def test_obvious_literal_secret_is_rejected_without_echoing_value(tmp_path: Path) -> None:
    value = "sk-" + "not-for-output"
    assignment = "client_" + "token_value = '" + value + "'\n"
    (tmp_path / "unsafe.py").write_text(assignment, encoding="utf-8")

    issues = validate_release(tmp_path)

    assert "credential-literal" in {issue.code for issue in issues}
    assert all(value not in field for issue in issues for field in (issue.path, issue.detail))


def test_missing_required_document_is_rejected(tmp_path: Path) -> None:
    assert "missing-required-path" in issue_codes(tmp_path)


def test_data_and_manifest_cannot_be_reblessed(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "data"
    shutil.copytree(source, tmp_path / "data")
    target = tmp_path / "data" / "LongCoQA_factoid.csv"
    target.write_bytes(target.read_bytes() + b"\nchanged")
    changed_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = tmp_path / "data" / "SHA256SUMS"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            "b2bd914f3e830c220e45ac6140930bc07d03b64c4c3b6fdd8c50b988bc087201",
            changed_hash,
        ),
        encoding="utf-8",
    )

    assert {"checksum-manifest-mismatch", "checksum-mismatch"} <= issue_codes(tmp_path)


def test_cli_success_output() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/verify_release.py", "--root", "."],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout == "release validation: ok\n"
