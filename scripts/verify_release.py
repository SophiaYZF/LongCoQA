from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

EXPECTED_HASHES = {
    "LongCoQA_factoid.csv": "b2bd914f3e830c220e45ac6140930bc07d03b64c4c3b6fdd8c50b988bc087201",
    "LongCoQA_non_factoid.csv": "f7bc1d59de921100c996272b6543c9bccf609942acd0ee9e238405ca78b71238",
    "LongCoQA_evidence.csv": "e22bee840b8a570bb157c79cf3a982b0a01ac21abfba713e3e0a6dd6607cc123",
    "splits/p0_full.json": "366f3005d421426dbd0c16ec0d590b414e446f7a2c4ca841ef873073e857c5e3",
    "splits/p1_dialogue_split.json": "e537b7c6944a6e9aebb967f6cb122a497f8fb9425239fb126d65203240fabe02",
}
FORBIDDEN_ROOTS = {
    "artifacts",
    "code",
    "dlc",
    "docs",
    "external",
    "outputs",
    "reports",
    "results",
    "runtime",
}
EXCLUDED_SCRIPTS = {
    "scripts/repair_p1_judge_fragment_failure.py",
    "scripts/repair_p1_judge_repeat_failure.py",
    "scripts/run_p0_nonfactoid_main_auto.sh",
    "scripts/run_p0_rag_formal_auto_1_5.sh",
}
REQUIRED_PATHS = {
    ".env.example",
    ".gitignore",
    "README.md",
    "LICENSE",
    "DATA_LICENSE",
    "DATASET_CARD.md",
    "THIRD_PARTY.md",
    "data/README.md",
    "data/SHA256SUMS",
    "pyproject.toml",
    *(f"data/{path}" for path in EXPECTED_HASHES),
}
MAX_TEXT_BYTES = 10 * 1024 * 1024
PRIVATE_PATHS = ("/" + "Users" + "/", "/" + "lustre" + "01")
PRIVATE_USERNAMES = ("feng" + "qin", "yu" + "zhengfeng")
EMAIL_ADDRESS = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SECRET_ASSIGNMENT = re.compile(
    r"(?im)^\s*['\"]?(?P<name>[A-Za-z_][A-Za-z0-9_-]*)['\"]?\s*[:=]\s*"
    r"(?P<value>[^\r\n#]+)"
)


@dataclass(frozen=True)
class ReleaseIssue:
    code: str
    path: str
    detail: str


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_secret_name(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return any(token in normalized for token in ("api_key", "token", "secret", "password"))


def _has_literal_secret(text: str) -> bool:
    for match in SECRET_ASSIGNMENT.finditer(text):
        value = match.group("value").strip()
        if (
            _is_secret_name(match.group("name"))
            and len(value) >= 2
            and value[0] in "'\""
            and value[-1] == value[0]
            and value[1:-1]
        ):
            return True
    return False


def _manifest_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    if not path.is_file():
        return hashes
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            hashes[parts[1].lstrip(" *")] = parts[0]
    return hashes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_release(root: Path) -> list[ReleaseIssue]:
    root = root.resolve()
    issues: set[ReleaseIssue] = set()

    for name in FORBIDDEN_ROOTS:
        if (root / name).is_dir():
            issues.add(ReleaseIssue("forbidden-path", name, "forbidden top-level directory"))
    if (root / "data" / "novel").exists():
        issues.add(ReleaseIssue("forbidden-path", "data/novel", "forbidden private data"))
    for relative in EXCLUDED_SCRIPTS:
        if (root / relative).exists():
            issues.add(ReleaseIssue("excluded-script", relative, "excluded script"))

    for relative in REQUIRED_PATHS:
        if not (root / relative).is_file():
            issues.add(ReleaseIssue("missing-required-path", relative, "required release file is missing"))

    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size >= MAX_TEXT_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = _relative(root, path)
        if any(token.casefold() in text.casefold() for token in PRIVATE_PATHS + PRIVATE_USERNAMES):
            issues.add(ReleaseIssue("private-path", relative, "private path or username"))
        if any(email.casefold() != "anonymous@example.invalid" for email in EMAIL_ADDRESS.findall(text)):
            issues.add(ReleaseIssue("email-address", relative, "non-anonymous email address"))
        if _has_literal_secret(text):
            issues.add(ReleaseIssue("credential-literal", relative, "literal credential assignment"))

    manifest = _manifest_hashes(root / "data" / "SHA256SUMS")
    for relative, expected in EXPECTED_HASHES.items():
        data_path = root / "data" / relative
        if manifest.get(relative) != expected:
            issues.add(ReleaseIssue("checksum-manifest-mismatch", "data/SHA256SUMS", relative))
        if data_path.is_file() and _sha256(data_path) != expected:
            issues.add(ReleaseIssue("checksum-mismatch", f"data/{relative}", "does not match locked hash"))

    return sorted(issues, key=lambda issue: (issue.path, issue.code, issue.detail))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    issues = validate_release(args.root)
    if not issues:
        print("release validation: ok")
        return 0
    for issue in issues:
        print(f"{issue.path}: {issue.code}: {issue.detail}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
