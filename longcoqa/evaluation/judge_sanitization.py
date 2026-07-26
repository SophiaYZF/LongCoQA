from __future__ import annotations


def sanitize_pathological_char_runs(
    text: str, *, threshold: int = 64, keep: int = 8
) -> tuple[str, list[dict[str, object]]]:
    """Compress pathological repeated-character runs for audited judge input."""
    if threshold < 2 or keep < 1 or keep >= threshold:
        raise ValueError("Require 1 <= keep < threshold")

    pieces: list[str] = []
    audits: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(text):
        end = cursor + 1
        while end < len(text) and text[end] == text[cursor]:
            end += 1
        run_length = end - cursor
        if run_length >= threshold:
            replacement = (
                f"［字符“{text[cursor]}”连续重复 {run_length} 次；"
                f"已保留前 {keep} 次］"
            )
            pieces.append(text[cursor] * keep)
            pieces.append(replacement)
            audits.append(
                {
                    "character": text[cursor],
                    "run_length": run_length,
                    "kept_characters": keep,
                    "replacement": replacement,
                    "start": cursor,
                    "end": end,
                }
            )
        else:
            pieces.append(text[cursor:end])
        cursor = end
    return "".join(pieces), audits


def sanitize_pathological_fragment_runs(
    text: str,
    *,
    threshold: int = 64,
    keep: int = 8,
    max_fragment: int = 16,
) -> tuple[str, list[dict[str, object]]]:
    """Compress pathological consecutive repetitions of short fragments."""
    if threshold < 2 or keep < 1 or max_fragment < 2:
        raise ValueError("Require threshold >= 2, keep >= 1, max_fragment >= 2")

    pieces: list[str] = []
    audits: list[dict[str, object]] = []
    cursor = 0
    while cursor < len(text):
        best: tuple[int, int, str] | None = None
        max_width = min(max_fragment, (len(text) - cursor) // 2)
        for width in range(2, max_width + 1):
            fragment = text[cursor : cursor + width]
            repeat_count = 1
            while text.startswith(fragment, cursor + repeat_count * width):
                repeat_count += 1
            covered = width * repeat_count
            if repeat_count <= keep or covered < threshold:
                continue
            if best is None or covered > best[0] or (
                covered == best[0] and width < best[1]
            ):
                best = (covered, width, fragment)

        if best is None:
            pieces.append(text[cursor])
            cursor += 1
            continue

        covered, width, fragment = best
        repeat_count = covered // width
        end = cursor + covered
        replacement = (
            f"［片段“{fragment}”连续重复 {repeat_count} 次；"
            f"已保留前 {keep} 次］"
        )
        pieces.append(fragment * keep)
        pieces.append(replacement)
        audits.append(
            {
                "fragment": fragment,
                "fragment_length": width,
                "repeat_count": repeat_count,
                "kept_repeats": keep,
                "replacement": replacement,
                "start": cursor,
                "end": end,
            }
        )
        cursor = end

    return "".join(pieces), audits
