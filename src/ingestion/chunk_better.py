import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROCESSED_TEXT_PATH


ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
ENGLISH_RE = re.compile(r"[A-Za-z]")
PUNCTUATION_END_RE = re.compile(r"[\.!\?:؛،]$")
TIME_RANGE_RE = re.compile(r"^\d{1,2}:\d{2}\s?(AM|PM)\s+to\s+\d{1,2}:\d{2}\s?(AM|PM)", re.IGNORECASE)


@dataclass
class Section:
    title: str
    section_index: int
    lines: list[tuple[int, str]]


@dataclass
class Unit:
    unit_type: str
    lines: list[tuple[int, str]]


def detect_language(text: str) -> str:
    has_ar = bool(ARABIC_RE.search(text))
    has_en = bool(ENGLISH_RE.search(text))
    if has_ar and not has_en:
        return "ar"
    if has_en and not has_ar:
        return "en"
    if has_en and has_ar:
        return "mixed"
    return "other"


def is_bullet_like(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith(("-", "*", "•")):
        return True
    if TIME_RANGE_RE.search(s):
        return True
    if re.match(r"^\d+[\.)-]\s+", s):
        return True
    # Preserve short label:value rows as list-like items.
    if ":" in s and len(s) < 180:
        return True
    return False


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if is_bullet_like(s):
        return False
    if len(s) > 110:
        return False
    if PUNCTUATION_END_RE.search(s):
        return False

    heading_keywords = (
        "track",
        "policy",
        "framework",
        "scope",
        "stack",
        "tasks",
        "requirements",
        "constraints",
        "deliverables",
        "milestones",
        "path",
        "paths",
        "schedule",
        "benefits",
        "notice",
        "graduation",
        "evaluation",
        "onboarding",
        "guide",
        "welcome",
        "design",
        "security",
        "infrastructure",
        "analysis",
        "operations",
    )
    heading_keywords_ar = (
        "مسار",
        "نظام",
        "حفل",
        "مزايا",
        "التقييم",
        "إرشادات",
        "التوجيه",
        "التخرج",
        "الأمن",
        "التطوير",
        "التصميم",
        "البرنامج",
    )

    low = s.lower()
    if any(k in low for k in heading_keywords):
        return True
    if any(k in s for k in heading_keywords_ar):
        return True

    # Title-like fallback: short sentence with few words and mostly title-cased words.
    words = s.split()
    if 2 <= len(words) <= 10 and all(len(w) <= 25 for w in words):
        alpha_words = [w for w in words if any(ch.isalpha() for ch in w)]
        if alpha_words and sum(1 for w in alpha_words if w[:1].isupper()) >= max(1, len(alpha_words) // 2):
            return True
    return False


def load_lines(input_path: str) -> list[tuple[int, str]]:
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = [ln.rstrip() for ln in f.readlines()]
    indexed = []
    for i, ln in enumerate(raw_lines):
        if ln.strip():
            indexed.append((i, ln.strip()))
    return indexed


def split_into_sections(indexed_lines: list[tuple[int, str]]) -> list[Section]:
    sections: list[Section] = []
    current_title = "General"
    current_lines: list[tuple[int, str]] = []
    section_index = 0
    i = 0

    while i < len(indexed_lines):
        line_id, line = indexed_lines[i]
        if is_heading(line):
            if current_lines:
                sections.append(Section(title=current_title, section_index=section_index, lines=current_lines))
                section_index += 1
                current_lines = []

            # Keep consecutive heading lines together so EN/AR titles stay paired.
            heading_lines = [line]
            j = i + 1
            while j < len(indexed_lines) and len(heading_lines) < 4:
                _, maybe_heading = indexed_lines[j]
                if not is_heading(maybe_heading):
                    break
                heading_lines.append(maybe_heading)
                j += 1

            current_title = " / ".join(heading_lines)
            i = j
            continue

        current_lines.append((line_id, line))
        i += 1

    if current_lines:
        sections.append(Section(title=current_title, section_index=section_index, lines=current_lines))

    return sections


def is_en_ar_pair(first: str, second: str) -> bool:
    l1 = detect_language(first)
    l2 = detect_language(second)
    if {l1, l2} != {"en", "ar"}:
        return False
    if is_bullet_like(first) or is_bullet_like(second):
        return False
    if is_heading(first) or is_heading(second):
        return False
    return True


def build_units(section: Section) -> list[Unit]:
    units: list[Unit] = []
    i = 0

    while i < len(section.lines):
        line_id, line = section.lines[i]

        # Preserve full list/rule blocks together.
        if is_bullet_like(line):
            block = [(line_id, line)]
            j = i + 1
            while j < len(section.lines):
                nid, nline = section.lines[j]
                if not is_bullet_like(nline):
                    break
                block.append((nid, nline))
                j += 1
            units.append(Unit(unit_type="list_block", lines=block))
            i = j
            continue

        # Keep EN/AR paragraph pairs together.
        if i + 1 < len(section.lines):
            next_id, next_line = section.lines[i + 1]
            if is_en_ar_pair(line, next_line):
                units.append(Unit(unit_type="paired_paragraph", lines=[(line_id, line), (next_id, next_line)]))
                i += 2
                continue

        units.append(Unit(unit_type="paragraph", lines=[(line_id, line)]))
        i += 1

    return units


def build_chunks(
    sections: list[Section],
    source_name: str,
    max_chars: int = 1200,
) -> list[dict]:
    chunks: list[dict] = []
    chunk_id = 0

    for section in sections:
        units = build_units(section)

        current_lines: list[tuple[int, str]] = []
        current_unit_types: list[str] = []

        for unit in units:
            unit_text = "\n".join(text for _, text in unit.lines)
            current_text = "\n".join(text for _, text in current_lines)

            if current_lines and len(current_text) + 2 + len(unit_text) > max_chars:
                line_ids = [lid for lid, _ in current_lines]
                content = "\n".join(text for _, text in current_lines)
                chunk_langs = sorted({detect_language(text) for _, text in current_lines})
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "content": content,
                        "metadata": {
                            "source": source_name,
                            "section_title": section.title,
                            "section_index": section.section_index,
                            "chunk_size": len(content),
                            "languages": chunk_langs,
                            "unit_types": sorted(set(current_unit_types)),
                            "line_ids": line_ids,
                        },
                    }
                )
                chunk_id += 1
                current_lines = []
                current_unit_types = []

            current_lines.extend(unit.lines)
            current_unit_types.append(unit.unit_type)

        if current_lines:
            line_ids = [lid for lid, _ in current_lines]
            content = "\n".join(text for _, text in current_lines)
            chunk_langs = sorted({detect_language(text) for _, text in current_lines})
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "metadata": {
                        "source": source_name,
                        "section_title": section.title,
                        "section_index": section.section_index,
                        "chunk_size": len(content),
                        "languages": chunk_langs,
                        "unit_types": sorted(set(current_unit_types)),
                        "line_ids": line_ids,
                    },
                }
            )
            chunk_id += 1

    return chunks


def find_expected_pairs(indexed_lines: list[tuple[int, str]]) -> list[tuple[int, int]]:
    pairs = []
    for i in range(len(indexed_lines) - 1):
        left_id, left = indexed_lines[i]
        right_id, right = indexed_lines[i + 1]
        if is_en_ar_pair(left, right):
            pairs.append((left_id, right_id))
    return pairs


def find_bullet_blocks(indexed_lines: list[tuple[int, str]]) -> list[list[int]]:
    blocks: list[list[int]] = []
    current: list[int] = []
    for line_id, line in indexed_lines:
        if is_bullet_like(line):
            current.append(line_id)
        else:
            if len(current) > 1:
                blocks.append(current)
            current = []
    if len(current) > 1:
        blocks.append(current)
    return blocks


def validate_chunks(indexed_lines: list[tuple[int, str]], chunks: list[dict]) -> dict:
    errors: list[str] = []

    if not chunks:
        errors.append("No chunks were created.")

    # Basic structure checks.
    for expected_id, ch in enumerate(chunks):
        if ch.get("chunk_id") != expected_id:
            errors.append(f"Chunk ID mismatch at index {expected_id}.")
            break
        if not ch.get("content", "").strip():
            errors.append(f"Chunk {expected_id} has empty content.")
        metadata = ch.get("metadata", {})
        for key in ["source", "section_title", "section_index", "chunk_size", "line_ids"]:
            if key not in metadata:
                errors.append(f"Chunk {expected_id} missing metadata field: {key}.")

    line_to_chunk: dict[int, int] = {}
    for ch in chunks:
        chunk_id = ch["chunk_id"]
        for line_id in ch["metadata"]["line_ids"]:
            line_to_chunk[line_id] = chunk_id

    # Verify EN/AR pair integrity.
    pair_violations = 0
    for left_id, right_id in find_expected_pairs(indexed_lines):
        if line_to_chunk.get(left_id) != line_to_chunk.get(right_id):
            pair_violations += 1
    if pair_violations:
        errors.append(f"Detected {pair_violations} EN/AR pair splits across chunks.")

    # Verify multi-line bullet/list blocks are not split.
    list_block_violations = 0
    for block in find_bullet_blocks(indexed_lines):
        chunk_ids = {line_to_chunk.get(line_id) for line_id in block}
        if len(chunk_ids) > 1:
            list_block_violations += 1
    if list_block_violations:
        errors.append(f"Detected {list_block_violations} split bullet/list blocks.")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "pair_violations": pair_violations,
        "list_block_violations": list_block_violations,
        "chunk_count": len(chunks),
    }


def save_chunks(chunks: list[dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def run_better_chunking(
    input_path: str = PROCESSED_TEXT_PATH,
    output_path: str = "data/processed/better_chunks.json",
) -> dict:
    indexed_lines = load_lines(input_path)
    sections = split_into_sections(indexed_lines)
    source_name = os.path.basename(input_path)
    chunks = build_chunks(sections=sections, source_name=source_name)
    validation = validate_chunks(indexed_lines=indexed_lines, chunks=chunks)

    save_chunks(chunks, output_path)

    print(f"Saved better chunks to {output_path}")
    print(f"Total source lines: {len(indexed_lines)}")
    print(f"Total sections: {len(sections)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Validation passed: {validation['valid']}")
    print(f"EN/AR pair violations: {validation['pair_violations']}")
    print(f"List block violations: {validation['list_block_violations']}")
    if validation["errors"]:
        print("Validation errors:")
        for err in validation["errors"]:
            print(f"- {err}")

    return {
        "sections": len(sections),
        "chunks": len(chunks),
        "validation": validation,
    }


if __name__ == "__main__":
    run_better_chunking()
