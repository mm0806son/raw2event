#!/usr/bin/env python3
"""
Check CIFAR10 dataset collection completeness and ordering.

Outputs a text report covering:
1) Index order vs cifar10_paths.txt
2) Broken collections (missing required files)
3) Not collected images (no files at all)
4) Duplicate collections (more than one timestamp per tag)
5) Completion status (exactly one valid collection per tag)

Raw/RGB .dat and .mkv are treated as equivalent.
"""
import argparse
import os
import re
from typing import Dict, List, Set, Tuple


TIMESTAMP_RE = r"\d{8}_\d{6}"
PREFIXES = ("metadata", "raw_frames", "rgb_frames", "dv_output", "preview")
REQUIRED_TYPES = ("metadata", "raw", "rgb", "dv", "preview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CIFAR10-XDVS collection completeness.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Dataset directory containing recorded files",
    )
    parser.add_argument(
        "--paths_file",
        type=str,
        default="./data/cifar10_paths.txt",
        help="CIFAR10 image paths list file",
    )
    parser.add_argument(
        "--index_start",
        type=int,
        default=1,
        help="Starting index used in tag numbering (default: 1)",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="./collection_check_report.txt",
        help="Output report text file",
    )
    return parser.parse_args()


def load_expected_tags(paths_file: str, index_start: int) -> Dict[int, str]:
    expected = {}
    with open(paths_file, "r", encoding="utf-8") as f:
        index = index_start
        for line in f:
            img_path = line.strip()
            if not img_path:
                continue
            parts = img_path.split("/")
            class_name = parts[-2] if len(parts) >= 2 else "unknown"
            img_filename = os.path.splitext(os.path.basename(img_path))[0]
            expected[index] = f"{index}_{class_name}_{img_filename}"
            index += 1
    return expected


def build_filename_regex(prefix: str) -> re.Pattern:
    # Allow optional suffix after timestamp (e.g., *_raw_10bit.mkv)
    pattern = rf"^{prefix}_(?P<tag>.+)_(?P<ts>{TIMESTAMP_RE})(?P<suffix>_[^.]*)?\.(?P<ext>.+)$"
    return re.compile(pattern)


def classify_file(prefix: str, ext: str) -> str:
    if prefix == "metadata" and ext == "dat":
        return "metadata"
    if prefix == "raw_frames" and ext in ("dat", "mkv"):
        return "raw"
    if prefix == "rgb_frames" and ext in ("dat", "mkv"):
        return "rgb"
    if prefix == "dv_output" and ext == "aedat4":
        return "dv"
    if prefix == "preview" and ext == "png":
        return "preview"
    return ""


def minute_key(ts: str) -> str:
    # Keep only date and minute: YYYYMMDD_HHMM
    return ts[:13]


def scan_dataset(dataset_dir: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Dict[str, Set[str]]]]:
    tag_to_types: Dict[str, Set[str]] = {}
    tag_to_files: Dict[str, Set[str]] = {}
    tag_to_ts_types: Dict[str, Dict[str, Set[str]]] = {}

    compiled = {p: build_filename_regex(p) for p in PREFIXES}

    for entry in os.scandir(dataset_dir):
        if not entry.is_file():
            continue
        name = entry.name
        for prefix, regex in compiled.items():
            match = regex.match(name)
            if not match:
                continue
            tag = match.group("tag")
            ts = match.group("ts")
            ts_min = minute_key(ts)
            ext = match.group("ext")
            file_type = classify_file(prefix, ext)
            if not file_type:
                continue
            tag_to_types.setdefault(tag, set()).add(file_type)
            tag_to_files.setdefault(tag, set()).add(name)
            tag_to_ts_types.setdefault(tag, {}).setdefault(ts_min, set()).add(file_type)
            break
    return tag_to_types, tag_to_files, tag_to_ts_types


def extract_index(tag: str) -> int:
    head = tag.split("_", 1)[0]
    if head.isdigit():
        return int(head)
    return -1


def write_report(
    output_path: str,
    expected: Dict[int, str],
    tag_to_types: Dict[str, Set[str]],
    tag_to_files: Dict[str, Set[str]],
    tag_to_ts_types: Dict[str, Dict[str, Set[str]]],
) -> None:
    expected_tags = set(expected.values())
    found_tags = set(tag_to_types.keys())

    missing_all = []
    broken = []
    duplicates = []
    completion_fail = []
    order_mismatch = []
    extras = []

    for index, tag in expected.items():
        if tag not in tag_to_types:
            missing_all.append(tag)
            continue
        missing_types = [t for t in REQUIRED_TYPES if t not in tag_to_types[tag]]
        if missing_types:
            broken.append((tag, missing_types))

        ts_map = tag_to_ts_types.get(tag, {})
        if len(ts_map) > 1:
            duplicates.append((tag, sorted(ts_map.keys())))

        valid_collections = [
            ts for ts, types in ts_map.items()
            if all(t in types for t in REQUIRED_TYPES)
        ]
        if len(valid_collections) != 1:
            completion_fail.append((tag, valid_collections, sorted(ts_map.keys())))

    for tag in found_tags:
        idx = extract_index(tag)
        if idx in expected and expected[idx] != tag:
            order_mismatch.append((idx, tag, expected[idx]))
        if tag not in expected_tags:
            extras.append(tag)

    def sort_key_tag(t: str) -> Tuple[int, str]:
        return (extract_index(t), t)

    missing_all.sort(key=sort_key_tag)
    broken.sort(key=lambda x: sort_key_tag(x[0]))
    duplicates.sort(key=lambda x: sort_key_tag(x[0]))
    completion_fail.sort(key=lambda x: sort_key_tag(x[0]))
    order_mismatch.sort(key=lambda x: x[0])
    extras.sort(key=sort_key_tag)

    total_expected = len(expected)
    total_found = len(found_tags)

    lines: List[str] = []
    lines.append("CIFAR10-XDVS Collection Check Report")
    lines.append("=" * 40)
    lines.append(f"Expected images: {total_expected}")
    lines.append(f"Found tags: {total_found}")
    lines.append("")
    lines.append("1) Index order vs cifar10_paths.txt")
    if order_mismatch:
        lines.append(f"- Order mismatches: {len(order_mismatch)}")
        for idx, found_tag, expected_tag in order_mismatch:
            lines.append(f"  {idx}: found={found_tag} expected={expected_tag}")
    else:
        lines.append("- No order mismatch detected based on index->tag mapping")
    lines.append("")
    lines.append("2) Broken collections (missing required files)")
    if broken:
        lines.append(f"- Broken collections: {len(broken)}")
        for tag, missing_types in broken:
            lines.append(f"  {tag} missing={','.join(missing_types)}")
    else:
        lines.append("- No broken collections detected")
    lines.append("")
    lines.append("3) Not collected images (no files found)")
    if missing_all:
        lines.append(f"- Not collected: {len(missing_all)}")
        for tag in missing_all:
            lines.append(f"  {tag}")
    else:
        lines.append("- All images have at least one file present")
    lines.append("")
    lines.append("Extras (files not in cifar10_paths.txt)")
    if extras:
        lines.append(f"- Extra tags: {len(extras)}")
        for tag in extras:
            lines.append(f"  {tag}")
    else:
        lines.append("- No extra tags detected")
    lines.append("")
    lines.append("Notes")
    lines.append("- raw_frames/rgb_frames: .dat and .mkv are treated as equivalent")
    lines.append("- expected tag format: {index}_{class}_{image_filename}")
    lines.append("")
    lines.append("4) Duplicate collections (multiple timestamps)")
    if duplicates:
        lines.append(f"- Duplicate tags: {len(duplicates)}")
        for tag, timestamps in duplicates:
            lines.append(f"  {tag} timestamps={','.join(timestamps)}")
    else:
        lines.append("- No duplicate collections detected")
    lines.append("")
    lines.append("5) Completion status (exactly one valid collection)")
    if completion_fail:
        lines.append(f"- Not completed or duplicated: {len(completion_fail)}")
        for tag, valid_ts, all_ts in completion_fail:
            valid_str = ",".join(valid_ts) if valid_ts else "none"
            lines.append(f"  {tag} valid={valid_str} all={','.join(all_ts)}")
    else:
        lines.append("- All tags have exactly one valid collection")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    expected = load_expected_tags(args.paths_file, args.index_start)
    tag_to_types, tag_to_files, tag_to_ts_types = scan_dataset(args.dataset_dir)
    write_report(args.output_report, expected, tag_to_types, tag_to_files, tag_to_ts_types)
    print(f"Report written to: {args.output_report}")


if __name__ == "__main__":
    main()

