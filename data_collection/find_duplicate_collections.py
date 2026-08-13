#!/usr/bin/env python3
"""
Find duplicate collections (same tag with multiple timestamps).

Outputs a report listing files with earlier timestamps (older duplicates).
"""
import argparse
import os
import re
from typing import Dict, List, Tuple


TIMESTAMP_RE = r"\d{8}_\d{6}"
PREFIXES = ("metadata", "raw_frames", "rgb_frames", "dv_output", "preview")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find duplicate collections by tag.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Dataset directory containing recorded files",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="./duplicate_collections_report.txt",
        help="Output report file",
    )
    parser.add_argument(
        "--remove_older",
        action="store_true",
        help="Remove files from older hours (keep latest hour per tag)",
    )
    parser.add_argument(
        "--compressed_report",
        type=str,
        default="./compressed_dat_report.txt",
        help="Output report file for .dat files that already have compressed .mkv",
    )
    parser.add_argument(
        "--remove_compressed_dat",
        action="store_true",
        help="Remove .dat files that already have compressed .mkv",
    )
    return parser.parse_args()


def build_filename_regex(prefix: str) -> re.Pattern:
    # Allow optional suffix after timestamp (e.g., *_raw_10bit.mkv)
    pattern = rf"^{prefix}_(?P<tag>.+)_(?P<ts>{TIMESTAMP_RE})(?P<suffix>_[^.]*)?\.(?P<ext>.+)$"
    return re.compile(pattern)


def hour_key(ts: str) -> str:
    # Keep only date and hour: YYYYMMDD_HH
    return ts[:11]


def scan_dataset(dataset_dir: str) -> Dict[str, Dict[str, List[str]]]:
    # tag -> hour_key -> [files]
    tag_map: Dict[str, Dict[str, List[str]]] = {}
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
            hour = hour_key(ts)
            tag_map.setdefault(tag, {}).setdefault(hour, []).append(name)
            break
    return tag_map


def find_compressed_dat(dataset_dir: str) -> List[Tuple[str, str]]:
    # Returns list of (dat_filename, mkv_filename)
    files = {entry.name for entry in os.scandir(dataset_dir) if entry.is_file()}
    compressed: List[Tuple[str, str]] = []
    for name in files:
        if not name.endswith(".dat"):
            continue
        if not (name.startswith("raw_frames_") or name.startswith("rgb_frames_")):
            continue
        base = os.path.splitext(name)[0]
        if name.startswith("raw_frames_"):
            candidates = [f"{base}_raw_10bit.mkv", f"{base}_raw.mkv"]
        else:
            candidates = [f"{base}_rgb.mkv"]
        for mkv in candidates:
            if mkv in files:
                compressed.append((name, mkv))
                break
    return sorted(compressed, key=lambda x: x[0])


def pick_latest_hour(hours: List[str]) -> str:
    # Hour key format: YYYYMMDD_HH, lexicographic order works
    return sorted(hours)[-1]


def write_report(output_path: str, tag_map: Dict[str, Dict[str, List[str]]]) -> None:
    duplicates = []
    for tag, ts_map in tag_map.items():
        if len(ts_map) <= 1:
            continue
        hours = sorted(ts_map.keys())
        latest = pick_latest_hour(hours)
        older = [h for h in hours if h != latest]
        duplicates.append((tag, latest, older))

    duplicates.sort(key=lambda x: x[0])

    lines: List[str] = []
    lines.append("Duplicate Collections Report")
    lines.append("=" * 30)
    lines.append(f"Tags with multiple timestamps: {len(duplicates)}")
    lines.append("")

    for tag, latest, older in duplicates:
        lines.append(f"Tag: {tag}")
        lines.append(f"  Latest hour: {latest}")
        lines.append(f"  Older hours: {', '.join(older)}")
        for ts in older:
            files = sorted(tag_map[tag][ts])
            for name in files:
                lines.append(f"    {name}")
        lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_compressed_report(output_path: str, dat_mkv_pairs: List[Tuple[str, str]]) -> None:
    lines: List[str] = []
    lines.append("Compressed DAT Report")
    lines.append("=" * 30)
    lines.append(f"DAT files with compressed MKV: {len(dat_mkv_pairs)}")
    lines.append("")
    for dat_name, mkv_name in dat_mkv_pairs:
        lines.append(f"{dat_name} -> {mkv_name}")
    lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def remove_older_files(tag_map: Dict[str, Dict[str, List[str]]], dataset_dir: str) -> int:
    removed = 0
    for tag, hour_map in tag_map.items():
        if len(hour_map) <= 1:
            continue
        hours = sorted(hour_map.keys())
        latest = pick_latest_hour(hours)
        for hour in hours:
            if hour == latest:
                continue
            for name in hour_map[hour]:
                path = os.path.join(dataset_dir, name)
                if os.path.exists(path):
                    try:
                        os.remove(path)
                        removed += 1
                    except Exception as exc:
                        print(f"Failed to remove {path}: {exc}")
    return removed


def remove_compressed_dat(dat_mkv_pairs: List[Tuple[str, str]], dataset_dir: str) -> int:
    removed = 0
    for dat_name, _ in dat_mkv_pairs:
        path = os.path.join(dataset_dir, dat_name)
        if os.path.exists(path):
            try:
                os.remove(path)
                removed += 1
            except Exception as exc:
                print(f"Failed to remove {path}: {exc}")
    return removed


def main() -> None:
    args = parse_args()
    tag_map = scan_dataset(args.dataset_dir)
    write_report(args.output_report, tag_map)
    print(f"Report written to: {args.output_report}")
    dat_mkv_pairs = find_compressed_dat(args.dataset_dir)
    write_compressed_report(args.compressed_report, dat_mkv_pairs)
    print(f"Compressed DAT report written to: {args.compressed_report}")
    if args.remove_older:
        removed = remove_older_files(tag_map, args.dataset_dir)
        print(f"Removed older duplicate files: {removed}")
    if args.remove_compressed_dat:
        removed = remove_compressed_dat(dat_mkv_pairs, args.dataset_dir)
        print(f"Removed compressed DAT files: {removed}")


if __name__ == "__main__":
    main()

