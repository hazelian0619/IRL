"""Audit REAL vs SIMULATED data snapshots based on metadata.json files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


def load_metadata(root: Path) -> Iterable[Tuple[Path, Dict]]:
    for meta_path in root.rglob('metadata.json'):
        try:
            data = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: failed to parse {meta_path}: {exc}")
            continue
        yield meta_path.parent, data


def audit(args: argparse.Namespace) -> int:
    roots = [Path(r).expanduser().resolve() for r in args.roots]
    summary = Counter()
    missing = []
    violations = []

    for root in roots:
        if not root.exists():
            print(f"WARNING: root {root} does not exist")
            continue
        for folder, meta in load_metadata(root):
            label = str(meta.get('data_type', 'UNKNOWN')).upper()
            summary[label] += 1
            if args.required_label and label != args.required_label.upper():
                violations.append((folder, label, 'expected ' + args.required_label))
            if label in args.fail_labels:
                violations.append((folder, label, 'explicitly forbidden'))
            if args.require_agent and not meta.get('agent'):
                missing.append((folder, 'agent'))
            if args.require_simulation and not meta.get('simulation'):
                missing.append((folder, 'simulation'))

        # 检查是否存在无metadata的最末级目录
        for sim_dir in root.glob('*/*'):
            if sim_dir.is_dir() and not (sim_dir / 'metadata.json').exists():
                missing.append((sim_dir, 'metadata'))

    print('──── Data Type Summary ────')
    for label, count in summary.most_common():
        print(f"  {label:<12} {count:>4}")

    if missing:
        print('\n──── Missing Fields ────')
        for folder, field in missing:
            print(f"  {folder}: missing {field}")

    if violations:
        print('\n──── Violations ────')
        for folder, label, reason in violations:
            print(f"  {folder}: label={label} ({reason})")
        return 1

    print('\nAudit completed without blockers.')
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Audit ingested data folders.')
    parser.add_argument('--roots', nargs='+', default=['data/real_ingest'],
                        help='待审计的根目录 (默认 data/real_ingest)')
    parser.add_argument('--required-label', default='REAL',
                        help='期望的data_type标签，若不检查则留空')
    parser.add_argument('--fail-labels', default='SIMULATED',
                        help='逗号分隔的禁止标签列表 (默认: SIMULATED)')
    parser.add_argument('--require-agent', action='store_true',
                        help='metadata必须包含agent字段')
    parser.add_argument('--require-simulation', action='store_true',
                        help='metadata必须包含simulation字段')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.fail_labels = {label.strip().upper() for label in args.fail_labels.split(',') if label.strip()}
    exit_code = audit(args)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
