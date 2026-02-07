from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SplitCount:
    split: str
    output_json: int
    error_json: int


def find_targets(run_dir: Path) -> list[Path]:
    return sorted([p for p in run_dir.iterdir() if p.is_dir()])


def find_splits(target_dir: Path) -> list[Path]:
    splits = []
    for p in target_dir.iterdir():
        if p.is_dir() and p.name not in {"_cache", "cache"}:
            splits.append(p)
    return sorted(splits)


def count_json_under_outputs(split_dir: Path) -> int:
    outputs_dir = split_dir / "outputs"
    if not outputs_dir.exists():
        return 0
    return sum(1 for p in outputs_dir.rglob("*.json") if p.is_file())


def count_json_under_errors(split_dir: Path) -> int:
    errors_dir = split_dir / "errors"
    if not errors_dir.exists():
        return 0
    return sum(1 for p in errors_dir.rglob("*.json") if p.is_file())


def collect_counts(run_dir: Path) -> dict[str, list[SplitCount]]:
    result: dict[str, list[SplitCount]] = {}
    for target_dir in find_targets(run_dir):
        target_name = target_dir.name
        split_counts: list[SplitCount] = []

        for split_dir in find_splits(target_dir):
            split_name = split_dir.name
            output_json = count_json_under_outputs(split_dir)
            error_json = count_json_under_errors(split_dir)
            split_counts.append(
                SplitCount(
                    split=split_name,
                    output_json=output_json,
                    error_json=error_json,
                )
            )

        result[target_name] = split_counts
    return result


def print_report(run_dir: Path, counts: dict[str, list[SplitCount]]):
    print(f"Run dir: {run_dir}")
    print("")

    for target_name in sorted(counts.keys()):
        rows = counts[target_name]
        print(f"[{target_name}]")

        if len(rows) == 0:
            print("  (no splits)")
            print("")
            continue

        split_w = max(len("split"), max(len(r.split) for r in rows))
        out_w = max(len("outputs_json"), max(len(str(r.output_json)) for r in rows))
        err_w = max(len("errors_json"), max(len(str(r.error_json)) for r in rows))

        header = f"  {'split'.ljust(split_w)}  {'outputs_json'.rjust(out_w)}  {'errors_json'.rjust(err_w)}"
        print(header)
        print(f"  {'-' * split_w}  {'-' * out_w}  {'-' * err_w}")

        for r in rows:
            line = f"  {r.split.ljust(split_w)}  {str(r.output_json).rjust(out_w)}  {str(r.error_json).rjust(err_w)}"
            print(line)

        total_out = sum(r.output_json for r in rows)
        total_err = sum(r.error_json for r in rows)
        print(f"  {'TOTAL'.ljust(split_w)}  {str(total_out).rjust(out_w)}  {str(total_err).rjust(err_w)}")
        print("")


def main():
    run_dir = Path("runs/SH17")  # 필요하면 여기만 바꾸시면 됩니다.
    counts = collect_counts(run_dir)
    print_report(run_dir, counts)


if __name__ == "__main__":
    main()
