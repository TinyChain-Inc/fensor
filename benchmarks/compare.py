"""Alternate frozen benchmark executables; emit validated long-form records."""

import argparse
import csv
import hashlib
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median

FIELDS = ["workload", "operation", "temperature", "metric", "unit", "value"]
CSV_FIELDS = ["revision", "executable_sha256", "phase", "threads", "pair", *FIELDS]


def records(text, prefix="FENSOR"):
    rows = []
    keys = set()
    for line in text.splitlines():
        _, marker, payload = line.partition(f"{prefix},")
        if not marker:
            continue
        fields = next(csv.reader([payload]))
        if len(fields) != len(FIELDS) or fields[4] not in ("ns", "count"):
            raise ValueError(f"invalid record: {line}")
        if not fields[5].isdigit() or not all(fields):
            raise ValueError(f"invalid value: {line}")
        key = tuple(fields[:-1])
        if key in keys:
            raise ValueError(f"duplicate measurement: {key}")
        keys.add(key)
        rows.append(fields)
    if not rows:
        raise ValueError("no benchmark records")
    return rows


def summarize(path):
    """Describe paired measurements; timings never determine pass/fail."""
    runs = defaultdict(dict)
    identities = {}
    with Path(path).open(newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError("unexpected comparison CSV columns")
        for row in reader:
            if None in row or any(value is None or not value.strip() for value in row.values()):
                raise ValueError("incomplete comparison row")
            phase = row["phase"]
            digest = row["executable_sha256"]
            if phase not in ("before", "after") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid comparison provenance")
            identity = row["revision"], digest
            if identities.setdefault(phase, identity) != identity:
                raise ValueError(f"inconsistent {phase} provenance")
            if any(not row[name].isdigit() for name in ("threads", "pair", "value")) or row["unit"] not in ("ns", "count"):
                raise ValueError("invalid comparison measurement")
            threads, pair, value = (int(row[name]) for name in ("threads", "pair", "value"))
            if threads < 1:
                raise ValueError("thread count must be positive")
            key = tuple(row[name] for name in FIELDS[:-1])
            run = runs[threads, pair, phase]
            if key in run:
                raise ValueError(f"duplicate measurement: {key}")
            run[key] = value
    if not runs:
        raise ValueError("no comparison measurements")
    expected = set(next(iter(runs.values())))
    if any(set(run) != expected for run in runs.values()):
        raise ValueError("measurement cases differ across runs")
    pairs = defaultdict(set)
    for threads, pair, phase in runs:
        pairs[threads].add(pair)
        if (threads, pair, "after" if phase == "before" else "before") not in runs:
            raise ValueError("missing paired run")
    expected_pairs = next(iter(pairs.values()))
    for selected in pairs.values():
        if selected != expected_pairs or selected != set(range(len(selected))):
            raise ValueError("pair numbers must match across threads and be contiguous from zero")
    admissions = {key[:3] for key in expected if key[3] in ("admitted", "capacity_rejected")}
    for group in admissions:
        admitted = (*group, "admitted", "count")
        rejected = (*group, "capacity_rejected", "count")
        if admitted not in expected or rejected not in expected:
            raise ValueError("incomplete admission outcome")
        for run in runs.values():
            if run[admitted] not in (0, 1) or run[rejected] != 1 - run[admitted]:
                raise ValueError("invalid admission outcome")

    lines = [f"{phase}: {identities[phase][0]} {identities[phase][1]}" for phase in ("before", "after")]
    lines.append("Medians describe observations, not statistical significance; admission-filtered timings include only pairs successful on both sides.")
    for threads, selected in sorted(pairs.items()):
        for key in sorted(expected):
            included = sorted(selected)
            if key[-1] == "ns" and key[:3] in admissions:
                admitted = (*key[:3], "admitted", "count")
                included = [pair for pair in included if all(runs[threads, pair, phase][admitted] for phase in ("before", "after"))]
            prefix = f"threads={threads} {' / '.join(key)}: pairs={len(selected)} n={len(included)} excluded={len(selected) - len(included)}"
            if not included:
                lines.append(prefix + " before=n/a after=n/a delta=n/a change=n/a")
                continue
            before, after = (median(runs[threads, pair, phase][key] for pair in included) for phase in ("before", "after"))
            change = f"{100 * (after - before) / before:+.2f}%" if before else "n/a (zero baseline)"
            lines.append(f"{prefix} before={before:g} after={after:g} delta={after - before:+g} change={change}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, nargs="?")
    parser.add_argument("after", type=Path, nargs="?")
    parser.add_argument("--entry", default="benchmark")
    parser.add_argument("--record-prefix", default="FENSOR")
    parser.add_argument("--suite-env", default="FENSOR_BENCH_SUITE")
    parser.add_argument("--suite", default="all")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--before-revision")
    parser.add_argument("--after-revision")
    parser.add_argument("--root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "comparison.csv")
    parser.add_argument("--summarize", type=Path, metavar="CSV", help="summarize existing paired records without running benchmarks")
    args = parser.parse_args()
    if args.summarize:
        if any((args.before, args.after, args.before_revision, args.after_revision)):
            parser.error("--summarize cannot be combined with executable or revision arguments")
        try:
            print(summarize(args.summarize))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        return
    if not all((args.before, args.after, args.before_revision, args.after_revision)):
        parser.error("before/after executables and revision labels are required")
    if any(name.endswith("_BENCH_SMOKE") for name in os.environ) or args.pairs < 1 or min(args.threads) < 1:
        parser.error("unset smoke and use positive pair/thread counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected = None
    with args.output.open("w") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(CSV_FIELDS)
        for threads in args.threads:
            for pair in range(args.pairs):
                versions = [("before", args.before, args.before_revision), ("after", args.after, args.after_revision)]
                for version, binary, revision in versions[::1 if pair % 2 == 0 else -1]:
                    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
                    print(threads, pair, version, flush=True)
                    log = args.output.with_name(f"{args.output.stem}-{threads}-{pair}-{version}.log")
                    with tempfile.TemporaryDirectory(prefix="fensor-benchmark-", dir=args.root) as root:
                        result = subprocess.run(
                            [str(binary.resolve()), args.entry, "--exact", "--ignored", "--nocapture", "--test-threads=1"],
                            env=os.environ | {"TMPDIR": root, "RAYON_NUM_THREADS": str(threads), args.suite_env: args.suite},
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        )
                    log.write_text(result.stdout)
                    result.check_returncode()
                    rows = records(result.stdout, args.record_prefix)
                    keys = {tuple(row[:-1]) for row in rows}
                    if expected is None:
                        expected = keys
                    elif keys != expected:
                        raise ValueError("measurement cases differ across runs")
                    writer.writerows([revision, digest, version, threads, pair, *row] for row in rows)
                    output.flush()

    print(summarize(args.output))


if __name__ == "__main__":
    main()
