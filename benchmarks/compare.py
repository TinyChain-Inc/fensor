"""Alternate frozen benchmark executables; emit validated long-form records."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile

FIELDS = ["workload", "operation", "temperature", "metric", "unit", "value"]


def records(text):
    rows = []
    keys = set()
    for line in text.splitlines():
        _, marker, payload = line.partition("FENSOR,")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--entry", choices=["benchmark", "profiling::profile"], default="benchmark")
    parser.add_argument("--suite", choices=["all", "matrix", "reduction", "completion", "pipeline", "copy"], default="all")
    parser.add_argument("--pairs", type=int, default=2)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--before-revision", required=True)
    parser.add_argument("--after-revision", required=True)
    parser.add_argument("--root", type=Path, default=Path(tempfile.gettempdir()))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results" / "comparison.csv")
    args = parser.parse_args()
    if "FENSOR_BENCH_SMOKE" in os.environ or args.pairs < 1 or min(args.threads) < 1:
        parser.error("unset smoke and use positive pair/thread counts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expected = None
    with args.output.open("w") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["revision", "executable_sha256", "version", "threads", "pair", *FIELDS])
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
                            env=os.environ | {"TMPDIR": root, "RAYON_NUM_THREADS": str(threads), "FENSOR_BENCH_SUITE": args.suite},
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        )
                    log.write_text(result.stdout)
                    result.check_returncode()
                    rows = records(result.stdout)
                    keys = {tuple(row[:-1]) for row in rows}
                    if expected is None:
                        expected = keys
                    elif keys != expected:
                        raise ValueError("measurement cases differ across runs")
                    writer.writerows([revision, digest, version, threads, pair, *row] for row in rows)
                    output.flush()


if __name__ == "__main__":
    main()
