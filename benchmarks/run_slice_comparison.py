"""Run separate release binaries sequentially; setup/copying is not timed.

Build slice_benchmark and read_benchmark in both target directories first.
Example: python3 benchmarks/run_slice_comparison.py BEFORE AFTER --root DIRECTORY
"""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
parser.add_argument("--root", type=Path, default=Path(tempfile.gettempdir()),
                    help="parent directory for temporary benchmark data")
parser.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "slices",
                    help="directory for CSV and logs (default: %(default)s)")
parser.add_argument("--pairs", type=int, default=2)
parser.add_argument("--start-pair", type=int, default=0,
                    help="append additional pairs starting at this index")
args = parser.parse_args()
if "FENSOR_BENCH_SMOKE" in os.environ:
    parser.error("unset FENSOR_BENCH_SMOKE before recording comparison measurements")
args.output.mkdir(parents=True, exist_ok=True)

with (args.output / "timings.csv").open("a" if args.start_pair else "w") as output:
    writer = csv.writer(output, lineterminator="\n")
    if not args.start_pair:
        writer.writerow([
            "threads", "pair", "version", "suite", "case", "mode",
            "temperature", "nanoseconds", "elements", "loads", "saves",
        ])
    for threads in [1, 4]:
        for pair in range(args.start_pair, args.pairs):
            versions = [("before", args.before), ("after", args.after)]
            if pair % 2:
                versions.reverse()
            for version, target in versions:
                for suite, prefix in [("slice", "SLICE"), ("read", "READ")]:
                    binaries = [p for p in (target / "release/deps").glob(
                        f"{suite}_benchmark-*") if p.is_file() and os.access(p, os.X_OK)]
                    if len(binaries) != 1:
                        raise RuntimeError(f"expected one {suite} binary in {target}")
                    label = f"{threads}-{pair}-{version}-{suite}"
                    print(label, flush=True)
                    with tempfile.TemporaryDirectory(prefix="fensor-slice-", dir=args.root) as tmp:
                        env = dict(os.environ, TMPDIR=tmp, RAYON_NUM_THREADS=str(threads))
                        result = subprocess.run(
                            [str(binaries[0]), "--ignored", "--nocapture"],
                            env=env, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                        )
                    (args.output / f"{label}.log").write_text(result.stdout)
                    result.check_returncode()
                    for line in result.stdout.splitlines():
                        # libtest may prefix the first record with "test NAME ... ".
                        _, marker, record = line.partition(prefix + ",")
                        if marker:
                            case, mode, temperature, elapsed, count, loads, saves = record.split(",")
                            nanos = int(elapsed) * (1000 if suite == "read" else 1)
                            writer.writerow([threads, pair, version, suite,
                                             case, mode, temperature, nanos, count, loads, saves])
                    output.flush()
