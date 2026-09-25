"""Alternate two frozen release executables; setup is outside printed timings."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("before", type=Path)
parser.add_argument("after", type=Path)
parser.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parent / "results" / "concurrency.csv",
                    help="CSV output file (default: %(default)s)")
parser.add_argument("--pairs", type=int, default=2)
parser.add_argument("--root", type=Path, default=Path(tempfile.gettempdir()),
                    help="parent directory for temporary benchmark data")
parser.add_argument("--focus-wide", action="store_true", help="repeat wide/dense/1 MB/capacity 31")
parser.add_argument("--expected-records", type=int, default=84, help="records per full run (208 for completion_benchmark)")
args = parser.parse_args()
if "FENSOR_BENCH_SMOKE" in os.environ:
    parser.error("unset FENSOR_BENCH_SMOKE for measurements")
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w") as output:
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["threads", "pair", "version", "case", "layout", "cache_bytes",
                     "capacity", "phase", "nanoseconds", "elements", "loads", "saves"])
    for threads in [1, 4]:
        for pair in range(args.pairs):
            versions = [("before", args.before), ("after", args.after)]
            if pair % 2:
                versions.reverse()
            for version, binary in versions:
                print(threads, pair, version, flush=True)
                environment = os.environ | {"RAYON_NUM_THREADS": str(threads)}
                if args.focus_wide:
                    environment["FENSOR_BENCH_FOCUS_WIDE"] = "1"
                else:
                    environment.pop("FENSOR_BENCH_FOCUS_WIDE", None)
                with tempfile.TemporaryDirectory(prefix="fensor-pipeline-", dir=args.root) as root:
                    result = subprocess.run(
                        [str(binary.resolve()), "--ignored", "--nocapture", "--test-threads=1"],
                        env=environment | {"TMPDIR": root},
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
                    )
                rows = []
                for line in result.stdout.splitlines():
                    _, found, payload = line.partition("PIPELINE,")
                    if found:
                        fields = payload.split(",")
                        if len(fields) != 9:
                            raise ValueError(line)
                        rows.append([threads, pair, version, *fields])
                expected = 3 if args.focus_wide else args.expected_records
                if len(rows) != expected:
                    raise ValueError(f"expected {expected} records, received {len(rows)}")
                writer.writerows(rows)
                output.flush()
