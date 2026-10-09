"""Runner schema checks, without starting benchmark processes."""

import csv
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compare import CSV_FIELDS, main, records, summarize


class Records(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(records("test benchmark ... FENSOR,case,row,warm,wall,ns,42\n"),
                         [["case", "row", "warm", "wall", "ns", "42"]])

    def test_explicit_caller_prefix(self):
        text = "COLLECTION,case,scan,warm,wall,ns,42\n"
        self.assertEqual(records(text, "COLLECTION"), [["case", "scan", "warm", "wall", "ns", "42"]])
        with self.assertRaises(ValueError):
            records(text)

    def test_rejects_malformed_missing_and_duplicate(self):
        row = "FENSOR,case,row,warm,wall,ns,42\n"
        for text in ["", row + row, row.replace(",42", ",-1"), row.replace(",ns,", ",seconds,"), "FENSOR,a,b"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                records(text)


class Summaries(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "comparison.csv"

    def row(self, value, phase="before", pair=0, **updates):
        return dict(zip(CSV_FIELDS, [phase, ("a" if phase == "before" else "b") * 64,
                                    phase, "1", str(pair), "case", "copy-attempt", "new",
                                    "wall", "ns", str(value)])) | updates

    def write(self, rows):
        with self.path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return self.path

    def test_medians_zero_baselines_and_thread_groups(self):
        for before, after, expected in [
            ([10, 30, 20], [30, 50, 40], "before=20 after=40 delta=+20 change=+100.00%"),
            ([0], [2], "before=0 after=2 delta=+2 change=n/a (zero baseline)"),
            ([0], [0], "before=0 after=0 delta=+0 change=n/a (zero baseline)"),
        ]:
            with self.subTest(before=before, after=after):
                rows = [self.row(value, phase, pair)
                        for phase, values in [("before", before), ("after", after)]
                        for pair, value in enumerate(values)]
                rows += [self.row(value, phase, pair, threads="4")
                         for pair in range(len(before))
                         for phase, value in [("before", 8), ("after", 4)]]
                summary = summarize(self.write(rows))
                self.assertIn(expected, summary)
                self.assertIn("threads=4", summary)
                self.assertIn("before=8 after=4 delta=-4 change=-50.00%", summary)

    def test_rejects_incomplete_duplicate_and_inconsistent_records(self):
        rows = [self.row(10), self.row(20, "after")]
        cases = [[], rows[:1], rows + rows[:1],
                 [rows[0], rows[1] | {"metric": "other"}],
                 [row | {"pair": "1"} for row in rows]]
        cases.append(rows + [self.row(10, pair=1), self.row(20, "after", 1),
                             self.row(10, threads="4"), self.row(20, "after", threads="4")])
        for column, value in [("phase", "unknown"), ("threads", "0"), ("pair", "-1"),
                              ("value", "-1"), ("value", "1.5"), ("unit", "seconds"),
                              ("revision", ""), ("executable_sha256", "bad")]:
            cases.append([rows[0] | {column: value}, rows[1]])
        for column, value in [("revision", "changed"), ("executable_sha256", "c" * 64)]:
            cases.append(rows + [self.row(10, pair=1, **{column: value}), self.row(20, "after", 1)])
        for case in cases:
            with self.subTest(rows=case), self.assertRaises(ValueError):
                summarize(self.write(case))
        for text in ["", "wrong,columns\n", ",".join(CSV_FIELDS) + "\nincomplete\n"]:
            self.path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                summarize(self.path)

    def admission_rows(self, outcomes):
        rows = []
        for pair, outcome in enumerate(outcomes):
            for phase, admitted in zip(["before", "after"], outcome):
                rows.extend([self.row(100 if phase == "before" else 200, phase, pair),
                             self.row(admitted, phase, pair, metric="admitted", unit="count"),
                             self.row(1 - admitted, phase, pair, metric="capacity_rejected", unit="count")])
        return rows

    def test_admission_filtering_and_mismatches(self):
        for outcomes, expected in [
            ([(1, 1), (0, 0), (1, 0), (0, 1)], "pairs=4 n=1 excluded=3 before=100 after=200"),
            ([(0, 0), (1, 0)], "pairs=2 n=0 excluded=2 before=n/a after=n/a"),
        ]:
            with self.subTest(outcomes=outcomes):
                summary = summarize(self.write(self.admission_rows(outcomes)))
                self.assertIn(expected, summary)
                self.assertIn("admitted / count", summary)
                self.assertIn("capacity_rejected / count", summary)
        rows = self.admission_rows([(1, 1)])
        for case in [[row for row in rows if row["metric"] != "capacity_rejected"],
                     [row | {"value": "2"} if row["metric"] == "admitted" else row for row in rows],
                     [row | {"value": "1"} if row["metric"] == "capacity_rejected" else row for row in rows]]:
            with self.subTest(rows=case), self.assertRaises(ValueError):
                summarize(self.write(case))

    def test_completed_comparison_prints_summary_and_preserves_csv(self):
        before = self.path.parent / "before"
        after = self.path.parent / "after"
        before.write_bytes(b"before executable")
        after.write_bytes(b"after executable")
        args = ["compare.py", str(before), str(after), "--pairs", "1", "--threads", "1",
                "--before-revision", "before", "--after-revision", "after", "--output", str(self.path)]
        result = subprocess.CompletedProcess([], 0, stdout="FENSOR,case,row,warm,wall,ns,10\n")
        with patch("sys.argv", args), patch("compare.subprocess.run", return_value=result) as run, \
                patch("builtins.print") as output:
            main()
            self.assertEqual(run.call_count, 2)
            self.assertIn("before=10 after=10 delta=+0", output.call_args.args[0])
        with self.path.open() as source:
            reader = csv.DictReader(source)
            self.assertEqual(reader.fieldnames, CSV_FIELDS)
            self.assertEqual([row["phase"] for row in reader], ["before", "after"])

    def test_offline_cli_does_not_run_benchmarks(self):
        self.write([self.row(10), self.row(20, "after")])
        with patch("sys.argv", ["compare.py", "--summarize", str(self.path)]), \
                patch("compare.subprocess.run") as run, patch("builtins.print") as output:
            main()
            run.assert_not_called()
            self.assertIn("before=10 after=20", output.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
