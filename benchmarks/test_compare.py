"""Runner schema checks, without starting benchmark processes."""
import unittest
from compare import records


class Records(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(records("test benchmark ... FENSOR,case,row,warm,wall,ns,42\n"),
                         [["case", "row", "warm", "wall", "ns", "42"]])

    def test_rejects_malformed_missing_and_duplicate(self):
        row = "FENSOR,case,row,warm,wall,ns,42\n"
        for text in ["", row + row, row.replace(",42", ",-1"), row.replace(",ns,", ",seconds,"), "FENSOR,a,b"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                records(text)


if __name__ == "__main__":
    unittest.main()
