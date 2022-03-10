# Lint as: python3
"""Tests for semeval2021."""

import io
import unittest

import semeval2021


gold_data = """0\t[12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
1\t[55, 56, 57, 58, 59, 60]
2\t[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 176, 177, 178, 179, 180]
3\t[]"""

class Semeval2021Test(unittest.TestCase):

  def test_evaluate_perfect(self):
    scores = semeval2021.evaluate(io.StringIO(
        "0\t[12, 13, 14, 15, 16, 17, 18, 19, 20, 21]\n"
        "1\t[55, 56, 57, 58, 59, 60]\n"
        "2\t[80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 176, 177, 178, 179, 180]\n"
        "3\t[]"),
        io.StringIO(gold_data))
    self.assertEqual(scores, (1.0, 0.0))

  def test_evaluate_empty(self):
    scores = semeval2021.evaluate(
        io.StringIO("0\t[]\n1\t[]\n2\t[]\n3\t[]"),
        io.StringIO(gold_data))
    self.assertEqual(scores, (0.25, 0.25))

  def test_evaluate_wrong_key(self):
    with self.assertRaises(ValueError):
      scores = semeval2021.evaluate(
          io.StringIO("8\t[]\n1\t[]\n2\t[]\n3\t[]"),
          io.StringIO(gold_data))


if __name__ == '__main__':
  unittest.main()
