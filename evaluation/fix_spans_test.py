# Lint as: python3
"""Tests for fix_spans."""

import unittest

import fix_spans


class FixSpansTest(unittest.TestCase):

    def test_leading_spaces(self):
        self.assertEqual(fix_spans.fix_spans(
            [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14],
            'To infinity and    beyond.'),
                         [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14])

    def test_ignore_space_in_span(self):
        self.assertEqual(fix_spans.fix_spans(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            'To infinity and   beyond.'),
                         [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])

    def test_trailing_spaces(self):
        self.assertEqual(fix_spans.fix_spans(
            [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22,
             23, 24], 'To infinity and   beyond.'),
                         [3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 19, 20, 21, 22,
                          23, 24])

    def test_drop_singletons(self):
        self.assertEqual(fix_spans.fix_spans(
            [3, 4, 5, 6, 7, 8, 9, 10, 13, 19, 20, 21, 22, 23, 24],
            'To infinity and    beyond.'),
                         [3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24])
    def test_special_characters(self):
        self.assertEqual(fix_spans.fix_spans(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22,
             23, 24, 25, 26], 'To \tinfinity and\n   beyond.'),
                         [4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 21, 22,
                          23, 24, 25, 26])


if __name__ == '__main__':
    unittest.main()
