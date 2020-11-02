# Lint as: python3
"""Automatic best effort span cleaner for SemEval 2021 Toxic Spans."""

import ast
import csv
import itertools
import string
import sys

SPECIAL_CHARACTERS = string.whitespace

def _contiguous_ranges(span_list):
    """Extracts continguous runs [1, 2, 3, 5, 6, 7] -> [(1,3), (5,7)]."""
    output = []
    for _, span in itertools.groupby(
        enumerate(span_list), lambda p: p[1] - p[0]):
        span = list(span)
        output.append((span[0][1], span[-1][1]))
    return output


def fix_spans(spans, text, special_characters=SPECIAL_CHARACTERS):
    """Applies minor edits to trim spans and remove singletons."""
    cleaned = []
    for begin, end in _contiguous_ranges(spans):
        while text[begin] in special_characters and begin < end:
            begin += 1
        while text[end] in special_characters and begin < end:
            end -= 1
        if end - begin > 1:
            cleaned.extend(range(begin, end + 1))
    return cleaned


def main():
    """Processes fixes by reading from stdin, writing to stdout."""
    reader = csv.reader(sys.stdin)
    writer = csv.writer(sys.stdout)
    header = True
    for row in reader:
        fixed_spans = row[0]
        if not header:
            fixed_spans = _fix_spans(ast.literal_eval(row[0]), row[1])
        writer.writerow([fixed_spans, row[1]])
        header = False


if __name__ == '__main__':
    main()
