"""Published values used only for comparison, read from data/published_values.json.

The file is not distributed with the code; data/README.md describes its source and format. Without
it, the scripts compute and write all of their own results and leave out the comparison.
"""
import json
import os

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "published_values.json")


def get(key):
    """Values stored under key, or None if the file or the key is absent."""
    try:
        with open(PATH) as f:
            return json.load(f).get(key)
    except FileNotFoundError:
        return None
