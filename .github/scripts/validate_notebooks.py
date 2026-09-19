"""Validate all notebook files in the repository.

Checks performed:
- File parses as nbformat 4.
- notebook has at least one cell.
- No cell has an empty source.
- Code cells do not contain obvious merge-conflict markers.
- The optional execution allowlist is respected by the workflow, not here.

Usage: python .github/scripts/validate_notebooks.py
Exits non-zero if any notebook fails.
"""

import glob
import sys

import nbformat

CONFLICT = ("<<<<<<<", "=======", ">>>>>>>")


def validate(path):
    with open(path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    nbformat.validate(nb)
    if not nb.cells:
        raise ValueError("notebook has no cells")
    for i, cell in enumerate(nb.cells):
        src = "".join(cell.get("source", ""))
        if cell.cell_type in ("markdown", "code") and not src.strip():
            raise ValueError(f"cell {i} is empty")
        if any(marker in src for marker in CONFLICT):
            raise ValueError(f"cell {i} contains a merge-conflict marker")
    return len(nb.cells)


def main():
    files = sorted(glob.glob("notebooks/**/*.ipynb", recursive=True))
    if not files:
        print("No notebooks found.")
        return 1
    failed = []
    for path in files:
        try:
            n = validate(path)
            print(f"OK   {path} ({n} cells)")
        except Exception as exc:  # noqa: BLE001
            failed.append(path)
            print(f"FAIL {path}: {exc}")
    print(f"\n{len(files) - len(failed)}/{len(files)} notebooks valid.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
