#!/usr/bin/env python3
"""Print the module and code-cell inventory from the canonical notebook."""
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "notebooks" / "AIML_A05_COMPLETE_END_TO_END_COLAB.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))
print(f"Notebook: {p.name} | cells: {len(nb.get('cells', []))}")
for i, cell in enumerate(nb.get("cells", []), start=1):
    source = "".join(cell.get("source", [])).strip()
    if not source:
        continue
    first = next((line.strip() for line in source.splitlines() if line.strip()), "")
    if cell.get("cell_type") == "markdown" and first.startswith("## "):
        print(f"\n{first}")
    elif cell.get("cell_type") == "code" and first.startswith("#"):
        print(f"  cell {i}: {first}")
