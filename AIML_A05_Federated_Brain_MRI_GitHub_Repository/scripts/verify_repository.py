#!/usr/bin/env python3
"""Basic structural and notebook-integrity checks; does not run ML experiments."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
notebook = root / "notebooks" / "AIML_A05_COMPLETE_END_TO_END_COLAB.ipynb"
assert notebook.is_file(), "Canonical notebook missing"
nb = json.loads(notebook.read_text(encoding="utf-8"))
assert len(nb.get("cells", [])) >= 100, "Notebook appears incomplete"
text = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for marker in ("## M0", "## M1", "## M2", "## M3", "## M4", "## M5", "## M6", "## M7", "## M8", "## M9"):
    assert marker in text, f"Missing notebook section: {marker}"
for required in ("README.md", "docs/MODULE_MAP.md", "requirements-colab.txt", ".gitignore"):
    assert (root / required).exists(), f"Missing repository file: {required}"
print(f"Repository structure OK. Notebook has {len(nb['cells'])} cells.")
print("This check does NOT execute training, validate data access, or certify deployment/security.")
