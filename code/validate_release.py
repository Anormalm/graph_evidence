"""Fail when public artifact claims drift past the evidence on hand."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    manifest = json.loads((ROOT / "protocol_manifest.json").read_text(encoding="utf-8"))
    assert manifest["raw_paper_generations_present"] is False
    assert manifest["paper_table_reproduction_status"].startswith("blocked_")
    assert manifest["semantic_relabeling"]["paper_intervention_match"] == "unverified"
    assert manifest["conditional_faithfulness"]["paper_implementation_recovered"] is False

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "RELEASE_STATUS.md").read_text(encoding="utf-8")
    required_readme_phrases = (
        "does not claim bit-for-bit numerical reproduction",
        "RELEASE_STATUS.md",
    )
    for phrase in required_readme_phrases:
        assert phrase in readme, f"README is missing required release boundary: {phrase}"
    assert "What must not be claimed yet" in status
    assert (ROOT / "LICENSE").is_file()
    print("Release-boundary checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
