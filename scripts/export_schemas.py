"""Regenerate the frozen JSON Schema artifacts under sciencerag/schemas/.

Run after any change to the Pydantic models in sciencerag/*/models.py:

    uv run python scripts/export_schemas.py
"""

import json
from pathlib import Path

from sciencerag.priors.models import PriorsRequest, PriorsResponse

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "sciencerag" / "schemas"


def main() -> None:
    priors_schema = {
        "PriorsRequest": PriorsRequest.model_json_schema(),
        "PriorsResponse": PriorsResponse.model_json_schema(),
    }
    out_path = SCHEMAS_DIR / "priors.schema.json"
    out_path.write_text(json.dumps(priors_schema, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
