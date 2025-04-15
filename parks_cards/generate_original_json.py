# %%
import json
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]

park_details = {
    "park_name": "",
    "costs": {
        "trees": 0,
        "mountains": 0,
        "sun": 0,
        "water": 0,
    },
}
orginal_json = {i: park_details for i in range(1, 101)}
with open(root_dir / "data/card_details/original_cards_costs.json", "w") as f:
    json.dump(orginal_json, f)
