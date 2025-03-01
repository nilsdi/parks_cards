"""
Go through the protected area shapes and filter the national parks
"""

# %%
import geopandas as gpd
import pandas as pd
import json


from datetime import datetime
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
# print(root_dir)
protected_areas_path = (
    root_dir / "data/protected_areas/Vern_0000_norge_4326_GEOJSON.json"
)

# open the protected areas as a json:
with open(protected_areas_path) as f:
    protected_areas = json.load(f)


protected_geo_objects = protected_areas["features"]


# protected_areas.keys()
# protected_areas["features"][0].keys()
# print(protected_areas["features"][100]["properties"].keys())
# protected_areas["features"][0]["type"]
# %%
def get_property_keys(geo_objects):
    property_keys = []
    for o in geo_objects:
        p_keys = o["properties"].keys()
        property_keys.extend(p_keys)
    return list(set(property_keys))


print(get_property_keys(protected_geo_objects))
# %%


def get_protection_statuses(geo_objects):
    protection_status = []
    protection_status_aggregated = []
    for o in geo_objects:
        p_keys = o["properties"].keys()
        if "verneform" in p_keys:
            protection_status.append(o["properties"]["verneform"])
        if "verneformAggregert" in p_keys:
            protection_status_aggregated.append(o["properties"]["verneformAggregert"])
    protection_status = list(set(protection_status))
    protection_status_aggregated = list(set(protection_status_aggregated))
    return protection_status, protection_status_aggregated


p_status, p_status_agg = get_protection_statuses(protected_geo_objects)
print(f"found {len(p_status)} protection statuses: {p_status}")
print(f"found {len(p_status_agg)} aggregated protection statuses: {p_status_agg}")


# %%
def get_national_parks(geo_objects) -> tuple[list]:
    national_parks = []
    for o in geo_objects:
        p_keys = o["properties"].keys()
        if "verneformAggregert" in p_keys:
            if o["properties"]["verneformAggregert"] == "nasjonalpark":
                national_parks.append(o)
    national_parks_names = [p["properties"]["navn"] for p in national_parks]
    return national_parks, national_parks_names


national_parks, national_parks_names = get_national_parks(protected_geo_objects)
print(f"found {len(national_parks)} national parks: {national_parks_names}")


# %%
def national_park_information(national_parks) -> dict:
    """
    find which properties are in all national parks, and print all these properties.
    """
    total_properties = []
    persistent_properties = []
    non_persistent_properties = []
    for p in national_parks:
        p_keys = p["properties"].keys()
        total_properties.extend(p_keys)
    total_properties = list(set(total_properties))
    for p in total_properties:
        persistent = True
        for n in national_parks:
            if p not in n["properties"].keys():
                persistent = False
        if persistent:
            persistent_properties.append(p)
        else:
            non_persistent_properties.append(p)

    print(f"persistent properties: {persistent_properties}")
    print(f"non persistent properties: {non_persistent_properties}")
    return


national_park_information(national_parks)


# %%
def get_park_properties(national_parks, property_keys):
    park_properties = {}
    for p in national_parks:
        p_name = p["properties"]["navn"]
        park_properties[p_name] = {}
        for k in property_keys:
            if k not in p["properties"].keys():
                raise ValueError(f"property {k} not in park properties")
            park_properties[p_name][k] = p["properties"][k]
    return park_properties


property_keys = [
    "offisieltNavn",
    "navn",
    "førstegangVernet",
    "vernedato",
    # "faktaark",
    # "kommune",
    "landareal",
    "sjøareal",
    "majorEcosystemType",
]
park_details = get_park_properties(national_parks, property_keys)

save_path = root_dir / "data/card_details/national_parks_raw_properties.json"

with open(save_path, "w") as f:
    json.dump(park_details, f)


# %% get the most interesting parts
def prepare_final_park_properties(national_parks):
    property_keys = [
        "landareal",
        "sjøareal",
        "majorEcosystemType",
        # "førstegangVernet",
        # "vernedato",
    ]
    park_properties = {}
    for p in national_parks:
        p_name = p["properties"]["navn"]
        park_properties[p_name] = {}
        for k in property_keys:
            park_properties[p_name][k] = p["properties"][k]
        f_vern_dato = p["properties"]["førstegangVernet"]
        vern_dato = p["properties"]["vernedato"]
        if f_vern_dato:
            park_properties[p_name]["creation_date"] = f_vern_dato[0:4]
        elif vern_dato:
            park_properties[p_name]["creation_date"] = vern_dato[0:4]
        # add empty strings for stuff we want to add later:
        park_properties[p_name]["AI image descriction"] = ""
        park_properties[p_name]["game_costs"] = {
            "suns": 0,
            "water": 0,
            "trees": 0,
            "mountains": 0,
        }
    return park_properties


ready_properties = prepare_final_park_properties(national_parks)
for p in ready_properties:
    print(p, ready_properties[p])

save_ready_path = root_dir / "data/card_details/national_parks_properties.json"
with open(save_ready_path, "w") as f:
    json.dump(ready_properties, f)


# %%
