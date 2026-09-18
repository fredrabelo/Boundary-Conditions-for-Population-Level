#!/usr/bin/env python3
"""Build the grounding personas and held-out human outcomes for the Ramos et al.
(2020) air-conditioning benchmark from the public survey file.

Reads:  ../data/raw/DATA.CSV  (Ramos et al. 2020, Mendeley Data, doi:10.17632/zwjxzgkkn7.1)
Writes: ../data/processed/respondents.json      248 sampled respondents: persona evidence
                                                 (grounding) and human outcomes (held out)
        ../data/processed/module3_subset.json   external keys of the 120 sampled respondents
                                                 whose bedroom has AC (Module 3 population)

Sampling: complete-case rows (city, income, bioclimatic zone present), stratified
proportionally across four climate groups, fixed seed, so no outcome ever
enters `evidence`. Each module sees a disjoint evidence/outcome split:
  Module 1 adoption:  preference + has-AC (held out)
  Module 2 behaviour: hot- and cold-weather actions (held out)
  Module 3 usage:     frequency, hours/day, setpoint among bedroom-AC owners (held out)

Usage: python3 01_build_sample.py
"""
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE.parent / "data" / "raw" / "DATA.CSV"
OUT_DIR = HERE.parent / "data" / "processed"

PILOT_SAMPLING_RATE = 250 / 3259
RANDOM_SEED = 42

A169_LABEL = {
    "0A": "Extremely hot humid",
    "0B": "Extremely hot dry",
    "1A": "Very hot humid",
    "2A": "Hot humid",
    "2B": "Hot dry",
    "3A": "Warm humid",
    "3C": "Warm marine",
}
# Paper's own 4-group climate stratification (Table 1/3): 0A | 1A | 2A+2B | 3A+3C.
CLIMATE_GROUP = {
    "0A": "extremely_hot", "0B": "extremely_hot",
    "1A": "very_hot",
    "2A": "hot", "2B": "hot",
    "3A": "warm", "3C": "warm",
}
GENDER_LABEL = {"1": "Female", "2": "Male", "3": "Rather not answer / other"}
FI_LABEL = {
    "1": "Up to 1 minimum wage",
    "2": "Between 1 and 2 minimum wages",
    "3": "Between 2 and 4 minimum wages",
    "4": "Between 4 and 10 minimum wages",
    "5": "Between 10 and 16 minimum wages",
    "6": "More than 16 minimum wages",
}
LE_LABEL = {
    "1": "Elementary school",
    "2": "High school (incomplete)",
    "3": "High school",
    "4": "Higher education (incomplete)",
    "5": "Higher education",
    "6": "Post-graduation",
    "7": "Other (did not attend)",
}
TR_LABEL = {
    "1": "One person living alone",
    "2": "Family",
    "3": "More than one adult, with a family relation",
    "4": "More than one adult, with no family relationship",
    "5": "Family and other adults with no family relationship",
}
OC_FS_LABEL = {
    "1": "We are most of the time at home",
    "2": "We almost don't stay at home",
    "3": "We leave part of the day",
    "4": "We leave the house for one day of the weekend",
    "5": "We do not have a routine",
    "6": "Similar to the rest of the week",
    "7": "We do not stay at home (travel, other residence)",
    "8": "Other",
}
BW_F_LABEL = {
    "1": "Always use",
    "2": "Often use",
    "3": "Sometimes use",
    "4": "Rarely use",
    "5": "Never use",
}

HOT_ACTION_COLS = {
    "H.CC": "I change my clothes",
    "H.CP": "I go to a cooler place at home",
    "H.F": "I turn on the fan",
    "H.OW": "I open windows and doors to ventilate",
    "H.AC": "I turn on the AC",
    "H.CD": "I take a cold drink",
    "H.CS": "I take a cold shower",
    "H.GP": "I go to the pool",
    "H.LH": "I leave the house to a more pleasant place",
}
COLD_ACTION_COLS = {
    "C.NA": "Does not apply to my city",
    "C.CC": "I change my clothes",
    "C.HP": "I go to a warmer place at home",
    "C.TB": "I use a blanket",
    "C.CW": "I close the windows",
    "C.AC": "I turn on the AC (heating)",
    "C.EH": "I turn on the electrical heating",
    "C.WS": "I take a warm shower",
    "C.WD": "I take a warm drink",
    "C.LH": "I leave the house for a more pleasant place",
}


def load_rows():
    with open(CSV_PATH, encoding="latin-1", newline="") as f:
        reader = csv.DictReader(f, delimiter=",")
        return list(reader)


def is_complete_case(row) -> bool:
    return bool(row["FI"].strip()) and bool(row["A169"].strip()) and bool(row["ZB"].strip()) and bool(row["CITY"].strip())


def flag(row, col) -> bool:
    return row.get(col, "").strip() == "1"


def build_evidence(row):
    a169 = row["A169"].strip()
    evidence = {
        "city": row["CITY"].strip(),
        "state": row["UF"].strip(),
        "region": row["REG"].strip(),
        "bioclimatic_zone": f"ZB{row['ZB'].strip()}",
        "koppen_climate_classification": row["KG.C"].strip(),
        "ashrae_climate_zone": f"{a169} ({A169_LABEL.get(a169, a169)})",
        "gender": GENDER_LABEL.get(row["GENDER"].strip(), row["GENDER"].strip()),
        "age": int(row["AGE"]) if row["AGE"].strip() else None,
        "household_type": TR_LABEL.get(row["TR"].strip(), row["TR"].strip()),
        "number_of_residents": int(row["N.RES"]) if row["N.RES"].strip() else None,
        "monthly_family_income": FI_LABEL.get(row["FI"].strip(), row["FI"].strip()),
        "highest_education_in_household": LE_LABEL.get(row["LE"].strip(), row["LE"].strip()),
        "weekend_occupancy_pattern": OC_FS_LABEL.get(row["Oc. FS"].strip(), row["Oc. FS"].strip()),
        "number_of_bedrooms": int(row["NB"]) if row["NB"].strip() else None,
        "house_is_sunny": flag(row, "P.S"),
        "house_is_well_ventilated": flag(row, "P.W"),
        "house_is_hot_in_summer": flag(row, "P.HS"),
        "house_is_hot_in_winter": flag(row, "P.HW"),
        "house_is_cool_in_summer": flag(row, "P.CoolS"),
        "house_is_cool_in_winter": flag(row, "P.CoolW"),
        "house_is_cold_in_winter": flag(row, "P.CW"),
        "has_ceiling_fan": flag(row, "Eq.CF"),
        "has_table_fan": flag(row, "Eq.TF"),
        "has_electric_heating": flag(row, "Eq.EH"),
        "has_fireplace": flag(row, "Eq.F"),
        "has_wood_stove": flag(row, "Eq.WS"),
    }
    return {k: v for k, v in evidence.items() if v is not None and v != ""}


def build_outcomes(row):
    hot_actions = [label for col, label in HOT_ACTION_COLS.items() if flag(row, col)]
    cold_actions = [label for col, label in COLD_ACTION_COLS.items() if flag(row, col)]
    bw_has_ac = row["BW"].strip() == "1"
    return {
        "module1": {
            "preference": "Naturally ventilated environment" if row["Pref."].strip() == "1" else "Conditioned environment",
            "hasAc": row["Eq.AC"].strip() == "1",
        },
        "module2": {
            "hotWeatherActions": hot_actions,
            "coldWeatherActions": cold_actions,
        },
        "module3": {
            "bwHasAc": bw_has_ac,
            "frequency": BW_F_LABEL.get(row["BW.F"].strip(), None) if bw_has_ac else None,
            "hoursPerDay": (float(row["BW.H"]) if row["BW.H"].strip() else None) if bw_has_ac else None,
            "setpointCelsius": (float(row["BW.Set"]) if row["BW.Set"].strip() else None) if bw_has_ac else None,
        },
    }


def stratified_sample(rows):
    groups = {}
    for row in rows:
        group = CLIMATE_GROUP.get(row["A169"].strip())
        if group is None:
            continue
        groups.setdefault(group, []).append(row)

    rng = random.Random(RANDOM_SEED)
    sampled = []
    strata_report = {}
    for group, group_rows in sorted(groups.items()):
        n = round(len(group_rows) * PILOT_SAMPLING_RATE)
        n = max(n, 1)
        chosen = rng.sample(group_rows, n)
        sampled.extend(chosen)
        strata_report[group] = {"frame_n": len(group_rows), "sampled_n": n}
    return sampled, strata_report


def main():
    all_rows = load_rows()
    if len(all_rows) != 3259:
        raise SystemExit(f"expected 3259 rows in DATA.CSV, got {len(all_rows)} — refusing to export")

    complete = [r for r in all_rows if is_complete_case(r)]
    print(f"complete-case frame: {len(complete)} of {len(all_rows)}")

    sampled, strata_report = stratified_sample(complete)
    print("strata (climate group -> frame_n, sampled_n):")
    for group, info in strata_report.items():
        print(f"  {group}: {info}")

    if len({r["ID"].strip() for r in sampled}) != len(sampled):
        raise SystemExit("duplicate ID detected in sample — refusing to export")

    respondents = []
    module3_subset = []
    for row in sampled:
        external_key = f"ramos-{row['ID'].strip()}"
        outcomes = build_outcomes(row)
        respondents.append(
            {
                "externalKey": external_key,
                "evidence": build_evidence(row),
                "outcomes": outcomes,
            }
        )
        if outcomes["module3"]["bwHasAc"]:
            module3_subset.append(external_key)

    if len({r["externalKey"] for r in respondents}) != len(respondents):
        raise SystemExit("duplicate externalKey detected — refusing to export")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "respondents.json").write_text(json.dumps(respondents, indent=2, ensure_ascii=False))
    (OUT_DIR / "module3_subset.json").write_text(json.dumps(module3_subset, indent=2, ensure_ascii=False))

    # Grounding spec: which evidence keys each module may see. Module 3 adds exactly one
    # fact (bedroom AC eligibility) to the 26 persona keys; no outcome is ever listed.
    persona_keys = list(respondents[0]["evidence"].keys())
    grounding_spec = {
        "module1_adoption": {"population": "all 248 sampled respondents", "evidence_keys": persona_keys,
                              "held_out": ["preference", "hasAc"]},
        "module2_behaviour": {"population": "all 248 sampled respondents", "evidence_keys": persona_keys,
                               "held_out": ["hotWeatherActions", "coldWeatherActions"]},
        "module3_usage": {"population": "120 sampled respondents with bedroom AC",
                           "evidence_keys": persona_keys + ["bedroom_air_conditioning"],
                           "bedroom_air_conditioning": True,
                           "held_out": ["frequency", "hoursPerDay", "setpointCelsius"]},
    }
    (OUT_DIR / "grounding_spec.json").write_text(json.dumps(grounding_spec, indent=2, ensure_ascii=False))

    print(f"\nwrote {len(respondents)} respondents to {OUT_DIR / 'respondents.json'}")
    print(f"wrote {len(module3_subset)} module3 (bedroom-AC) respondents to {OUT_DIR / 'module3_subset.json'}")


if __name__ == "__main__":
    main()
