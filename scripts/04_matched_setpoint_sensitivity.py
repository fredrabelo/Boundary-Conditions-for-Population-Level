#!/usr/bin/env python3
"""Matched-profile setpoint sensitivity. Uses only files in this repository.
Compares human setpoints (n = 108 observed) with synthetic setpoints for (a) all 120
Module 3 profiles (unmatched comparison) and (b) exactly the same 108 profiles.
Population SD (ddof = 0) throughout. Writes results/matched108_setpoint.csv."""
import csv, json, random, statistics as st
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
resp = json.load(open(ROOT/"data/processed/respondents.json", encoding="utf-8"))
human = {r["externalKey"]: r["outcomes"]["module3"].get("setpointCelsius")
         for r in resp if r["outcomes"]["module3"].get("bwHasAc")}
human = {k: v for k, v in human.items() if v is not None}
hsd = st.pstdev(human.values())
syn = {}
for r in csv.DictReader(open(ROOT/"data/model_outputs/synthetic_responses.csv", encoding="utf-8")):
    if r["question_key"] == "setpoint_celsius":
        prov = "claude" if r["provider"] == "anthropic" else r["provider"]
        syn.setdefault((prov, r["condition"]), {})[r["external_key"]] = float(json.loads(r["value_json"]))
rows = ["provider,condition,n_all,mean_all,sd_all,ratio_all,n_matched,mean_matched,sd_matched,ratio_matched,ratio_ci95_lo,ratio_ci95_hi"]
rng = random.Random(20260907)
for (p, c), d in sorted(syn.items()):
    keys = [k for k in human if k in d]
    h = [human[k] for k in keys]; m = [d[k] for k in keys]
    boot = []
    for _ in range(10000):
        ix = [rng.randrange(len(keys)) for _ in keys]
        hs = st.pstdev([h[i] for i in ix])
        if hs > 0: boot.append(st.pstdev([m[i] for i in ix]) / hs)
    boot.sort(); allv = list(d.values())
    rows.append(f"{p},{c},{len(allv)},{st.fmean(allv):.3f},{st.pstdev(allv):.3f},{st.pstdev(allv)/hsd:.3f},"
                f"{len(m)},{st.fmean(m):.3f},{st.pstdev(m):.3f},{st.pstdev(m)/st.pstdev(h):.3f},"
                f"{boot[250]:.3f},{boot[9749]:.3f}")
(ROOT/"results/matched108_setpoint.csv").write_text("\n".join(rows)+"\n")
print("human n = %d, mean = %.3f, SD (ddof=0) = %.3f" % (len(human), st.fmean(human.values()), hsd))
print("\n".join(rows))
