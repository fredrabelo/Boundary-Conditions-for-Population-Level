#!/usr/bin/env python3
"""Sensitivity of the recovery criteria to the improvement threshold (20/25/30%).
Uses only results/*.csv in this repository. Writes results/criteria_sensitivity.csv."""
import csv, math
from pathlib import Path
R = Path(__file__).resolve().parent.parent/"results"
rd = lambda f: list(csv.DictReader(open(R/f, encoding="utf-8")))
def pearson(x, y):
    n = len(x); mx, my = sum(x)/n, sum(y)/n
    return sum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0]*len(v); i = 0
    while i < len(v):
        j = i
        while j+1 < len(v) and v[o[j+1]] == v[o[i]]: j += 1
        for k in range(i, j+1): r[o[k]] = (i+j)/2+1
        i = j+1
    return r
drop = lambda old, new: (old-new)/old*100
out = ["comparison,metric,cell,reference,new,improvement_pct,pass_20,pass_25,pass_30"]
def add(cmp_, metric, cell, a, b):
    d = drop(a, b); out.append(f"{cmp_},{metric},{cell},{a:.4f},{b:.4f},{d:.1f},{d>=20},{d>=25},{d>=30}")
for wx, f in (("hot", "module2_hot_actions.csv"), ("cold", "module2_cold_actions.csv")):
    rows = rd(f); h = [float(r["human_pct"]) for r in rows]; hm = sum(h)/100
    for prov in ("claude", "gemini"):
        s0 = [float(r[f"baseline_{prov}_pct"]) for r in rows]; s1 = [float(r[f"instrument_{prov}_pct"]) for r in rows]
        mae = lambda s: sum(abs(a-b) for a, b in zip(h, s))/len(h)
        add("instrument_vs_baseline", "item-prevalence MAE (pp)", f"{prov}-{wx}", mae(s0), mae(s1))
        add("instrument_vs_baseline", "selected-count error", f"{prov}-{wx}", abs(sum(s0)/100-hm), abs(sum(s1)/100-hm))
fr = rd("module3_frequency.csv"); hf = [float(r["human_pct"]) for r in fr]
tvd = lambda col: 0.5*sum(abs(a-float(r[col])) for a, r in zip(hf, fr))/100
sp = {r["config"]: r for r in rd("module3_setpoint.csv")}; hr = {r["config"]: r for r in rd("module3_hours.csv")}
m1 = {(r["config"], r["provider"]): r for r in rd("module1_prevalence.csv")}
hs, hh = float(sp["human"]["sd"]), float(hr["human"]["sd"])
for prov in ("claude", "gemini"):
    e = lambda c, k: abs(float(m1[(c, prov)][k]))
    add("narrative_vs_instrument", "1 ownership error (pp)", prov, e("instrument", "has_ac_err_pp"), e("narrative", "has_ac_err_pp"))
    add("narrative_vs_instrument", "2 ventilation-preference error (pp)", prov, e("instrument", "nv_pref_err_pp"), e("narrative", "nv_pref_err_pp"))
    add("narrative_vs_instrument", "3 frequency TVD", prov, tvd(f"instrument_{prov}"), tvd(f"narrative_{prov}"))
    add("narrative_vs_instrument", "4 hours-SD error", prov, abs(hh-float(hr[f"instrument_{prov}"]["sd"])), abs(hh-float(hr[f"narrative_{prov}"]["sd"])))
    add("narrative_vs_instrument", "5 setpoint-SD error", prov, abs(hs-float(sp[f"instrument_{prov}"]["sd"])), abs(hs-float(sp[f"narrative_{prov}"]["sd"])))
(R/"criteria_sensitivity.csv").write_text("\n".join(out)+"\n"); print("\n".join(out))
