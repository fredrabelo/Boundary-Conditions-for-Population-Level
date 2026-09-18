#!/usr/bin/env python3
"""Recompute every population-level number in the paper from the released data.

Reads only:
  ../data/processed/respondents.json         sampled respondents + held-out human outcomes
  ../data/model_outputs/synthetic_responses.csv   16 runs, 7,120 item responses
  ../data/manifests/run_costs.json            metered cost per run

Writes CSV tables to ../results/ (the inputs of the manuscript figures) and prints
the numbers quoted in the paper. Needs numpy only.

Unit of validation is the population. Individual-level agreement (Jaccard) is
printed as a diagnostic and supports no claim.

Bootstrap: resample respondents (external keys, in sorted order) with replacement, 10,000
resamples, numpy.random.default_rng(20260907), percentile 95% intervals. Comparisons between two
configurations on the same respondents are PAIRED (the same drawn indices are applied
to both). Each comparison restarts the generator with the same seed, so a given
population size always sees the same resamples.

Usage: python3 02_analyze_results.py
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)

SEED = 20260907
N_BOOT = 10_000
CONDITIONS = ["baseline", "instrument", "narrative"]
PROVIDERS = ["claude", "gemini"]
PROVIDER_OF = {"anthropic": "claude", "gemini": "gemini"}
COND_LABEL = {"baseline": "Baseline", "instrument": "Instrument", "narrative": "Narrative"}

# ---- answer options (Portuguese, as shown to the models) <-> survey labels (English) ----
NV_OPTION = "Ambiente naturalmente ventilado"
NV_HUMAN = "Naturally ventilated environment"
AC_OPTION = "Sim, tenho ar-condicionado em casa"
HOT = {
    "Troco de roupa": "I change my clothes",
    "Vou para um lugar mais fresco dentro de casa": "I go to a cooler place at home",
    "Ligo o ventilador": "I turn on the fan",
    "Abro janelas e portas": "I open windows and doors to ventilate",
    "Ligo o ar-condicionado": "I turn on the AC",
    "Tomo uma bebida gelada": "I take a cold drink",
    "Tomo um banho frio": "I take a cold shower",
    "Vou para a piscina": "I go to the pool",
    "Saio de casa para um lugar mais agradável": "I leave the house to a more pleasant place",
}
COLD = {
    "Não se aplica à minha cidade": "Does not apply to my city",
    "Troco de roupa": "I change my clothes",
    "Vou para um lugar mais quente dentro de casa": "I go to a warmer place at home",
    "Uso um cobertor/manta": "I use a blanket",
    "Fecho as janelas": "I close the windows",
    "Ligo o ar-condicionado (modo aquecer)": "I turn on the AC (heating)",
    "Ligo o aquecedor elétrico": "I turn on the electrical heating",
    "Tomo um banho quente": "I take a warm shower",
    "Tomo uma bebida quente": "I take a warm drink",
    "Saio de casa para um lugar mais agradável": "I leave the house for a more pleasant place",
}
FREQ = [("Sempre uso", "Always use"), ("Uso frequentemente", "Often use"), ("Uso às vezes", "Sometimes use"),
        ("Uso raramente", "Rarely use"), ("Nunca uso", "Never use")]
INCOME_ORDER = [
    "Up to 1 minimum wage", "Between 1 and 2 minimum wages", "Between 2 and 4 minimum wages",
    "Between 4 and 10 minimum wages", "Between 10 and 16 minimum wages", "More than 16 minimum wages",
]
CLIMATE_OF_ASHRAE = {"0": "extremely_hot", "1": "very_hot", "2": "hot", "3": "warm"}
CLIMATE_ORDER = ["extremely_hot", "very_hot", "hot", "warm"]


# ---------------------------------------------------------------- loading
def load():
    people = json.load(open(DATA / "processed/respondents.json", encoding="utf-8"))
    keys = sorted(p["externalKey"] for p in people)  # fixed order -> reproducible resampling
    human = {p["externalKey"]: p for p in people}
    m3_keys = json.load(open(DATA / "processed/module3_subset.json"))
    # syn[(condition, provider, question_key)][external_key] = parsed value
    syn = defaultdict(dict)
    n_rows = 0
    with open(DATA / "model_outputs/synthetic_responses.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = json.loads(r["value_json"])
            syn[(r["condition"], PROVIDER_OF[r["provider"]], r["question_key"])][r["external_key"]] = v
            n_rows += 1
    return keys, human, m3_keys, syn, n_rows


# ---------------------------------------------------------------- statistics helpers
def rank(a):
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(rank(a), rank(b))


def boot_delta(x_from, x_to, stat=np.mean):
    """Paired bootstrap of stat(x_to) - stat(x_from) over respondents.
    x_* are aligned per-respondent arrays. Returns (point, lo, hi)."""
    x_from, x_to = np.asarray(x_from, float), np.asarray(x_to, float)
    n = len(x_from)
    idx = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    if stat is np.mean:
        d = x_to[idx].mean(1) - x_from[idx].mean(1)
    else:
        d = np.apply_along_axis(stat, 1, x_to[idx]) - np.apply_along_axis(stat, 1, x_from[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(stat(x_to) - stat(x_from)), float(lo), float(hi)


def tvd(p, q):
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def pct(x):
    return 100.0 * float(np.mean(x))


def write_csv(name, header, rows):
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def split_actions(v):
    return [s.strip() for s in v.split(",") if s.strip()] if isinstance(v, str) else list(v)


# ================================================================= main
def main():
    keys, human, m3_keys, syn, n_rows = load()
    n = len(keys)
    assert n == 248 and len(m3_keys) == 120 and n_rows == 7120, (n, len(m3_keys), n_rows)
    print(f"Respondents: {n} (Module 3: {len(m3_keys)}); model responses: {n_rows}")

    costs = json.load(open(DATA / "manifests/run_costs.json"))
    print(f"Metered cost: {costs['cost_by_condition_usd']}  total US${costs['total_cost_usd']}\n")

    # ---------------------------------------------------------- Module 1
    print("=" * 78 + "\nMODULE 1 - adoption\n" + "=" * 78)
    h_nv = np.array([human[k]["outcomes"]["module1"]["preference"] == NV_HUMAN for k in keys])
    h_ac = np.array([human[k]["outcomes"]["module1"]["hasAc"] for k in keys])
    print(f"Human: prefers natural ventilation {pct(h_nv):.2f}% | household has AC {pct(h_ac):.2f}%")
    m1 = {}
    rows = [["Human", "-", round(pct(h_nv), 4), 0, round(pct(h_ac), 4), 0, n]]
    for c in CONDITIONS:
        for p in PROVIDERS:
            nv = np.array([syn[(c, p, "preference")][k] == NV_OPTION for k in keys])
            ac = np.array([syn[(c, p, "has_ac")][k] == AC_OPTION for k in keys])
            m1[(c, p)] = (nv, ac)
            e_nv, e_ac = abs(pct(nv) - pct(h_nv)), abs(pct(ac) - pct(h_ac))
            rows.append([c, p, round(pct(nv), 4), round(e_nv, 4), round(pct(ac), 4), round(e_ac, 4), n])
            print(f"  {COND_LABEL[c]:10s} {p:6s} NV {pct(nv):6.2f}% (err {e_nv:5.2f}pp) | AC {pct(ac):6.2f}% (err {e_ac:5.2f}pp)")
    write_csv("module1_prevalence.csv", ["config", "provider", "nv_pref_pct", "nv_pref_err_pp", "has_ac_pct", "has_ac_err_pp", "n"], rows)

    print("\nPaired bootstrap changes in prevalence (percentage points, 95% CI):")
    for p in PROVIDERS:
        for a, b in [("baseline", "instrument"), ("instrument", "narrative")]:
            for label, j in [("NV preference", 0), ("has AC", 1)]:
                d, lo, hi = boot_delta(m1[(a, p)][j], m1[(b, p)][j])
                print(f"  {p:6s} {a}->{b:10s} {label:14s} {100*d:+7.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")

    # gradients in AC ownership
    income = np.array([human[k]["evidence"]["monthly_family_income"] for k in keys])
    climate = np.array([CLIMATE_OF_ASHRAE[human[k]["evidence"]["ashrae_climate_zone"][0]] for k in keys])

    def by_group(vec, groups, order):
        return [(g, pct(vec[groups == g]) if (groups == g).any() else float("nan"), int((groups == g).sum())) for g in order]

    def nondecreasing(vals):
        v = [x for x in vals if not np.isnan(x)]
        return all(b >= a for a, b in zip(v, v[1:]))

    print("\nAC ownership by income tier (share %, n):")
    h_inc = by_group(h_ac, income, INCOME_ORDER)
    print("  Human      ", " | ".join(f"{v:5.1f} (n={m})" for _, v, m in h_inc), "| monotone non-decreasing:", nondecreasing([v for _, v, _ in h_inc]))
    for c in CONDITIONS:
        for p in PROVIDERS:
            g = by_group(m1[(c, p)][1], income, INCOME_ORDER)
            mono = nondecreasing([v for _, v, _ in g])
            rho = spearman([v for _, v, _ in g], list(range(len(g))))
            print(f"  {COND_LABEL[c]:10s} {p:6s}", " | ".join(f"{v:5.1f}" for _, v, _ in g), f"| non-decreasing: {mono} | Spearman vs tier {rho:+.2f}")
    write_csv("gradients_income.csv", ["condition", "provider", "tier"] + [t for t in INCOME_ORDER],
              [["Human", "-", "share_pct"] + [round(v, 2) for _, v, _ in h_inc]] +
              [[c, p, "share_pct"] + [round(v, 2) for _, v, _ in by_group(m1[(c, p)][1], income, INCOME_ORDER)] for c in CONDITIONS for p in PROVIDERS])

    print("\nAC ownership by climate group (share %):", CLIMATE_ORDER)
    h_cl = [v for _, v, _ in by_group(h_ac, climate, CLIMATE_ORDER)]
    h_rank = tuple(np.argsort(-np.array(h_cl), kind="stable"))
    print("  Human      ", " | ".join(f"{v:5.1f}" for v in h_cl), "| ranking (best->worst):", [CLIMATE_ORDER[i] for i in h_rank])
    for c in CONDITIONS:
        for p in PROVIDERS:
            g = [v for _, v, _ in by_group(m1[(c, p)][1], climate, CLIMATE_ORDER)]
            same = tuple(np.argsort(-np.array(g), kind="stable")) == h_rank
            print(f"  {COND_LABEL[c]:10s} {p:6s}", " | ".join(f"{v:5.1f}" for v in g), f"| same ordering as human: {same} | Spearman {spearman(g, h_cl):+.2f}")

    hi_income = np.isin(income, INCOME_ORDER[-2:])
    for label, mask in [("top two income tiers", hi_income), ("top income tier", income == INCOME_ORDER[-1])]:
        cell = mask & (climate == "extremely_hot")
        print(f"\nHuman AC ownership, extremely hot x {label}: {pct(h_ac[cell]):.1f}% (n={int(cell.sum())})")

    # ---------------------------------------------------------- Module 2
    print("\n" + "=" * 78 + "\nMODULE 2 - habitual actions (multi-select)\n" + "=" * 78)
    m2_summary = {}
    for wk, qkey, opts, hkey in [("hot", "hot_weather_actions", HOT, "hotWeatherActions"),
                                 ("cold", "cold_weather_actions", COLD, "coldWeatherActions")]:
        labels = list(opts)
        H = np.array([[opts[o] in human[k]["outcomes"]["module2"][hkey] for o in labels] for k in keys], float)
        hp = H.mean(0) * 100
        hcount = H.sum(1)
        print(f"\n{wk.upper()} weather - human mean #actions {hcount.mean():.2f} (sd {hcount.std():.2f})")
        S = {}
        for c in ["baseline", "instrument"]:
            for p in PROVIDERS:
                arr = []
                for k in keys:
                    sel = split_actions(syn[(c, p, qkey)][k])
                    assert set(sel) <= set(labels), (c, p, k, sel)
                    arr.append([o in sel for o in labels])
                S[(c, p)] = np.array(arr, float)
        for c in ["baseline", "instrument"]:
            for p in PROVIDERS:
                A = S[(c, p)]
                sp = A.mean(0) * 100
                cnt = A.sum(1)
                mae = float(np.abs(sp - hp).mean())
                jac = np.mean([(a * h).sum() / max(((a + h) > 0).sum(), 1) for a, h in zip(A, H)])
                m2_summary[(wk, c, p)] = dict(mae=mae, pearson=pearson(hp, sp), spearman=spearman(hp, sp),
                                              mean_count=float(cnt.mean()), sd_count=float(cnt.std()),
                                              count_err=abs(float(cnt.mean()) - float(hcount.mean())), jaccard=float(jac))
                s = m2_summary[(wk, c, p)]
                print(f"  {COND_LABEL[c]:10s} {p:6s} MAE {mae:5.2f}pp | Pearson {s['pearson']:.3f} | Spearman {s['spearman']:.3f} | "
                      f"mean #actions {s['mean_count']:.2f} (sd {s['sd_count']:.2f}, err {s['count_err']:.2f}) | Jaccard {jac:.2f} [diagnostic]")
        print("  Change in mean #actions, baseline -> instrument (paired bootstrap):")
        for p in PROVIDERS:
            d, lo, hi = boot_delta(S[("baseline", p)].sum(1), S[("instrument", p)].sum(1))
            b, i = m2_summary[(wk, "baseline", p)], m2_summary[(wk, "instrument", p)]
            red = 100 * (b["count_err"] - i["count_err"]) / b["count_err"]
            print(f"    {p:6s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] | error in mean #actions falls {red:.2f}%")
        # per-action table
        tab = []
        for j, o in enumerate(labels):
            tab.append([o, round(hp[j], 4)] + [round(S[(c, p)].mean(0)[j] * 100, 4) for c in ["baseline", "instrument"] for p in PROVIDERS])
        order = np.argsort(-hp, kind="stable")
        write_csv(f"module2_{wk}_actions.csv",
                  ["action_label", "human_pct", "baseline_claude_pct", "baseline_gemini_pct", "instrument_claude_pct", "instrument_gemini_pct"],
                  [tab[j] for j in order])

    # ---------------------------------------------------------- Module 3
    print("\n" + "=" * 78 + "\nMODULE 3 - usage among bedroom-AC owners\n" + "=" * 78)
    m3k = m3_keys
    hf = [human[k]["outcomes"]["module3"]["frequency"] for k in m3k]
    hfreq = np.array([sum(f == en for f in hf) / len(hf) for _, en in FREQ])
    hh = np.array([human[k]["outcomes"]["module3"]["hoursPerDay"] for k in m3k], float)
    hs = np.array([human[k]["outcomes"]["module3"]["setpointCelsius"] for k in m3k if human[k]["outcomes"]["module3"]["setpointCelsius"] is not None], float)
    print("Human frequency (%):", {en: round(float(100 * v), 1) for (_, en), v in zip(FREQ, hfreq)})
    print(f"Human hours/day mean {hh.mean():.2f} sd {hh.std():.2f} (n={len(hh)}) | setpoint mean {hs.mean():.2f} sd {hs.std():.2f} (n={len(hs)})")
    m3 = {}
    freq_rows = {pt: [round(100 * v, 4)] for (pt, _), v in zip(FREQ, hfreq)}
    hours_rows, set_rows = [["human", len(hh), round(hh.mean(), 4), round(hh.std(), 4), np.median(hh), *np.percentile(hh, [25, 75])]], \
        [["human", len(hs), round(hs.mean(), 4), round(hs.std(), 4), np.median(hs), *np.percentile(hs, [25, 75])]]
    header_cfg = []
    for c in CONDITIONS:
        for p in PROVIDERS:
            fq = [syn[(c, p, "frequency")][k] for k in m3k]
            sf = np.array([sum(f == pt for f in fq) / len(fq) for pt, _ in FREQ])
            sh = np.array([float(syn[(c, p, "hours_per_day")][k]) for k in m3k])
            ss = np.array([float(syn[(c, p, "setpoint_celsius")][k]) for k in m3k])
            m3[(c, p)] = dict(tvd=tvd(hfreq, sf), sh=sh, ss=ss, sf=sf)
            for (pt, _), v in zip(FREQ, sf):
                freq_rows[pt].append(round(100 * v, 4))
            header_cfg.append(f"{c}_{p}")
            hours_rows.append([f"{c}_{p}", len(sh), round(sh.mean(), 4), round(sh.std(), 4), np.median(sh), *np.percentile(sh, [25, 75])])
            set_rows.append([f"{c}_{p}", len(ss), round(ss.mean(), 4), round(ss.std(), 4), np.median(ss), *np.percentile(ss, [25, 75])])
            print(f"  {COND_LABEL[c]:10s} {p:6s} TVD {m3[(c, p)]['tvd']:.3f} | freq % {np.round(100 * sf, 1).tolist()} | "
                  f"hours {sh.mean():.2f} (sd {sh.std():.2f}) | setpoint {ss.mean():.2f} (sd {ss.std():.2f}; {100 * ss.std() / hs.std():.0f}% of human sd)")
    write_csv("module3_frequency.csv", ["category", "human_pct"] + header_cfg, [[k] + v for k, v in freq_rows.items()])
    write_csv("module3_hours.csv", ["config", "n", "mean", "sd", "median", "q1", "q3"], hours_rows)
    write_csv("module3_setpoint.csv", ["config", "n", "mean", "sd", "median", "q1", "q3"], set_rows)
    ratios = [m3[k]["ss"].std() / hs.std() for k in m3]
    print(f"\nSetpoint sd as share of human sd across the six configurations: {100 * min(ratios):.0f}%-{100 * max(ratios):.0f}%")
    smeans = [m3[k]["ss"].mean() for k in m3]
    ssds = [m3[k]["ss"].std() for k in m3]
    print(f"Setpoint means {min(smeans):.2f}-{max(smeans):.2f} C; sds {min(ssds):.2f}-{max(ssds):.2f} C (human {hs.mean():.2f}, sd {hs.std():.2f})")

    # ---------------------------------------------------------- criteria
    print("\n" + "=" * 78 + "\nFROZEN CRITERIA (improvement = error falls by >=25%)\n" + "=" * 78)

    def drop(before, after):
        return (before - after) / before if before else float("nan")

    print("Instrument Calibration vs Baseline")
    c_rows = []
    for p in PROVIDERS:
        for wk in ["hot", "cold"]:
            b, i = m2_summary[(wk, "baseline", p)], m2_summary[(wk, "instrument", p)]
            c_rows.append((f"M2 count error {p}-{wk}", drop(b["count_err"], i["count_err"]) >= .25, drop(b["count_err"], i["count_err"])))
            c_rows.append((f"M2 prevalence MAE {p}-{wk}", drop(b["mae"], i["mae"]) >= .25, drop(b["mae"], i["mae"])))
            c_rows.append((f"M2 Pearson&Spearman not worse {p}-{wk}", i["pearson"] >= b["pearson"] and i["spearman"] >= b["spearman"], i["spearman"] - b["spearman"]))
        t0, t1 = m3[("baseline", p)]["tvd"], m3[("instrument", p)]["tvd"]
        c_rows.append((f"M3 frequency TVD {p}", drop(t0, t1) >= .25, drop(t0, t1)))
        e0 = abs(hs.std() - m3[("baseline", p)]["ss"].std())
        e1 = abs(hs.std() - m3[("instrument", p)]["ss"].std())
        c_rows.append((f"M3 setpoint sd error {p}", drop(e0, e1) >= .25, drop(e0, e1)))
    for name, ok, val in c_rows:
        print(f"  [{'PASS' if ok else 'fail'}] {name:42s} {100 * val:+7.1f}%" if "Pearson" not in name else f"  [{'PASS' if ok else 'fail'}] {name:42s} dSpearman {val:+.3f}")

    print("\nNarrative Grounding vs Instrument Calibration")
    d_rows = []
    for p in PROVIDERS:
        nv_i, ac_i = m1[("instrument", p)]
        nv_n, ac_n = m1[("narrative", p)]
        for label, a, b, h in [("M1 has-AC error", ac_i, ac_n, h_ac), ("M1 NV-preference error", nv_i, nv_n, h_nv)]:
            e0, e1 = abs(pct(a) - pct(h)), abs(pct(b) - pct(h))
            d_rows.append((f"{label} {p}", drop(e0, e1) >= .25, drop(e0, e1)))
        t0, t1 = m3[("instrument", p)]["tvd"], m3[("narrative", p)]["tvd"]
        d_rows.append((f"M3 frequency TVD {p}", drop(t0, t1) >= .25, drop(t0, t1)))
        for label, key, hsd in [("M3 hours sd error", "sh", hh.std()), ("M3 setpoint sd error", "ss", hs.std())]:
            e0, e1 = abs(hsd - m3[("instrument", p)][key].std()), abs(hsd - m3[("narrative", p)][key].std())
            d_rows.append((f"{label} {p}", drop(e0, e1) >= .25, drop(e0, e1)))
    for name, ok, val in d_rows:
        print(f"  [{'PASS' if ok else 'fail'}] {name:42s} {100 * val:+7.1f}%")
    by_metric = defaultdict(list)
    for name, ok, _ in d_rows:
        by_metric[name.rsplit(" ", 1)[0]].append(ok)
    met = {k: all(v) for k, v in by_metric.items()}
    # Criterion 6: a mean may not improve while the corresponding sd compresses further.
    violations = []
    for p in PROVIDERS:
        for label, key, hmean in [("hours", "sh", hh.mean()), ("setpoint", "ss", hs.mean())]:
            i_, n_ = m3[("instrument", p)][key], m3[("narrative", p)][key]
            if abs(hmean - n_.mean()) < abs(hmean - i_.mean()) and n_.std() < i_.std():
                violations.append(f"{p}/{label}")
    # Criterion 7: any change must point the same way for both providers.
    signs = defaultdict(set)
    for name, _, val in d_rows:
        signs[name.rsplit(" ", 1)[0]].add(val > 0)
    same_direction = all(len(v) == 1 for v in signs.values())
    print("  Criteria 1-5 (met by BOTH providers):", met)
    print(f"  Criterion 6 (means do not improve at the cost of tighter compression): {'met' if not violations else 'not met - ' + ', '.join(violations)}")
    print(f"  Criterion 7 (same direction for both providers): {'met' if same_direction else 'not met'}")
    n_met = sum(met.values()) + (not violations) + same_direction
    print(f"  Pre-specified criteria met by Narrative Grounding: {n_met} of 7")
    write_csv("criteria_checks.csv", ["comparison", "check", "passed", "relative_change"],
              [["instrument_vs_baseline", n_, ok, round(v, 4)] for n_, ok, v in c_rows] +
              [["narrative_vs_instrument", n_, ok, round(v, 4)] for n_, ok, v in d_rows])
    print("\nWrote tables to results/")


if __name__ == "__main__":
    main()
