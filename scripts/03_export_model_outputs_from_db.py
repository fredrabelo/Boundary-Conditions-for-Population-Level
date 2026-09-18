#!/usr/bin/env python3
"""
Provenance script: documents how data/model_outputs/synthetic_responses.csv,
data/manifests/run_manifests.json, data/manifests/run_costs.json and
data/instruments/instruments.json were extracted from the run database. It needs
access to that private database, so it is included for audit only; the paper's
tables and figures are reproduced from the exported files by 02_analyze_results.py.

The export is a plain SELECT and reshape. Internal identifiers of populations,
studies, instruments and stimuli are not exported.

Credentials are read from environment variables (never hardcode them):
  DB_HOST (default 127.0.0.1), DB_USER, DB_PASSWORD, DB_NAME

Usage (authors only):
  DB_USER=... DB_PASSWORD=... DB_NAME=... python3 03_export_model_outputs_from_db.py
"""
import csv
import json
import os
from pathlib import Path

import pymysql

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"

# Run id -> (condition, module). Ids are local to the run database; provider and
# model come from each run's own manifest.
RUN_META = {
    96: ("baseline", 1), 97: ("baseline", 1),
    98: ("baseline", 2), 99: ("baseline", 2),
    100: ("baseline", 3), 101: ("baseline", 3),
    118: ("instrument", 1), 119: ("instrument", 1),
    120: ("instrument", 2), 121: ("instrument", 2),
    122: ("instrument", 3), 123: ("instrument", 3),
    124: ("narrative", 1), 125: ("narrative", 1),
    126: ("narrative", 3), 127: ("narrative", 3),
}
EXPECTED_ROWS = {1: 496, 2: 496, 3: 360}  # agents x items per run

# Instrument id -> (condition, module)
INSTRUMENTS = {
    25: ("baseline", 1), 26: ("baseline", 2), 27: ("baseline", 3),
    34: ("instrument", 1), 35: ("instrument", 2), 36: ("instrument", 3),
    37: ("narrative", 1), 38: ("narrative", 3),
}

MANIFEST_KEEP = [
    "llmProvider", "llmModel", "groundingCompilerKey", "groundingCompilerVersion",
    "groundingNarrativeGroups", "evidenceKeyAllowlist", "exposureMode", "frozenAt",
]


def jload(x):
    return json.loads(x) if isinstance(x, str) else x


def main():
    conn = pymysql.connect(
        host=os.environ.get("DB_HOST", "127.0.0.1"),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
    )
    cur = conn.cursor()
    run_ids = ",".join(str(k) for k in RUN_META)

    # ---- run manifests (whitelisted fields only) ----
    cur.execute(f"SELECT id, manifest_json, started_at, finished_at FROM study_runs WHERE id IN ({run_ids})")
    manifests = {}
    for run_id, mj, started, finished in cur.fetchall():
        m = jload(mj)
        condition, module = RUN_META[run_id]
        entry = {"run_id": run_id, "condition": condition, "module": module}
        entry.update({k: m[k] for k in MANIFEST_KEEP if k in m})
        entry["started_at"] = started.isoformat() if started else None
        entry["finished_at"] = finished.isoformat() if finished else None
        manifests[run_id] = entry
    assert set(manifests) == set(RUN_META), "missing runs"
    (DATA / "manifests").mkdir(parents=True, exist_ok=True)
    (DATA / "manifests" / "run_manifests.json").write_text(
        json.dumps([manifests[k] for k in sorted(manifests)], indent=2, ensure_ascii=False))

    # ---- responses ----
    cur.execute(
        f"""
        SELECT rr.run_id, a.external_key, rr.question_key, rr.draw_index, rr.value_json, rr.explanation
        FROM run_responses rr JOIN agents a ON a.id = rr.agent_id
        WHERE rr.run_id IN ({run_ids})
        ORDER BY rr.run_id, a.external_key, rr.question_key
        """
    )
    rows = cur.fetchall()
    per_run = {}
    for r in rows:
        per_run[r[0]] = per_run.get(r[0], 0) + 1
    for run_id, (_, module) in RUN_META.items():
        assert per_run.get(run_id) == EXPECTED_ROWS[module], (run_id, per_run.get(run_id))
    assert len(rows) == 7120, len(rows)

    out = DATA / "model_outputs" / "synthetic_responses.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "condition", "module", "provider", "model",
                    "external_key", "question_key", "draw_index", "value_json", "explanation"])
        for run_id, external_key, qk, draw_idx, value_json, explanation in rows:
            condition, module = RUN_META[run_id]
            mf = manifests[run_id]
            w.writerow([run_id, condition, module, mf["llmProvider"], mf["llmModel"],
                        external_key, qk, draw_idx, json.dumps(jload(value_json), ensure_ascii=False),
                        explanation or ""])
    print(f"Wrote {len(rows)} rows to {out}")

    # ---- metered cost / tokens per run ----
    cur.execute(
        f"""SELECT run_id, COUNT(*), SUM(input_tokens), SUM(output_tokens), SUM(cost_usd), MIN(is_batch), MAX(is_batch)
            FROM token_usage WHERE run_id IN ({run_ids}) GROUP BY run_id"""
    )
    costs = []
    for run_id, n_calls, tin, tout, cost, b_min, b_max in cur.fetchall():
        assert b_min == b_max == 1, f"run {run_id} not fully batch"
        costs.append({"run_id": run_id, "condition": RUN_META[run_id][0], "module": RUN_META[run_id][1],
                      "provider": manifests[run_id]["llmProvider"], "model": manifests[run_id]["llmModel"],
                      "calls": n_calls, "input_tokens": int(tin), "output_tokens": int(tout),
                      "cost_usd": round(cost, 6)})
    costs.sort(key=lambda c: c["run_id"])
    total = round(sum(c["cost_usd"] for c in costs), 4)
    by_condition = {cond: round(sum(c["cost_usd"] for c in costs if c["condition"] == cond), 4)
                    for cond in ("baseline", "instrument", "narrative")}
    (DATA / "manifests" / "run_costs.json").write_text(
        json.dumps({"runs": costs, "cost_by_condition_usd": by_condition, "total_cost_usd": total}, indent=2))
    print(f"Metered cost over {len(costs)} runs: ${total}")

    # ---- instruments (question wording, options, ranges, instrument-level switches) ----
    instruments = []
    for ins_id, (condition, module) in sorted(INSTRUMENTS.items()):
        cur.execute("SELECT item_type, item_key, prompt_text, config FROM instrument_items "
                    "WHERE instrument_id=%s ORDER BY sort_order", (ins_id,))
        instruments.append({
            "condition": condition, "module": module,
            "items": [{"type": t, "key": k, "prompt_pt": p, "config": jload(c)} for t, k, p, c in cur.fetchall()],
        })
    (DATA / "instruments").mkdir(parents=True, exist_ok=True)
    (DATA / "instruments" / "instruments.json").write_text(
        json.dumps(instruments, indent=2, ensure_ascii=False))
    conn.close()


if __name__ == "__main__":
    main()
