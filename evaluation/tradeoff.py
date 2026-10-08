"""
tradeoff.py - trade-off analysis of the designs over the design constraints in constraints.json.
Run after evaluate.py / train.py; re-run whenever results or constraints.json change.

    python evaluation/tradeoff.py
    python evaluation/tradeoff.py --selftest

1. Normalized score (0-10, best design = 10 on each constraint):
       lower is better:  score = best / value x 10
       higher is better: score = value / best x 10
2. Overall score = sum(weight x score), weights scaled to sum to 1.
3. Pareto: a design is Pareto-optimal if no other design is at least as good on every constraint and
   strictly better on one.
4. Sensitivity: each constraint's weight is swept 0 -> 1 (the others keep their ratios); the weight where
   the overall winner changes is reported. No flip = the winner is robust to that constraint's importance.

Repeated runs (evaluate.py --runs): values are the mean of the newest batch per design; runs.png plots every
run, one angle per run, for each constraint that has more than one run.

Prints raw values with the winner per constraint, then the scores.
Outputs in evaluation/results/: tradeoff.csv, radar.png, sensitivity.png, runs.png
"""
import argparse
import csv
import json
import os
import statistics

import matplotlib
import numpy as np

matplotlib.use("Agg")   # save to file, no window
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def load_constraints(path=f"{HERE}/constraints.json"):
    with open(path) as f:
        cons = json.load(f)["constraints"]
    for c in cons:
        assert c["better"] in ("lower", "higher"), f"{c['name']}: better must be lower or higher"
    return cons


def latest(path):
    """{design: rows of its newest batch}. No batch column (e.g. training_time.csv): the last row only."""
    if not os.path.exists(path):
        return {}
    by_design = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            by_design.setdefault(r["design"], []).append(r)
    return {d: [r for r in rows if r.get("batch") and r["batch"] == rows[-1]["batch"]] or rows[-1:]
            for d, rows in by_design.items()}


def gather(cons, results=f"{HERE}/results"):
    """{design: {constraint name: [value per run]}} from the results CSVs; [] = not measured yet."""
    tables = {c["file"]: latest(os.path.join(results, c["file"])) for c in cons}
    designs = sorted(set().union(*tables.values()))
    return {d: {c["name"]: [float(r[c["column"]]) for r in tables[c["file"]].get(d, []) if r.get(c["column"], "") != ""]
                for c in cons} for d in designs}


def scores(raw, cons):
    """Normalized 0-10 per constraint, higher always better."""
    out = {d: {} for d in raw}
    for c in cons:
        vals = {d: raw[d][c["name"]] for d in raw}
        best = min(vals.values()) if c["better"] == "lower" else max(vals.values())
        for d, v in vals.items():
            if c["better"] == "lower":
                out[d][c["name"]] = 10.0 if v == best else 10 * best / v   # v == best covers best == 0
            else:
                out[d][c["name"]] = 10 * v / best if best else 10.0
    return out


def overall(sc, weights):
    return {d: sum(weights[n] * s[n] for n in weights) for d, s in sc.items()}


def pareto(sc):
    def dominates(a, b):
        return all(sc[a][n] >= sc[b][n] for n in sc[a]) and any(sc[a][n] > sc[b][n] for n in sc[a])
    return {d: not any(dominates(o, d) for o in sc if o != d) for d in sc}


def sensitivity(sc, cons, steps=101):
    """{constraint: (ws, {design: overall at each w}, [(w, old winner, new winner)])}."""
    out = {}
    base = {c["name"]: c["weight"] for c in cons}
    for c in cons:
        k, rest = c["name"], 1 - base[c["name"]]
        ws = np.linspace(0, 1, steps)
        curves = {d: [] for d in sc}
        winners = []
        for w in ws:
            weights = {n: w if n == k else ((1 - w) * b / rest if rest else (1 - w) / (len(base) - 1))
                       for n, b in base.items()}
            ov = overall(sc, weights)
            for d in sc:
                curves[d].append(ov[d])
            winners.append(max(ov, key=ov.get))
        flips = [(round(float(ws[i]), 2), winners[i - 1], winners[i]) for i in range(1, steps) if winners[i] != winners[i - 1]]
        out[k] = (ws, curves, flips)
    return out


def radar(sc, cons, path):
    names = [c["name"] for c in cons]
    ang = np.linspace(0, 2 * np.pi, len(names), endpoint=False).tolist()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for d, s in sc.items():
        vals = [s[n] for n in names]
        ax.plot(ang + ang[:1], vals + vals[:1], label=d, linewidth=2)
        ax.fill(ang + ang[:1], vals + vals[:1], alpha=0.1)
    ax.set_xticks(ang)
    ax.set_xticklabels(names)
    ax.tick_params(axis="x", pad=14)
    ax.set_ylim(0, 10)
    ax.set_title("Normalized score per design constraint (10 = best)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def runs_plot(series, cons, path):
    """One polar chart per constraint with repeated runs: angle = run number, radius = that run's value."""
    cons = [c for c in cons if any(len(set(series[d][c["name"]])) > 1 for d in series)]   # skip constants (e.g. MI)
    if not cons:
        return False
    fig, axes = plt.subplots(1, len(cons), figsize=(5.5 * len(cons), 5.5), subplot_kw=dict(polar=True), squeeze=False)
    for ax, c in zip(axes[0], cons):
        for d in series:
            vals = series[d][c["name"]]
            ang = np.linspace(0, 2 * np.pi, len(vals), endpoint=False).tolist()
            ax.plot(ang + ang[:1], vals + vals[:1], label=f"{d} (mean {statistics.mean(vals):.4g})", linewidth=1)
        ax.set_xticks([])   # angle is only the run number
        allv = [v for d in series for v in series[d][c["name"]]]
        pad = 0.1 * (max(allv) - min(allv))
        ax.set_ylim(min(allv) - pad, max(allv) + pad)   # zoomed: radius does not start at 0, read the ring labels
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_title(f"{c['metric']}\n{max(len(series[d][c['name']]) for d in series)} runs, {c['better']} is better")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def sensitivity_plot(sens, cons, path):
    fig, axes = plt.subplots(1, len(cons), figsize=(4 * len(cons), 3.5), sharey=True, squeeze=False)
    for ax, c in zip(axes[0], cons):
        ws, curves, flips = sens[c["name"]]
        for d, ys in curves.items():
            ax.plot(ws, ys, label=d)
        ax.axvline(c["weight"], color="gray", linestyle="--", linewidth=1)   # current weight
        for w, _, _ in flips:
            ax.axvline(w, color="red", linestyle=":", linewidth=1)           # winner changes here
        ax.set_title(c["name"])
        ax.set_xlabel("weight")
    axes[0][0].set_ylabel("overall score")
    axes[0][0].legend()
    fig.suptitle("Sensitivity: overall score vs. one constraint's weight (gray = current, red = winner flips)", y=1.04)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(results=f"{HERE}/results", constraints=f"{HERE}/constraints.json", plots=True):
    cons = load_constraints(constraints)
    series = gather(cons, results)
    assert series, f"no results in {results}: run evaluate.py / train.py first"
    raw = {d: {n: statistics.mean(v) if v else None for n, v in s.items()} for d, s in series.items()}
    print(f"{'Constraint':46}" + "".join(f"{d:>12}" for d in raw) + "   winner")   # raw values
    for c in cons:
        vals = {d: raw[d][c["name"]] for d in raw if raw[d][c["name"]] is not None}
        best = min if c["better"] == "lower" else max
        winner = "/".join(d for d in vals if vals[d] == best(vals.values())) or "-"   # ties: all of them
        print(f"{c['name'] + ': ' + c['metric']:46}" + "".join(f"{vals[d]:>12.5g}" if d in vals else f"{'-':>12}"
                                                             for d in raw) + f"   {winner}")
    runs = {d: max(len(v) for v in s.values()) for d, s in series.items()}
    print("runs averaged: " + ", ".join(f"{d} {n}" for d, n in runs.items()))
    print()
    missing = [c for c in cons if any(raw[d][c["name"]] is None for d in raw)]
    for c in missing:
        print(f"skipped {c['name']}: no value for every design yet ({c['file']} {c['column']})")
    cons = [c for c in cons if c not in missing]
    total = sum(c["weight"] for c in cons)
    for c in cons:
        c["weight"] /= total

    sc = scores(raw, cons)
    ov = overall(sc, {c["name"]: c["weight"] for c in cons})
    par = pareto(sc)
    sens = sensitivity(sc, cons)
    ranked = sorted(ov, key=ov.get, reverse=True)

    with open(os.path.join(results, "tradeoff.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design"] + [f"{c['name']} raw" for c in cons] + [f"{c['name']} score" for c in cons]
                   + ["overall", "pareto_optimal"])
        for d in ranked:
            w.writerow([d] + [raw[d][c["name"]] for c in cons] + [round(sc[d][c["name"]], 2) for c in cons]
                       + [round(ov[d], 2), par[d]])

    print(f"{'Constraint':18}{'weight':>8}" + "".join(f"{d:>10}" for d in ranked))
    for c in cons:
        print(f"{c['name']:18}{c['weight']:>8.2f}" + "".join(f"{sc[d][c['name']]:>10.2f}" for d in ranked))
    print(f"{'Overall':18}{'':>8}" + "".join(f"{ov[d]:>10.2f}" for d in ranked))
    print(f"{'Pareto-optimal':18}{'':>8}" + "".join(f"{str(par[d]):>10}" for d in ranked))
    if len(ranked) > 1:
        print(f"\nwinner: {ranked[0]}, {ov[ranked[0]] - ov[ranked[1]]:.2f} points ahead of {ranked[1]} (out of 10)")
    for c in cons:
        flips = sens[c["name"]][2]
        print(f"sensitivity {c['name']:18}" + ("; ".join(f"weight {w}: {a} -> {b}" for w, a, b in flips)
                                                if flips else "winner never changes"))
    if plots:
        radar(sc, cons, os.path.join(results, "radar.png"))
        sensitivity_plot(sens, cons, os.path.join(results, "sensitivity.png"))
        has_runs = runs_plot(series, cons, os.path.join(results, "runs.png"))
        print(f"-> {results}/tradeoff.csv, radar.png, sensitivity.png" + (", runs.png" if has_runs else ""))
    return dict(raw=raw, scores=sc, overall=ov, pareto=par, sensitivity=sens, ranked=ranked)


def selftest():
    import tempfile

    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "evaluation.csv"), "w", newline="") as f:
        f.write("design,misclassification_pct,inference_mean_s,memory_mean_pct,maintainability_index,batch\n"
                "d1,50,0.5,9,10,old\n"                                       # older batch: ignored
                "d1,10,0.008,2,60,b1\nd1,10,0.012,2,60,b1\n"                 # newest batch: mean 0.01
                "d2,20,0.02,4,90,b1\nd3,5,0.04,4,60,b1\nd4,20,0.02,4,60,b1\n")   # d4 dominated by d2
    with open(os.path.join(tmp, "training_time.csv"), "w", newline="") as f:
        f.write("design,training_time_s\nd1,100\nd2,50\nd3,200\nd4,50\n")
    r = run(tmp, plots=True)
    sc = r["scores"]
    assert abs(r["raw"]["d1"]["Performance"] - 0.01) < 1e-12, r["raw"]["d1"]
    assert sc["d1"]["Performance"] == 10 and abs(sc["d3"]["Performance"] - 2.5) < 1e-9  # 0.01 / 0.04 x 10
    assert sc["d2"]["Sustainability"] == 10 and abs(sc["d1"]["Sustainability"] - 60 / 9) < 1e-9
    assert r["pareto"] == dict(d1=True, d2=True, d3=True, d4=False), r["pareto"]
    flips = r["sensitivity"]["Reliability"][2]
    assert flips and flips[-1][2] == "d3", flips                                       # reliability-heavy -> d3
    assert all(os.path.exists(os.path.join(tmp, f)) for f in ("radar.png", "sensitivity.png", "runs.png"))
    os.remove(os.path.join(tmp, "training_time.csv"))
    r = run(tmp, plots=False)                                                          # missing constraint skipped
    assert "Manufacturability" not in r["scores"]["d1"]
    print("selftest: ALL PASS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{HERE}/results")
    ap.add_argument("--constraints", default=f"{HERE}/constraints.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else run(a.results, a.constraints)
