"""Phase 2 (car_follow) data and result analysis for the phase2-analysis skill: distribution tables (summary.md and
summary.json) and paper-style figures (PNG, 300 dpi) from the trial lists in configs/ and the results in outputs/.

Sealed lists ('main', 'calibration') are refused unless --allow-sealed is given. Give it only when the user has
explicitly allowed opening that list in the conversation. Without it, a completed evaluation run on a sealed list
shows its summary only: no per-trial or per-flight rows, no evidence or traffic files, no figure.

Run from the car_follow folder:
    uv run python ../.claude/skills/phase2-analysis/scripts/analyze.py inventory
    uv run python ../.claude/skills/phase2-analysis/scripts/analyze.py trials --splits dev pilot smoke
    uv run python ../.claude/skills/phase2-analysis/scripts/analyze.py run --run config3_dev_rec1_qlora1_v2
    uv run python ../.claude/skills/phase2-analysis/scripts/analyze.py labels --runs config3_dev_rec1_qlora1_v2
    uv run python ../.claude/skills/phase2-analysis/scripts/analyze.py signals --runs config3_dev_rec1_qlora1_v2
    uv run python ../.claude/skills/phase2-analysis/scripts/analyze.py signals --runs config3_dev_rec1_qlora1_v2 \
        --flight dev-017_ko
    uv run python ../.claude/skills/phase2-analysis/scripts/analyze.py events --name a4_dev
    uv run python ../.claude/skills/phase2-analysis/scripts/analyze.py lanes --run config3_a4_dev_guard-on_rfdetr-small_qlora1
Outputs go to outputs/analysis/<command>_<name>/ (local only, like the rest of outputs/).
"""
import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
CF = REPO / "car_follow"
sys.path.insert(0, str(CF))
CONFIGS, OUTPUTS = CF / "configs", CF / "outputs"
EVAL, LABELS, TRAFFIC = OUTPUTS / "demo1_eval", OUTPUTS / "failure_labels", OUTPUTS / "demo1_follow" / "traffic"
OUT = OUTPUTS / "analysis"
SEALED = ("main", "calibration")
DT = 0.05
WINDOW_S = 2.0

# Paper style from the dataviz reference palette: the first three categorical slots (blue, orange, aqua) pass every
# check for all pairs on white; aqua is under 3:1 contrast, so every figure ships with its table (summary.md).
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a")
INK, INK2, MUTED, GRID, AXIS, SURFACE, SHADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff", "#f0efec"


class Refused(SystemExit):
    pass


def refuse(what):
    raise Refused(f"refused: {what} is sealed. Open it only with the user's explicit permission in the conversation "
                  "(then pass --allow-sealed).")


# --- loading, with the sealing rule ---

def sealed_files():
    """Config files that hold a sealed list: those an index names as sealed, and any whose name says so."""
    names = {p.name for p in CONFIGS.glob("*.json") if any(k in p.stem for k in ("_main", "calibration", "sealed"))}
    for p in CONFIGS.glob("*.json"):
        if p.name in names:
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict) and "splits" in d:
            names |= {v["file"] for v in d.values() if isinstance(v, dict) and v.get("sealed") and "file" in v}
    return names


def load_list(index_path, split, allow):
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    if split in index.get("splits", {}):
        if split in SEALED and not allow:
            refuse(f"the {split} list")
        return index["splits"][split]
    entry = index.get(split)
    if isinstance(entry, dict) and entry.get("sealed"):
        if not allow:
            refuse(f"the {split} list ({entry['file']})")
        path = Path(index_path).parent / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"{path} does not match its sealed sha256")
        return json.loads(path.read_text(encoding="utf-8"))["trials"]
    raise SystemExit(f"{index_path} has no split {split!r}")


def load_run(name, allow):
    """(result, sealed, opened): a sealed run without `allow` keeps only its summaries."""
    r = json.loads((EVAL / f"{name}.json").read_text(encoding="utf-8"))
    sealed = r.get("split") in SEALED
    if sealed and not allow:
        if "languages" in r:
            r["languages"] = {lang: {k: v for k, v in part.items() if k != "trials"}
                              for lang, part in r["languages"].items()}
        r.pop("trials", None)
    return r, sealed, sealed and allow


def flight_rows(result):
    """[(trial, language or None, row)]."""
    if "languages" in result:
        return [(t, lang, row) for lang, part in result["languages"].items() for t, row in part.get("trials", {}).items()]
    return [(t, None, row) for t, row in result.get("trials", {}).items()]


def need_open(result, sealed, allow, what):
    if sealed and not allow:
        refuse(f"{what} of {result['split']} run")


# --- tables and figures ---

def md_table(header, rows):
    out = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join("" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Malgun Gothic", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
        "legend.frameon": False, "lines.linewidth": 1.2, "axes.unicode_minus": False,
    })
    return plt


def bars(ax, categories, series, title, xlabel="count", legend=True):
    """Horizontal bars, one group per category; `series` {name: [value per category]} with at most 3 series (one
    colour each, legend only for two or more)."""
    if len(series) > len(SLOTS):
        raise SystemExit("at most three series per panel (fold the rest or facet)")
    cats = list(categories)
    y = np.arange(len(cats))
    h = 0.8 / max(1, len(series))
    for i, (name, vals) in enumerate(series.items()):
        ax.barh(y - 0.4 + h * (i + 0.5), vals, height=h, color=SLOTS[i], edgecolor=SURFACE, linewidth=0.8,
                label=name)
    ax.set_yticks(y, cats)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    if len(series) > 1 and legend:
        ax.legend(loc="best")


def strip(ax, groups, title, xlabel, ref=None, colors=None):
    """One row per group: the values as dots (jittered), the median as a tick. `ref` (value, label) draws a solid
    reference line."""
    rng = np.random.default_rng(0)
    names = list(groups)
    for i, name in enumerate(names):
        v = np.asarray([x for x in groups[name] if x is not None and np.isfinite(x)], dtype=float)
        c = (colors or {}).get(name, SLOTS[0])
        if len(v):
            ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=9, color=c, edgecolor=SURFACE, linewidth=0.4,
                       zorder=3)
            ax.plot([np.median(v)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=1.0, zorder=4)
    ax.set_yticks(range(len(names)), [f"{n} (n={sum(1 for x in groups[n] if x is not None)})" for n in names])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    if ref is not None:
        ax.axvline(ref[0], color=MUTED, linewidth=0.8, zorder=2)
        lo, hi = ax.get_xlim()
        right = (ref[0] - lo) / (hi - lo) > 0.8   # near the right edge: label on the line's left
        ax.annotate(ref[1], (ref[0], 0.0), xycoords=("data", "axes fraction"), xytext=(-3 if right else 3, 3),
                    textcoords="offset points", fontsize=6.5, color=INK2, va="bottom",
                    ha="right" if right else "left")
    ax.set_xlabel(xlabel)
    ax.set_title(title)


def write(name, md, data, figures):
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    (out / "summary.md").write_text(f"# {name}\n\n생성: {stamp}, `{Path(__file__).name}`\n\n" + md + "\n",
                                    encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(dict(data, generated=stamp), indent=1, ensure_ascii=False,
                                                 default=str), encoding="utf-8")
    for fname, fig in figures:
        fig.savefig(out / fname)
    print(f"wrote {out.relative_to(CF)}: summary.md, summary.json" + "".join(f", {f}" for f, _ in figures))
    return out


def runs_name(runs):
    return runs[0] if len(runs) == 1 else f"{runs[0]}+{len(runs) - 1}"


def stats(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not len(v):
        return [None] * 5
    return [round(float(x), 3) for x in (np.min(v), np.percentile(v, 10), np.median(v), np.percentile(v, 90),
                                         np.max(v))]


# --- commands ---

def cmd_inventory(args):
    sealed = sealed_files()
    md, data = [], {"configs": [], "runs": [], "labels": [], "traffic": []}
    rows = []
    for p in sorted(CONFIGS.glob("*.json")):
        if p.name in sealed:
            rows.append([p.name, "sealed list", "(not opened)"])
            data["configs"].append({"file": p.name, "sealed": True})
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(d, dict) and "splits" in d:
            parts = [f"{k} {len(v)}" for k, v in d["splits"].items()]
            for k, v in d.items():
                if isinstance(v, dict) and v.get("sealed"):
                    ok = hashlib.sha256((p.parent / v["file"]).read_bytes()).hexdigest() == v["sha256"]
                    parts.append(f"{k} {v['count']} (sealed, sha256 {'matches' if ok else 'DIFFERS'})")
            rows.append([p.name, "trial index", ", ".join(parts)])
        else:
            rows.append([p.name, "config", ", ".join(list(d)[:6]) if isinstance(d, dict) else type(d).__name__])
        data["configs"].append({"file": p.name, "summary": rows[-1][2]})
    md += ["## 1. configs", md_table(["파일", "종류", "내용"], rows)]
    rows, sealed_recordings = [], set()
    for p in sorted(EVAL.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(r, dict) or "split" not in r:
            continue
        is_sealed = r["split"] in SEALED
        if is_sealed:
            sealed_recordings.add(r.get("recording"))
        if "languages" in r:
            n = sum(part["summary"]["trials"] for part in r["languages"].values())
            gate = r.get("gate", {}).get("passed")
        else:
            n, gate = r.get("summary", {}).get("trials"), None
        rows.append([p.stem, r.get("config"), r["split"] + (" (sealed: summary only)" if is_sealed else ""),
                     r.get("recording"), n, gate])
        data["runs"].append(dict(zip(("run", "config", "split", "recording", "flights", "gate"), rows[-1])))
    md += ["## 2. 평가 실행 (outputs/demo1_eval)", md_table(["실행", "구성", "목록", "교통 기록", "비행", "관문"], rows)]
    rows = []
    for p in sorted(LABELS.glob("*.json")) if LABELS.exists() else []:
        d = json.loads(p.read_text(encoding="utf-8"))
        c = d["counts"]
        fails = ", ".join(f"{k} {v}" for k, v in c["by_failure"].items() if v) or "none"
        rows.append([p.stem, d["split"] + (" (sealed)" if d["split"] in SEALED else ""), c["flights"], fails,
                     c["near_miss_flights"]])
        data["labels"].append(dict(zip(("file", "split", "flights", "failures", "near_miss_flights"), rows[-1])))
    md += ["## 3. 실패 라벨 (outputs/failure_labels)", md_table(["파일", "목록", "비행", "실패", "위기 비행"], rows)]
    rows = []
    for d in sorted(p for p in TRAFFIC.iterdir() if p.is_dir()):
        s = d.name in sealed_recordings
        rows.append([d.name, len(list(d.glob("*.npz"))), "sealed (not opened)" if s else "open"])
        data["traffic"].append(dict(zip(("recording", "trials", "state"), rows[-1])))
    md += ["## 4. 교통 기록 (outputs/demo1_follow/traffic)", md_table(["기록", "시험", "상태"], rows)]
    write("inventory", "\n\n".join(md), data, [])


def trial_features(trial, types, trials_mod, labels_mod):
    t = trial["target"]
    cond = labels_mod.conditions(trial)
    x, y = t["transform"][0], t["transform"][1]
    event = cond["event"] or {}
    return {
        "target type": types[t["blueprint"]]["type"],
        "colour family": trials_mod.color_family(t["color"]),
        "expression": trial["sentence"]["expression"],
        "look-alikes": str(len(trial["lookalikes"])),
        "lanes": str(t["lanes"]),
        "map cell": trials_mod.cell_split(trials_mod.cell_of(x, y)),
        "condition": cond["condition"],
        "distractor": cond["distractor"],
        "event": event.get("kind", "none") if isinstance(event, dict) else str(event),
    }


def cmd_trials(args):
    from src.car_follow import labels as labels_mod, trials as trials_mod
    index = CF / args.index
    if len(args.splits) > len(SLOTS):
        raise SystemExit("at most three splits per figure")
    types = trials_mod.load_types()
    feats = {s: [trial_features(t, types, trials_mod, labels_mod) for t in load_list(index, s, args.allow_sealed)]
             for s in args.splits}
    keys = list(next(iter(feats.values()))[0]) if any(feats.values()) else []
    md, data = [f"목록: {', '.join(f'{s} {len(v)}개' for s, v in feats.items())} (`{args.index}`)"], {"counts": {}}
    plt = pyplot()
    shown = [k for k in keys if len({f[k] for v in feats.values() for f in v}) > 1]
    cols = 3
    rows_n = math.ceil(len(shown) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(7.2, 2.0 * rows_n), squeeze=False)
    for ax, k in zip(axes.flat, shown):
        cats = sorted({f[k] for v in feats.values() for f in v},
                      key=lambda c: -sum(f[k] == c for v in feats.values() for f in v))
        counts = {s: [sum(f[k] == c for f in v) for c in cats] for s, v in feats.items()}
        bars(ax, cats, counts, k, legend=False)
        data["counts"][k] = {s: dict(zip(cats, c)) for s, c in counts.items()}
        md.append(f"## {k}\n\n" + md_table([k] + list(counts), [[c] + [counts[s][i] for s in counts]
                                                                  for i, c in enumerate(cats)]))
    for ax in list(axes.flat)[len(shown):]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if len(feats) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="upper right", ncol=len(feats),
                   bbox_to_anchor=(1.0, 1.0))
    same = [k for k in keys if k not in shown]
    if same:
        md.append("모든 시험에서 같은 값: " + ", ".join(f"{k} = {next(iter(feats.values()))[0][k]}" for k in same))
    md.append("![trials](fig_trials.png)")
    write(f"trials_{'_'.join(args.splits)}", "\n\n".join(md), data, [("fig_trials.png", fig)])


def cmd_run(args):
    r, sealed, opened = load_run(args.run, args.allow_sealed)
    md = [f"실행 `{args.run}`: 구성 {r.get('config')}, 목록 {r['split']}, 교통 기록 {r.get('recording')}"
          + (" (봉인 목록, 사용자 허락으로 열었음)" if opened else " (봉인 목록: 요약만)" if sealed else "")]
    parts = r["languages"] if "languages" in r else {"-": {"summary": r["summary"]}}
    keys = ("trials", "first_pick", "switch_free", "hold", "distance_error_m", "horizontal_error_m")
    md.append("## 1. 요약\n\n" + md_table(["언어"] + list(keys), [[lang] + [p["summary"].get(k) for k in keys]
                                                                 for lang, p in parts.items()]))
    data = {"run": args.run, "split": r["split"], "sealed": sealed, "opened_sealed": opened,
            "summary": {lang: p["summary"] for lang, p in parts.items()}}
    if sealed and not opened:
        write(f"run_{args.run}", "\n\n".join(md), data, [])
        return
    rows = flight_rows(r)
    by_lang = {}
    for t, lang, row in rows:
        pick = row.get("pick") or {}
        by_lang.setdefault(lang or "-", []).append({
            "trial": t, "switches": len(row["switches"]), "hold": row["hold"],
            "distance_error_m": row["distance_error_m"], "hidden_s": row["hidden_s"], "main_s": row["main_s"],
            "log_ratio": pick.get("log_ratio"), "pick_s": pick.get("seconds"),
            "climb_s": (row.get("avoidance") or {}).get("time_s")})
    metrics = ("hold", "distance_error_m", "hidden_s", "log_ratio", "climb_s")
    table = [[lang, m] + stats([x[m] for x in v]) for lang, v in by_lang.items() for m in metrics]
    md.append("## 2. 시험별 분포 (최소, 10%, 중앙값, 90%, 최대)\n\n"
              + md_table(["언어", "값", "최소", "p10", "중앙값", "p90", "최대"], table))
    md.append("## 3. 시험별 값\n\n" + md_table(
        ["언어", "시험", "바뀜", "유지", "거리 오차 m", "가림 s", "본 구간 s", "log(p1/p2)", "고도 회피 s"],
        [[lang, x["trial"], x["switches"], x["hold"], x["distance_error_m"], x["hidden_s"], x["main_s"],
          x["log_ratio"], x["climb_s"]] for lang, v in by_lang.items() for x in v]))
    data["trials"] = by_lang
    plt = pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 3.6))
    colors = {lang: SLOTS[i % 3] for i, lang in enumerate(by_lang)}
    pass_lines = r.get("pass_lines", {"hold": 0.95, "distance_error_m": 3.0})
    thr = r.get("ambiguity_threshold")
    thr = thr.get("threshold") if isinstance(thr, dict) else thr
    for ax, (m, title, xl, ref) in zip(axes.flat, (
            ("hold", "Hold per trial (main window)", "share of visible ticks on the target",
             (pass_lines.get("hold", 0.95), "pass line")),
            ("distance_error_m", "Distance error per trial", "m", (pass_lines.get("distance_error_m", 3.0),
                                                                    "pass line")),
            ("log_ratio", "First-pick ambiguity log(p1/p2)", "log ratio", (thr, "threshold") if thr else None),
            ("hidden_s", "Target hidden per trial", "s", None))):
        strip(ax, {lang: [x[m] for x in v] for lang, v in by_lang.items()}, title, xl, ref, colors)
    fig.tight_layout()
    md.append("![run](fig_run.png)")
    write(f"run_{args.run}", "\n\n".join(md), data, [("fig_run.png", fig)])


def load_labels(run, allow):
    p = LABELS / f"{run}.json"
    if not p.exists():
        raise SystemExit(f"{p.relative_to(CF)} is missing: run scripts/failure_labels.py --run {run} first")
    d = json.loads(p.read_text(encoding="utf-8"))
    sealed = d["split"] in SEALED
    if sealed and not allow:
        d["flights"] = []
    return d, sealed


def cmd_labels(args):
    flights, md, notes = [], [], []
    counts = {}
    for run in args.runs:
        d, sealed = load_labels(run, args.allow_sealed)
        counts[run] = d["counts"]
        if sealed and not args.allow_sealed:
            notes.append(f"`{run}` ({d['split']}): 봉인 목록이라 집계만 씀")
        flights += [dict(f, run=run) for f in d["flights"]]
    md.append("## 1. 실행별 집계\n\n" + md_table(
        ["실행", "비행", "실패 없음"] + [k for k in counts[args.runs[0]]["by_failure"]] + ["위기 비행"],
        [[run, c["flights"], c["no_failure"]] + list(c["by_failure"].values()) + [c["near_miss_flights"]]
         for run, c in counts.items()]))
    md += notes
    data = {"runs": args.runs, "counts": counts}
    if not flights:
        write(f"labels_{runs_name(args.runs)}", "\n\n".join(md), data, [])
        return
    from src.car_follow import labels as labels_mod
    langs = sorted({f["language"] or "-" for f in flights})
    kinds = [k for k in labels_mod.OUTCOMES if any(f["failure"] == k for f in flights)]
    chains = {}
    for f in flights:
        key = " -> ".join(e["kind"] for e in f["chain"]) or "none"
        chains[key] = chains.get(key, 0) + 1
    md.append("## 2. 사슬\n\n" + md_table(["사슬", "비행"], sorted(chains.items(), key=lambda x: -x[1])))
    fails = [f for f in flights if f["failure"]]
    md.append("## 3. 실패 비행\n\n" + md_table(
        ["실행", "시험", "언어", "실패", "t_fail s", "원인", "복구 s"],
        [[f["run"], f["trial"], f["language"], f["failure"], f["t_fail_s"], f["chain"][0].get("cause"),
          f["recovered_s"]] for f in fails]))
    expo = labels_mod.EXPOSURES
    md.append("## 4. 노출 (비행당 초)\n\n" + md_table(
        ["노출", "합", "비행 수 (>0)", "최소", "p10", "중앙값", "p90", "최대"],
        [[k, round(sum(f["exposure_s"][k] for f in flights), 2), sum(f["exposure_s"][k] > 0 for f in flights)]
         + stats([f["exposure_s"][k] for f in flights]) for k in expo]))
    data.update(chains=chains, failures=[{k: f[k] for k in ("run", "trial", "language", "failure", "t_fail_s")}
                                         for f in fails])
    plt = pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1, 1.3, 1]})
    colors = {lang: SLOTS[i % 3] for i, lang in enumerate(langs)}
    cats = ["none"] + kinds
    bars(axes[0], cats, {lang: [sum((f["failure"] or "none") == c and (f["language"] or "-") == lang
                                    for f in flights) for c in cats] for lang in langs}, "Outcome per flight",
         "flights")
    if fails:
        groups = {k: [f["t_fail_s"] for f in fails if f["failure"] == k] for k in kinds}
        strip(axes[1], groups, "Failure onset t_fail", "s after the start")
        axes[1].set_xlim(0, max(60.0, max(f["ticks"] for f in flights) * DT))
    else:
        axes[1].set_axis_off()
        axes[1].text(0.5, 0.5, "no failure", ha="center", va="center", color=INK2, transform=axes[1].transAxes)
    bars(axes[2], list(expo), {lang: [round(sum(f["exposure_s"][k] for f in flights
                                                if (f["language"] or "-") == lang), 1) for k in expo]
                               for lang in langs}, "Exposure (total)", "s")
    fig.tight_layout()
    md.append("![labels](fig_labels.png)")
    write(f"labels_{runs_name(args.runs)}", "\n\n".join(md), data, [("fig_labels.png", fig)])


def rolling_mean(x, w):
    """Trailing mean over `w` ticks, NaN ignored (NaN where the window has no value)."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    c = np.cumsum(np.r_[0.0, np.where(ok, x, 0.0)])
    n = np.cumsum(np.r_[0, ok.astype(int)])
    lo = np.maximum(0, np.arange(len(x)) + 1 - w)
    s, m = c[1:] - c[lo], n[1:] - n[lo]
    return np.where(m > 0, s / np.maximum(m, 1), np.nan)


def online_signals(ev, lock):
    """Signals a monitor could compute in flight (docs/failure_plan.md 2, table 1): share of the last 2 s with the
    target's track not matched, the estimate's one-tick jump off a constant-velocity step, the target track's
    detection score over the last 2 s, the estimate's angle below the drone's horizon."""
    n = len(ev["owner"])
    w = int(round(WINDOW_S / DT))
    est = np.asarray(ev["est_pos"], dtype=float)
    alive = (np.asarray(ev["track_id"]) >= 0) & np.isfinite(est).all(axis=1)
    alive[: lock if lock is not None else n] = False
    unmatched = np.where(alive, (~np.isfinite(ev["target_score"])).astype(float), np.nan)
    jump = np.full(n, np.nan)
    ok = alive[2:] & alive[1:-1] & alive[:-2]
    pred = 2 * est[1:-1, :2] - est[:-2, :2]
    jump[2:][ok] = np.linalg.norm(est[2:, :2] - pred, axis=1)[ok]
    rel = np.asarray(ev["log_position"], dtype=float) - est
    below = np.where(alive, np.degrees(np.arctan2(rel[:, 2], np.linalg.norm(rel[:, :2], axis=1))), np.nan)
    score = np.where(alive, ev["target_score"], np.nan)
    return {"unmatched_2s": np.where(alive, rolling_mean(unmatched, w), np.nan), "estimate_jump_m": jump,
            "det_score_2s": np.where(alive, rolling_mean(score, w), np.nan), "below_est_deg": below}


SIGNALS = (("unmatched_2s", "max", "Unmatched share, last 2 s", "share"),
           ("estimate_jump_m", "max", "Estimate jump off a CV step", "m"),
           ("det_score_2s", "min", "Detection score, last 2 s (min)", "score"),
           ("below_est_deg", "max", "Estimate below the horizon", "deg"),
           ("drift_m", "max", "Estimate error (ground truth)", "m"),
           ("below_true_deg", "max", "Target below the horizon (ground truth)", "deg"))


def flight_signals(run, result, trial, lang, row, by_id, types):
    from scripts.failure_labels import lock_and_pick, word_match
    from src.car_follow import labels as labels_mod
    stem = trial if lang is None else f"{trial}_{lang}"
    ev = dict(np.load(EVAL / run / f"{stem}.npz"))
    rec = dict(np.load(TRAFFIC / result["recording"] / f"{trial}.npz"))
    lock, pick = lock_and_pick(row, ev)
    lab, arr = labels_mod.label_flight(ev, rec, word_match(rec, by_id[trial], types), lock, pick, with_arrays=True)
    sig = online_signals(ev, lock)
    sig["drift_m"], sig["below_true_deg"] = arr["drift"], np.where(arr["alive"], arr["below"], np.nan)
    return stem, ev, lab, sig


def cmd_signals(args):
    from src.car_follow import trials as trials_mod
    types = trials_mod.load_types()
    per, md, data = [], [], {"runs": args.runs}
    plt = pyplot()
    figures = []
    for run in args.runs:
        r, sealed, opened = load_run(run, args.allow_sealed)
        need_open(r, sealed, args.allow_sealed, "the flights")
        by_id = {t["id"]: t for t in load_list(CF / args.index, r["split"], args.allow_sealed)}
        for trial, lang, row in flight_rows(r):
            stem = trial if lang is None else f"{trial}_{lang}"
            if args.flight and stem != args.flight:
                continue
            stem, ev, lab, sig = flight_signals(run, r, trial, lang, row, by_id, types)
            end = lab["t_fail_tick"] if lab["t_fail_tick"] is not None else lab["ticks"]
            cat = lab["failure"] and "failure" or ("near-miss" if lab["near_miss_flight"] else "normal")
            vals = {}
            for key, how, _, _ in SIGNALS:
                v = sig[key][:end]
                v = v[np.isfinite(v)]
                vals[key] = (float(v.max()) if how == "max" else float(v.min())) if len(v) else None
            per.append({"run": run, "flight": stem, "category": cat, "failure": lab["failure"], **vals})
            if args.flight:
                figures.append(("fig_flight.png", flight_figure(plt, stem, lab, sig)))
    if not per:
        raise SystemExit("no flight matched")
    cats = [c for c in ("normal", "near-miss", "failure") if any(p["category"] == c for p in per)]
    colors = dict(zip(("normal", "near-miss", "failure"), SLOTS))
    md.append("값은 비행마다 잠금부터 t_fail 전까지(실패가 없으면 끝까지)의 최댓값 또는 최솟값이다. "
              "추정 오차와 실제 각도는 정답이라 감시기는 볼 수 없다.")
    md.append("## 1. 분류별 분포 (최소, 10%, 중앙값, 90%, 최대)\n\n" + md_table(
        ["신호", "분류", "비행", "최소", "p10", "중앙값", "p90", "최대"],
        [[key, c, sum(p["category"] == c for p in per)] + stats([p[key] for p in per if p["category"] == c])
         for key, _, _, _ in SIGNALS for c in cats]))
    md.append("## 2. 비행별 값\n\n" + md_table(["실행", "비행", "분류", "실패"] + [k for k, _, _, _ in SIGNALS],
                                            [[p["run"], p["flight"], p["category"], p["failure"]]
                                             + [None if p[k] is None else round(p[k], 3) for k, _, _, _ in SIGNALS]
                                             for p in per]))
    data["flights"] = per
    if not args.flight:
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.4))
        for ax, (key, how, title, xl) in zip(axes.flat, SIGNALS):
            strip(ax, {c: [p[key] for p in per if p["category"] == c] for c in cats}, title, f"{how}, {xl}",
                  colors=colors)
        for ax in axes[:, 1:].flat:
            ax.set_yticklabels([])
        fig.tight_layout()
        figures.append(("fig_signals.png", fig))
    md += [f"![{f}]({f})" for f, _ in figures]
    name = f"signals_{runs_name(args.runs)}" + (f"_{args.flight}" if args.flight else "")
    write(name, "\n\n".join(md), data, figures)


def flight_figure(plt, stem, lab, sig):
    """Time series of one flight: the signals, t_fail as a solid line, near-miss runs shaded."""
    keys = ("drift_m", "unmatched_2s", "estimate_jump_m", "det_score_2s")
    titles = {k: t.replace(" (min)", "") for k, _, t, _ in SIGNALS}
    units = {k: u for k, _, _, u in SIGNALS}
    t = np.arange(lab["ticks"]) * DT
    fig, axes = plt.subplots(len(keys), 1, figsize=(7.2, 4.6), sharex=True)
    for ax, k in zip(axes, keys):
        ax.plot(t, sig[k], color=SLOTS[0], linewidth=1.0)
        for nm in lab["near_miss"]:
            ax.axvspan(nm["start_s"], nm["start_s"] + nm["length_s"], color=SHADE, zorder=0, linewidth=0)
        if lab["t_fail_s"] is not None:
            ax.axvline(lab["t_fail_s"], color=INK2, linewidth=0.8)
        ax.set_ylabel(units[k])
        ax.set_title(titles[k])
    if lab["t_fail_s"] is not None:
        axes[0].annotate(f"t_fail {lab['t_fail_s']:.2f} s ({lab['failure']})", (lab["t_fail_s"], 1.0),
                         xycoords=("data", "axes fraction"), xytext=(-3, -10), textcoords="offset points",
                         fontsize=6.5, color=INK2, ha="right")
    if lab["near_miss"]:
        axes[0].annotate("shaded: near-miss", (0.0, 1.0), xycoords="axes fraction", xytext=(3, -10),
                         textcoords="offset points", fontsize=6.5, color=INK2)
    axes[-1].set_xlabel("s after the start")
    fig.suptitle(stem, x=0.01, ha="left", fontsize=9, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def cmd_events(args):
    """The failure-trial generator's runs (outputs/failure_trials/<name>/pilot.json, dev cells only): event yield per
    condition and level, event times, and one example of each event over time."""
    path = OUTPUTS / "failure_trials" / args.name / "pilot.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d["trials"]
    if any(r.get("split", "generator") not in ("generator", "dev") for r in rows):
        refuse(f"the generator run {args.name} holds a list other than dev")
    md = [f"생성기 실행 `{args.name}` (seed 기준 {d['seed_base']} + {d.get('seed_offset', 0)}), 사건 창 {d['window_s']} s, "
          f"통과 기준 {d['pass_yield']:.0%}"]
    md.append("## 1. 조건별 사건 비율\n\n" + md_table(
        ["조건", "기록", "일어남", "비율", "통과", "단계별", "기록 전에 버린 뽑기"],
        [[c, s["recorded"], s["happened"], s["yield"], s["passes"], s["by_level"], s["draws_rejected"]]
         for c, s in d["summary"].items()]))
    md.append("## 2. 시험별\n\n" + md_table(
        ["시험", "조건", "단계", "대본 상태", "사건", "시각 s", "값"],
        [[r["id"], r["condition"], r["level"], r["script"].get("state", "")[:40], r["check"].get("happened"),
          r["check"].get("t_s"), {k: v for k, v in r["check"].items() if k not in ("happened", "t_s", "tick")}]
         for r in rows]))
    plt = pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.8), gridspec_kw={"height_ratios": [1.3, 1]})
    from src.car_follow import failure_trials as ft
    examples = {}
    for r in rows:
        if r["check"].get("happened") and r["condition"] not in examples:
            examples[r["condition"]] = r
    shown = [c for c in d["summary"]][:3]
    for col in range(3):
        ax = axes[0, col]
        cond = shown[col] if col < len(shown) else None
        r = examples.get(cond)
        if r is None:
            ax.set_axis_off()
            continue
        rec = dict(np.load(OUTPUTS / "failure_trials" / args.name / "traffic" / f"{r['id']}.npz"))
        t = np.arange(len(rec["pos"])) * DT
        if cond.startswith("A4"):   # the target's lateral position from its start lane (and the swap partner's)
            lane_example(ax, rec, r, cond, t)
            ax.set_xlabel("s")
            continue
        if cond in ("A1", "A2"):
            di = int(np.flatnonzero(rec["kind"] == "distractor")[0])
            along, lateral, _, vt, vd = ft.pair_relation(rec, 0, di)
            if cond == "A1":   # one quantity per panel: the distractor's offset to the side of the target
                ax.plot(t, np.abs(lateral), color=SLOTS[0])
                ax.set_ylabel("distractor to the side, m")
                ax.set_ylim(0, 8)
                ax.set_title(f"A1 cut-in ({r['id']}, {r['level']})")
            else:
                ax.plot(t, vt, color=SLOTS[0], label="target")
                ax.plot(t, vd, color=SLOTS[1], label="distractor")
                ax.set_ylabel("speed, m/s")
                ax.legend(loc="upper left", fontsize=6)
                ax.set_title(f"A2 queue ({r['id']}, {r['level']})")
        else:
            pos = rec["pos"][:, 0, :2].astype(float)
            speed = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1) / DT]
            ax.plot(t, speed, color=SLOTS[0])
            ax.set_ylabel("target speed, m/s")
            ax.set_title(f"A3 occlusion ({r['id']})")
        ax.axvspan(r["check"]["t_s"], r["check"]["t_s"] + r["check"].get("hidden_s", r["check"].get("stood_s", 1.0)),
                   color=SHADE, zorder=0, linewidth=0)
        ax.axvline(r["check"]["t_s"], color=INK2, linewidth=0.8)
        ax.set_xlabel("s")
    for ax in axes[0]:
        ax.set_xlim(0, 60)
    gs = axes[1, 0].get_gridspec()
    for ax in axes[1]:
        ax.remove()
    bottom = fig.add_subplot(gs[1, :])
    strip(bottom, {c: [r["check"].get("t_s") for r in rows if r["condition"] == c and r["check"].get("happened")]
                   for c in d["summary"]}, "Event onset per condition (the window shaded)", "s after the start")
    bottom.axvspan(*d["window_s"], color=SHADE, zorder=0, linewidth=0)
    bottom.set_xlim(0, 60)
    fig.tight_layout()
    md.append("위 줄의 회색 띠는 확인된 사건 구간(가림, 정지)이고 세로선은 사건 시작이다. ![events](fig_events.png)")
    write(f"events_{args.name}", "\n\n".join(md), {"run": args.name, "summary": d["summary"]},
          [("fig_events.png", fig)])


LANE_NAMES = {"A4a": "single", "A4b": "weave", "A4c": "swap"}


def lane_example(ax, rec, r, cond, t):
    """One A4 trial: the target's lateral position from its start lane (+ right), and in a swap the partner's
    (from the same lane), with each change's span shaded."""
    from src.car_follow import failure_trials as ft
    ax.plot(t, ft.lateral_track(rec, 0), color=SLOTS[0], label="target")
    if cond == "A4c":
        di = int(np.flatnonzero(rec["kind"] == "distractor")[0])
        _, lateral, _, _, _ = ft.pair_relation(rec, 0, di)
        start = float(np.median(lateral[:20]))   # the partner's side and lane at the start
        ax.plot(t, ft.lateral_track(rec, di) + start, color=SLOTS[1], label="near-twin")
        ax.legend(loc="best", fontsize=6)
    ch = ft.lane_changes(rec, 0)
    for c in ch:
        ax.axvspan(c["start_s"], c["end_s"], color=SHADE, zorder=0, linewidth=0)
    if ch:
        ax.set_xlim(max(0.0, ch[0]["start_s"] - 8.0), min(t[-1], ch[-1]["end_s"] + 8.0))
    ax.set_ylabel("lateral from start lane, m (+ right)")
    ax.set_title(f"{cond} {LANE_NAMES.get(cond, '')} ({r['id']})")


def cmd_lanes(args):
    """The lane-change comparison of an evaluation run over A4 trials (scripts/lane_change_report.py writes
    outputs/failure_labels/<run>_lanes.json): switches, losses, errors and jitter before, during, between and after
    the target's lane changes, per condition, and one flight per condition over time."""
    d = json.loads((LABELS / f"{args.run}_lanes.json").read_text(encoding="utf-8"))
    if set(d["split"].split("+")) & set(SEALED) and not args.allow_sealed:
        refuse(f"the lane comparison of {args.run}")
    conds = list(d["by_condition"])
    phases = [p for p in ("before", "during", "between", "after")
              if any(p in part["pooled"] for part in d["by_condition"].values())]
    md = [f"평가 실행 `{args.run}` (실패 시험 `{d['failure_run']}`, {d['split']}), 구간 규칙 {d['rules']}"]
    md.append("## 1. 같은 차 유지\n\n" + md_table(
        ["조건", "비행", "처음 지정 맞음", "같은 차 유지", "차선 변경 수", "첫 실패 (시험, 언어, 종류, 시각, 구간)"],
        [[f"{c} {LANE_NAMES.get(c, '')}", part["flights"], part["first_pick_right"], part["kept_same_car"],
          part["changes"], "; ".join(f"{f['trial']} {f['language']} {f['kind']} {f['t_s']} s {f['phase']}"
                                     for f in part["first_failures"]) or "없음"]
         for c, part in d["by_condition"].items()]))
    rows, motion = [], []
    for c, part in d["by_condition"].items():
        for ph in phases:
            r = part["pooled"].get(ph)
            if r is None:
                continue
            e = r["events"]
            rows.append([c, ph, r["flights"], r["seconds"], r["wrong_box_s"], r["no_box_s"], r["hold"], r["lost_s"],
                         f"{e['switch_box']}/{e['switch_estimate']}/{e['drift']}/{e['loss']}",
                         r["estimate_error_mean"], r["estimate_error_p95_worst"]])
            motion.append([c, ph, r["distance_error_mean"], r["distance_error_p95_worst"], r["across_error_mean"],
                           r["along_error_mean"], r["accel_rms"], r["accel_across_rms"], r["accel_along_rms"],
                           r["jerk_rms"], r["yaw_rate_rms"], r["target_yaw_rate"], r["target_accel_rms"]])
    md.append("## 2. 구간별 추적 (비행을 합침)\n\n사건 열은 상자 바뀜/추정 바뀜/이탈/소실 수다.\n\n" + md_table(
        ["조건", "구간", "비행", "초", "다른 차 상자 s", "상자 없음 s", "유지율", "트랙 없음 s", "사건",
         "추정 오차 평균 m", "p95 최대 m"], rows))
    md.append("## 3. 구간별 비행 (비행을 합침)\n\n옆과 진행은 목표 진행 방향 기준이다. 목표 yaw 속도는 굽은 길, 목표 가속은 "
              "목표의 속도 변화를 보인다.\n\n" + md_table(
                  ["조건", "구간", "거리 오차 평균 m", "p95 최대 m", "옆 m", "진행 m", "가속 RMS m/s²", "옆 가속",
                   "진행 가속", "저크 RMS m/s³", "드론 yaw 속도 RMS °/s", "목표 yaw 속도 RMS °/s",
                   "목표 가속 RMS m/s²"], motion))
    plt = pyplot()
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), gridspec_kw={"height_ratios": [1.15, 1, 1]})
    for col in range(3):
        ax = axes[0, col]
        cond = conds[col] if col < len(conds) else None
        fl = next((f for f in d["flights"] if f["condition"] == cond), None)
        if fl is None:
            ax.set_axis_off()
            continue
        rec = dict(np.load(OUTPUTS / "failure_trials" / d["failure_run"] / "traffic" / f"{fl['trial']}.npz"))
        lane_example(ax, rec, {"id": fl["trial"]}, cond, np.arange(len(rec["pos"])) * DT)
        ax.set_xlabel("s")
    panels = [("wrong_box_s", True, "Box on another car", "s per min"),
              ("lost_s", True, "Target track gone", "s per min"),
              ("estimate_error_mean", False, "Estimate error, mean", "m"),
              ("across_error_mean", False, "Distance error across heading", "m, mean"),
              ("accel_across_rms", False, "Drone accel. across heading", "m/s², RMS"),
              ("target_yaw_rate", False, "Target yaw rate (curves)", "deg/s, RMS")]
    x = np.arange(len(phases))
    for i, (key, per_min, title, unit) in enumerate(panels):
        ax = axes[1 + i // 3, i % 3]
        for j, c in enumerate(conds[:3]):
            pooled = d["by_condition"][c]["pooled"]
            ys = []
            for ph in phases:
                r = pooled.get(ph)
                v = None if r is None else r[key]
                if v is not None and per_min:
                    v = 60.0 * v / r["seconds"] if r["seconds"] else None
                ys.append(np.nan if v is None else v)
            ok = [i for i, y in enumerate(ys) if np.isfinite(y)]   # phases a condition has (no 'between' in one change)
            ax.plot(x[ok], np.asarray(ys)[ok], color=SLOTS[j], marker="o", markersize=4, markeredgecolor=SURFACE,
                    markeredgewidth=0.6, label=f"{c} {LANE_NAMES.get(c, '')}")
        ax.set_xticks(x, phases)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(bottom=0)
        if per_min:
            ax.set_ylim(top=max(1.0, ax.get_ylim()[1]))   # all zero: keep a readable scale
        ax.set_ylabel(unit)
        ax.set_title(title)
        if i == 0 and len(conds) > 1:
            ax.legend(loc="best", fontsize=6)
    fig.tight_layout()
    md.append("위 줄은 조건마다 비행 하나의 목표 횡위치(시작 차선 기준, 오른쪽 +)이고 회색 띠는 차선 변경 구간이다. 아래 두 줄은 "
              "구간별로 합친 값이다. 거리 오차와 드론 가속은 목표 진행 방향에 수직인 성분이다(차선 변경이 만드는 쪽). 마지막 칸은 "
              "목표 자신의 yaw 속도로, 굽은 길이 섞인 정도를 보인다. ![lanes](fig_lanes.png)")
    write(f"lanes_{args.run}", "\n\n".join(md), {"run": args.run, "by_condition": d["by_condition"]},
          [("fig_lanes.png", fig)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--allow-sealed", action="store_true",
                        help="open a sealed list; only with the user's explicit permission in the conversation")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inventory", parents=[common],
                   help="what lists, runs, labels and recordings exist (sealed ones by count only)")
    s = sub.add_parser("trials", parents=[common],
                       help="distribution of a trial list (types, colours, expressions, conditions)")
    s.add_argument("--splits", nargs="+", default=["dev", "pilot", "smoke"])
    s.add_argument("--index", default="configs/demo1_trials.json")
    s = sub.add_parser("run", parents=[common], help="per-trial results of one evaluation run")
    s.add_argument("--run", required=True)
    s = sub.add_parser("labels", parents=[common], help="failure labels: outcomes, chains, t_fail, exposures")
    s.add_argument("--runs", nargs="+", required=True)
    s = sub.add_parser("signals", parents=[common],
                       help="monitor signal candidates per flight, or one flight over time")
    s.add_argument("--runs", nargs="+", required=True)
    s.add_argument("--flight", default=None, help="one flight (<trial>_<lang>) as a time series")
    s.add_argument("--index", default="configs/demo1_trials.json", help="the trial index the runs flew")
    s = sub.add_parser("events", parents=[common], help="the failure-trial generator's event yield and examples")
    s.add_argument("--name", default="confirm", help="a run under outputs/failure_trials")
    s = sub.add_parser("lanes", parents=[common],
                       help="lane-change comparison (A4): before / during / after the target's lane changes")
    s.add_argument("--run", required=True, help="an evaluation run with outputs/failure_labels/<run>_lanes.json")
    args = p.parse_args()
    {"inventory": cmd_inventory, "trials": cmd_trials, "run": cmd_run, "labels": cmd_labels,
     "signals": cmd_signals, "events": cmd_events, "lanes": cmd_lanes}[args.cmd](args)


if __name__ == "__main__":
    main()
