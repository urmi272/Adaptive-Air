"""Reproducible evaluation of Adaptive Air against the rule-based baseline.

PROTOCOL (per seed x simulated user profile; every number is computed from simulation output):
  1. TRAINING PHASE  - 14 simulated days. Each occupied 15-min step: the baseline and the adaptive
     engine each recommend a level; the simulated user's preferred level (with 10 % noise) is the
     answer. Every occupied step is logged as feedback. The model is retrained at the END of each day.
  2. HELD-OUT PHASE - 10 NEW days drawn from independent random streams. The model is frozen (never
     fitted on these days). Baseline and adaptive are scored on identical conditions and identical
     user preferences.
  Metrics (held-out days only):
     corrections/day : occupied decisions where the recommended level != the user's preferred level
     agreement       : share of occupied decisions within one level of the preferred level (also exact)
     empty fan-on    : empty-room decisions where the fan is not IDLE; denominator = empty-room decisions
                       of the conventional fan, which has no occupancy awareness and so stays on.
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .controller import comfort_level_from_conditions
from .occupancy import OperatingState
from .personalization import PersonalizationEngine, ProfileModel
from .simulator import generate_day, make_rng, preferred_level
from .storage import make_event
from .validation import clip_level


# ------------------------------------------------------------------ helpers
def agreement_rates(predicted, chosen) -> dict:
    p, c = np.asarray(predicted, dtype=float), np.asarray(chosen, dtype=float)
    n = int(len(c))
    if n == 0:
        return {"n": 0, "exact": None, "within_one": None}
    return {"n": n, "exact": float(np.mean(p == c)), "within_one": float(np.mean(np.abs(p - c) <= 1))}


def reduction(baseline_value: float, new_value: float):
    """Fractional reduction versus the baseline; None when the baseline is zero (undefined)."""
    return None if baseline_value == 0 else 1.0 - new_value / baseline_value


def classify(value, target: float, baseline_value: float = 0.0) -> str:
    """ACHIEVED if value >= target; PARTIALLY ACHIEVED if it improves on the baseline but misses the target."""
    if value is None:
        return "NOT EVALUABLE"
    if value >= target:
        return "ACHIEVED"
    return "PARTIALLY ACHIEVED" if value > baseline_value else "NOT ACHIEVED"


# ------------------------------------------------------------------ training phase
def run_online_phase(profile: str, n_days: int, env_rng, user_rng, initial_events: pd.DataFrame | None = None,
                     source: str = "simulated_profile", day_offset: int = 0):
    """Simulate n_days of a user correcting the adaptive engine. Returns (events, daily_stats, model).

    `events` is a list of validated feedback dicts (each also carries a `day` number for auditing).
    """
    model = ProfileModel(profile)
    events: list[dict] = []
    if initial_events is not None and len(initial_events):
        events = initial_events[initial_events["profile"] == profile].to_dict("records")
        model.fit(initial_events)
    daily = []
    for d in range(n_days):
        day_no = day_offset + d + 1
        frame = generate_day(env_rng)
        n_occ = base_corr = adapt_corr = 0
        for hour, t, h, occupied in frame[["hour", "temperature", "humidity", "occupied"]].itertuples(index=False):
            if not occupied:
                continue
            base = comfort_level_from_conditions(t, h)
            adaptive = clip_level(base + model.predict_offset(t, h, hour).offset)
            pref = preferred_level(profile, t, h, hour, user_rng)
            n_occ += 1
            base_corr += pref != base
            adapt_corr += pref != adaptive
            ev = make_event(profile, hour, t, h, "occupied", base, adaptive, pref, source=source)
            ev["day"] = day_no
            events.append(ev)
        daily.append({"day": day_no, "occupied_decisions": n_occ,
                      "baseline_corrections": int(base_corr), "adaptive_corrections": int(adapt_corr),
                      "model_active": model.active})
        model.fit(pd.DataFrame(events))          # retrain once per day, on past days only
    return events, daily, model


# ------------------------------------------------------------------ held-out phase
def evaluate_heldout(model: ProfileModel, profile: str, n_days: int, env_rng, user_rng,
                     idle_delay_minutes: float = C.IDLE_DELAY_MINUTES) -> dict:
    """Score a FROZEN model on new days. The model is never fitted here."""
    engine = PersonalizationEngine(idle_delay_minutes)
    engine.models[profile] = model
    base_pred, adapt_pred, chosen = [], [], []
    empty_steps = conv_on = adaptive_on = forgetful_on = 0
    runs: list[int] = []
    for _ in range(n_days):
        frame = generate_day(env_rng)
        run = 0
        for hour, t, h, occupied in frame[["hour", "temperature", "humidity", "occupied"]].itertuples(index=False):
            if occupied:
                if run:
                    runs.append(run)
                run = 0
                base = comfort_level_from_conditions(t, h)
                adaptive = engine.decide(profile, t, h, hour, "occupied", explain=False).comfort_level
                base_pred.append(base)
                adapt_pred.append(adaptive)
                chosen.append(preferred_level(profile, t, h, hour, user_rng))
            else:
                minutes_empty = run * C.STEP_MINUTES
                d = engine.decide(profile, t, h, hour, "empty", minutes_empty, explain=False)
                empty_steps += 1
                conv_on += 1                                           # conventional fan never switches itself off
                adaptive_on += d.state is not OperatingState.IDLE
                forgetful_on += minutes_empty < C.FORGETFUL_USER_OFF_MINUTES
                run += 1
        if run:
            runs.append(run)
    base_pred, adapt_pred, chosen = map(np.asarray, (base_pred, adapt_pred, chosen))
    return {
        "occupied_decisions": int(len(chosen)),
        "baseline_corrections": int(np.sum(base_pred != chosen)),
        "adaptive_corrections": int(np.sum(adapt_pred != chosen)),
        "baseline_agreement": agreement_rates(base_pred, chosen),
        "adaptive_agreement": agreement_rates(adapt_pred, chosen),
        "empty_decisions": empty_steps,
        "conventional_on_empty": conv_on,
        "adaptive_on_empty": int(adaptive_on),
        "forgetful_on_empty": int(forgetful_on),
        "empty_run_lengths": runs,
    }


def _on_steps_for_delay(run_lengths, delay_minutes: float) -> int:
    k = math.ceil(delay_minutes / C.STEP_MINUTES)
    return int(sum(min(r, k) for r in run_lengths))


# ------------------------------------------------------------------ full experiment
def run_experiment(seeds=C.EVAL_SEEDS, train_days: int = C.TRAIN_DAYS, test_days: int = C.TEST_DAYS,
                   profiles=C.PROFILES, idle_delay_minutes: float = C.IDLE_DELAY_MINUTES) -> dict:
    runs = []
    for seed in seeds:
        for p_idx, profile in enumerate(C.PROFILES):
            if profile not in profiles:
                continue
            events, daily, model = run_online_phase(profile, train_days, make_rng(seed, p_idx, 1), make_rng(seed, p_idx, 2))
            held = evaluate_heldout(model, profile, test_days, make_rng(seed, p_idx, 3), make_rng(seed, p_idx, 4),
                                    idle_delay_minutes)
            runs.append({"seed": seed, "profile": profile, "train_events": len(events),
                         "model_active": model.active, "daily": daily, "heldout": held})

    def pool(rs):
        g = lambda k: int(sum(r["heldout"][k] for r in rs))
        occ, empty = g("occupied_decisions"), g("empty_decisions")
        base_c, adapt_c = g("baseline_corrections"), g("adaptive_corrections")
        n_days = len(rs) * test_days
        wr = lambda key, which: (sum(r["heldout"][which][key] * r["heldout"][which]["n"] for r in rs) / occ) if occ else None
        return {
            "runs": len(rs), "test_days": n_days, "occupied_decisions": occ, "empty_decisions": empty,
            "baseline_corrections": base_c, "adaptive_corrections": adapt_c,
            "baseline_corrections_per_day": base_c / n_days if n_days else None,
            "adaptive_corrections_per_day": adapt_c / n_days if n_days else None,
            "correction_reduction": reduction(base_c, adapt_c),
            "baseline_within_one": wr("within_one", "baseline_agreement"),
            "adaptive_within_one": wr("within_one", "adaptive_agreement"),
            "baseline_exact": wr("exact", "baseline_agreement"),
            "adaptive_exact": wr("exact", "adaptive_agreement"),
            "conventional_on_empty": g("conventional_on_empty"),
            "adaptive_on_empty": g("adaptive_on_empty"),
            "empty_fan_reduction": reduction(g("conventional_on_empty"), g("adaptive_on_empty")),
            "forgetful_on_empty": g("forgetful_on_empty"),
            "empty_fan_reduction_vs_forgetful": reduction(g("forgetful_on_empty"), g("adaptive_on_empty")),
        }

    overall = pool(runs)
    per_profile = {p: pool([r for r in runs if r["profile"] == p]) for p in profiles}
    seed_reductions = [pool([r for r in runs if r["seed"] == s])["correction_reduction"] for s in seeds]
    seed_reductions = [x for x in seed_reductions if x is not None]
    overall["correction_reduction_per_seed"] = seed_reductions
    overall["correction_reduction_seed_std"] = float(np.std(seed_reductions, ddof=1)) if len(seed_reductions) > 1 else None

    curves = {}
    for p in profiles:
        rs = [r for r in runs if r["profile"] == p]
        curves[p] = {
            "baseline": [float(np.mean([r["daily"][d]["baseline_corrections"] for r in rs])) for d in range(train_days)],
            "adaptive": [float(np.mean([r["daily"][d]["adaptive_corrections"] for r in rs])) for d in range(train_days)],
        }

    all_runs = [x for r in runs for x in r["heldout"]["empty_run_lengths"]]
    sensitivity = []
    for delay in (0, 15, 30, 45, 60):
        on = _on_steps_for_delay(all_runs, delay)
        sensitivity.append({"idle_delay_minutes": delay, "on_steps": on,
                            "reduction_vs_conventional": reduction(overall["conventional_on_empty"], on)})

    for r in runs:
        r["heldout"].pop("empty_run_lengths")
    targets = build_targets(overall)
    return {
        "config": {"seeds": list(seeds), "train_days": train_days, "test_days": test_days,
                   "profiles": list(profiles), "steps_per_day": C.STEPS_PER_DAY, "step_minutes": C.STEP_MINUTES,
                   "idle_delay_minutes": idle_delay_minutes, "preference_noise": C.PREFERENCE_NOISE,
                   "min_feedback_events": C.MIN_FEEDBACK_EVENTS},
        "overall": overall, "per_profile": per_profile, "learning_curve": curves,
        "energy_sensitivity": sensitivity, "targets": targets, "runs": runs,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def build_targets(overall: dict) -> list[dict]:
    cr, ag, ef = overall["correction_reduction"], overall["adaptive_within_one"], overall["empty_fan_reduction"]
    return [
        {"name": "Manual corrections per day (after 14 days)", "target": C.TARGET_CORRECTION_REDUCTION,
         "kind": "reduction", "baseline": overall["baseline_corrections_per_day"],
         "adaptive": overall["adaptive_corrections_per_day"], "measured": cr,
         "status": classify(cr, C.TARGET_CORRECTION_REDUCTION, 0.0)},
        {"name": "Agreement within one level (held-out days)", "target": C.TARGET_AGREEMENT_WITHIN_ONE,
         "kind": "agreement", "baseline": overall["baseline_within_one"], "adaptive": ag, "measured": ag,
         "status": classify(ag, C.TARGET_AGREEMENT_WITHIN_ONE, overall["baseline_within_one"] or 0.0)},
        {"name": "Fan-on time while room is empty", "target": C.TARGET_EMPTY_FAN_REDUCTION,
         "kind": "reduction", "baseline": overall["conventional_on_empty"], "adaptive": overall["adaptive_on_empty"],
         "measured": ef, "status": classify(ef, C.TARGET_EMPTY_FAN_REDUCTION, 0.0)},
    ]


# ------------------------------------------------------------------ live-feedback reports (dashboard)
def rolling_agreement(events: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Online (prequential) exact-match rate. Each logged prediction was made BEFORE the user answered."""
    d = events.sort_values("step").reset_index(drop=True)
    out = pd.DataFrame({"step": d["step"]})
    out["Baseline"] = (d["baseline_level"] == d["chosen_level"]).astype(float).rolling(window, min_periods=window).mean()
    out["Adaptive"] = (d["adaptive_level"] == d["chosen_level"]).astype(float).rolling(window, min_periods=window).mean()
    return out.dropna()


def live_feedback_report(events: pd.DataFrame, profile: str) -> dict:
    """Summaries of one profile's logged feedback, with a chronological hold-out when enough data exists."""
    d = events[events["profile"] == profile].sort_values("step") if len(events) else events
    n = len(d)
    if n == 0:
        return {"n_events": 0, "message": "No feedback recorded for this profile yet."}
    online = {"baseline": agreement_rates(d["baseline_level"], d["chosen_level"]),
              "adaptive": agreement_rates(d["adaptive_level"], d["chosen_level"])}
    rep = {"n_events": n, "n_corrections": int(d["corrected"].sum()), "online": online, "holdout": None}
    if n < C.HOLDOUT_MIN_EVENTS:
        rep["message"] = (f"Held-out evaluation needs at least {C.HOLDOUT_MIN_EVENTS} events for this profile "
                          f"({n} so far). No results are shown rather than guessing.")
        return rep
    cut = int(n * (1 - C.HOLDOUT_FRACTION))
    train, test = d.iloc[:cut], d.iloc[cut:]
    model = ProfileModel(profile)
    summary = model.fit(train)                         # trained on the EARLIER events only
    if not model.active:
        rep["message"] = f"Hold-out split made, but the model needs {C.MIN_FEEDBACK_EVENTS} training events ({cut} available)."
        return rep
    adapt = [clip_level(b + model.predict_offset(t, h, hr).offset)
             for b, t, h, hr in zip(test["baseline_level"], test["temperature"], test["humidity"], test["sim_hour"])]
    rep["holdout"] = {
        "n_train": cut, "n_test": len(test),
        "train_exact": summary.train_exact, "train_within_one": summary.train_within_one,
        "test_baseline": agreement_rates(test["baseline_level"], test["chosen_level"]),
        "test_adaptive": agreement_rates(adapt, test["chosen_level"]),
    }
    return rep


# ------------------------------------------------------------------ CLI
def format_markdown(res: dict) -> str:
    o, cfg = res["overall"], res["config"]
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines = [
        f"Seeds {cfg['seeds']} | {len(cfg['profiles'])} profiles | {cfg['train_days']} training days + "
        f"{cfg['test_days']} held-out days | {cfg['steps_per_day']} decisions/day ({cfg['step_minutes']} min) | "
        f"idle delay {cfg['idle_delay_minutes']:.0f} min",
        f"Held-out sample: {o['runs']} runs, {o['test_days']} days, {o['occupied_decisions']} occupied decisions, "
        f"{o['empty_decisions']} empty-room decisions", "",
        "| Metric | Target | Baseline | Adaptive | Measured | Status |", "|---|---|---|---|---|---|"]
    for t in res["targets"]:
        if t["kind"] == "agreement":
            b, a = pct(t["baseline"]), pct(t["adaptive"])
        elif "corrections" in t["name"]:
            b, a = f"{t['baseline']:.1f}/day", f"{t['adaptive']:.1f}/day"
        else:
            b, a = f"{t['baseline']} steps", f"{t['adaptive']} steps"
        lines.append(f"| {t['name']} | {'>= ' + pct(t['target'])} | {b} | {a} | {pct(t['measured'])} | {t['status']} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Run the Adaptive Air evaluation experiment.")
    ap.add_argument("--out", default=str(C.EXPERIMENT_DIR / "evaluation_results.json"))
    args = ap.parse_args()
    res = run_experiment()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, default=float))
    print(format_markdown(res))
    print(f"\nFull results written to {out}")


if __name__ == "__main__":
    main()
