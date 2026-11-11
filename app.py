"""Adaptive Air - Streamlit dashboard (SIMULATION ONLY: no real sensors, no real fan).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.controller import RuleBasedController
from src.evaluation import live_feedback_report, rolling_agreement, run_experiment, run_online_phase
from src.occupancy import Occupancy, OperatingState
from src.personalization import PersonalizationEngine
from src.simulator import PROFILE_DESCRIPTIONS, make_rng
from src.storage import append_events, load_feedback, make_event, reset_feedback
from src.ui_components import ADAPT_COLOR, BASE_COLOR, CSS, clean, esc, fan_html, stat_card
from src.validation import ValidationError

st.set_page_config(page_title="Adaptive Air", page_icon="🌀", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

OCC_CHOICES = {"Occupied": Occupancy.OCCUPIED, "Empty": Occupancy.EMPTY, "Unknown (sensor missing)": Occupancy.UNKNOWN}
ss = st.session_state


# ------------------------------------------------------------------ state
def reload_feedback(profile: str | None = None) -> None:
    """Re-read the CSV (source of truth) and retrain one profile (or all). Called ONLY when data changes."""
    res = load_feedback()
    ss.feedback, ss.load_notes = res.data, res.notes
    if profile:
        ss.engine.retrain(profile, ss.feedback)
    else:
        ss.engine.retrain_all(ss.feedback)


def init_state() -> None:
    if "engine" not in ss:
        ss.engine = PersonalizationEngine()
        ss.last_effect, ss.flash = None, None
        reload_feedback()
    ss.setdefault("live_temp", 30.0)
    ss.setdefault("live_hum", 70.0)
    ss.setdefault("live_hour", 19.0)
    ss.setdefault("live_occ", "Occupied")


def set_preset(temp: float, hum: float, hour: float) -> None:
    ss.live_temp, ss.live_hum, ss.live_hour, ss.live_occ = temp, hum, hour, "Occupied"


def pct(x) -> str:
    if x is None:
        return "n/a"
    return f"{100 * x:.2f}%" if 0.999 <= x < 1 else f"{100 * x:.1f}%"


def style(fig: go.Figure, height: int = 340) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(color="#12222B"), legend=dict(orientation="h", y=-0.2))
    return fig


init_state()
engine: PersonalizationEngine = ss.engine

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Adaptive Air")
    profile = st.selectbox("Simulated user profile", C.PROFILES, help="Each profile has its own separate learned model.")
    st.caption(PROFILE_DESCRIPTIONS[profile])
    idle_delay = st.slider("Idle delay when room is empty (min)", 0, 120, int(C.IDLE_DELAY_MINUTES), 5,
                           help="How long the room must be empty before the simulated fan goes IDLE.")
    engine.idle_delay_minutes = float(idle_delay)
    engine.baseline.idle_delay_minutes = float(idle_delay)
    st.divider()
    st.caption("All temperatures, occupancy and fan behaviour in this app are SIMULATED. "
               "No physical fan is connected or controlled.")
    with st.expander("Reset demonstration data"):
        st.write("This clears saved feedback. A timestamped backup copy is written to `data/backups/` first.")
        confirm = st.checkbox("I understand and want to reset")
        if st.button("Reset feedback data", disabled=not confirm):
            backup = reset_feedback()
            reload_feedback()
            ss.last_effect = None
            ss.flash = ("success", f"Feedback reset. Backup saved as {backup.name}." if backup else "Feedback file was already empty.")
            st.rerun()
    for note in ss.load_notes:
        st.warning(note)

# ------------------------------------------------------------------ header
st.markdown(clean("""<div class="aa-hero"><h1>Adaptive Air</h1>
<p>A fan that reads the room, decides on its own, and learns from your corrections.</p>
<span class="aa-sim">SIMULATION PROTOTYPE - simulated sensors, simulated fan</span></div>"""), unsafe_allow_html=True)
if ss.flash:
    getattr(st, ss.flash[0])(ss.flash[1])
    ss.flash = None

tab_live, tab_learn, tab_energy, tab_compare, tab_about = st.tabs(
    ["Live control", "Learning", "Energy saving", "Baseline comparison", "About & limits"])

# ================================================================== LIVE CONTROL (A, B, C)
with tab_live:
    left, mid, right = st.columns([1.15, 0.9, 1.35], gap="large")
    with left:
        st.subheader("Simulated room")
        p1, p2, p3 = st.columns(3)
        p1.button("Hot, humid evening", on_click=set_preset, args=(31.0, 75.0, 20.0), width="stretch")
        p2.button("Mild afternoon", on_click=set_preset, args=(26.0, 50.0, 14.0), width="stretch")
        p3.button("Cool night", on_click=set_preset, args=(23.0, 55.0, 2.0), width="stretch")
        temp = st.slider("Temperature (C)", 15.0, 45.0, key="live_temp", step=0.5)
        hum = st.slider("Relative humidity (%)", 20, 95, key="live_hum", step=1)
        hour = st.slider("Time of day (simulated, hours)", 0.0, 23.75, key="live_hour", step=0.25)
        st.caption(f"Simulated clock: {int(hour):02d}:{int(round((hour % 1) * 60)):02d}")
        occ_label = st.radio("Occupancy", list(OCC_CHOICES), key="live_occ", horizontal=True)
        occ = OCC_CHOICES[occ_label]
        minutes_empty = 0.0
        if occ is Occupancy.EMPTY:
            minutes_empty = float(st.slider("Minutes the room has been empty", 0, 180, 0, 5))

    base_d = engine.baseline.decide(temp, hum, occ, minutes_empty)
    adapt_d = engine.decide(profile, temp, hum, hour, occ, minutes_empty)

    with mid:
        st.subheader("Fan controller")
        st.markdown(fan_html(adapt_d.applied_level, adapt_d.state), unsafe_allow_html=True)

    with right:
        st.subheader("Recommendation")
        c1, c2 = st.columns(2)
        base_val = "OFF" if base_d.applied_level == 0 else f"Level {base_d.applied_level}"
        adapt_val = "OFF" if adapt_d.applied_level == 0 else f"Level {adapt_d.applied_level}"
        c1.markdown(stat_card("Baseline (rules)", base_val, f"comfort level {base_d.comfort_level}", base_d.state, BASE_COLOR), unsafe_allow_html=True)
        c2.markdown(stat_card("Adaptive Air", adapt_val, f"comfort level {adapt_d.comfort_level}", adapt_d.state, ADAPT_COLOR), unsafe_allow_html=True)
        st.markdown(f'<div class="aa-expl" style="margin-top:10px"><b>Why (Adaptive Air):</b> {esc(adapt_d.explanation)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="aa-expl" style="margin-top:8px;border-left-color:{BASE_COLOR}"><b>Why (baseline):</b> {esc(base_d.explanation)}</div>', unsafe_allow_html=True)
        st.caption(f"Learning model for '{profile}': {engine.model_for(profile).summary.message}")

    # ---------------- C. user feedback
    st.divider()
    st.subheader("Your feedback")
    can_learn = occ is Occupancy.OCCUPIED
    if not can_learn:
        st.info("Feedback is only learned while the room is occupied. When the room is empty or occupancy is unknown, "
                "the occupancy policy decides the fan state, not personal preference.")
    fb1, fb2, fb3 = st.columns([1.6, 1, 1.3], gap="large")
    with fb1:
        pref = st.radio("Which speed do you actually want here?", [1, 2, 3, 4, 5], index=adapt_d.comfort_level - 1,
                        horizontal=True, format_func=lambda x: f"Level {x}", key=f"pref-{profile}-{adapt_d.comfort_level}",
                        disabled=not can_learn)
    with fb2:
        repeats = st.number_input("Repeat as separate events", 1, 10, 1, disabled=not can_learn,
                                  help="Simulates the same correction on several similar evenings.")
    with fb3:
        b1, b2 = st.columns(2)
        apply_clicked = b1.button("Apply Correction", type="primary", disabled=(not can_learn) or pref == adapt_d.comfort_level, width="stretch")
        accept_clicked = b2.button("Accept as is", disabled=not can_learn, width="stretch",
                                   help="Log that the recommendation was fine (also useful training signal).")

    if apply_clicked or accept_clicked:
        chosen = pref if apply_clicked else adapt_d.comfort_level
        try:
            events = [make_event(profile, hour, temp, hum, occ, base_d.comfort_level, adapt_d.comfort_level, chosen,
                                 "manual" if apply_clicked else "accepted") for _ in range(int(repeats))]
            before = adapt_d.comfort_level
            append_events(events)
            reload_feedback(profile)
            after = engine.decide(profile, temp, hum, hour)
            ss.last_effect = {"n": len(events), "chosen": chosen, "before": before, "after": after,
                              "cond": f"{temp:.1f} C, {hum}% RH, {int(hour):02d}:{int(round((hour % 1) * 60)):02d}", "profile": profile}
        except ValidationError as exc:
            st.error(f"Feedback not saved: {exc}")
        else:
            st.rerun()

    eff = ss.last_effect
    if eff:
        m = engine.model_for(eff["profile"])
        st.success(f"Saved {eff['n']} feedback event(s) for '{eff['profile']}' (you chose Level {eff['chosen']}).")
        if eff["after"].model_active:
            st.info(f"Under the same conditions ({eff['cond']}) the recommendation went from Level {eff['before']} to "
                    f"Level {eff['after'].comfort_level} (learned offset {eff['after'].learned_offset:+d}). "
                    "This shows what the model now recommends; it is not proof the new level is more comfortable.")
        else:
            st.info(f"Learning is not active yet: {m.summary.n_events} of {m.min_events} events. Keep giving feedback, "
                    "or use the repeat box to log several similar events. Recommendations stay at the baseline until then.")

# ================================================================== LEARNING (D)
with tab_learn:
    fb = ss.feedback
    rep = live_feedback_report(fb, profile)
    model = engine.model_for(profile)
    st.subheader(f"Learning dashboard - {profile}")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Feedback events", rep["n_events"])
    k2.metric("Corrections", rep.get("n_corrections", 0), help="Events where you chose a different level than Adaptive Air recommended.")
    k3.metric("Model", "Active" if model.active else "Not active yet")
    k4.metric("Events, all profiles", len(fb))
    st.caption(model.summary.message)

    with st.expander("Generate simulated-user feedback (synthetic data, labelled source = simulated_profile)"):
        st.write("A simulated user with this profile's hidden preference reacts to Adaptive Air for several simulated days. "
                 "The events are saved like real feedback, so you can watch the model learn without clicking hundreds of times.")
        days = st.number_input("Simulated days", 1, 14, 3)
        if st.button(f"Simulate {int(days)} day(s) for {profile}"):
            prior = fb[fb["profile"] == profile] if len(fb) else fb
            seed = 100_000 + len(prior)
            idx = C.PROFILES.index(profile)
            events, _, _ = run_online_phase(profile, int(days), make_rng(seed, idx, 1), make_rng(seed, idx, 2),
                                            initial_events=fb if len(fb) else None)
            new = events[len(prior):]
            append_events(new)
            reload_feedback(profile)
            ss.flash = ("success", f"Added {len(new)} simulated feedback events for '{profile}'.")
            st.rerun()

    if rep["n_events"] == 0:
        st.info(rep["message"])
    else:
        st.markdown("#### Recommendations vs your chosen speed")
        o = rep["online"]
        table = pd.DataFrame({
            "Controller": ["Baseline (rules)", "Adaptive Air (as recommended at the time)"],
            "Exact match": [pct(o["baseline"]["exact"]), pct(o["adaptive"]["exact"])],
            "Within one level": [pct(o["baseline"]["within_one"]), pct(o["adaptive"]["within_one"])],
            "Events": [o["baseline"]["n"], o["adaptive"]["n"]]})
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Online results: each recommendation was made BEFORE the model saw that answer, so they are not leaked. "
                   "Early events mostly come from the untrained baseline, which lowers the adaptive score.")

        st.markdown("#### Held-out evaluation")
        h = rep["holdout"]
        if h is None:
            st.info(rep["message"])
        else:
            st.caption(f"Chronological split: the model was trained on the first {h['n_train']} events and tested on the "
                       f"last {h['n_test']} events it never saw.")
            ho = pd.DataFrame({
                "Measure": ["Training data (in-sample, optimistic)", "Held-out: baseline", "Held-out: Adaptive Air"],
                "Exact match": [pct(h["train_exact"]), pct(h["test_baseline"]["exact"]), pct(h["test_adaptive"]["exact"])],
                "Within one level": [pct(h["train_within_one"]), pct(h["test_baseline"]["within_one"]), pct(h["test_adaptive"]["within_one"])]})
            st.dataframe(ho, hide_index=True, width="stretch")
            d_exact = h["test_adaptive"]["exact"] - h["test_baseline"]["exact"]
            verdict = "higher than" if d_exact > 0 else "lower than" if d_exact < 0 else "equal to"
            st.write(f"On held-out events, Adaptive Air's exact-match rate is **{verdict}** the baseline's "
                     f"({d_exact * 100:+.1f} percentage points, n = {h['n_test']}). Small samples are noisy.")

        st.markdown("#### Agreement as feedback accumulates")
        pdf = fb[fb["profile"] == profile]
        roll = rolling_agreement(pdf, 10)
        if len(roll) < 5:
            st.info("The learning curve needs at least 14 events for this profile.")
        else:
            fig = go.Figure()
            fig.add_scatter(x=roll["step"], y=roll["Baseline"], name="Baseline", line=dict(color=BASE_COLOR))
            fig.add_scatter(x=roll["step"], y=roll["Adaptive"], name="Adaptive Air", line=dict(color=ADAPT_COLOR, width=3))
            fig.update_layout(title="Exact agreement with your chosen speed (rolling window of 10 events)",
                              xaxis_title="Feedback event number", yaxis=dict(title="Share matching", range=[0, 1.02], tickformat=".0%"))
            st.plotly_chart(style(fig), width="stretch")

        st.markdown("#### History")
        show_all = st.checkbox("Show all profiles", value=False)
        hist = (fb if show_all else pdf).copy()
        cols = ["step", "profile", "source", "sim_hour", "temperature", "humidity", "baseline_level", "adaptive_level", "chosen_level", "corrected", "recorded_at"]
        st.dataframe(hist[cols].iloc[::-1].head(200), hide_index=True, width="stretch")
        st.caption(f"Showing the latest {min(200, len(hist))} of {len(hist)} events.")

    st.markdown("#### What the model is, and what it is not")
    st.markdown(f"""
- **Model:** one small decision tree (depth {C.TREE_MAX_DEPTH}) per profile. It predicts your *offset* from the rule baseline using temperature, humidity and time of day.
- **Guards:** no learning before {C.MIN_FEEDBACK_EVENTS} events; offset limited to +/-{C.MAX_OFFSET}; final level always 1-5; occupancy policy always wins.
- **Retraining** happens only when feedback is saved, never on a page refresh.
- **Limits:** tiny datasets give noisy rules; the model cannot learn preferences for conditions it has not seen; simulated users are an approximation of real people.""")
    with st.expander("Learned rules for this profile (readable tree)"):
        st.code(model.rules_text())

# ================================================================== ENERGY SAVING (E)
with tab_energy:
    st.subheader("Occupancy-based energy saving (simulated)")
    st.write("A scripted scenario: the room is occupied, then empty, then occupied again. Both fans see the same room. "
             "The conventional fan has no occupancy awareness; Adaptive Air applies the occupancy policy.")
    e1, e2, e3 = st.columns(3)
    e_temp = e1.slider("Temperature (C)", 15.0, 45.0, 30.0, 0.5, key="e_temp")
    e_hum = e2.slider("Humidity (%)", 20, 95, 65, 1, key="e_hum")
    e_hour = e3.slider("Time of day (hours)", 0.0, 23.75, 19.0, 0.25, key="e_hour")
    e4, e5, e6 = st.columns(3)
    before_min = e4.slider("Occupied before (min)", 15, 240, 60, 15)
    empty_min = e5.slider("Room empty for (min)", 15, 360, 120, 15)
    after_min = e6.slider("Occupied again after (min)", 0, 240, 45, 15)
    conv = RuleBasedController(energy_saving=False)
    n_before, n_empty, n_after = before_min // C.STEP_MINUTES, empty_min // C.STEP_MINUTES, after_min // C.STEP_MINUTES
    rows, run = [], 0
    for i in range(n_before + n_empty + n_after):
        is_empty = n_before <= i < n_before + n_empty
        occ_i = Occupancy.EMPTY if is_empty else Occupancy.OCCUPIED
        mins = run * C.STEP_MINUTES if is_empty else 0.0
        run = run + 1 if is_empty else 0
        a = engine.decide(profile, e_temp, e_hum, e_hour, occ_i, mins, explain=False)
        c = conv.decide(e_temp, e_hum, occ_i, mins, explain=False)
        rows.append({"minute": i * C.STEP_MINUTES, "occupancy": occ_i.value, "conventional_level": c.applied_level,
                     "conventional_state": c.state.value, "adaptive_level": a.applied_level, "adaptive_state": a.state.value,
                     "empty": is_empty})
    sim = pd.DataFrame(rows)
    empty_rows = sim[sim["empty"]]
    conv_on = int((empty_rows["conventional_state"] != OperatingState.IDLE.value).sum()) * C.STEP_MINUTES
    adapt_on = int((empty_rows["adaptive_state"] != OperatingState.IDLE.value).sum()) * C.STEP_MINUTES
    saving = None if conv_on == 0 else 1 - adapt_on / conv_on

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Time room was empty", f"{len(empty_rows) * C.STEP_MINUTES} min")
    m2.metric("Conventional fan ON while empty", f"{conv_on} min")
    m3.metric("Adaptive Air ON while empty", f"{adapt_on} min", help="REDUCED_SPEED counts as on; only IDLE counts as off.")
    m4.metric("Reduction", pct(saving) if saving is not None else "n/a")
    fig = go.Figure()
    if len(empty_rows):
        fig.add_vrect(x0=empty_rows["minute"].min(), x1=empty_rows["minute"].max() + C.STEP_MINUTES, fillcolor="#E2E8F0",
                      opacity=0.6, line_width=0, annotation_text="Room empty", annotation_position="top left")
    fig.add_scatter(x=sim["minute"], y=sim["conventional_level"], name="Conventional fan", mode="lines", line=dict(color=BASE_COLOR, shape="hv", dash="dot"))
    fig.add_scatter(x=sim["minute"], y=sim["adaptive_level"], name="Adaptive Air", mode="lines", line=dict(color=ADAPT_COLOR, shape="hv", width=3))
    fig.update_layout(title="Simulated fan level over time (0 = IDLE)", xaxis_title="Simulated minutes", yaxis=dict(title="Level", range=[-0.2, 5.4], dtick=1))
    st.plotly_chart(style(fig), width="stretch")
    if empty_min <= idle_delay:
        st.warning(f"The room was empty for {empty_min} min, which is not longer than the {idle_delay} min idle delay, so the fan never reached IDLE and no saving is possible.")
    st.caption(f"Computed from {len(sim)} simulated decisions ({C.STEP_MINUTES} min each) using the same controllers as the rest of the app. "
               "Energy is represented by simulated fan-on time only; no power or kWh is measured or claimed.")
    with st.expander("Decision log"):
        st.dataframe(sim.drop(columns="empty"), hide_index=True, width="stretch")

# ================================================================== BASELINE COMPARISON (F)
with tab_compare:
    st.subheader("Baseline vs Adaptive Air - reproducible experiment")
    st.write(f"Simulated days have **{C.STEPS_PER_DAY} decisions** ({C.STEP_MINUTES} min each). For each seed and each of the 3 profiles, "
             f"Adaptive Air learns for **{C.TRAIN_DAYS} days**, then is frozen and scored on **{C.TEST_DAYS} new days** it never trained on. "
             "Baseline and Adaptive Air see identical rooms and identical simulated user preferences.")
    n_seeds = st.slider("Number of random seeds", 1, 10, len(C.EVAL_SEEDS), help="Default (5) matches the documented results.")
    seeds = tuple(range(11, 11 + n_seeds))

    @st.cache_data(show_spinner="Running simulation experiment...")
    def cached_experiment(seeds, delay):
        return run_experiment(seeds=seeds, idle_delay_minutes=delay)

    if st.button("Run experiment", type="primary", help="Takes about 5-10 seconds the first time; cached afterwards."):
        ss.exp_ran = True
    if not ss.get("exp_ran"):
        st.info("Press **Run experiment** to simulate the training and held-out days. Nothing is pre-computed or hard-coded.")
    else:
        res = cached_experiment(seeds, float(idle_delay))
        default_run = seeds == tuple(C.EVAL_SEEDS) and float(idle_delay) == C.IDLE_DELAY_MINUTES
        if not default_run:
            st.warning("Custom settings: these numbers differ from the documented run (5 seeds, 30 min idle delay).")
        o = res["overall"]
        icon = {"ACHIEVED": "✅", "PARTIALLY ACHIEVED": "🟡", "NOT ACHIEVED": "❌", "NOT EVALUABLE": "⚪"}
        rows = []
        for t in res["targets"]:
            if t["kind"] == "agreement":
                b, a, meas, tgt = pct(t["baseline"]), pct(t["adaptive"]), pct(t["measured"]), f">= {pct(t['target'])} within one level"
            elif "corrections" in t["name"]:
                b, a, meas, tgt = f"{t['baseline']:.1f} / day", f"{t['adaptive']:.1f} / day", f"{pct(t['measured'])} fewer", f">= {pct(t['target'])} fewer"
            else:
                b, a, meas, tgt = f"{t['baseline']} decisions", f"{t['adaptive']} decisions", f"{pct(t['measured'])} less", f">= {pct(t['target'])} less"
            rows.append({"Metric": t["name"], "Abstract target": tgt, "Baseline": b, "Adaptive Air": a, "Measured": meas,
                         "Status": f"{icon[t['status']]} {t['status']}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        std = o["correction_reduction_seed_std"]
        st.caption(f"Sample: {o['runs']} runs, {o['test_days']} held-out days, {o['occupied_decisions']:,} occupied decisions, "
                   f"{o['empty_decisions']:,} empty-room decisions. Correction reduction varied by seed with a standard deviation of "
                   f"{'n/a' if std is None else f'{100 * std:.1f} percentage points'} across {len(seeds)} seeds. "
                   "Targets are the abstract's goals, not established results.")

        cc1, cc2 = st.columns(2)
        with cc1:
            fig = go.Figure()
            profs = list(res["per_profile"])
            fig.add_bar(x=profs, y=[res["per_profile"][p]["baseline_corrections_per_day"] for p in profs], name="Baseline", marker_color=BASE_COLOR)
            fig.add_bar(x=profs, y=[res["per_profile"][p]["adaptive_corrections_per_day"] for p in profs], name="Adaptive Air", marker_color=ADAPT_COLOR)
            fig.update_layout(title="Corrections per held-out day", barmode="group", yaxis_title="Corrections / day")
            st.plotly_chart(style(fig), width="stretch")
        with cc2:
            fig = go.Figure()
            palette = {"Warm-preferring": "#B45309", "Cool-preferring": "#1D4ED8", "Evening-oriented": "#7C3AED"}
            for p, cur in res["learning_curve"].items():
                days_x = list(range(1, len(cur["adaptive"]) + 1))
                fig.add_scatter(x=days_x, y=cur["baseline"], name=f"{p} - baseline", line=dict(color=palette[p], dash="dot"))
                fig.add_scatter(x=days_x, y=cur["adaptive"], name=f"{p} - adaptive", line=dict(color=palette[p], width=3))
            fig.update_layout(title="Training days: corrections per day", xaxis_title="Simulated day", yaxis_title="Corrections / day")
            st.plotly_chart(style(fig), width="stretch")

        st.markdown("#### Per profile")
        pp = pd.DataFrame([{"Profile": p, "Baseline corrections/day": round(v["baseline_corrections_per_day"], 1),
                            "Adaptive corrections/day": round(v["adaptive_corrections_per_day"], 1),
                            "Fewer corrections": pct(v["correction_reduction"]),
                            "Baseline within one": pct(v["baseline_within_one"]), "Adaptive within one": pct(v["adaptive_within_one"]),
                            "Baseline exact": pct(v["baseline_exact"]), "Adaptive exact": pct(v["adaptive_exact"]),
                            "Empty-room fan-on reduction": pct(v["empty_fan_reduction"])} for p, v in res["per_profile"].items()])
        st.dataframe(pp, hide_index=True, width="stretch")

        st.markdown("#### How sensitive is the energy result?")
        sens = pd.DataFrame([{"Idle delay (min)": s["idle_delay_minutes"], "Decisions with fan on while empty": s["on_steps"],
                              "Reduction vs conventional fan": pct(s["reduction_vs_conventional"])} for s in res["energy_sensitivity"]])
        st.dataframe(sens, hide_index=True, width="stretch")
        st.write(f"If the conventional-fan user manually switched off after {int(C.FORGETFUL_USER_OFF_MINUTES)} min instead of never, the "
                 f"reduction at the current idle delay would be **{pct(o['empty_fan_reduction_vs_forgetful'])}**. "
                 "The 'conventional fan never switches off' baseline follows the abstract and is the most favourable comparison.")
        with st.expander("Assumptions and honest caveats"):
            st.markdown(f"""
    - **Synthetic users are learnable by design.** Each simulated user's preference is a simple rule of the same inputs the model sees (plus {int(C.PREFERENCE_NOISE * 100)}% noise). High agreement shows the learning loop works on such users; it does not prove real people are this consistent.
    - **Noise floor:** with {int(C.PREFERENCE_NOISE * 100)}% random +/-1 noise, no controller can reach 100% exact agreement.
    - **Within-one agreement is a lenient metric.** The rule baseline already scores {pct(o['baseline_within_one'])} on these profiles, so exact agreement and corrections/day are more informative.
    - **"Correction"** means one occupied decision where the recommended level differs from the simulated user's preferred level. A real person would not re-correct every {C.STEP_MINUTES} minutes, so these counts are an upper bound on real interventions.
    - **Energy** is fan-on time only. The result depends on the idle delay and on the simulated absence pattern (70% of days have a 2-8 h absence plus short ones). It is not a measurement of electricity.
    - Results come from synthetic data and do not demonstrate real-world comfort, savings or commercial readiness.""")

# ================================================================== ABOUT
with tab_about:
    st.subheader("About this prototype")
    st.markdown(f"""
**Adaptive Air** (team Breezient, atomQUEST 2026, Track 1) is a software comfort engine for fans. It runs entirely in simulation.

**Decision flow:** room state -> rule baseline level -> learned personal offset (per profile) -> occupancy policy (always last, always wins) -> simulated fan.

**Baseline rule (prototype assumption):** effective temperature = temperature + {C.HUMIDITY_WEIGHT} x (humidity - {int(C.HUMIDITY_REFERENCE)}); thresholds {C.TEMP_THRESHOLDS} give Levels 1-5.
These thresholds are not validated comfort standards.

**Occupancy policy:** occupied -> run; empty -> reduce by {C.REDUCTION_LEVELS} levels, then IDLE after the idle delay; unknown or missing -> keep running (never assumed empty).

**Limitations**
- Everything is simulated: sensors, occupancy, users and the fan. Nothing here controls real hardware.
- Rule thresholds, simulated preferences and occupancy patterns are assumptions that need real-world validation.
- Connecting a real fan would need a separate hardware safety design and testing process; software tests alone do not establish physical safety.
- Not a claim of real-world comfort, energy saving or commercial readiness.""")
