"""Small presentation helpers for the Streamlit dashboard (HTML/CSS only, no logic)."""
from __future__ import annotations

import html

from .occupancy import OperatingState

INK, MUTED, TEAL, AMBER, SLATE = "#12222B", "#4B5B66", "#0F766E", "#B45309", "#64748B"
BASE_COLOR, ADAPT_COLOR = SLATE, TEAL

CSS = """
<style>
.block-container{padding-top:1.6rem;max-width:1250px}
.aa-hero{background:linear-gradient(120deg,#0B3C49 0%,#0F766E 100%);color:#fff;border-radius:16px;padding:20px 26px;margin-bottom:14px}
.aa-hero h1{margin:0;font-size:1.9rem;letter-spacing:-.5px;color:#fff}
.aa-hero p{margin:4px 0 0 0;color:#E2F3F1;font-size:.97rem}
.aa-sim{display:inline-block;background:#FEF3C7;color:#7C2D12;border-radius:999px;padding:3px 12px;font-size:.78rem;font-weight:700;margin-top:10px;letter-spacing:.3px}
.aa-card{background:#fff;border:1px solid #DDE5E9;border-radius:14px;padding:14px 16px;height:100%}
.aa-label{font-size:.76rem;text-transform:uppercase;letter-spacing:.8px;color:#4B5B66;font-weight:700}
.aa-value{font-size:2rem;font-weight:800;color:#12222B;line-height:1.15}
.aa-sub{font-size:.85rem;color:#4B5B66}
.aa-badge{display:inline-block;border-radius:999px;padding:2px 10px;font-size:.78rem;font-weight:700}
.aa-run{background:#CCFBF1;color:#134E4A}.aa-red{background:#FEF3C7;color:#7C2D12}.aa-idle{background:#E2E8F0;color:#1E293B}
.aa-fan{background:#fff;border:1px solid #DDE5E9;border-radius:14px;padding:12px;text-align:center}
.aa-pips{display:flex;gap:8px;justify-content:center;margin:6px 0 2px 0}
.aa-pip{width:34px;height:10px;border-radius:5px;background:#D5DEE3}
.aa-note{font-size:.8rem;color:#4B5B66}
.aa-expl{background:#fff;border-left:4px solid #0F766E;border-radius:6px;padding:10px 14px;font-size:.92rem;color:#12222B;border-top:1px solid #DDE5E9;border-right:1px solid #DDE5E9;border-bottom:1px solid #DDE5E9}
@keyframes aa-spin{to{transform:rotate(360deg)}}
@media (prefers-reduced-motion: reduce){.aa-blades{animation:none!important}}
</style>
"""


def clean(markup: str) -> str:
    """Strip indentation/blank lines so Streamlit's markdown never turns HTML into a code block."""
    return "".join(line.strip() for line in markup.splitlines() if line.strip())


def esc(text) -> str:
    return html.escape(str(text))


def badge(state: OperatingState) -> str:
    cls = {OperatingState.RUNNING: "aa-run", OperatingState.REDUCED_SPEED: "aa-red", OperatingState.IDLE: "aa-idle"}[state]
    return f'<span class="aa-badge {cls}">{state.value}</span>'


def stat_card(label: str, value: str, sub: str = "", state: OperatingState | None = None, accent: str = "#DDE5E9") -> str:
    extra = badge(state) if state else ""
    return clean(f"""<div class="aa-card" style="border-top:4px solid {accent}">
        <div class="aa-label">{esc(label)}</div><div class="aa-value">{esc(value)}</div>
        <div>{extra}</div><div class="aa-sub">{esc(sub)}</div></div>""")


_SPIN_SECONDS = {1: 3.0, 2: 1.9, 3: 1.2, 4: 0.75, 5: 0.45}
_BLADE = "M100 100 C 78 70, 80 28, 100 16 C 120 28, 122 70, 100 100 Z"


def fan_html(applied_level: int, state: OperatingState) -> str:
    """Inline SVG fan. CSS-only animation (no JS, no reruns), disabled for reduced-motion users."""
    idle = state is OperatingState.IDLE
    color = SLATE if idle else (AMBER if state is OperatingState.REDUCED_SPEED else TEAL)
    anim = "" if idle else f"animation:aa-spin {_SPIN_SECONDS.get(applied_level, 1.2)}s linear infinite;"
    blades = "".join(f'<path d="{_BLADE}" fill="{color}" opacity="0.88" transform="rotate({a} 100 100)"/>' for a in (0, 120, 240))
    pips = "".join(f'<div class="aa-pip" style="background:{color if i <= applied_level else "#D5DEE3"}"></div>' for i in range(1, 6))
    big = "OFF" if idle else f"Level {applied_level}"
    return clean(f"""<div class="aa-fan">
        <div class="aa-label">Simulated fan</div>
        <svg viewBox="0 0 200 200" width="210" height="210" role="img" aria-label="Simulated fan, {esc(big)}, {state.value}">
        <circle cx="100" cy="100" r="92" fill="none" stroke="#B8C6CD" stroke-width="3"/>
        <circle cx="100" cy="100" r="62" fill="none" stroke="#E2E8EC" stroke-width="1.5"/>
        <g class="aa-blades" style="transform-origin:100px 100px;{anim}">{blades}</g>
        <circle cx="100" cy="100" r="13" fill="#12222B"/></svg>
        <div class="aa-value" style="font-size:1.6rem">{esc(big)}</div>
        <div class="aa-pips">{pips}</div><div>{badge(state)}</div>
        <div class="aa-note">Simulated fan. No real hardware is controlled.</div></div>""")
