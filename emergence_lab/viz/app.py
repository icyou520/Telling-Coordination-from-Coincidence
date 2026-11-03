"""Interactive Emergence Lab window (matplotlib widgets).

Animation only redraws the changing artists (blitting), and the slow
analyses (layer ladder, tests) run in a background thread, so the window
stays responsive while they compute.
"""

from __future__ import annotations

import textwrap
import time
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

from .views import ALL_VIEWS, SystemView

EXPLORE, TESTS = "Explore layers", "Tests"

HELP = (
    "F = the micro rules\n"
    "C = coarse-graining (zoom out)\n"
    "G = law of the zoomed-out layer\n\n"
    "A layer is emergent when\n"
    "C(F(x)) = G(C(x)): zooming out\n"
    "then applying G gives the same\n"
    "answer as simulating every part."
)


def verdict(row: dict) -> str:
    if row["scale"] == 1:
        return "Micro layer: the exact rules live here."
    own, gap = row["r2_own"], row["gap"]
    if gap > 0.05:
        return "NOT CLOSED: finer detail improves the prediction, so this layer is missing hidden variables."
    if own >= 0.9:
        return "STRONG EMERGENT LAYER: predicts itself almost perfectly without finer detail."
    if own >= 0.5:
        return "CANDIDATE LAYER: closed (finer detail doesn't help) and partly predictable."
    return "WEAK: closed but mostly unpredictable here; looks like noise at this scale."


def _compute(view_cls, param, kind: str):
    """Run an analysis on a private copy of the system, safe to call from a worker thread."""
    view = view_cls()
    view.set_param(param)
    return view.ladder() if kind == "ladder" else view.compute_tests()


class Explorer:
    FRAME_MS = 40

    def __init__(self, interactive: bool = True):
        self.interactive = interactive
        self.fig = plt.figure(figsize=(16, 9.5))
        if self.fig.canvas.manager is not None:
            self.fig.canvas.manager.set_window_title("Emergence Lab")
        self.views: dict[str, SystemView] = {v.name: v() for v in ALL_VIEWS}
        self.sys: SystemView = next(iter(self.views.values()))
        self.mode = EXPLORE
        self.scale_idx = 2
        self.playing = False
        self.content: list = []
        self.live_artists: list = []
        self.background = None
        self.rng = np.random.default_rng(0)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs: dict = {}

        canvas = self.fig.canvas
        self.timer = canvas.new_timer(interval=self.FRAME_MS)
        self.timer.add_callback(self._tick)
        self.debounce = canvas.new_timer(interval=600)
        self.debounce.single_shot = True
        self.debounce.add_callback(self._after_param_change)
        self.poller = canvas.new_timer(interval=150)
        self.poller.add_callback(self._poll_jobs)
        canvas.mpl_connect("draw_event", self._on_draw)
        canvas.mpl_connect("close_event", lambda _e: self._shutdown())

        self._build_controls()
        self._build_sliders()
        self._build_content()

    # ------------------------------------------------------------ controls
    def _build_controls(self):
        f = self.fig
        f.text(0.015, 0.955, "Emergence Lab", fontsize=17, weight="bold")
        f.text(0.015, 0.93, "When does a zoomed-out view get its own laws?", fontsize=8.5, color="0.35")

        f.text(0.015, 0.895, "VIEW", fontsize=8, weight="bold", color="0.4")
        self.view_radio = RadioButtons(f.add_axes([0.015, 0.815, 0.15, 0.075]), [EXPLORE, TESTS])
        self.view_radio.on_clicked(self._on_mode)

        f.text(0.015, 0.79, "SYSTEM", fontsize=8, weight="bold", color="0.4")
        self.sys_radio = RadioButtons(f.add_axes([0.015, 0.68, 0.15, 0.105]), list(self.views))
        self.sys_radio.on_clicked(self._on_system)

        def button(rect, label, cb):
            b = Button(f.add_axes(rect), label)
            b.label.set_fontsize(9)
            b.on_clicked(cb)
            return b

        self.play_btn = button([0.015, 0.47, 0.072, 0.04], "Play", self._on_play)
        self.step_btn = button([0.093, 0.47, 0.072, 0.04], "Step", self._on_step)
        self.perturb_btn = button([0.015, 0.42, 0.072, 0.04], "Perturb", self._on_perturb)
        self.reset_btn = button([0.093, 0.42, 0.072, 0.04], "Reset", self._on_reset)
        self.tests_btn = button([0.015, 0.36, 0.15, 0.045], "Run tests", self._on_run_tests)

        self.status = f.text(0.015, 0.325, "", fontsize=8.5, color="tab:red")
        f.text(0.015, 0.03, HELP, fontsize=8.5, family="monospace", color="0.25", va="bottom")

    def _build_sliders(self):
        for name in ("param_slider", "scale_slider"):
            if hasattr(self, name):
                getattr(self, name).ax.remove()
        lo, hi, _, step = self.sys.param_range
        f = self.fig
        self.param_slider = Slider(f.add_axes([0.06, 0.615, 0.075, 0.025]),
                                   self.sys.param_label, lo, hi, valinit=self.sys.param, valstep=step)
        self.param_slider.label.set_fontsize(8)
        self.param_slider.on_changed(self._on_param)

        n = len(self.sys.scales)
        self.scale_idx = min(self.scale_idx, n - 1)
        self.scale_slider = Slider(f.add_axes([0.06, 0.565, 0.075, 0.025]), "layer",
                                   0, n - 1, valinit=self.scale_idx, valstep=1)
        self.scale_slider.label.set_fontsize(8)
        self.scale_slider.on_changed(self._on_scale)
        self.scale_slider.valtext.set_text(f"{self.sys.scales[self.scale_idx]}")

    # ------------------------------------------------------------- content
    def _clear_content(self):
        for a in self.content:
            a.remove()
        self.content = []
        self.live_artists = []

    def _ax(self, rect):
        ax = self.fig.add_axes(rect)
        self.content.append(ax)
        return ax

    def _build_content(self):
        self._clear_content()
        self.header = self.fig.text(0.22, 0.955, "", fontsize=14, weight="bold")
        self.content.append(self.header)
        self._update_header()
        if self.mode == EXPLORE:
            self.ax_micro = self._ax([0.21, 0.54, 0.235, 0.36])
            self.ax_layer = self._ax([0.485, 0.54, 0.235, 0.36])
            self.ax_macro = self._ax([0.76, 0.54, 0.22, 0.36])
            self.ax_ladder = self._ax([0.21, 0.07, 0.45, 0.37])
            self.ax_text = self._ax([0.70, 0.07, 0.28, 0.37])
            self._setup_live()
            self._draw_ladder()
        else:
            rects = [[0.22, 0.58, 0.33, 0.30], [0.64, 0.58, 0.33, 0.30],
                     [0.22, 0.135, 0.33, 0.30], [0.64, 0.135, 0.33, 0.30]]
            self.test_axes = [self._ax(r) for r in rects]
            self._draw_tests()
        self.fig.canvas.draw_idle()

    def _update_header(self):
        self.header.set_text(f"{self.mode}:  {self.sys.title()}")

    def _setup_live(self):
        s = self.sys.scales[self.scale_idx]
        for ax in (self.ax_micro, self.ax_layer, self.ax_macro):
            ax.cla()
        self.live_artists = self.sys.setup_live(self.ax_micro, self.ax_layer, self.ax_macro, s)
        self.ax_micro.set_title("LAYER 0 · micro (every part)", fontsize=10, loc="left")
        self.ax_layer.set_title(f"LAYER · scale {s}  (C = zoom out)", fontsize=10, loc="left")
        self.ax_macro.set_title(f"TOP · {self.sys.macro_name}", fontsize=10, loc="left")
        if self.interactive:
            for a in self.live_artists:
                a.set_animated(True)
        self.background = None

    def _refresh_live(self):
        """Redraw only the live artists when possible (fast path)."""
        self.sys.update_live()
        canvas = self.fig.canvas
        if not self.interactive or self.background is None or not canvas.supports_blit:
            canvas.draw_idle()
            return
        canvas.restore_region(self.background)
        for a in self.live_artists:
            self.fig.draw_artist(a)
        canvas.blit(self.fig.bbox)

    def _on_draw(self, _event):
        if not self.interactive:
            return
        canvas = self.fig.canvas
        self.background = canvas.copy_from_bbox(self.fig.bbox)
        for a in self.live_artists:
            self.fig.draw_artist(a)

    def _placeholder(self, ax, msg):
        ax.cla()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="0.4")

    def _draw_ladder(self):
        ax, tx = self.ax_ladder, self.ax_text
        tx.cla()
        tx.axis("off")
        if not self.sys.ladder_ready():
            self._placeholder(ax, "computing layer ladder…  (the simulation keeps running)")
            self._request("ladder")
            return
        ax.cla()
        rows = self.sys.ladder()
        xs = np.arange(len(rows))
        own = np.clip([r["r2_own"] for r in rows], -0.1, 1)
        det = np.clip([r["r2_detail"] for r in rows], -0.1, 1)
        comp = [r["compression"] for r in rows]
        ax.fill_between(xs, own, det, color="tab:orange", alpha=0.3,
                        label="information hidden below (gap)")
        ax.plot(xs, det, "s--", color="tab:orange", label="+ finer detail")
        ax.plot(xs, own, "o-", color="tab:blue", lw=2, label="layer predicts itself (R²)")
        ax.plot(xs, comp, ":", color="0.5", label="compression")
        good = [i for i, r in enumerate(rows)
                if r["scale"] > 1 and r["r2_own"] >= 0.5 and r["gap"] <= 0.05]
        if good:
            ax.plot(xs[good], own[good], "*", color="tab:green", ms=16, zorder=5,
                    label="emergent-layer candidate")
        ax.axvline(self.scale_idx, color="tab:red", alpha=0.35, lw=8)
        ax.set_xticks(xs, [str(r["scale"]) for r in rows])
        ax.set_xlabel(f"zoom level: {self.sys.scale_unit} per block  (left = micro, right = whole system)")
        ax.set_ylim(-0.12, 1.08)
        ax.set_ylabel("R² for the next state")
        ax.set_title("LAYER LADDER — which zoom levels have their own laws?", fontsize=10, loc="left")
        ax.legend(fontsize=7.5, loc="lower left", ncol=2)
        ax.grid(alpha=0.2)

        r = rows[self.scale_idx]
        s = r["scale"]
        mono = dict(family="monospace")
        lines = [
            ("LAYER AT SCALE " + str(s), dict(weight="bold", fontsize=11)),
            (f"{r['cells']} coarse cells  ·  compression {100 * r['compression']:.0f}%", {}),
            ("", {}),
            ("F  " + self.sys.F_desc, {}),
            ("C  " + self.sys.C_desc(s), {}),
            ("G  fitted law: a block + its neighbours → its next value", {}),
            ("", {}),
            (f"predicts itself      R² = {r['r2_own']:.2f}", mono),
            (f"with finer detail    R² = {r['r2_detail']:.2f}", mono),
            (f"hidden information   gap = {r['gap']:.2f}", mono),
            ("", {}),
            (verdict(r), dict(weight="bold", color="tab:green" if self.scale_idx in good else "0.15")),
        ]
        y = 0.98
        for text, style in lines:
            style = dict(style)
            wrapped = textwrap.fill(text, 52, subsequent_indent="   ") if text else ""
            tx.text(0.0, y, wrapped, transform=tx.transAxes, va="top",
                    fontsize=style.pop("fontsize", 9), **style)
            y -= 0.075 * (wrapped.count("\n") + 1) if text else 0.04

    def _draw_tests(self):
        if not self.sys.tests_ready():
            for ax in self.test_axes:
                self._placeholder(ax, "")
            self._placeholder(self.test_axes[0], "running tests…  (takes a few seconds)")
            self._request("tests")
            return
        for ax in self.test_axes:
            ax.cla()
        self.sys.draw_tests(self.test_axes)
        for ax, cap in zip(self.test_axes, self.sys.test_captions):
            ax.text(0.0, -0.17, textwrap.fill(cap, 92), transform=ax.transAxes, fontsize=7.8,
                    va="top", color="0.2")

    # ------------------------------------------------------- background jobs
    def _request(self, kind: str):
        key = (self.sys.name, kind, self.sys.param)
        if key in self.jobs:
            return
        self.jobs[key] = self.executor.submit(_compute, type(self.sys), self.sys.param, kind)
        self._set_status(f"computing {kind}…")
        if self.interactive:
            self.poller.start()
        else:
            self.jobs[key].result()
            self._poll_jobs()

    def _poll_jobs(self):
        for key, fut in list(self.jobs.items()):
            if not fut.done():
                continue
            del self.jobs[key]
            name, kind, param = key
            view = self.views[name]
            try:
                result = fut.result()
            except Exception as exc:  # surface errors in the window instead of dying silently
                self._set_status(f"{kind} failed: {exc}")
                continue
            (view._ladder if kind == "ladder" else view._tests)[param] = result
            if view is self.sys and param == view.param:
                if kind == "ladder" and self.mode == EXPLORE:
                    self._draw_ladder()
                elif kind == "tests" and self.mode == TESTS:
                    self._draw_tests()
                self.fig.canvas.draw_idle()
        if not self.jobs:
            self.poller.stop()
            self._set_status("")

    def _set_status(self, msg: str):
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    def _shutdown(self):
        self.timer.stop()
        self.poller.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)

    # -------------------------------------------------------------- events
    def _on_mode(self, label):
        self.mode = label
        if label == TESTS:
            self._set_playing(False)
        self._build_content()

    def _on_system(self, label):
        self._set_playing(False)
        self.sys = self.views[label]
        self._build_sliders()
        self._build_content()

    def _on_param(self, val):
        val = int(val) if isinstance(self.sys.param_range[2], int) else round(float(val), 5)
        if val == self.sys.param:
            return
        self.sys.set_param(val)
        self._update_header()
        if self.mode == EXPLORE:
            self.sys.update_live()
            if not self.sys.ladder_ready():
                self._placeholder(self.ax_ladder, "release the slider to recompute…")
                self.ax_text.cla()
                self.ax_text.axis("off")
        self.fig.canvas.draw_idle()
        self.debounce.stop()
        self.debounce.start()

    def _after_param_change(self):
        if self.mode == EXPLORE:
            self._draw_ladder()
        else:
            self._draw_tests()
        self.fig.canvas.draw_idle()

    def _on_scale(self, val):
        self.scale_idx = int(val)
        self.scale_slider.valtext.set_text(f"{self.sys.scales[self.scale_idx]}")
        if self.mode == EXPLORE:
            self._setup_live()
            self._draw_ladder()
        self.fig.canvas.draw_idle()

    def _set_playing(self, on: bool):
        self.playing = on
        self.play_btn.label.set_text("Pause" if on else "Play")
        if self.interactive:
            (self.timer.start if on else self.timer.stop)()
        self.fig.canvas.draw_idle()

    def _on_play(self, _event):
        if self.mode != EXPLORE:
            self._switch_mode(EXPLORE)
        self._set_playing(not self.playing)

    def _switch_mode(self, mode):
        self.view_radio.set_active([EXPLORE, TESTS].index(mode))

    def _on_step(self, _event):
        if self.mode == EXPLORE:
            self.sys.advance()
            self._refresh_live()

    def _on_perturb(self, _event):
        self.sys.perturb(self.rng)
        if self.mode == EXPLORE:
            self._refresh_live()

    def _on_reset(self, _event):
        self.sys.seed += 1
        self.sys.reset()
        if self.mode == EXPLORE:
            self._refresh_live()

    def _on_run_tests(self, _event):
        self._set_playing(False)
        if self.mode != TESTS:
            self._switch_mode(TESTS)
        else:
            self._build_content()

    def _tick(self):
        if not self.playing or self.mode != EXPLORE:
            return
        start = time.perf_counter()
        self.sys.advance()
        self._refresh_live()
        # leave the event loop at least as much idle time as a frame took, so clicks get through
        spent_ms = (time.perf_counter() - start) * 1000
        self.timer.interval = int(max(self.FRAME_MS, 2 * spent_ms))

    # ------------------------------------------------------------ headless
    def select(self, system: str | None = None, mode: str | None = None):
        if system is not None:
            self.sys_radio.set_active(list(self.views).index(system))
        if mode is not None:
            self._switch_mode(mode)


def main():
    Explorer()
    plt.show()
