// Live simulations. Each entry renders into an element and returns {stop()}.
(function () {
  const { LiveChart, bars, setup } = window.Charts;
  const C = { blue: "#2f6fd6", orange: "#e8892b", red: "#d24b43", green: "#2e9b5f", teal: "#13a39a",
              purple: "#7b5bd6", grey: "#8b95a7", navy: "#1f3a68", yellow: "#e6b422" };

  let spare = null;
  function randn() {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    v = Math.random();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));

  function slider(id, label, min, max, step, val) {
    return `<div class="ctl"><label for="${id}">${label} <b id="${id}-v">${val}</b></label>
      <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${val}"></div>`;
  }
  function chartbox(id, title, cap, legend) {
    const lg = legend ? `<div class="legend">${legend.map(([c, t, d]) =>
      `<span><i style="background:${c};${d ? "opacity:.7" : ""}"></i>${t}</span>`).join("")}</div>` : "";
    return `<div class="chartbox"><h4>${title}</h4>${cap ? `<p class="cap">${cap}</p>` : ""}${lg}<canvas id="${id}"></canvas></div>`;
  }
  function bind(root, ids, onchange) {
    const vals = {};
    ids.forEach(id => {
      const el = root.querySelector("#" + id), out = root.querySelector("#" + id + "-v");
      const read = () => { vals[id] = parseFloat(el.value); out.textContent = el.value; };
      read();
      el.addEventListener("input", () => { read(); onchange && onchange(id); });
    });
    return vals;
  }
  function loop(fn) {
    let on = true, id;
    const tick = () => { if (!on) return; fn(); id = requestAnimationFrame(tick); };
    id = requestAnimationFrame(tick);
    return () => { on = false; cancelAnimationFrame(id); };
  }
  class EMA {  // running mean that forgets slowly, so estimates follow the sliders
    constructor(n) { this.n = n; this.v = 0; this.w = 0; }
    add(x) { const a = 1 / this.n; this.v = (1 - a) * this.v + a * x; this.w = (1 - a) * this.w + a; }
    get() { return this.w ? this.v / this.w : 0; }
    reset() { this.v = 0; this.w = 0; }
  }
  const share = (full, twin) => (twin.get() > 0 ? 1 - full.get() / twin.get() : 0);
  const pct = v => (isFinite(v) ? v.toFixed(3) : "–");

  // ======================================================== 1. walkers & twin
  function walkers(root) {
    const N = 10;
    root.innerHTML = `
    <div class="sim">
      <div>
        <div class="stage"><canvas id="w-canvas" style="height:380px"></canvas></div>
        <div class="controls">
          ${slider("w-gamma", "Coupling γ (spring strength)", 0, 0.45, 0.01, 0.2)}
          ${slider("w-theta", "Leak θ (pull back to 0)", 0.02, 0.3, 0.01, 0.1)}
          ${slider("w-speed", "Steps per second", 2, 240, 1, 30)}
        </div>
        <div class="btnrow"><button class="btn" id="w-pause">Pause</button>
          <button class="btn" id="w-ff">Fast-forward 5 000 steps</button>
          <button class="btn" id="w-reset">Reset</button></div>
        <div class="explain">
          <h4>What you are seeing</h4>
          <p>Ten random walkers on a line (top), joined by springs to their neighbours. Below, the <b>twin</b>: the same ten walkers with the springs cut. Both get <em>exactly the same random kicks</em>. The dashed line in each panel is the centre of mass (the average position).</p>
          <div class="try"><h4>Try this</h4><ul>
            <li>Watch the two dashed lines: they are always at the same height. The springs change every walker, but never the average.</li>
            <li>Turn the coupling γ up. The coupled chain straightens out (zig-zags get damped) while the twin stays ragged.</li>
            <li>Set γ to 0: every bar in the spectrum drops to zero. No interaction, no interaction share.</li>
          </ul></div>
          <h4>What it tells you</h4>
          <p>The <b>interaction share</b> asks: how much better does the real law forecast a variable than the twin's law? For the centre of mass the answer is exactly 0, because internal forces cancel (Newton's third law, proved in Lean). It is an <em>inherited</em> variable: it would behave the same without interaction. The zig-zag mode is the opposite: almost entirely made by the springs. The popular score Ψ gets this backwards (e23).</p>
          <p class="small muted">Related: <a href="#/exp/e23">e23</a>, <a href="#/exp/e24">e24</a>, <a href="#/exp/e45">e45</a>.</p>
        </div>
      </div>
      <div>
        <div class="readout">
          <div><b id="w-scm">0</b><span>S centre of mass (live)</span></div>
          <div><b id="w-szz">0</b><span>S zig-zag mode (live)</span></div>
          <div><b id="w-sze">0</b><span>S zig-zag (exact formula)</span></div>
        </div>
        ${chartbox("w-c1", "Centre of mass: coupled vs twin", "The two lines lie exactly on top of each other.",
                   [[C.blue, "coupled chain"], [C.orange, "twin (no springs)"]])}
        ${chartbox("w-c2", "Zig-zag mode: coupled vs twin", "Springs damp the zig-zag; the twin's is much larger.",
                   [[C.blue, "coupled chain"], [C.orange, "twin"]])}
        ${chartbox("w-c3", "Interaction-share spectrum, one variable per mode",
                   "Bars: measured live from the simulation. Black ticks: exact formula. CM = centre of mass, 1 = smoothest wave, 9 = zig-zag.")}
      </div>
    </div>`;
    const cv = root.querySelector("#w-canvas");
    const p = bind(root, ["w-gamma", "w-theta", "w-speed"], id => { if (id !== "w-speed") resetStats(); });
    const lam = [...Array(N)].map((_, k) => 2 - 2 * Math.cos(Math.PI * k / N));
    const modes = [...Array(N)].map((_, k) => {
      const v = [...Array(N)].map((_, i) => Math.cos(Math.PI * k * (i + 0.5) / N));
      const nrm = Math.hypot(...v); return v.map(x => x / nrm);
    });
    let x, xt, full, twin, steps, paused = false, acc = 0;
    const c1 = new LiveChart(root.querySelector("#w-c1"), { window: 300, series: [{ color: C.blue, width: 4 }, { color: C.orange, width: 1.6, dash: [5, 4] }] });
    const c2 = new LiveChart(root.querySelector("#w-c2"), { window: 300, series: [{ color: C.blue, width: 2 }, { color: C.orange, width: 1.6 }] });
    function resetStats() { full = modes.map(() => new EMA(3000)); twin = modes.map(() => new EMA(3000)); steps = 0; }
    function reset() { x = Array(N).fill(0); xt = Array(N).fill(0); c1.clear(); c2.clear(); resetStats(); }
    reset();
    const proj = (v, y) => v.reduce((s, vi, i) => s + vi * y[i], 0);
    function step(record) {
      const g = p["w-gamma"], th = p["w-theta"];
      const kick = x.map(() => randn() * 0.5);
      const nx = x.map((xi, i) => {
        const l = i > 0 ? x[i - 1] - xi : 0, r = i < N - 1 ? x[i + 1] - xi : 0;
        return (1 - th) * xi + g * (l + r) + kick[i];
      });
      modes.forEach((v, k) => {
        const y0 = proj(v, x), y1 = proj(v, nx), a = 1 - th - g * lam[k], b = 1 - th;
        full[k].add((y1 - a * y0) ** 2); twin[k].add((y1 - b * y0) ** 2);
      });
      xt = xt.map((xi, i) => (1 - th) * xi + kick[i]);
      x = nx; steps++;
      if (record) {
        const cm = x.reduce((s, v) => s + v, 0) / N, cmt = xt.reduce((s, v) => s + v, 0) / N;
        c1.push([cm, cmt]); c2.push([proj(modes[N - 1], x), proj(modes[N - 1], xt)]);
      }
    }
    function exact() {
      const th = p["w-theta"], g = p["w-gamma"], b = 1 - th;
      return lam.map(l => { const a = 1 - th - g * l; return (a - b) ** 2 / ((1 - a * a) + (a - b) ** 2); });
    }
    function drawStage() {
      const { ctx, w, h } = setup(cv);
      const rows = [["Coupled chain (springs on)", x, "#6fb1ff"], ["Twin (springs cut, same kicks)", xt, "#ffb36b"]];
      const sc = h / 2 / 9;
      rows.forEach(([lab, arr, col], r) => {
        const mid = h / 4 + r * h / 2, X = i => 40 + (w - 80) * i / (N - 1);
        ctx.fillStyle = "#9fb0cc"; ctx.font = "12px -apple-system, sans-serif"; ctx.textAlign = "left";
        ctx.fillText(lab, 12, r * h / 2 + 18);
        ctx.strokeStyle = "rgba(255,255,255,.12)"; ctx.beginPath(); ctx.moveTo(20, mid); ctx.lineTo(w - 20, mid); ctx.stroke();
        if (r === 0) {
          ctx.strokeStyle = "rgba(160,190,240,.7)"; ctx.lineWidth = 1 + 6 * p["w-gamma"];
          ctx.beginPath(); arr.forEach((v, i) => { const yy = mid - v * sc; i ? ctx.lineTo(X(i), yy) : ctx.moveTo(X(i), yy); }); ctx.stroke();
        }
        const cm = arr.reduce((s, v) => s + v, 0) / N;
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.setLineDash([6, 5]);
        ctx.beginPath(); ctx.moveTo(20, mid - cm * sc); ctx.lineTo(w - 20, mid - cm * sc); ctx.stroke(); ctx.setLineDash([]);
        arr.forEach((v, i) => {
          ctx.fillStyle = col; ctx.beginPath(); ctx.arc(X(i), mid - Math.max(-8.5, Math.min(8.5, v)) * sc, 8, 0, 7); ctx.fill();
        });
        ctx.fillStyle = "#fff"; ctx.textAlign = "right"; ctx.fillText("centre of mass " + cm.toFixed(2), w - 14, r * h / 2 + 18);
      });
    }
    function render() {
      drawStage(); c1.draw(); c2.draw();
      const S = full.map((f, k) => share(f, twin[k])), ex = exact();
      root.querySelector("#w-scm").textContent = pct(S[0]);
      root.querySelector("#w-szz").textContent = pct(S[N - 1]);
      root.querySelector("#w-sze").textContent = pct(ex[N - 1]);
      bars(root.querySelector("#w-c3"), S, { exact: ex, min: -0.05, max: 1, labels: ["CM", 1, 2, 3, 4, 5, 6, 7, 8, 9],
        colors: S.map((_, k) => k === 0 ? C.grey : `hsl(${220 - 220 * k / (N - 1)},70%,52%)`) });
    }
    root.querySelector("#w-pause").onclick = e => { paused = !paused; e.target.textContent = paused ? "Play" : "Pause"; };
    root.querySelector("#w-ff").onclick = () => { for (let i = 0; i < 5000; i++) step(i > 4700); render(); };
    root.querySelector("#w-reset").onclick = reset;
    const stop = loop(() => {
      if (!paused) { acc += p["w-speed"] / 60; while (acc >= 1) { step(true); acc--; } }
      render();
    });
    return { stop };
  }

  // ======================================================== 2. flock
  class Flock {
    constructor(n, L) { this.n = n; this.L = L; this.reset(); }
    reset() {
      this.x = []; this.y = []; this.th = [];
      for (let i = 0; i < this.n; i++) { this.x.push(Math.random() * this.L); this.y.push(Math.random() * this.L); this.th.push((Math.random() * 2 - 1) * Math.PI); }
    }
    step(p) {  // p: {ka, kc, sigma, R, v}
      const { n, L } = this, u = new Float64Array(n), xi = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          let dx = this.x[j] - this.x[i], dy = this.y[j] - this.y[i];
          dx -= L * Math.round(dx / L); dy -= L * Math.round(dy / L);
          const d2 = dx * dx + dy * dy;
          if (d2 > p.R * p.R) continue;
          s += p.ka * Math.sin(this.th[j] - this.th[i]);
          if (p.kc) s += p.kc * Math.sin(Math.atan2(dy, dx) - this.th[i]);
        }
        u[i] = s; xi[i] = p.sigma * randn();
      }
      for (let i = 0; i < n; i++) {
        this.th[i] = wrap(this.th[i] + u[i] + xi[i]);
        this.x[i] = (this.x[i] + p.v * Math.cos(this.th[i]) + L) % L;
        this.y[i] = (this.y[i] + p.v * Math.sin(this.th[i]) + L) % L;
      }
      return { u, xi };
    }
    order() {
      let c = 0, s = 0;
      for (const t of this.th) { c += Math.cos(t); s += Math.sin(t); }
      return { r: Math.hypot(c, s) / this.n, psi: Math.atan2(s, c) };
    }
    draw(ctx, w, h, opts = {}) {
      const sc = Math.min(w, h) / this.L;
      const sx = opts.fill ? w / this.L : sc, sy = opts.fill ? h / this.L : sc;
      const ox = (w - this.L * sx) / 2, oy = (h - this.L * sy) / 2;
      const size = opts.size || 6;
      for (let i = 0; i < this.n; i++) {
        const X = ox + this.x[i] * sx, Y = oy + this.y[i] * sy, t = this.th[i];
        ctx.fillStyle = `hsl(${(t * 180 / Math.PI + 360) % 360},75%,62%)`;
        ctx.beginPath();
        ctx.moveTo(X + size * Math.cos(t), Y + size * Math.sin(t));
        ctx.lineTo(X + size * 0.6 * Math.cos(t + 2.5), Y + size * 0.6 * Math.sin(t + 2.5));
        ctx.lineTo(X + size * 0.6 * Math.cos(t - 2.5), Y + size * 0.6 * Math.sin(t - 2.5));
        ctx.fill();
      }
      return { sc, ox, oy };
    }
  }

  function flock(root) {
    root.innerHTML = `
    <div class="sim">
      <div>
        <div class="stage"><canvas id="f-canvas" style="height:440px"></canvas></div>
        <div class="controls">
          ${slider("f-ka", "Alignment (turn toward neighbours' heading)", 0, 0.2, 0.005, 0.08)}
          ${slider("f-kc", "Attraction (turn toward neighbours)", 0, 0.04, 0.002, 0)}
          ${slider("f-sigma", "Noise (random turning)", 0, 0.6, 0.01, 0.15)}
          ${slider("f-speed", "Steps per frame", 1, 6, 1, 2)}
        </div>
        <div class="btnrow"><button class="btn" id="f-int">Interactions: ON</button>
          <button class="btn" id="f-pause">Pause</button><button class="btn" id="f-reset">Scatter agents</button></div>
        <div class="explain">
          <h4>What you are seeing</h4>
          <p>120 agents moving at constant speed in a box that wraps around. Colour = heading. Each step, every agent turns by a <b>social turn</b> (from neighbours within a small radius) plus a <b>random turn</b>. The twin is the same agents with the social turn switched off.</p>
          <div class="try"><h4>Try this</h4><ul>
            <li>With alignment only, watch the red line: the share of the <em>group's total turn</em> made by interaction stays at 0, even while each agent's turn (blue) is strongly social. Alignment is a pairwise, equal-and-opposite exchange, so it cancels in the total.</li>
            <li>Now add a little attraction (about 0.01). The red line jumps up: attraction is not symmetric, so the group's total turn is now shaped by interaction (e45: S = 0.66).</li>
            <li>Raise the noise until the flock melts (polarization falls). Switch interactions off: everything drops to 0.</li>
          </ul></div>
          <h4>What it tells you</h4>
          <p>Whether a collective variable is shaped by interaction depends on <em>which</em> interaction. That is what lets cheap tests rule out whole families of explanations (the stage-0 test in e39–e44), and why the spectrum in e45–e46 finds the total turn as the interaction-blind variable for aligning agents and for real fish.</p>
          <p class="small muted">Related: <a href="#/exp/e9">e9</a>, <a href="#/exp/e45">e45</a>, <a href="#/exp/e46">e46</a>, <a href="#/exp/e44">e44</a>.</p>
        </div>
      </div>
      <div>
        <div class="readout">
          <div><b id="f-pol">0</b><span>polarization (0 = random, 1 = all aligned)</span></div>
          <div><b id="f-s1">0</b><span>S of one agent's turn</span></div>
          <div><b id="f-st">0</b><span>S of the group's total turn</span></div>
        </div>
        ${chartbox("f-c1", "Polarization", "How aligned the whole group is.", [[C.teal, "polarization"]])}
        ${chartbox("f-c2", "Interaction share, measured live",
                   "1 − (error of the real law) / (error of the twin's law). 0 = interaction does not shape it.",
                   [[C.blue, "one agent's turn"], [C.red, "group's total turn"]])}
      </div>
    </div>`;
    const cv = root.querySelector("#f-canvas"), fl = new Flock(120, 30);
    let on = true, paused = false;
    const s1f = new EMA(400), s1t = new EMA(400), stf = new EMA(400), stt = new EMA(400);
    const resetS = () => [s1f, s1t, stf, stt].forEach(e => e.reset());
    const p = bind(root, ["f-ka", "f-kc", "f-sigma", "f-speed"], id => id !== "f-speed" && resetS());
    const c1 = new LiveChart(root.querySelector("#f-c1"), { window: 500, min: 0, max: 1, series: [{ color: C.teal, width: 2 }] });
    const c2 = new LiveChart(root.querySelector("#f-c2"), { window: 500, min: -0.1, max: 1, series: [{ color: C.blue, width: 2 }, { color: C.red, width: 2.5 }] });
    root.querySelector("#f-int").onclick = e => { on = !on; e.target.textContent = "Interactions: " + (on ? "ON" : "OFF (twin)"); e.target.classList.toggle("on", !on); resetS(); };
    root.querySelector("#f-pause").onclick = e => { paused = !paused; e.target.textContent = paused ? "Play" : "Pause"; };
    root.querySelector("#f-reset").onclick = () => { fl.reset(); resetS(); };
    const stop = loop(() => {
      if (!paused) for (let k = 0; k < p["f-speed"]; k++) {
        const { u, xi } = fl.step({ ka: on ? p["f-ka"] : 0, kc: on ? p["f-kc"] : 0, sigma: p["f-sigma"], R: 3, v: 0.12 });
        let su = 0, sx = 0;
        for (let i = 0; i < fl.n; i++) { s1f.add(xi[i] ** 2); s1t.add((u[i] + xi[i]) ** 2); su += u[i]; sx += xi[i]; }
        stf.add(sx * sx); stt.add((su + sx) ** 2);
        c1.push([fl.order().r]); c2.push([share(s1f, s1t), share(stf, stt)]);
      }
      const { ctx, w, h } = setup(cv);
      fl.draw(ctx, w, h, { size: 7 });
      const o = fl.order();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 3;
      const cx = w - 60, cy = 60, R = 40 * o.r;
      ctx.beginPath(); ctx.arc(cx, cy, 40, 0, 7); ctx.strokeStyle = "rgba(255,255,255,.2)"; ctx.lineWidth = 1; ctx.stroke();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(cx + R * Math.cos(o.psi), cy + R * Math.sin(o.psi)); ctx.stroke();
      ctx.fillStyle = "#9fb0cc"; ctx.font = "11px -apple-system, sans-serif"; ctx.textAlign = "center"; ctx.fillText("mean heading", cx, cy + 56);
      root.querySelector("#f-pol").textContent = o.r.toFixed(2);
      root.querySelector("#f-s1").textContent = pct(share(s1f, s1t));
      root.querySelector("#f-st").textContent = pct(share(stf, stt));
      c1.draw(); c2.draw();
    });
    return { stop };
  }

  // ======================================================== 3. Kuramoto
  function kuramoto(root) {
    const N = 60;
    root.innerHTML = `
    <div class="sim">
      <div>
        <div class="stage"><canvas id="k-canvas" style="height:420px"></canvas></div>
        <div class="controls">
          ${slider("k-K", "Coupling K", 0, 4, 0.05, 1.5)}
          ${slider("k-spread", "Spread of natural frequencies", 0, 1.5, 0.05, 0.5)}
          ${slider("k-sigma", "Noise", 0, 1, 0.05, 0.3)}
          ${slider("k-speed", "Steps per frame", 1, 10, 1, 2)}
        </div>
        <div class="btnrow"><button class="btn" id="k-pause">Pause</button><button class="btn" id="k-reset">Randomize phases</button></div>
        <div class="explain">
          <h4>What you are seeing</h4>
          <p>60 oscillators (fireflies, neurons, metronomes), each a dot running around the circle at its own natural speed (colour). Coupling pulls each one toward the others' phases. The white arrow is the <b>order parameter</b>: its length r is how synchronized they are, its angle is the average phase.</p>
          <div class="try"><h4>Try this</h4><ul>
            <li>Slowly raise K from 0. Past a threshold the dots clump and r jumps up: synchronization emerges.</li>
            <li>Widen the frequency spread: you need more coupling to synchronize.</li>
            <li>Watch the red line: the share of the <em>mean phase</em> made by interaction stays at exactly 0 at every setting. The blue line, synchrony r, is strongly interaction-made once K is on.</li>
          </ul></div>
          <h4>What it tells you</h4>
          <p>The same system has an inherited variable (mean phase: the pulls are equal and opposite, so they cancel) and an emergent one (synchrony). The e45 spectrum found the mean phase on its own among 30 candidate observables. In e47, modules of oscillators formed a closed layer whose interaction share is tiny at one step but 0.49 over long horizons.</p>
          <p class="small muted">Related: <a href="#/exp/e30">e30</a>, <a href="#/exp/e34">e34</a>, <a href="#/exp/e45">e45</a>, <a href="#/exp/e47">e47</a>.</p>
        </div>
      </div>
      <div>
        <div class="readout">
          <div><b id="k-r">0</b><span>synchrony r</span></div>
          <div><b id="k-sr">0</b><span>S of synchrony r</span></div>
          <div><b id="k-sm">0</b><span>S of mean phase</span></div>
        </div>
        ${chartbox("k-c1", "Synchrony r over time", "", [[C.teal, "r"]])}
        ${chartbox("k-c2", "Interaction share, measured live", "Real law vs the twin's law (each oscillator at its own speed).",
                   [[C.blue, "synchrony r"], [C.red, "mean phase"]])}
      </div>
    </div>`;
    const cv = root.querySelector("#k-canvas"), dt = 0.05;
    const p = bind(root, ["k-K", "k-spread", "k-sigma", "k-speed"], id => id !== "k-speed" && resetS());
    const z = [...Array(N)].map(() => randn());
    let th = [...Array(N)].map(() => Math.random() * 2 * Math.PI), paused = false;
    const rf = new EMA(600), rt = new EMA(600), mf = new EMA(600), mt = new EMA(600);
    const resetS = () => [rf, rt, mf, mt].forEach(e => e.reset());
    const c1 = new LiveChart(root.querySelector("#k-c1"), { window: 500, min: 0, max: 1, series: [{ color: C.teal, width: 2 }] });
    const c2 = new LiveChart(root.querySelector("#k-c2"), { window: 500, min: -0.1, max: 1, series: [{ color: C.blue, width: 2 }, { color: C.red, width: 2.5 }] });
    const R = a => { let c = 0, s = 0; for (const t of a) { c += Math.cos(t); s += Math.sin(t); } return [Math.hypot(c, s) / a.length, Math.atan2(s, c)]; };
    root.querySelector("#k-pause").onclick = e => { paused = !paused; e.target.textContent = paused ? "Play" : "Pause"; };
    root.querySelector("#k-reset").onclick = () => { th = th.map(() => Math.random() * 2 * Math.PI); resetS(); };
    const stop = loop(() => {
      if (!paused) for (let k = 0; k < p["k-speed"]; k++) {
        const K = p["k-K"], om = z.map(v => 1 + p["k-spread"] * v), [r, psi] = R(th);
        const coup = th.map(t => K * r * Math.sin(psi - t));
        const full = th.map((t, i) => t + dt * (om[i] + coup[i])), twin = th.map((t, i) => t + dt * om[i]);
        const nth = full.map(t => t + p["k-sigma"] * Math.sqrt(dt) * randn());
        const rn = R(nth)[0];
        rf.add((rn - R(full)[0]) ** 2); rt.add((rn - R(twin)[0]) ** 2);
        const dm = nth.reduce((s, t, i) => s + t - th[i], 0) / N;
        const pf = full.reduce((s, t, i) => s + t - th[i], 0) / N, pt = twin.reduce((s, t, i) => s + t - th[i], 0) / N;
        mf.add((dm - pf) ** 2); mt.add((dm - pt) ** 2);
        th = nth.map(t => t % (2 * Math.PI));
        c1.push([rn]); c2.push([share(rf, rt), share(mf, mt)]);
      }
      const { ctx, w, h } = setup(cv), cx = w / 2, cy = h / 2, rad = Math.min(w, h) * 0.38;
      ctx.strokeStyle = "rgba(255,255,255,.18)"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(cx, cy, rad, 0, 7); ctx.stroke();
      th.forEach((t, i) => {
        ctx.fillStyle = `hsl(${200 + 60 * Math.tanh(z[i])},80%,${55 + 15 * Math.tanh(z[i])}%)`;
        ctx.beginPath(); ctx.arc(cx + rad * Math.cos(t), cy + rad * Math.sin(t), 7, 0, 7); ctx.fill();
      });
      const [r, psi] = R(th);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(cx, cy);
      ctx.lineTo(cx + rad * r * Math.cos(psi), cy + rad * r * Math.sin(psi)); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 7); ctx.fill();
      ctx.fillStyle = "#9fb0cc"; ctx.font = "12px -apple-system, sans-serif"; ctx.textAlign = "left";
      ctx.fillText("colour: natural frequency (purple slow → green fast)", 12, h - 12);
      root.querySelector("#k-r").textContent = r.toFixed(2);
      root.querySelector("#k-sr").textContent = pct(share(rf, rt));
      root.querySelector("#k-sm").textContent = pct(share(mf, mt));
      c1.draw(); c2.draw();
    });
    return { stop };
  }

  // ======================================================== 4. hierarchy & layers
  function jacobi(A) {  // symmetric eigen-decomposition: {values, vectors (columns)}
    const n = A.length, a = A.map(r => r.slice()), V = A.map((_, i) => A.map((_, j) => +(i === j)));
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] ** 2;
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const phi = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]), c = Math.cos(phi), s = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
    return { values: a.map((r, i) => r[i]), vectors: V };
  }

  function hierarchy(root) {
    const N = 12, mod = i => Math.floor(i / 3), sup = i => Math.floor(i / 6);
    const MODC = ["#6fb1ff", "#4f7fe0", "#ffb36b", "#e8792b"];
    root.innerHTML = `
    <div class="sim">
      <div>
        <div class="stage"><canvas id="h-canvas" style="height:300px"></canvas></div>
        <div class="controls">
          ${slider("h-c1", "Coupling inside a module (strong)", 0.02, 0.3, 0.005, 0.25)}
          ${slider("h-c2", "Between modules (medium)", 0.001, 0.15, 0.001, 0.02)}
          ${slider("h-c3", "Between super-modules (weak)", 0.0002, 0.05, 0.0002, 0.002)}
          ${slider("h-speed", "Steps per frame", 1, 20, 1, 4)}
        </div>
        <div class="btnrow"><button class="btn" id="h-pause">Pause</button></div>
        ${chartbox("h-spec", "Interaction-share spectrum (exact, log scale)",
                   "Every collective variable, most interaction-blind on the left. Red dashed lines: gaps found by the fixed rule (next bar ≥ 4× the current one). Each block left of a gap is a layer. Variable 1, the global average, has S = 0 exactly, so its bar is empty.")}
        <p id="h-layers" class="small"></p>
        <div class="explain">
          <h4>What you are seeing</h4>
          <p>12 walkers wired as <b>2 super-modules × 2 modules × 3 walkers</b>. Walkers in the same module pull hard on each other, modules in the same super-module pull more weakly, and the two halves barely interact. Lines between walkers show coupling strength. The four level charts show the system at four zoom levels.</p>
          <div class="try"><h4>Try this</h4><ul>
            <li>Compare the four level charts: each level up moves more slowly. That separation of timescales is what makes each level a usable description on its own.</li>
            <li>In the spectrum, find the gaps: 1 global variable, then the super-module difference, then 2 module differences, then 8 fast within-module variables. The layers are 1, 2 and 4 variables, as e47 found from data.</li>
            <li>Drag "between modules" up toward "inside a module". The gap closes and two layers merge into one: there is no longer a module level.</li>
          </ul></div>
          <h4>What it tells you</h4>
          <p>Herbert Simon's idea of <em>near-decomposability</em>: when interactions are organized by strength, each level is the set of variables the stronger interactions cannot touch. e47 showed these layers can be <b>discovered</b> from a flat list of trajectories with no labels, and that they are <b>closed</b> (forecastable from themselves). It also showed that the interaction share of a slow layer is tiny at one step but large at the layer's own timescale.</p>
          <p class="small muted">Related: <a href="#/exp/e47">e47</a>, <a href="#/concepts">concepts: layers</a>.</p>
        </div>
      </div>
      <div>
        ${chartbox("h-l0", "Level 0: walkers within their module", "Each walker minus its module's average: fast jitter.")}
        ${chartbox("h-l1", "Level 1: modules within their super-module", "Module average minus super-module average.", MODC.map((c, i) => [c, "module " + (i + 1)]))}
        ${chartbox("h-l2", "Level 2: the two super-modules", "Super-module average minus the global average: slower.", [[C.blue, "left half"], [C.orange, "right half"]])}
        ${chartbox("h-l3", "Level 3: the global average", "The slowest variable, untouched by any coupling.", [[C.purple, "global"]])}
      </div>
    </div>`;
    const theta = 0.004, cv = root.querySelector("#h-canvas");
    let x = Array(N).fill(0), W, Lap, paused = false;
    const p = bind(root, ["h-c1", "h-c2", "h-c3", "h-speed"], id => id !== "h-speed" && rebuild());
    const L0 = new LiveChart(root.querySelector("#h-l0"), { window: 600, series: [...Array(N)].map((_, i) => ({ color: MODC[mod(i)], width: 1 })) });
    const L1 = new LiveChart(root.querySelector("#h-l1"), { window: 600, series: MODC.map(c => ({ color: c, width: 1.6 })) });
    const L2 = new LiveChart(root.querySelector("#h-l2"), { window: 600, series: [{ color: C.blue, width: 2 }, { color: C.orange, width: 2 }] });
    const L3 = new LiveChart(root.querySelector("#h-l3"), { window: 600, series: [{ color: C.purple, width: 2.4 }] });
    function rebuild() {
      W = [...Array(N)].map((_, i) => [...Array(N)].map((_, j) =>
        i === j ? 0 : mod(i) === mod(j) ? p["h-c1"] : sup(i) === sup(j) ? p["h-c2"] : p["h-c3"]));
      Lap = W.map((r, i) => r.map((v, j) => (i === j ? r.reduce((s, u) => s + u, 0) : -v)));
      const { values } = jacobi(Lap);
      const S = values.map(l => { l = Math.max(l, 0); const a = 1 - theta - l; return l * l / ((1 - a * a) + l * l); }).sort((a, b) => a - b);
      const cuts = [];
      for (let k = 0; k < N - 1; k++) if (S[k + 1] >= 4 * Math.max(S[k], 5e-4)) cuts.push(k + 1);
      const pal = ["#7b5bd6", "#2f6fd6", "#13a39a", "#e8892b", "#d24b43"];
      const colors = S.map((_, i) => pal[Math.min(cuts.filter(c => c <= i).length, pal.length - 1)]);
      bars(root.querySelector("#h-spec"), S, { log: true, floor: 1e-5, colors, cuts, labels: S.map((_, i) => i + 1) });
      root.querySelector("#h-layers").innerHTML = cuts.length
        ? `<b>Layers found:</b> the ${cuts.map(c => c === 1 ? "most blind variable" : `${c} most blind`).join(", then the ")} variables. ` +
          (cuts.join(",") === "1,2,4" ? "Exactly the built hierarchy: global → super-modules → modules." : "Not the built 1-2-4 hierarchy: the coupling scales are too close to separate every level.")
        : "<b>No gaps:</b> no level of description stands out.";
    }
    rebuild();
    root.querySelector("#h-pause").onclick = e => { paused = !paused; e.target.textContent = paused ? "Play" : "Pause"; };
    const mean = idx => idx.reduce((s, i) => s + x[i], 0) / idx.length;
    const stop = loop(() => {
      if (!paused) for (let k = 0; k < p["h-speed"]; k++) {
        x = x.map((xi, i) => (1 - theta) * xi - Lap[i].reduce((s, l, j) => s + l * x[j], 0) + 0.3 * randn());
        const m = [0, 1, 2, 3].map(q => mean([0, 1, 2].map(r => 3 * q + r)));
        const s = [0, 1].map(q => (m[2 * q] + m[2 * q + 1]) / 2), g = (s[0] + s[1]) / 2;
        L0.push(x.map((xi, i) => xi - m[mod(i)])); L1.push(m.map((v, q) => v - s[Math.floor(q / 2)]));
        L2.push(s.map(v => v - g)); L3.push([g]);
      }
      [L0, L1, L2, L3].forEach(c => c.draw());
      const { ctx, w, h } = setup(cv);
      const pos = [...Array(N)].map((_, i) => {
        const sx = sup(i) ? 0.73 : 0.27, mx = (mod(i) % 2 ? 0.11 : -0.11), a = (i % 3) * 2.1 + 0.6;
        return [w * (sx + mx) + 26 * Math.cos(a), h * 0.55 + 26 * Math.sin(a)];
      });
      const cmax = Math.max(p["h-c1"], 1e-9);
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        ctx.strokeStyle = `rgba(160,190,240,${Math.min(1, 0.08 + 0.9 * Math.sqrt(W[i][j] / cmax))})`;
        ctx.lineWidth = 0.5 + 4 * W[i][j] / cmax;
        ctx.beginPath(); ctx.moveTo(...pos[i]); ctx.lineTo(...pos[j]); ctx.stroke();
      }
      pos.forEach(([X, Y], i) => {
        ctx.fillStyle = MODC[mod(i)]; ctx.beginPath(); ctx.arc(X, Y - Math.max(-30, Math.min(30, 4 * x[i])), 9, 0, 7); ctx.fill();
      });
      ctx.fillStyle = "#9fb0cc"; ctx.font = "12px -apple-system, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("super-module A", w * 0.27, 26); ctx.fillText("super-module B", w * 0.73, 26);
      ctx.fillText("(dots move up/down with each walker's value)", w / 2, h - 10);
    });
    return { stop };
  }

  window.SIMS = {
    walkers: { title: "Twin & interaction share", icon: "⛓", blurb: "Coupled walkers and their no-interaction twin: what S measures.", mount: walkers },
    flock: { title: "Flock", icon: "🐟", blurb: "Alignment and attraction: which group variables does interaction shape?", mount: flock },
    kuramoto: { title: "Synchronizing oscillators", icon: "◷", blurb: "Synchrony emerges; the mean phase is inherited.", mount: kuramoto },
    hierarchy: { title: "Layers in a hierarchy", icon: "▤", blurb: "Discover levels of organization from the spectrum's gaps.", mount: hierarchy },
  };
  window.SimUtil = { Flock, randn, loop };
})();
