// Replay of real fish schools (Puy et al. 2024) and their virtual (never-met) twins.
(function () {
  const { LiveChart, scatter, setup } = window.Charts;
  const wrap = a => Math.atan2(Math.sin(a), Math.cos(a));
  const LEAD = 10;  // frames ahead for the school's turn (1 s at 10 Hz)

  function analyse(clip) {
    const { n, frames, xy } = clip, P = (t, i, c) => xy[(t * n + i) * 2 + c] / 10;
    const th = new Float64Array(frames * n), out = [];
    for (let i = 0; i < n; i++) {
      let prev = 0;
      for (let t = 0; t < frames; t++) {
        const a = Math.max(0, t - 2), b = Math.min(frames - 1, t + 2);
        const dx = P(b, i, 0) - P(a, i, 0), dy = P(b, i, 1) - P(a, i, 1);
        if (Math.hypot(dx, dy) / ((b - a) / clip.hz) > 0.5) prev = Math.atan2(dy, dx);
        th[t * n + i] = prev;
      }
    }
    for (let t = 0; t < frames; t++) {
      let cx = 0, cy = 0, hc = 0, hs = 0;
      for (let i = 0; i < n; i++) { cx += P(t, i, 0); cy += P(t, i, 1); hc += Math.cos(th[t * n + i]); hs += Math.sin(th[t * n + i]); }
      cx /= n; cy /= n;
      const pol = Math.hypot(hc, hs) / n, psi = Math.atan2(hs, hc);
      let spread = 0, lead = 0;
      const front = new Uint8Array(n);
      for (let i = 0; i < n; i++) {
        const dx = P(t, i, 0) - cx, dy = P(t, i, 1) - cy;
        spread += Math.hypot(dx, dy);
        const xp = dx * Math.cos(psi) + dy * Math.sin(psi);
        front[i] = xp > 0;
        lead += xp * Math.sin(wrap(th[t * n + i] - psi));
      }
      spread /= n;
      out.push({ cx, cy, pol, psi, spread, L: lead / n / Math.max(spread, 1e-6), front });
    }
    const pts = [];
    for (let t = 0; t + LEAD < frames; t += 2) pts.push([out[t].L, wrap(out[t + LEAD].psi - out[t].psi)]);
    const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length, my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    let sxy = 0, sxx = 0, syy = 0;
    pts.forEach(([a, b]) => { sxy += (a - mx) * (b - my); sxx += (a - mx) ** 2; syy += (b - my) ** 2; });
    return { th, frames: out, pts, corr: sxy / Math.sqrt(sxx * syy), P };
  }

  function mount(root) {
    const keys = Object.keys(window.FISH || {});
    if (!keys.length) {
      root.innerHTML = `<h1>Real fish</h1><div class="panel"><p>The fish recordings are not in this build. Download Puy et al. 2024 (Zenodo 10890112) into <code>data/puy2024/</code> and run <code>uv run python -m lab.build</code> again.</p></div>`;
      return { stop() {} };
    }
    const label = k => { const [n, kind] = k.split("_"); return `${n} fish · ${kind === "real" ? "real school" : "virtual group"}`; };
    root.innerHTML = `
      <h1>Real fish schools</h1>
      <p class="lead">Real recorded fish in a 1 m tank (Puy et al. 2024), replayed at 10 frames per second. Switch to the <b>virtual group</b>: the same fish, each shifted in time so they never actually swam together. Anything a real school has and its virtual twin lacks is made by interaction.</p>
      <div class="tabs" id="fr-clips">${keys.sort((a, b) => parseInt(a) - parseInt(b) || (a < b ? 1 : -1)).map(k =>
        `<button class="btn" data-k="${k}">${label(k)}</button>`).join("")}</div>
      <div class="sim">
        <div>
          <div class="stage"><canvas id="fr-canvas" style="height:560px"></canvas></div>
          <div class="btnrow">
            <button class="btn primary" id="fr-play">Pause</button>
            <button class="btn" data-sp="1">1×</button><button class="btn on" data-sp="2">2×</button><button class="btn" data-sp="5">5×</button><button class="btn" data-sp="15">15×</button>
            <button class="btn on" id="fr-follow">Follow school</button>
            <button class="btn on" id="fr-front">Colour front / back</button>
            <button class="btn on" id="fr-trail">Trails</button>
          </div>
          <input type="range" id="fr-time" min="0" max="1" value="0" step="1" style="width:100%;margin-top:10px">
          <p class="small muted" id="fr-clock"></p>
        </div>
        <div>
          <div class="readout">
            <div><b id="fr-pol">0</b><span>polarization</span></div>
            <div><b id="fr-spread">0</b><span>spread (cm)</span></div>
            <div><b id="fr-L">0</b><span>leadership L</span></div>
          </div>
          <div class="chartbox"><h4>Polarization and spread (last 30 s)</h4><p class="cap">Polarization 1 = all fish heading the same way. Spread = mean distance to the centre (cm, right scale ÷ 50).</p>
            <div class="legend"><span><i style="background:#13a39a"></i>polarization</span><span><i style="background:#7b5bd6"></i>spread / 50 cm</span></div><canvas id="fr-c1"></canvas></div>
          <div class="chartbox"><h4>Do the front fish steer?</h4><p class="cap">Orange: leadership L now (front fish heading left of the school = positive). Blue: how much the school actually turns over the <em>next</em> 1 s. In real schools blue follows orange.</p>
            <div class="legend"><span><i style="background:#e8892b"></i>L now</span><span><i style="background:#2f6fd6"></i>school turn, next 1 s (rad)</span></div><canvas id="fr-c2"></canvas></div>
          <div class="chartbox"><h4>Whole clip: L now vs turn over the next 1 s</h4><p class="cap" id="fr-corr"></p><canvas id="fr-sc" style="height:190px"></canvas></div>
        </div>
      </div>
      <div class="grid g3" style="margin-top:20px">
        <div class="panel"><h3>Interaction lives in the turns</h3><p>In real schools about 7 combinations of the 8 fish's turns are shaped by interaction: one per fish, minus the group's total (8 − 1), which alignment cannot change. Virtual groups have none. In heading variables the count is 2 × (8 − 1) = 14. <a href="#/exp/e46">e46</a>, <a href="#/exp/e48">e48</a></p></div>
        <div class="panel"><h3>The school is the self-contained unit</h3><p>A real school's state can be forecast from itself, while a single fish cannot be forecast without the others. In virtual groups it is the reverse. The school's shape (polarization, spread) is 65–99% made by interaction. <a href="#/exp/e49">e49</a></p></div>
        <div class="panel"><h3>Turns start at the front</h3><p>When the fish ahead of the centre are heading left, the whole school turns left over the next second. This holds in all 6 real 8-fish schools (t = 11–22) and both 39-fish schools, and in no virtual group. <a href="#/exp/e50">e50</a></p></div>
      </div>`;

    const cv = root.querySelector("#fr-canvas"), time = root.querySelector("#fr-time");
    const cache = {};
    let key = keys.find(k => k.startsWith("8_real")) || keys[0], A, clip, t = 0, playing = true, speed = 2, follow = true, colourFront = true, trail = true, acc = 0;
    const c1 = new LiveChart(root.querySelector("#fr-c1"), { window: 300, min: 0, max: 1, series: [{ color: "#13a39a", width: 2 }, { color: "#7b5bd6", width: 2 }] });
    const c2 = new LiveChart(root.querySelector("#fr-c2"), { window: 300, min: -1, max: 1, series: [{ color: "#e8892b", width: 2 }, { color: "#2f6fd6", width: 2 }] });

    function select(k) {
      key = k; clip = window.FISH[k]; A = cache[k] || (cache[k] = analyse(clip));
      t = 0; time.max = clip.frames - 1; c1.clear(); c2.clear();
      root.querySelectorAll("#fr-clips .btn").forEach(b => b.classList.toggle("primary", b.dataset.k === k));
      const other = k.endsWith("real") ? k.replace("real", "virtual") : k.replace("virtual", "real");
      const oc = window.FISH[other] ? (cache[other] || (cache[other] = analyse(window.FISH[other]))).corr : null;
      root.querySelector("#fr-corr").innerHTML = `Correlation <b>${A.corr.toFixed(2)}</b> in this ${k.endsWith("real") ? "real school" : "virtual group"}` +
        (oc !== null ? ` (its ${k.endsWith("real") ? "virtual twin" : "real school"}: <b>${oc.toFixed(2)}</b>).` : ".") +
        ` A positive slope means the front fish's headings predict where the school goes.`;
      scatter(root.querySelector("#fr-sc"), A.pts, { fit: true, xlabel: "leadership L now", ylabel: "turn next 1 s",
        color: k.endsWith("real") ? "rgba(47,111,214,.22)" : "rgba(139,149,167,.3)" });
    }
    root.querySelectorAll("#fr-clips .btn").forEach(b => b.onclick = () => select(b.dataset.k));
    root.querySelector("#fr-play").onclick = e => { playing = !playing; e.target.textContent = playing ? "Pause" : "Play"; };
    root.querySelectorAll("[data-sp]").forEach(b => b.onclick = () => {
      speed = +b.dataset.sp; root.querySelectorAll("[data-sp]").forEach(x => x.classList.toggle("on", x === b));
    });
    const tog = (id, f) => root.querySelector(id).onclick = e => e.target.classList.toggle("on", f());
    tog("#fr-follow", () => (follow = !follow));
    tog("#fr-front", () => (colourFront = !colourFront));
    tog("#fr-trail", () => (trail = !trail));
    time.oninput = () => { t = +time.value; c1.clear(); c2.clear(); };
    select(key);

    function draw() {
      const { ctx, w, h } = setup(cv), F = A.frames[t], n = clip.n;
      const view = follow ? Math.max(30, F.spread * 5) : 100;
      const x0 = follow ? F.cx - view / 2 : 0, y0 = follow ? F.cy - view / 2 : 0;
      const sc = Math.min(w, h) / view, ox = (w - view * sc) / 2, oy = (h - view * sc) / 2;
      const X = v => ox + (v - x0) * sc, Y = v => oy + (v - y0) * sc;
      ctx.strokeStyle = "rgba(255,255,255,.35)"; ctx.lineWidth = 2; ctx.strokeRect(X(0), Y(0), 100 * sc, 100 * sc);
      if (follow) {
        ctx.fillStyle = "#9fb0cc"; ctx.font = "12px -apple-system, sans-serif"; ctx.textAlign = "left";
        ctx.fillText(`view ${view.toFixed(0)} cm wide, centred on the school`, 12, 20);
      }
      if (trail) for (let i = 0; i < n; i++) {
        ctx.strokeStyle = "rgba(160,190,240,.25)"; ctx.lineWidth = 1.2; ctx.beginPath();
        for (let s = Math.max(0, t - 30); s <= t; s++) {
          const px = X(A.P(s, i, 0)), py = Y(A.P(s, i, 1));
          s === Math.max(0, t - 30) ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
      ctx.setLineDash([4, 4]); ctx.strokeStyle = "rgba(255,255,255,.3)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(X(F.cx), Y(F.cy), F.spread * sc, 0, 7); ctx.stroke(); ctx.setLineDash([]);
      const size = Math.max(6, 1.6 * sc);
      for (let i = 0; i < n; i++) {
        const px = X(A.P(t, i, 0)), py = Y(A.P(t, i, 1)), a = A.th[t * n + i];
        ctx.fillStyle = colourFront ? (F.front[i] ? "#ffb36b" : "#6fb1ff") : `hsl(${(a * 180 / Math.PI + 360) % 360},75%,62%)`;
        ctx.beginPath();
        ctx.moveTo(px + size * Math.cos(a), py + size * Math.sin(a));
        ctx.lineTo(px + size * 0.55 * Math.cos(a + 2.5), py + size * 0.55 * Math.sin(a + 2.5));
        ctx.lineTo(px + size * 0.55 * Math.cos(a - 2.5), py + size * 0.55 * Math.sin(a - 2.5));
        ctx.fill();
      }
      const al = Math.max(F.spread * 1.6, 4) * F.pol * sc;
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(F.cx), Y(F.cy));
      ctx.lineTo(X(F.cx) + al * Math.cos(F.psi), Y(F.cy) + al * Math.sin(F.psi)); ctx.stroke();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(X(F.cx), Y(F.cy), 4, 0, 7); ctx.fill();
      if (colourFront) {
        ctx.fillStyle = "#9fb0cc"; ctx.font = "12px -apple-system, sans-serif"; ctx.textAlign = "right";
        ctx.fillText("orange = ahead of the centre, blue = behind; white arrow = school heading", w - 12, h - 12);
      }
      const s = t / clip.hz + clip.start;
      root.querySelector("#fr-clock").textContent = `${label(key)} · recording ${clip.rec} · t = ${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
      root.querySelector("#fr-pol").textContent = F.pol.toFixed(2);
      root.querySelector("#fr-spread").textContent = F.spread.toFixed(1);
      root.querySelector("#fr-L").textContent = F.L.toFixed(2);
    }

    const stop = window.SimUtil.loop(() => {
      if (playing) {
        acc += speed * clip.hz / 60;
        while (acc >= 1) {
          acc--; t = (t + 1) % clip.frames;
          const F = A.frames[t], fut = A.frames[Math.min(t + LEAD, clip.frames - 1)];
          c1.push([F.pol, Math.min(1, F.spread / 50)]);
          c2.push([Math.max(-1, Math.min(1, F.L)), Math.max(-1, Math.min(1, wrap(fut.psi - F.psi)))]);
        }
        time.value = t;
      }
      draw(); c1.draw(); c2.draw();
    });
    return { stop };
  }

  window.FishReplay = { mount };
})();
