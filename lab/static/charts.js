// Small canvas charts: scrolling time series, bar spectra and scatter plots.
(function () {
  const AX = "#8b95a7", GRID = "#edf0f6", TXT = "#5d6779";

  function setup(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 300, h = canvas.clientHeight || 150;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h };
  }

  function fmt(v) {
    const a = Math.abs(v);
    if (a === 0) return "0";
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 0.01) return v.toFixed(2);
    return v.toExponential(0);
  }

  function ticks(lo, hi, n) {
    const span = hi - lo || 1, raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(12));
    return out;
  }

  function frame(ctx, w, h, lo, hi, pad, ylab) {
    ctx.font = "11px -apple-system, sans-serif";
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    const y = v => pad.t + (h - pad.t - pad.b) * (1 - (v - lo) / (hi - lo || 1));
    for (const t of ticks(lo, hi, 4)) {
      ctx.strokeStyle = t === 0 ? "#c9d0dc" : GRID; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, y(t)); ctx.lineTo(w - pad.r, y(t)); ctx.stroke();
      ctx.fillStyle = TXT; ctx.fillText(fmt(t), pad.l - 5, y(t));
    }
    if (ylab) {
      ctx.save(); ctx.translate(10, pad.t + (h - pad.t - pad.b) / 2); ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center"; ctx.fillText(ylab, 0, 0); ctx.restore();
    }
    return y;
  }

  class LiveChart {
    constructor(canvas, opts) {
      this.c = canvas; this.o = Object.assign({ window: 400, series: [] }, opts);
      this.data = this.o.series.map(() => []);
    }
    push(vals) {
      vals.forEach((v, i) => {
        const d = this.data[i]; d.push(v);
        if (d.length > this.o.window) d.shift();
      });
    }
    clear() { this.data = this.o.series.map(() => []); }
    draw() {
      const { ctx, w, h } = setup(this.c);
      const pad = { l: this.o.ylabel ? 46 : 36, r: 8, t: 8, b: 8 };
      let lo = this.o.min, hi = this.o.max;
      if (lo === undefined || hi === undefined) {
        let a = Infinity, b = -Infinity;
        for (const d of this.data) for (const v of d) if (isFinite(v)) { a = Math.min(a, v); b = Math.max(b, v); }
        if (!isFinite(a)) { a = 0; b = 1; }
        const m = (b - a) * 0.1 || Math.abs(a) * 0.1 || 0.1;
        if (lo === undefined) lo = a - m;
        if (hi === undefined) hi = b + m;
      }
      const y = frame(ctx, w, h, lo, hi, pad, this.o.ylabel);
      const n = this.o.window, x = i => pad.l + (w - pad.l - pad.r) * i / (n - 1);
      this.o.series.forEach((s, k) => {
        const d = this.data[k]; if (!d.length) return;
        const off = n - d.length;
        ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.6;
        ctx.setLineDash(s.dash || []);
        ctx.beginPath();
        d.forEach((v, i) => {
          const yy = y(Math.max(lo, Math.min(hi, v)));
          i ? ctx.lineTo(x(off + i), yy) : ctx.moveTo(x(off + i), yy);
        });
        ctx.stroke(); ctx.setLineDash([]);
      });
    }
  }

  // values: array; opts: {colors, exact, log, min, max, cuts, labels, ylabel}
  function bars(canvas, values, opts = {}) {
    const { ctx, w, h } = setup(canvas);
    const pad = { l: 46, r: 8, t: 8, b: opts.labels ? 18 : 8 };
    const tr = opts.log ? v => Math.log10(Math.max(v, opts.floor || 1e-5)) : v => v;
    const all = values.concat(opts.exact || []).filter(v => isFinite(v)).map(tr);
    let lo = opts.min !== undefined ? tr(opts.min) : Math.min(0, ...all);
    let hi = opts.max !== undefined ? tr(opts.max) : Math.max(...all) * 1.08 || 1;
    if (opts.log) { lo = Math.floor(Math.min(lo, ...all)); hi = Math.ceil(Math.max(hi, ...all)); }
    const y = frame(ctx, w, h, lo, hi, pad, opts.ylabel);
    if (opts.log) {  // relabel log ticks
      ctx.clearRect(0, 0, pad.l - 2, h);
      ctx.fillStyle = TXT; ctx.textAlign = "right";
      for (let t = Math.ceil(lo); t <= hi; t++) ctx.fillText("1e" + t, pad.l - 5, y(t));
    }
    const n = values.length, bw = (w - pad.l - pad.r) / n;
    const base = y(opts.log ? lo : Math.max(lo, 0));
    values.forEach((v, i) => {
      const yy = y(Math.max(lo, Math.min(hi, tr(v))));
      ctx.fillStyle = (opts.colors && opts.colors[i]) || "#2f6fd6";
      ctx.fillRect(pad.l + i * bw + bw * 0.12, Math.min(yy, base), bw * 0.76, Math.abs(base - yy));
      if (opts.exact) {
        const ye = y(Math.max(lo, Math.min(hi, tr(opts.exact[i]))));
        ctx.strokeStyle = "#1b2333"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(pad.l + i * bw + 2, ye); ctx.lineTo(pad.l + (i + 1) * bw - 2, ye); ctx.stroke();
      }
      if (opts.labels) {
        ctx.fillStyle = TXT; ctx.textAlign = "center"; ctx.textBaseline = "top";
        ctx.fillText(opts.labels[i], pad.l + (i + 0.5) * bw, h - pad.b + 4);
      }
    });
    (opts.cuts || []).forEach(c => {
      const xx = pad.l + c * bw;
      ctx.strokeStyle = "#d24b43"; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(xx, pad.t); ctx.lineTo(xx, h - pad.b); ctx.stroke(); ctx.setLineDash([]);
    });
  }

  // pts: [[x,y],...]; opts: {color, fit:true, xlabel, ylabel, lim}
  function scatter(canvas, pts, opts = {}) {
    const { ctx, w, h } = setup(canvas);
    const pad = { l: 46, r: 10, t: 10, b: 30 };
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    const q = (a, f) => { const s = a.slice().sort((u, v) => u - v); return s[Math.floor(f * (s.length - 1))]; };
    const xl = opts.xlim || [q(xs, 0.01), q(xs, 0.99)], yl = opts.ylim || [q(ys, 0.01), q(ys, 0.99)];
    const y = frame(ctx, w, h, yl[0], yl[1], pad, opts.ylabel);
    const x = v => pad.l + (w - pad.l - pad.r) * (v - xl[0]) / (xl[1] - xl[0] || 1);
    ctx.fillStyle = opts.color || "rgba(47,111,214,.25)";
    for (const [a, b] of pts) {
      if (a < xl[0] || a > xl[1] || b < yl[0] || b > yl[1]) continue;
      ctx.fillRect(x(a) - 1.5, y(b) - 1.5, 3, 3);
    }
    ctx.fillStyle = TXT; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const t of ticks(xl[0], xl[1], 4)) ctx.fillText(fmt(t), x(t), h - pad.b + 4);
    if (opts.xlabel) ctx.fillText(opts.xlabel, pad.l + (w - pad.l - pad.r) / 2, h - 12);
    if (opts.fit && pts.length > 2) {
      const mx = xs.reduce((s, v) => s + v, 0) / xs.length, my = ys.reduce((s, v) => s + v, 0) / ys.length;
      let sxy = 0, sxx = 0;
      pts.forEach(([a, b]) => { sxy += (a - mx) * (b - my); sxx += (a - mx) ** 2; });
      const k = sxy / sxx;
      ctx.strokeStyle = opts.fitColor || "#d24b43"; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(x(xl[0]), y(my + k * (xl[0] - mx))); ctx.lineTo(x(xl[1]), y(my + k * (xl[1] - mx))); ctx.stroke();
    }
  }

  window.Charts = { LiveChart, bars, scatter, setup };
})();
