// Emergence Lab: routing and pages.
(function () {
  const LAB = window.LAB, app = document.getElementById("app");
  const EXP = LAB.experiments, byId = Object.fromEntries(EXP.map(e => [e.id, e]));
  const THEME = Object.fromEntries(LAB.themes.map(t => [t.id, t]));
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const RELATED = {
    walkers: ["e23", "e24", "e25", "e26", "e27", "e28", "e29"], flock: ["e3", "e9", "e10", "e11", "e12", "e13", "e39", "e40", "e41", "e42", "e43", "e44", "e45"],
    kuramoto: ["e30", "e34"], hierarchy: ["e47"], fish: ["e31", "e32", "e33", "e46", "e48", "e49", "e50"],
  };
  const simFor = id => Object.keys(RELATED).find(k => RELATED[k].includes(id));
  let current = null, lbList = [], lbIdx = 0;

  const chip = e => `<span class="chip tone-${e.tone}">${esc(e.verdict)}</span>`;
  const frac = v => { const m = /(\d+)\s*\/\s*(\d+)/.exec(v); return m ? [+m[1], +m[2]] : null; };

  // ------------------------------------------------------------ lightbox
  const lb = document.getElementById("lightbox");
  function openLb(list, i) { lbList = list; lbIdx = i; showLb(); lb.classList.remove("hidden"); }
  function showLb() {
    const f = lbList[lbIdx];
    lb.querySelector("img").src = f.src;
    lb.querySelector(".lb-text").innerHTML = `<p class="small muted">${esc(f.exp || "")} · ${lbIdx + 1} / ${lbList.length}</p><h3>${esc(f.title)}</h3>${explainer(f)}`;
  }
  const closeLb = () => lb.classList.add("hidden");
  lb.querySelector(".lb-close").onclick = closeLb;
  lb.querySelector(".lb-prev").onclick = () => { lbIdx = (lbIdx - 1 + lbList.length) % lbList.length; showLb(); };
  lb.querySelector(".lb-next").onclick = () => { lbIdx = (lbIdx + 1) % lbList.length; showLb(); };
  lb.onclick = e => { if (e.target === lb) closeLb(); };
  document.addEventListener("keydown", e => {
    if (lb.classList.contains("hidden")) return;
    if (e.key === "Escape") closeLb();
    if (e.key === "ArrowRight") lb.querySelector(".lb-next").click();
    if (e.key === "ArrowLeft") lb.querySelector(".lb-prev").click();
  });
  const explainer = f => `<div class="ex what"><b>What it shows</b>${esc(f.what)}</div>
    <div class="ex read"><b>How to read it</b>${esc(f.read)}</div>
    <div class="ex tells"><b>What it tells us</b>${esc(f.tells)}</div>`;

  // ------------------------------------------------------------ home
  function home() {
    const nFig = EXP.reduce((s, e) => s + e.figures.length, 0);
    const pre = EXP.map(e => frac(e.verdict)).filter(Boolean);
    const passed = pre.reduce((s, f) => s + f[0], 0), total = pre.reduce((s, f) => s + f[1], 0);
    app.innerHTML = `
    <section class="hero">
      <div>
        <h1>Emergence Lab</h1>
        <p class="lead">When does a group become <em>more</em> than its parts? This lab tests one precise answer: a zoomed-out description is a real level of organization when it has <b>its own law</b>, and when that law is <b>made by the interactions</b> between the parts, not just by averaging them.</p>
        <div class="equation">C(F(x)) ≈ G(C(x)) &nbsp;and&nbsp; S = 1 − error(real law) / error(no-interaction twin) &gt; 0</div>
        <p class="muted">Every experiment states its predictions in advance and reports PASS or FAIL honestly. Below you can play with the ideas live, replay real fish schools, and browse every chart with an explanation of what it is telling you.</p>
        <div class="btnrow"><a class="btn primary" href="#/concepts">Start with the concepts →</a><a class="btn" href="#/sims">Play with simulations</a><a class="btn" href="#/fish">Watch real fish</a></div>
      </div>
      <canvas id="hero-canvas"></canvas>
    </section>
    <div class="grid g4" style="margin-bottom:26px">
      <div class="panel stat"><b>${EXP.length}</b><span>experiments (e0–e50)</span></div>
      <div class="panel stat"><b>${nFig}</b><span>charts, each explained</span></div>
      <div class="panel stat"><b>${passed}/${total}</b><span>pre-registered predictions passed</span></div>
      <div class="panel stat"><b>5</b><span>live simulations and replays</span></div>
    </div>
    <h2>What we have learned, in five steps</h2>
    <div class="grid g3">
      ${[
        ["1", "A popular score fails", "The information score Ψ rated agents that ignore each other as <em>more</em> emergent than a flock. It cannot see interaction.", "e8"],
        ["2", "A measure that works", "The interaction share S compares the real law with a twin that has interactions switched off. Flock: S ≈ 1. Twin: S = 0. Centre of mass of springs: exactly 0.", "e23"],
        ["3", "Choose the right variables", "Global order alone is not self-contained; global + local order is, and it has an equation you can run without agents. At a ~100-agent scale the law is the same at every size.", "e16"],
        ["4", "Let the data choose", "One calculation ranks every combination of parts by how much interaction shapes it. Gaps in that spectrum reveal the levels of a hierarchy, which are closed.", "e47"],
        ["5", "Real fish", "In real schools the school, not the fish, is the self-contained unit; its shape is made by interaction, and its turns start at the front.", "e49"],
        ["?", "Still open", "The flock's fluctuations near the transition, a size test in 39-fish schools, and a closure test that is both flexible and regularized.", null],
      ].map(([n, t, d, e]) => `<a class="card" href="${e ? "#/exp/" + e : "#/results/open"}"><span class="chip tone-explore">${n}</span><h3>${t}</h3><p>${d}</p></a>`).join("")}
    </div>
    <h2>The journey: every experiment</h2>
    <p class="muted">Green = predictions passed, orange = partly, red = failed, grey = exploratory. Click any experiment.</p>
    <div class="journey">${LAB.themes.map(t => `
      <div class="jrow"><div><h4>${esc(t.title)}</h4><p>${esc(t.question)}</p></div>
      <div class="jchips">${EXP.filter(e => e.theme === t.id).map(e =>
        `<a class="bg-${e.tone}" href="#/exp/${e.id}" title="${esc(e.title)}: ${esc(e.verdict)}">${e.id}</a>`).join("")}</div></div>`).join("")}
    </div>`;
    const cv = document.getElementById("hero-canvas"), fl = new window.SimUtil.Flock(90, 30);
    return window.SimUtil.loop(() => {
      fl.step({ ka: 0.09, kc: 0.004, sigma: 0.12, R: 3.5, v: 0.12 });
      const { ctx, w, h } = window.Charts.setup(cv);
      fl.draw(ctx, w, h, { size: 6, fill: true });
    });
  }

  // ------------------------------------------------------------ concepts
  const svgWrap = (body, vb = "0 0 380 220") => `<svg viewBox="${vb}" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system, sans-serif">${body}</svg>`;
  const dots = (cx, cy, n, r, col, seed = 1) => [...Array(n)].map((_, i) => {
    const a = i * 2.39996 + seed, d = r * Math.sqrt((i + 0.5) / n);
    return `<circle cx="${(cx + d * Math.cos(a)).toFixed(1)}" cy="${(cy + d * Math.sin(a)).toFixed(1)}" r="4" fill="${col}"/>`;
  }).join("");
  const arrow = (x1, y1, x2, y2, col = "#1f3a68", lab = "") => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="2" marker-end="url(#ah)"/>` +
    (lab ? `<text x="${(x1 + x2) / 2}" y="${(y1 + y2) / 2 - 9}" text-anchor="middle" font-size="12" fill="${col}">${lab}</text>` : "");
  const defs = `<defs><marker id="ah" markerUnits="userSpaceOnUse" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#1f3a68"/></marker></defs>`;

  const CONCEPTS = [
    ["Parts and wholes", `A system is made of <b>parts</b> (agents, fish, oscillators, cells): the <em>micro</em> level. A <b>coarse-graining</b> C summarizes the parts in a few numbers, the <em>macro</em> level, like a flock's average heading or a school's spread. The question is never just "is there a pattern?" but "is this summary a level with its own rules?"`,
      svgWrap(defs + dots(80, 110, 26, 60, "#2f6fd6") + arrow(160, 110, 230, 110, "#1f3a68", "C (summarize)") +
        `<rect x="245" y="80" width="110" height="60" rx="10" fill="#eef3fb" stroke="#2f6fd6"/><text x="300" y="107" text-anchor="middle" font-size="13" fill="#1f3a68">macro state</text><text x="300" y="125" text-anchor="middle" font-size="11" fill="#5d6779">e.g. polarization</text><text x="80" y="200" text-anchor="middle" font-size="12" fill="#5d6779">micro: every part</text>`), null],
    ["Closure: a law of its own", `A macro description is <b>closed</b> when its future can be predicted from the macro state alone: there is a law G with C(F(x)) = G(C(x)). Going "micro step then summarize" gives the same answer as "summarize then macro step". Then you can forget the parts. Cellular automaton rule 105, seen in pairs of cells, is exactly rule 150 (e1). Closure is measured by the <b>closure gap</b> G: 0 = closed.`,
      svgWrap(defs + `<text x="60" y="40" text-anchor="middle" font-size="14" fill="#1b2333">x</text><text x="320" y="40" text-anchor="middle" font-size="14" fill="#1b2333">F(x)</text>
        <text x="60" y="190" text-anchor="middle" font-size="14" fill="#1b2333">C(x)</text><text x="320" y="190" text-anchor="middle" font-size="14" fill="#1b2333">C(F(x))</text>` +
        arrow(85, 36, 290, 36, "#1f3a68", "micro dynamics F") + arrow(60, 52, 60, 170, "#1f3a68") + arrow(320, 52, 320, 170, "#1f3a68") +
        arrow(95, 186, 280, 186, "#13a39a", "macro law G") + `<text x="45" y="115" text-anchor="end" font-size="12" fill="#1f3a68">C</text><text x="335" y="115" font-size="12" fill="#1f3a68">C</text>
        <text x="190" y="118" text-anchor="middle" font-size="12" fill="#5d6779">both routes agree</text><text x="190" y="134" text-anchor="middle" font-size="12" fill="#5d6779">→ closed</text>`), "e1"],
    ["The twin", `To ask what interaction does, compare the system with its <b>no-interaction twin</b>: the same parts, each with its own habits, but with every interaction switched off. In simulations the twin is a real intervention. On real data it is estimated: each part's own terms are kept and the social terms removed. For real animals the control is a <b>virtual group</b>: the same animals, time-shifted so they never actually met.`,
      svgWrap(`<text x="95" y="24" text-anchor="middle" font-size="13" fill="#1f3a68" font-weight="600">real system</text><text x="285" y="24" text-anchor="middle" font-size="13" fill="#e8892b" font-weight="600">twin</text>` +
        [[60, 70], [120, 60], [90, 120], [140, 140], [50, 160], [110, 190]].map(([x, y], i, a) => (i ? `<line x1="${a[i - 1][0]}" y1="${a[i - 1][1]}" x2="${x}" y2="${y}" stroke="#9db7e8" stroke-width="2"/>` : "")).join("") +
        `<line x1="60" y1="70" x2="90" y2="120" stroke="#9db7e8" stroke-width="2"/><line x1="140" y1="140" x2="110" y2="190" stroke="#9db7e8" stroke-width="2"/>` +
        [[60, 70], [120, 60], [90, 120], [140, 140], [50, 160], [110, 190]].map(([x, y]) => `<circle cx="${x}" cy="${y}" r="8" fill="#2f6fd6"/><circle cx="${x + 190}" cy="${y}" r="8" fill="#e8892b"/>`).join("") +
        `<line x1="190" y1="40" x2="190" y2="205" stroke="#e2e6ef" stroke-width="2" stroke-dasharray="4 4"/><text x="285" y="215" text-anchor="middle" font-size="11" fill="#5d6779">same parts, no links</text>`), "e24"],
    ["Interaction share S", `<b>S = 1 − error(real law) / error(twin's law)</b>, measured on the real data. S = 0: the twin's law forecasts the macro variable just as well, so interaction does not shape it (the variable is just an average of independent parts). S near 1: interaction makes almost all of the macro law. Unlike Ψ, S is 0 for non-interacting parts by construction.`,
      svgWrap(`<defs><linearGradient id="gS" x1="0" x2="1"><stop offset="0" stop-color="#8b95a7"/><stop offset="1" stop-color="#d24b43"/></linearGradient></defs>
        <rect x="30" y="90" width="320" height="26" rx="13" fill="url(#gS)"/><text x="30" y="80" font-size="13" fill="#1b2333">0</text><text x="350" y="80" text-anchor="end" font-size="13" fill="#1b2333">1</text>
        <text x="30" y="140" font-size="12" fill="#5d6779">just averaging</text><text x="30" y="156" font-size="11" fill="#5d6779">(twin explains it)</text>
        <text x="350" y="140" text-anchor="end" font-size="12" fill="#5d6779">made by interaction</text>
        <path d="M70 90 v-14" stroke="#1b2333"/><text x="70" y="60" text-anchor="middle" font-size="11" fill="#1b2333">centre of mass: 0</text>
        <path d="M335 90 v-14" stroke="#1b2333"/><text x="300" y="45" text-anchor="middle" font-size="11" fill="#1b2333">ordered flock: ≈ 1</text>`), "walkers"],
    ["Emergent vs inherited", `Some macro variables are closed only because interaction never touches them: the centre of mass of particles joined by springs moves exactly as if there were no springs (internal forces cancel; proved in Lean). That layer is <b>inherited</b>. An <b>emergent</b> layer is closed <em>and</em> shaped by interaction. A careful catch from e47: a slow emergent layer can look untouched at one step and be strongly interaction-made at its own timescale.`,
      svgWrap(`<rect x="40" y="20" width="300" height="170" fill="#fafbfe" stroke="#e2e6ef"/><line x1="190" y1="20" x2="190" y2="190" stroke="#e2e6ef"/><line x1="40" y1="105" x2="340" y2="105" stroke="#e2e6ef"/>
        <rect x="41" y="21" width="148" height="83" fill="#e1f4e8"/><text x="115" y="58" text-anchor="middle" font-size="13" fill="#1d7a45" font-weight="600">emergent layer</text><text x="115" y="76" text-anchor="middle" font-size="11" fill="#1d7a45">closed + interaction-made</text>
        <rect x="41" y="106" width="148" height="83" fill="#e8ecf5"/><text x="115" y="143" text-anchor="middle" font-size="13" fill="#4a5671" font-weight="600">inherited layer</text><text x="115" y="161" text-anchor="middle" font-size="11" fill="#4a5671">closed, untouched</text>
        <text x="265" y="65" text-anchor="middle" font-size="12" fill="#a8322b">not closed:</text><text x="265" y="81" text-anchor="middle" font-size="12" fill="#a8322b">needs the parts</text>
        <text x="190" y="210" text-anchor="middle" font-size="11" fill="#5d6779">closure gap →</text><text x="22" y="105" text-anchor="middle" font-size="11" fill="#5d6779" transform="rotate(-90 22 105)">interaction share →</text>`), "e47"],
    ["The spectrum: let the data choose", `Instead of picking a variable, take every combination of the parts and ask how much interaction shapes it. One generalized eigenvalue problem ranks them all, from most interaction-made to completely blind (e45). It found the centre of mass, an influence-weighted average nobody specified, the mean phase of oscillators, and the zig-zag modes of a chain, from data alone.`,
      svgWrap([0.92, 0.85, 0.8, 0.74, 0.66, 0.6, 0.31, 0.27, 0.24, 0.05, 0.03, 0.0].map((v, i) =>
        `<rect x="${30 + i * 27}" y="${180 - 150 * v}" width="20" height="${150 * v + 1}" rx="3" fill="${i < 6 ? "#d24b43" : i < 9 ? "#e8892b" : "#8b95a7"}"/>`).join("") +
        `<line x1="25" y1="181" x2="355" y2="181" stroke="#8b95a7"/><text x="30" y="202" font-size="11" fill="#5d6779">most interaction-made</text><text x="352" y="202" text-anchor="end" font-size="11" fill="#5d6779">interaction-blind</text>
        <line x1="${30 + 6 * 27 - 4}" y1="20" x2="${30 + 6 * 27 - 4}" y2="182" stroke="#d24b43" stroke-dasharray="4 3" stroke-width="2"/><line x1="${30 + 9 * 27 - 4}" y1="20" x2="${30 + 9 * 27 - 4}" y2="182" stroke="#d24b43" stroke-dasharray="4 3" stroke-width="2"/>
        <text x="${30 + 9 * 27}" y="30" font-size="11" fill="#d24b43">gaps = layers</text>`), "hierarchy"],
    ["Pre-registration and controls", `Every prediction is written down (and hashed) <em>before</em> the run, so a failure cannot be quietly explained away. Each method is first tested where the answer is known (synthetic data), and every real-data claim must also pass a <b>negative control</b>: the same pipeline on parts that never interacted must find nothing. Several headline failures in this project (e8, e31, e39–e43) were caught exactly this way.`,
      svgWrap(`<rect x="30" y="30" width="95" height="60" rx="9" fill="#eef3fb" stroke="#2f6fd6"/><text x="77" y="57" text-anchor="middle" font-size="12" fill="#1f3a68">predictions</text><text x="77" y="73" text-anchor="middle" font-size="11" fill="#5d6779">hashed first</text>` + defs +
        arrow(127, 60, 160, 60) + `<rect x="163" y="30" width="80" height="60" rx="9" fill="#eef3fb" stroke="#2f6fd6"/><text x="203" y="64" text-anchor="middle" font-size="12" fill="#1f3a68">run</text>` + arrow(245, 60, 278, 60) +
        `<rect x="281" y="20" width="75" height="34" rx="8" fill="#e1f4e8"/><text x="318" y="42" text-anchor="middle" font-size="12" fill="#1d7a45">PASS</text><rect x="281" y="66" width="75" height="34" rx="8" fill="#fbe3e1"/><text x="318" y="88" text-anchor="middle" font-size="12" fill="#a8322b">FAIL</text>
        <rect x="30" y="130" width="150" height="60" rx="9" fill="#fdf0d9" stroke="#e8892b"/><text x="105" y="157" text-anchor="middle" font-size="12" fill="#a2620c">known-answer test</text><text x="105" y="173" text-anchor="middle" font-size="11" fill="#a2620c">synthetic data</text>
        <rect x="206" y="130" width="150" height="60" rx="9" fill="#e8ecf5" stroke="#8b95a7"/><text x="281" y="157" text-anchor="middle" font-size="12" fill="#4a5671">negative control</text><text x="281" y="173" text-anchor="middle" font-size="11" fill="#4a5671">must find nothing</text>`), "e44"],
  ];

  function concepts() {
    app.innerHTML = `<h1>Concepts</h1>
      <p class="lead">Seven ideas are enough to read every chart in this lab. Each one links to a live simulation or the experiment where it matters most.</p>
      ${CONCEPTS.map(([t, body, svg, link], i) => `
        <div class="panel concept"><div><h2 style="margin-top:0"><span class="num">${i + 1}</span>${t}</h2><p>${body}</p>
        ${link ? `<p><a class="btn" href="${byId[link] ? "#/exp/" + link : "#/sims/" + link}">${byId[link] ? "See " + link + ": " + esc(byId[link].title) : "Try the live simulation"} →</a></p>` : ""}</div>${svg}</div>`).join("")}
      ${LAB.video ? `<h2>The explainer video</h2><div class="panel"><video controls preload="metadata" src="interaction_share_explainer.mp4"></video></div>` : ""}
      <h2>Figures from the white paper</h2>
      <div class="grid g2">${["fig1_twin", "fig2_walkers", "fig3_transitions", "fig4_fish"].map(f =>
        `<div class="panel"><img src="figures/${f}.png" style="width:100%" alt="${f}" onerror="this.parentNode.remove()"></div>`).join("")}</div>`;
  }

  // ------------------------------------------------------------ simulations
  function sims(sub) {
    const id = window.SIMS[sub] ? sub : "walkers";
    app.innerHTML = `<h1>Simulations</h1>
      <p class="lead">Live, in your browser. Every number on the right is measured from the running simulation, the same way the experiments measure it.</p>
      <div class="simnav">${Object.entries(window.SIMS).map(([k, s]) =>
        `<a href="#/sims/${k}" class="${k === id ? "active" : ""}">${s.icon} ${s.title}</a>`).join("")}
        <a href="#/fish">🐠 Real fish replay</a></div>
      <p class="muted">${esc(window.SIMS[id].blurb)}</p><div id="simroot"></div>`;
    return window.SIMS[id].mount(document.getElementById("simroot")).stop;
  }

  // ------------------------------------------------------------ experiments
  function sidebar(active) {
    return `<nav class="sidebar">${LAB.themes.map(t => `<h5>${esc(t.title)}</h5>` + EXP.filter(e => e.theme === t.id).map(e =>
      `<a href="#/exp/${e.id}" class="${e.id === active ? "active" : ""}"><span class="dot ${e.tone}"></span><span class="eid">${e.id}</span>${esc(e.title)}</a>`).join("")).join("")}</nav>`;
  }

  function experimentsIndex() {
    app.innerHTML = `<h1>Experiments</h1>
      <p class="lead">All ${EXP.length} experiments, grouped by the question they ask. Each page has a plain-language summary, every chart with an explainer, and the full write-up from FINDINGS.md.</p>
      ${LAB.themes.map(t => `<h2>${esc(t.title)}</h2><p class="muted" style="margin-top:-6px">${esc(t.question)}</p>
        <div class="grid g3">${EXP.filter(e => e.theme === t.id).map(e => `
          <a class="card" href="#/exp/${e.id}"><div class="exp-head"><span class="chip tone-explore" style="font-family:monospace">${e.id}</span>${chip(e)}
          ${e.figures.length ? `<span class="small muted">${e.figures.length} chart${e.figures.length > 1 ? "s" : ""}</span>` : ""}</div>
          <h3>${esc(e.title)}</h3><p>${esc(e.summary)}</p></a>`).join("")}</div>`).join("")}`;
  }

  function experiment(id) {
    const e = byId[id];
    if (!e) return experimentsIndex();
    const i = EXP.indexOf(e), prev = EXP[i - 1], next = EXP[i + 1], sim = simFor(id);
    const figs = e.figures.map(f => ({ ...f, exp: `${e.id}: ${e.title}` }));
    app.innerHTML = `<div class="layout">${sidebar(id)}<div>
      <div class="exp-head"><span class="eid">${e.id}</span>${chip(e)}<span class="muted small">${esc(THEME[e.theme].title)}</span></div>
      <h1>${esc(e.title)}</h1>
      <div class="plain"><b>In plain words.</b> ${esc(e.summary)}</div>
      ${sim ? `<p><a class="btn" href="${sim === "fish" ? "#/fish" : "#/sims/" + sim}">▶ See this idea live: ${sim === "fish" ? "real fish replay" : esc(window.SIMS[sim].title)}</a></p>` : ""}
      ${figs.length ? figs.map((f, k) => `
        <div class="panel figcard"><div><img src="${f.src}" alt="${esc(f.title)}" data-k="${k}" loading="lazy"></div>
        <div><h3>${esc(f.title)}</h3>${explainer(f)}<p class="small muted">Click the chart to enlarge. Use ← → to step through.</p></div></div>`).join("")
        : `<div class="panel"><p class="muted">This experiment has no saved charts; its results are numerical. The full write-up is below.</p></div>`}
      ${e.writeup ? `<details class="writeup" ${figs.length ? "" : "open"}><summary>Full write-up from FINDINGS.md: ${esc(e.writeup_title)}</summary><div class="panel prose" style="margin-top:10px">${e.writeup}</div></details>` : ""}
      <div class="pager">${prev ? `<a class="btn" href="#/exp/${prev.id}">← ${prev.id}: ${esc(prev.title)}</a>` : "<span></span>"}
        ${next ? `<a class="btn" href="#/exp/${next.id}">${next.id}: ${esc(next.title)} →</a>` : ""}</div>
    </div></div>`;
    app.querySelectorAll(".figcard img").forEach(img => img.onclick = () => openLb(figs, +img.dataset.k));
    const act = app.querySelector(".sidebar a.active"), side = app.querySelector(".sidebar");
    if (act) side.scrollTop = act.offsetTop - side.clientHeight / 2;
  }

  // ------------------------------------------------------------ results
  function results(tab) {
    const tabs = [["scoreboard", "Scoreboard"], ["gallery", "All charts"], ["summary", "Summary"], ["picture", "Big picture"],
                  ["measures", "Measures"], ["equations", "Equations"], ["lessons", "Lessons"], ["open", "Open problems"]];
    tab = tabs.some(t => t[0] === tab) ? tab : "scoreboard";
    let body = "";
    if (tab === "scoreboard") {
      const col = { pass: "#2e9b5f", partial: "#e8892b", fail: "#d24b43", explore: "#8b95a7" };
      body = `<p class="muted">Share of pre-registered predictions that passed, per experiment. Experiments without a k/n count are shown by their verdict. Failures are part of the record: most of them led directly to the next fix.</p>
        <div class="grid g2">${LAB.themes.map(t => `<div class="panel bars"><h3 style="margin-top:0">${esc(t.title)}</h3>${EXP.filter(e => e.theme === t.id).map(e => {
          const f = frac(e.verdict), w = f ? 100 * f[0] / f[1] : 0;
          return `<div class="row"><a href="#/exp/${e.id}" title="${esc(e.title)}">${e.id}</a><div class="track" title="${esc(e.title)}"><div class="fill" style="width:${f ? Math.max(w, 2) : 100}%;background:${col[e.tone]};opacity:${f ? 1 : 0.3}"></div></div><span class="small">${esc(e.verdict)}</span></div>`;
        }).join("")}</div>`).join("")}</div>`;
    } else if (tab === "gallery") {
      const all = EXP.flatMap(e => e.figures.filter(f => !/full_results_text|scorecard/.test(f.path)).map(f => ({ ...f, exp: `${e.id}: ${e.title}` })));
      body = `<p class="muted">${all.length} charts (scorecards and raw text pages omitted). Click any chart for its explanation; use ← → to browse.</p>
        <div class="grid g4">${all.map((f, k) => `<div class="panel" style="padding:10px;cursor:zoom-in" data-k="${k}">
          <img src="${f.src}" loading="lazy" style="width:100%;border-radius:6px;border:1px solid #e2e6ef" alt="">
          <p class="small" style="margin:6px 0 0"><b>${esc(f.title)}</b><br><span class="muted">${esc(f.exp)}</span></p></div>`).join("")}</div>`;
      setTimeout(() => app.querySelectorAll("[data-k]").forEach(d => d.onclick = () => openLb(all, +d.dataset.k)));
    } else {
      const pg = LAB.pages[tab];
      body = pg ? `<div class="panel prose"><h2 style="margin-top:0">${esc(pg.title)}</h2>${pg.html}</div>` : "<p>Missing.</p>";
    }
    app.innerHTML = `<h1>Results</h1><div class="tabs">${tabs.map(([k, t]) => `<a class="btn ${k === tab ? "primary" : ""}" href="#/results/${k}">${t}</a>`).join("")}</div>${body}`;
  }

  // ------------------------------------------------------------ router
  function route() {
    if (current) { current(); current = null; }
    closeLb();
    const [page, sub] = (location.hash.replace(/^#\/?/, "") || "home").split("/");
    document.querySelectorAll("#nav a").forEach(a => a.classList.toggle("active",
      a.dataset.route === page || (page === "exp" && a.dataset.route === "experiments")));
    const pages = { home, concepts, sims, fish: () => window.FishReplay.mount(app).stop, experiments: experimentsIndex,
                    exp: experiment, results };
    current = (pages[page] || home)(sub) || null;
    if (typeof current !== "function") current = null;
    window.scrollTo(0, 0);
  }
  window.addEventListener("hashchange", route);
  route();
})();
