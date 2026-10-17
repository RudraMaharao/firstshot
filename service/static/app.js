"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const LIGHTS = [
  { key: "D65", label: "Daylight", tol: 1.0 },
  { key: "A", label: "Incandescent", tol: 2.0 },
  { key: "FL11", label: "Store fluorescent", tol: 2.0 },
  { key: "LED-B4", label: "White LED", tol: 2.0 },
];
const LIGHT_LABEL = Object.fromEntries(LIGHTS.map((l) => [l.key, l.label]));

const PRIORITY = {
  fewest: { label: "Fewest pigments", hint: "Every recipe must pass first. Then the fewest pigments wins." },
  cost: { label: "Lowest cost", hint: "Every recipe must pass first. Then the lowest pigment cost wins." },
  carbon: { label: "Lowest carbon", hint: "Every recipe must pass first. Then the lowest embodied carbon wins." },
  closest: { label: "Closest match", hint: "Every recipe must pass first. Then the smallest color difference wins." },
};
const MODE_HINT = {
  ai: "A network picks the five most likely pigments and only their combinations are searched. If none passes, the search widens.",
  exhaustive: "Searches every combination of up to three colorants. Several seconds; use it to check the shortlist.",
};
const MODEL_NOTE = {
  physics: "Kubelka-Munk, pigment strengths re-fitted to the current lot. The usual software approach.",
  hybrid: "Kubelka-Munk plus the learned correction, trained on older pigment lots.",
  tuned: "The same, fine-tuned to the current lot from one calibration panel per pigment.",
};

const state = {
  kind: "patch",
  patch: "orange",
  priority: "fewest",
  mode: "ai",
  exterior: false,
  light: "D65",
  match: null,
  legacy: null,
  reformPriority: "cost",
};

/* ---------- helpers ---------- */
async function api(path, body) {
  const init = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : {};
  const res = await fetch(path, init);
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const j = await res.json();
      detail = typeof j.detail === "string" ? j.detail : j.detail?.[0]?.msg || detail;
    } catch (_) { /* keep status text */ }
    throw new Error(detail);
  }
  return res.json();
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d));
const pct = (x) => `${Math.round(x * 100)}%`;
const signed = (x) => `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x).toFixed(1)}%`;

function luminance(hex) {
  const n = parseInt(hex.slice(1), 16);
  const ch = [(n >> 16) & 255, (n >> 8) & 255, n & 255].map((v) => {
    const c = v / 255;
    return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
}
const inkOn = (hex) => (luminance(hex) > 0.3 ? "#1b1b1a" : "#f6f6f4");

function setStatus(text, isError = false) {
  const el = $("#status");
  el.textContent = text;
  el.classList.toggle("error", isError);
}

function setBusy(busy, label) {
  const btn = $("#run");
  btn.disabled = busy;
  btn.textContent = busy ? label : "Match color";
}

function worstOther(deltaE) {
  const vals = LIGHTS.slice(1).map((l) => deltaE[l.key]).filter((v) => v !== undefined);
  return vals.length ? Math.max(...vals) : null;
}

function inlineRecipe(pigments, { skipWhite = true } = {}) {
  return `<span class="recipe-inline">${pigments
    .filter((p) => !(skipWhite && p.key === "white"))
    .map((p) => `<span><i style="background:${p.chip}"></i>${esc(p.name)} <small>${fmt(p.share, p.share < 1 ? 2 : 1)}%</small></span>`)
    .join("")}</span>`;
}

function pigmentRows(pigments) {
  return `<ul class="pig">${pigments
    .map(
      (p) => `<li>
        <i style="background:${p.chip}" aria-hidden="true"></i>
        <span class="pn">${esc(p.name)}<small>${esc(p.ci)}</small>${p.status === "restricted" ? '<span class="restricted">Restricted</span>' : ""}</span>
        <span class="bar" aria-hidden="true"><b style="width:${Math.min(100, p.share)}%"></b></span>
        <span class="num">${fmt(p.share, p.share < 1 ? 2 : 1)}%</span>
      </li>`
    )
    .join("")}</ul>`;
}

/* ---------- segmented controls ---------- */
function bindSeg(name, onChange) {
  const group = $(`.seg[data-name="${name}"]`);
  const buttons = $$("button", group);
  const select = (btn, focus) => {
    buttons.forEach((b) => {
      b.setAttribute("aria-checked", String(b === btn));
      b.tabIndex = b === btn ? 0 : -1;
    });
    if (focus) btn.focus();
    onChange(btn.dataset.value);
  };
  buttons.forEach((b) => {
    b.tabIndex = b.getAttribute("aria-checked") === "true" ? 0 : -1;
    b.addEventListener("click", () => select(b, false));
    b.addEventListener("keydown", (e) => {
      const i = buttons.indexOf(b);
      if (e.key === "ArrowRight" || e.key === "ArrowDown") { e.preventDefault(); select(buttons[(i + 1) % buttons.length], true); }
      if (e.key === "ArrowLeft" || e.key === "ArrowUp") { e.preventDefault(); select(buttons[(i - 1 + buttons.length) % buttons.length], true); }
    });
  });
}

/* ---------- target inputs ---------- */
function renderPatches(patches) {
  const box = $("#patches");
  box.innerHTML = patches
    .map((p) => `<button type="button" role="radio" aria-checked="${p.name === state.patch}" aria-label="${esc(p.name)}" title="${esc(p.name)}" data-name="${esc(p.name)}" style="background:${p.hex}"></button>`)
    .join("");
  const buttons = $$("button", box);
  const pick = (btn) => {
    buttons.forEach((b) => b.setAttribute("aria-checked", String(b === btn)));
    state.patch = btn.dataset.name;
    $("#patch-name").textContent = state.patch;
  };
  buttons.forEach((b) => {
    b.tabIndex = b.dataset.name === state.patch ? 0 : -1;
    b.addEventListener("click", () => { pick(b); runMatch(); });
    b.addEventListener("keydown", (e) => {
      const i = buttons.indexOf(b);
      const step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 6, ArrowUp: -6 }[e.key];
      if (step) {
        e.preventDefault();
        const next = buttons[(i + step + buttons.length) % buttons.length];
        buttons.forEach((x) => (x.tabIndex = -1));
        next.tabIndex = 0;
        next.focus();
        pick(next);
      }
    });
  });
  $("#patch-name").textContent = state.patch;
}

function currentTarget() {
  if (state.kind === "patch") return { kind: "patch", value: state.patch };
  if (state.kind === "hex") {
    let v = $("#hex-input").value.trim();
    if (!v.startsWith("#")) v = `#${v}`;
    if (!/^#[0-9a-fA-F]{6}$/.test(v)) throw new Error("Enter a six-digit hex code, like #3a6ea5.");
    return { kind: "hex", value: v };
  }
  const lab = ["#lab-l", "#lab-a", "#lab-b"].map((id) => parseFloat($(id).value));
  if (lab.some(Number.isNaN)) throw new Error("Enter numbers for L, a and b.");
  return { kind: "lab", value: lab };
}

/* ---------- match ---------- */
async function runMatch(e) {
  if (e) e.preventDefault();
  let target;
  try {
    target = currentTarget();
  } catch (err) {
    setStatus(err.message, true);
    return;
  }
  setBusy(true, state.mode === "exhaustive" ? "Searching every set" : "Matching");
  setStatus(state.mode === "exhaustive" ? "Searching every pigment set. This takes several seconds." : "");
  try {
    const m = await api("/api/match", {
      target, priority: state.priority, exterior_only: state.exterior, mode: state.mode,
    });
    state.match = m;
    renderMatch(m);
    const widened = m.search.widened ? " The shortlist found nothing that passes, so the search widened." : "";
    setStatus(
      `${m.viable ? "Matched" : "No passing recipe"} in ${m.search.ms.toLocaleString()} ms, ${m.search.sets_checked} of ${m.search.sets_total} pigment sets checked.${widened}`
    );
    $("#trial-list").innerHTML = '<p class="empty">Spraying and measuring trial panels.</p>';
    const t = await api("/api/trials", { target, priority: state.priority, exterior_only: state.exterior });
    renderTrials(t);
  } catch (err) {
    setStatus(`Could not match: ${err.message}`, true);
  } finally {
    setBusy(false);
  }
}

function renderMatch(m) {
  document.documentElement.style.setProperty("--t", m.target.appearance.D65);
  document.documentElement.style.setProperty("--r", m.recipe.appearance.D65);
  if (!m.target.spectral && state.light !== "D65") state.light = "D65";
  renderBooth(false);
  renderTicket(m);
  renderFigures(m);
  renderCounts(m);
}

function renderBooth(animate = true) {
  const m = state.match;
  if (!m) return;
  const r = m.recipe;
  $("#lights").innerHTML = LIGHTS.map((l) => {
    const de = r.delta_e[l.key];
    const known = de !== undefined;
    const cls = known ? (de <= l.tol ? "ok" : "bad") : "";
    return `<button type="button" role="tab" aria-selected="${state.light === l.key}" data-light="${l.key}">
      <span class="l-name">${l.label}</span>
      <span class="l-de ${cls}">${known ? `<span class="flag" aria-hidden="true"></span>dE ${fmt(de)}` : "Recipe only"}</span>
    </button>`;
  }).join("");
  $$("#lights button").forEach((b) =>
    b.addEventListener("click", () => {
      if (state.light === b.dataset.light) return;
      state.light = b.dataset.light;
      renderBooth(true);
    })
  );

  const light = state.light;
  const tHalf = $("#half-target");
  const rHalf = $("#half-recipe");
  const tHex = m.target.appearance[light];
  const rHex = r.appearance[light];
  tHalf.classList.toggle("unknown", !tHex);
  tHalf.style.backgroundColor = tHex || "#9b9b98";
  tHalf.style.color = tHex ? inkOn(tHex) : "#f6f6f4";
  tHalf.innerHTML = tHex
    ? `<span>Target<span class="sub">${esc(m.target.name)}</span></span>`
    : `<span>Target<span class="sub">A screen color has no spectrum, so it can only be seen in daylight.</span></span>`;
  rHalf.style.backgroundColor = rHex;
  rHalf.style.color = inkOn(rHex);
  const parts = r.colorants === 0 ? "White base only" : `${r.colorants} colorant${r.colorants === 1 ? "" : "s"} plus white`;
  rHalf.innerHTML = `<span>Recipe<span class="sub">${parts}</span></span>`;

  const de = r.delta_e[light];
  const tol = LIGHTS.find((l) => l.key === light).tol;
  $("#booth-note").textContent =
    de === undefined
      ? `${LIGHT_LABEL[light]}. Shows how the recipe itself will look under this light.`
      : `${LIGHT_LABEL[light]}. Difference ${fmt(de)} against a tolerance of ${fmt(tol, 1)}: ${de <= tol ? "passes" : "fails"}.`;

  if (animate && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    const dd = $("#drawdown");
    dd.classList.remove("sweeping");
    void dd.offsetWidth;
    dd.classList.add("sweeping");
  }
}

function renderTicket(m) {
  const r = m.recipe;
  const tag = m.viable
    ? '<span class="tag pass">Passes under all lights</span>'
    : '<span class="tag fail">Outside tolerance</span>';
  const note = m.viable
    ? ""
    : `<p class="empty" style="margin-top:10px">No recipe from this palette passes robustly. This is the closest one.</p>`;
  $("#ticket").innerHTML = `
    <div class="ticket-head"><h2>Recipe</h2>${tag}</div>
    ${pigmentRows(r.pigments)}
    ${note}
    <p class="ticket-note">Shares are by weight of the pigment blend. Predicted color, CIELAB under daylight: L ${fmt(r.lab[0], 1)}, a ${fmt(r.lab[1], 1)}, b ${fmt(r.lab[2], 1)}.</p>`;
}

function renderFigures(m) {
  const r = m.recipe;
  const s = m.search;
  $("#figures").innerHTML = `
    <div><dt>Pass probability</dt><dd>${pct(r.pass_probability)}<small>across model members and pigment lots</small></dd></div>
    <div><dt>Worst other light</dt><dd>${m.target.spectral ? fmt(worstOther(r.delta_e)) : "n/a"}<small>${m.target.spectral ? "dE, tolerance 2.0" : "needs a measured sample"}</small></dd></div>
    <div><dt>Pigment cost</dt><dd>${fmt(r.cost_index)}<small>per kg, titanium white = 1.00</small></dd></div>
    <div><dt>Carbon index</dt><dd>${fmt(r.carbon_index)}<small>illustrative, per kg</small></dd></div>
    <div><dt>Search</dt><dd>${s.ms.toLocaleString()} ms<small>${s.sets_checked} of ${s.sets_total} pigment sets</small></dd></div>
    <div><dt>Closest daylight match</dt>${closestFigure(m)}</div>`;
}

function closestFigure(m) {
  const c = m.closest;
  if (m.versus_closest.same_recipe) return `<dd>This one<small>no other recipe is closer in daylight</small></dd>`;
  if (c.passes) {
    return `<dd>Passes too<small>dE ${fmt(c.delta_e.D65)} with ${c.colorants} colorants; chosen recipe cost ${signed(m.versus_closest.cost_pct)}</small></dd>`;
  }
  if (!m.target.spectral) return `<dd>Fails<small>dE ${fmt(c.delta_e.D65)}, pass probability ${pct(c.pass_probability)}</small></dd>`;
  const worst = LIGHTS.slice(1).reduce((a, l) => (c.delta_e[l.key] > c.delta_e[a.key] ? l : a), LIGHTS[1]);
  return `<dd>Fails<small>dE ${fmt(c.delta_e.D65)} in daylight, ${fmt(c.delta_e[worst.key])} under ${worst.label.toLowerCase()}</small></dd>`;
}

function renderCounts(m) {
  const spectral = m.target.spectral;
  const rows = m.by_count
    .map(
      (c) => `<tr class="${c.chosen ? "chosen" : ""}">
        <td>${c.colorants}${c.chosen ? ' <span class="tag pass">Chosen</span>' : ""}</td>
        <td>${inlineRecipe(c.pigments)}</td>
        <td class="num">${fmt(c.delta_e.D65)}</td>
        <td class="num">${spectral ? fmt(worstOther(c.delta_e)) : "n/a"}</td>
        <td class="num">${pct(c.pass_probability)}</td>
        <td class="num">${fmt(c.cost_index)}</td>
        <td class="num">${fmt(c.carbon_index)}</td>
      </tr>`
    )
    .join("");
  $("#count-table").innerHTML = `
    <thead><tr>
      <th scope="col">Colorants</th><th scope="col">Recipe, white not shown</th>
      <th scope="col" class="num">dE daylight</th><th scope="col" class="num">Worst other light</th>
      <th scope="col" class="num">Pass probability</th><th scope="col" class="num">Cost</th><th scope="col" class="num">Carbon</th>
    </tr></thead>
    <tbody>${rows}</tbody>`;
}

/* ---------- lab trials ---------- */
function renderTrials(t) {
  $("#trial-list").innerHTML = t.runs
    .map((run) => {
      const shots = run.history
        .map((h) => {
          const other = h.de_other_lights === null ? "" : `<small>other lights ${fmt(h.de_other_lights)}</small>`;
          return `<span class="shot ${h.passed ? "ok" : ""}"><b>Trial ${h.trial}</b>dE ${fmt(h.de_daylight)}${other}</span>`;
        })
        .join("");
      const verdict = run.passed
        ? run.trials === 1 ? "Right first time" : `Passed on trial ${run.trials}`
        : `Not passed in ${run.trials}`;
      return `<div class="trial">
        <div><h3>${esc(run.label)}</h3><p>${esc(MODEL_NOTE[run.model] || "")}</p></div>
        <div class="shots" aria-label="Trial results">${shots}</div>
        <div class="verdict ${run.passed ? "pass-txt" : "fail-txt"}">${verdict}</div>
      </div>`;
    })
    .join("");
}

/* ---------- safer materials ---------- */
function renderLegacyList(items) {
  const box = $("#legacy-list");
  box.innerHTML = items
    .map((l) => `<button type="button" class="legacy" role="radio" aria-checked="false" data-key="${esc(l.key)}">
        <i style="background:${l.hex}" aria-hidden="true"></i>
        <span><strong>${esc(l.name)}</strong><span>${esc(l.use)}</span><span>${esc(l.reason)}</span></span>
      </button>`)
    .join("");
  const buttons = $$(".legacy", box);
  buttons.forEach((b, i) => {
    b.tabIndex = i === 0 ? 0 : -1;
    b.addEventListener("click", () => {
      buttons.forEach((x) => { x.setAttribute("aria-checked", String(x === b)); x.tabIndex = x === b ? 0 : -1; });
      state.legacy = b.dataset.key;
      runReform();
    });
    b.addEventListener("keydown", (e) => {
      const step = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 }[e.key];
      if (step) {
        e.preventDefault();
        const next = buttons[(i + step + buttons.length) % buttons.length];
        next.focus();
        next.click();
      }
    });
  });
}

async function runReform() {
  if (!state.legacy) return;
  const out = $("#reform-out");
  out.setAttribute("aria-busy", "true");
  try {
    const r = await api("/api/reformulate", { legacy: state.legacy, priority: state.reformPriority });
    renderReform(r);
  } catch (err) {
    out.innerHTML = `<p class="status error">Could not rebuild this formula: ${esc(err.message)}</p>`;
  } finally {
    out.removeAttribute("aria-busy");
  }
}

function renderReform(r) {
  const rep = r.replacement;
  const lab = r.lab_check;
  const strip = LIGHTS.map(
    (l) => `<figure>
      <div class="duo"><span style="background:${r.target.appearance[l.key]}"></span><span style="background:${rep.appearance[l.key]}"></span></div>
      <figcaption>${l.label}<br>dE ${fmt(rep.delta_e[l.key])}</figcaption>
    </figure>`
  ).join("");
  const cls = (x) => (x < 0 ? "down" : x > 0 ? "up" : "");
  const labText = lab.passed
    ? lab.trials === 1 ? "Right first time in the lab" : `Passed on lab trial ${lab.trials}`
    : `Not passed in ${lab.trials} lab trials`;
  // Priorities that land on the same recipe share one row.
  const groups = [];
  for (const o of r.options) {
    const sig = o.pigments.map((p) => `${p.key}:${Math.round(p.share)}`).join("|");
    const g = groups.find((x) => x.sig === sig);
    if (g) g.labels.push(o.priority);
    else groups.push({ sig, labels: [o.priority], o });
  }
  const options = groups
    .map(({ labels, o }) => ({ ...o, labels }))
    .map(
      (o) => `<tr class="${o.labels.includes(r.settings.priority) ? "chosen" : ""}">
        <td>${o.labels.map((k, i) => (i ? PRIORITY[k].label.toLowerCase() : PRIORITY[k].label)).join(", ")}</td>
        <td>${inlineRecipe(o.pigments)}</td>
        <td class="num">${fmt(o.delta_e.D65)}</td>
        <td class="num">${pct(o.pass_probability)}</td>
        <td class="num ${cls(o.cost_pct)}">${signed(o.cost_pct)}</td>
        <td class="num ${cls(o.carbon_pct)}">${signed(o.carbon_pct)}</td>
      </tr>`
    )
    .join("");

  let notice = "";
  if (!r.viable) {
    const worst = LIGHTS.reduce((a, l) => (rep.delta_e[l.key] - l.tol > rep.delta_e[a.key] - a.tol ? l : a), LIGHTS[0]);
    notice = `<p class="notice"><strong>No standard-pigment recipe passes under all four lights.</strong>
      The closest one is shown. It misses by dE ${fmt(rep.delta_e[worst.key])} under ${worst.label.toLowerCase()},
      which customers would see. Keep the legacy pigment for this color, or accept a visible shift.</p>`;
  }
  $("#reform-out").innerHTML = `${notice}
    <div class="compare">
      <div class="side">
        <h3>Legacy formula <span class="tag fail">Replace ${esc(r.legacy.replaced.join(", "))}</span></h3>
        ${pigmentRows(r.legacy.pigments)}
        <p class="reason">${esc(r.legacy.reason)}</p>
        <dl class="deltas">
          <div><dt>Pigment cost</dt><dd>${fmt(r.legacy.cost_index)}</dd></div>
          <div><dt>Carbon index</dt><dd>${fmt(r.legacy.carbon_index)}</dd></div>
        </dl>
      </div>
      <div class="side">
        <h3>Replacement ${r.viable ? '<span class="tag pass">Passes under all lights</span>' : '<span class="tag fail">Outside tolerance</span>'}</h3>
        ${pigmentRows(rep.pigments)}
        <dl class="deltas">
          <div><dt>Pigment cost</dt><dd class="${cls(r.change.cost_pct)}">${signed(r.change.cost_pct)}</dd></div>
          <div><dt>Carbon index</dt><dd class="${cls(r.change.carbon_pct)}">${signed(r.change.carbon_pct)}</dd></div>
        </dl>
        <p class="reason">${r.viable ? `No ${esc(r.legacy.replaced.join(", ").toLowerCase())}. ` : ""}${labText}. ${r.settings.exterior_only ? "Exterior-grade pigments only." : "Interior use, so all standard pigments allowed."}</p>
      </div>
    </div>
    <div class="side">
      <h3>Legacy against replacement, under each light</h3>
      <div class="lightstrip">${strip}</div>
    </div>
    <div class="table-wrap" ${r.viable ? "" : "hidden"}><table class="data">
      <thead><tr><th scope="col">Priority</th><th scope="col">Recipe, white not shown</th><th scope="col" class="num">dE daylight</th>
      <th scope="col" class="num">Pass probability</th><th scope="col" class="num">Cost change</th><th scope="col" class="num">Carbon change</th></tr></thead>
      <tbody>${options}</tbody>
    </table></div>`;
}

/* ---------- model card ---------- */
function renderMetrics(card) {
  const t = card.training?.current_lot_test;
  const box = $("#metrics");
  if (!t) { box.innerHTML = ""; return; }
  const rows = [
    ["Physics only", t.physics_uncalibrated],
    ["Physics, calibrated", t.physics_calibrated],
    ["Physics + ML", t.hybrid_untuned],
    ["Physics + ML, tuned", t.hybrid_tuned],
  ]
    .map(([name, v]) => `<tr><td>${name}</td><td class="num">${fmt(v.median)}</td><td class="num">${fmt(v.p90)}</td><td class="num">${pct(v.within_1)}</td></tr>`)
    .join("");
  let bench = "";
  const g = card.benchmark?.groups?.["customer samples"]?.summary;
  if (g) {
    const tr = g.trials;
    bench = `<h3>Benchmark on ${g.targets} customer samples</h3>
      <div class="table-wrap"><table class="data">
        <thead><tr><th scope="col">Model</th><th scope="col" class="num">Right first time</th><th scope="col" class="num">Mean lab trials</th></tr></thead>
        <tbody>${["physics", "hybrid", "tuned"].map((k) => `<tr><td>${esc(tr[k].label)}</td><td class="num">${pct(tr[k].first_shot_pass)}</td><td class="num">${fmt(tr[k].mean_trials)}</td></tr>`).join("")}</tbody>
      </table></div>
      <p>The AI shortlist checked ${fmt(g.search.ai_sets, 0)} of ${fmt(g.search.all_sets, 0)} pigment sets, ${fmt(g.search.all_seconds / g.search.ai_seconds, 0)} times faster, and found a passing recipe for ${pct(g.search.ai_found_when_reachable)} of the samples that full search could match.</p>`;
  }
  box.innerHTML = `
    <h3>Prediction error on 1,500 fresh panels from the current pigment lot</h3>
    <div class="table-wrap"><table class="data">
      <thead><tr><th scope="col">Model</th><th scope="col" class="num">Median dE</th><th scope="col" class="num">90th percentile</th><th scope="col" class="num">Within 1.0</th></tr></thead>
      <tbody>${rows}</tbody>
    </table></div>
    ${bench}`;
}

/* ---------- start ---------- */
function bindControls() {
  bindSeg("kind", (v) => {
    state.kind = v;
    $$(".pane").forEach((p) => (p.hidden = p.dataset.pane !== v));
  });
  bindSeg("priority", (v) => {
    state.priority = v;
    $("#priority-hint").textContent = PRIORITY[v].hint;
  });
  bindSeg("mode", (v) => {
    state.mode = v;
    $("#mode-hint").textContent = MODE_HINT[v];
  });
  bindSeg("reform-priority", (v) => {
    state.reformPriority = v;
    runReform();
  });
  $("#exterior").addEventListener("change", (e) => (state.exterior = e.target.checked));
  $("#controls").addEventListener("submit", runMatch);
  const picker = $("#hex-picker");
  const text = $("#hex-input");
  picker.addEventListener("input", () => (text.value = picker.value));
  text.addEventListener("input", () => {
    const v = text.value.trim().startsWith("#") ? text.value.trim() : `#${text.value.trim()}`;
    if (/^#[0-9a-fA-F]{6}$/.test(v)) picker.value = v.toLowerCase();
  });
}

async function start() {
  bindControls();
  try {
    const [patches, legacy, card] = await Promise.all([api("/api/targets"), api("/api/legacy"), api("/api/model-card")]);
    renderPatches(patches);
    renderLegacyList(legacy);
    renderMetrics(card);
    runMatch();
    const first = $(".legacy");
    if (first) first.click();
  } catch (err) {
    setStatus(`Could not reach the API: ${err.message}`, true);
  }
}

start();
