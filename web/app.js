// Jacobian Space Explorer 前端:格子渲染、lens 切換、預計算進度輪詢
const $ = (id) => document.getElementById(id);

let state = {
  data: null,        // /api/analyze 回應
  lens: "jlens",     // "jlens" | "logit"
  selected: null,    // {layer, pos}
  pollTimer: null,
};

async function loadModels() {
  const res = await fetch("/api/models");
  const { models } = await res.json();
  const sel = $("model-select");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const m of models) {
    const opt = document.createElement("option");
    const badge = { ready: "✓ J-lens", computing: "⏳ 計算中", partial: "◐ 部分", missing: "logit only" }[m.state] || m.state;
    opt.value = m.model_id;
    opt.textContent = `${m.model_id}  [${badge}]`;
    opt.dataset.state = m.state;
    sel.appendChild(opt);
  }
  if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
  updateBanner();
}

function currentModel() { return $("model-select").value; }

function updateBanner() {
  const opt = $("model-select").selectedOptions[0];
  const banner = $("precompute-banner");
  if (!opt) return banner.classList.add("hidden");
  const st = opt.dataset.state;
  if (st === "ready") return banner.classList.add("hidden");
  banner.classList.remove("hidden");
  if (st === "computing") {
    $("precompute-msg").textContent = "平均 Jacobian 預計算進行中,期間可先用 logit lens。";
    $("precompute-btn").classList.add("hidden");
    $("progress-wrap").classList.remove("hidden");
    startPolling();
  } else {
    $("precompute-msg").textContent = `此模型尚未預計算平均 Jacobian(J-lens 不可用,可先用 logit lens)。`;
    $("precompute-btn").classList.remove("hidden");
    $("progress-wrap").classList.add("hidden");
  }
}

async function startPrecompute() {
  await fetch("/api/precompute", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_id: currentModel() }),
  });
  $("precompute-btn").classList.add("hidden");
  $("progress-wrap").classList.remove("hidden");
  startPolling();
}

function startPolling() {
  if (state.pollTimer) return;
  state.pollTimer = setInterval(async () => {
    const res = await fetch(`/api/precompute/status?model_id=${encodeURIComponent(currentModel())}`);
    const p = await res.json();
    if (p.state === "running" || p.state === "starting") {
      const pct = p.total ? (100 * (p.done || 0) / p.total) : 0;
      $("progress-bar").style.width = pct.toFixed(1) + "%";
      $("precompute-msg").textContent = `預計算中 ${p.done || 0}/${p.total}(可先用 logit lens)`;
    } else {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
      if (p.state === "error") {
        $("precompute-msg").textContent = "預計算失敗:" + (p.error || "未知錯誤");
        $("precompute-btn").classList.remove("hidden");
      } else {
        await loadModels();
        if (state.data) analyze(); // 重新分析以取得 J-lens 結果
      }
    }
  }, 3000);
}

async function analyze() {
  const text = $("text-input").value;
  if (!text.trim()) return;
  $("analyze-btn").disabled = true;
  $("analyze-btn").textContent = "…";
  try {
    const res = await fetch("/api/analyze", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: currentModel(), text, top_k: 10 }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    state.data = await res.json();
    state.selected = null;
    if (!state.data.jlens_available && state.lens === "jlens") setLens("logit");
    renderGrid();
  } catch (e) {
    const err = Object.assign(document.createElement("div"), { id: "grid-empty" });
    err.textContent = "分析失敗:" + e.message;
    $("grid-wrap").replaceChildren(err);
  } finally {
    $("analyze-btn").disabled = false;
    $("analyze-btn").textContent = "分析";
  }
}

function setLens(lens) {
  state.lens = lens;
  document.querySelectorAll("#lens-toggle button").forEach(b =>
    b.classList.toggle("active", b.dataset.lens === lens));
  if (state.data) renderGrid();
}

function grid() { return state.data.lenses[state.lens]; }

function renderGrid() {
  const d = state.data;
  const g = grid();
  if (!g) {
    $("grid-wrap").innerHTML = `<div id="grid-empty">此模型的 ${state.lens} 尚不可用</div>`;
    return;
  }
  const L = g.length;
  const table = document.createElement("table");
  table.className = "grid";

  const thead = document.createElement("tr");
  thead.appendChild(Object.assign(document.createElement("th"), { className: "layer", textContent: "層 \\ 位置" }));
  d.tokens.forEach((tok, i) => {
    const th = Object.assign(document.createElement("th"), { className: "tok", textContent: tok || "·" });
    th.dataset.pos = i;
    th.title = `輸入 token #${i}`;
    th.onclick = () => highlightColumn(i);
    thead.appendChild(th);
  });
  table.appendChild(thead);

  for (let l = L - 1; l >= 0; l--) {  // 後期層在上
    const tr = document.createElement("tr");
    tr.appendChild(Object.assign(document.createElement("th"), { className: "layer", textContent: `L${l + 1}` }));
    g[l].forEach((cell, pos) => {
      const td = document.createElement("td");
      const top = cell[0];
      const tok = Object.assign(document.createElement("div"), { className: "cell-tok" });
      tok.textContent = top.token.trim() || "␣";
      const prob = Object.assign(document.createElement("div"), { className: "cell-p" });
      prob.textContent = top.p >= 0.001 ? (top.p * 100).toFixed(1) + "%" : "<0.1%";
      td.append(tok, prob);
      td.style.background = `rgba(94, 234, 212, ${Math.min(top.p, 1) * 0.55})`;
      if (top.p > 0.5) td.classList.add("strong");
      td.title = `L${l + 1} pos ${pos}: "${top.token}" p=${top.p}`;
      td.dataset.layer = l; td.dataset.pos = pos;
      td.onclick = () => selectCell(l, pos, td);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  }
  $("grid-wrap").innerHTML = "";
  $("grid-wrap").appendChild(table);
  if (state.selected) selectCell(state.selected.layer, state.selected.pos);
}

function selectCell(l, pos, td) {
  state.selected = { layer: l, pos };
  document.querySelectorAll(".grid td.selected").forEach(e => e.classList.remove("selected"));
  (td || document.querySelector(`td[data-layer="${l}"][data-pos="${pos}"]`))?.classList.add("selected");

  const cell = grid()[l][pos];
  $("sidebar").classList.remove("hidden");
  $("sidebar-title").textContent = `L${l + 1} · 位置 ${pos}「${state.data.tokens[pos]}」 · ${state.lens === "jlens" ? "J-lens" : "logit lens"} top-${cell.length}`;
  const maxP = cell[0].p || 1e-9;
  $("sidebar-bars").innerHTML = "";
  for (const e of cell) {
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `<span class="tok"></span><div class="bar"></div><span class="p"></span>`;
    row.querySelector(".tok").textContent = JSON.stringify(e.token).slice(1, -1);
    row.querySelector(".bar").style.width = `${(e.p / maxP) * 140}px`;
    row.querySelector(".p").textContent = (e.p * 100).toFixed(2) + "%";
    $("sidebar-bars").appendChild(row);
  }
}

function highlightColumn(pos) {
  document.querySelectorAll(".col-hi").forEach(e => e.classList.remove("col-hi"));
  document.querySelectorAll(`[data-pos="${pos}"]`).forEach(e => e.classList.add("col-hi"));
}

// --- 事件繫結 ---
$("analyze-btn").onclick = analyze;
$("text-input").addEventListener("keydown", e => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});
$("model-select").onchange = () => { updateBanner(); if (state.data) analyze(); };
$("precompute-btn").onclick = startPrecompute;
document.querySelectorAll("#lens-toggle button").forEach(b => b.onclick = () => setLens(b.dataset.lens));
$("custom-model").addEventListener("keydown", e => {
  if (e.key !== "Enter") return;
  const id = e.target.value.trim();
  if (!id) return;
  const opt = Object.assign(document.createElement("option"), { value: id, textContent: `${id}  [custom]` });
  opt.dataset.state = "missing";
  $("model-select").appendChild(opt);
  $("model-select").value = id;
  e.target.value = "";
  updateBanner();
});

loadModels();
