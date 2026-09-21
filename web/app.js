/* StyleSense — front-end logic.
   Talks to: /api/recommend, /api/suggest, /api/facets, /api/brands,
   /api/collab/{asin}, /health. No build step, no dependencies. */

const $ = (id) => document.getElementById(id);
let USD_INR = 83;
const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");

// Titles/brands come from scraped Amazon metadata (untrusted). Escape before
// innerHTML — raw titles contain '<br />' tags and '&', which would break the
// card layout (or be an XSS vector).
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// "u.s. polo assn." -> "U.S. Polo Assn." (dot-initials and short tokens go
// fully uppercase, the rest title-case).
function capBrand(s) {
  if (!s) return "";
  return s.toLowerCase().split(/\s+/).map((w) =>
    w.includes(".") || w.length <= 2 ? w.toUpperCase() : w[0].toUpperCase() + w.slice(1)
  ).join(" ");
}

function fmtCount(n) {
  if (n >= 1000) {
    const k = (n / 1000).toFixed(1);
    return (k.endsWith(".0") ? k.slice(0, -2) : k) + "K";
  }
  return String(n);
}

/* Small inline icons for the filter rows (stroke = currentColor). */
const ICO = {
  type: '<svg class="fico" viewBox="0 0 24 24"><path d="M8 3 4 6.5 6.8 10 9 8.8V21h6V8.8L17.2 10 20 6.5 16 3c-1 1.6-2.4 2.5-4 2.5S9 4.6 8 3z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/></svg>',
  brand: '<svg class="fico" viewBox="0 0 24 24"><path d="M20.6 13.4 12 22 2 12V2h10l8.6 8.6a1.8 1.8 0 0 1 0 2.8z" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round"/><circle cx="7" cy="7" r="1.4" fill="currentColor"/></svg>',
  size: '<svg class="fico" viewBox="0 0 24 24"><path d="M3 8V3h5M21 16v5h-5M3 3l5 5M21 21l-5-5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>',
  price: '<span class="fico rp">₹</span>',
  star: '<span class="fico rp">★</span>',
};

// Active clickable filters + the raw query text last typed.
const state = { brand: "", size: "", priceKey: "", typeKey: "", rating: 0, baseQuery: "", sort: "best" };
let facets = null, searchSeq = 0, lastData = null;
let recent = [];
try { recent = JSON.parse(localStorage.getItem("ss_recent") || "[]"); } catch (_) { recent = []; }

// Garment words a type-chip must strip before appending its own word, so
// "black polo" + [T-Shirts] becomes "black tshirt", not "black polo tshirt".
const TYPE_STRIP = /\b(t-?\s?shir\w*|tshir\w*|tee\w*|shir\w*|polo\w*|henley\w*|hoodie\w*|sweatshirt\w*|tank\w*|vest\w*)\b/gi;
function composeQuery() {
  let q = state.baseQuery.trim();
  if (state.typeKey && facets) {
    const t = facets.types.find((x) => x.key === state.typeKey);
    if (t && t.query_word) {
      q = q.replace(TYPE_STRIP, "").replace(/\s+/g, " ").trim();
      q = q ? q + " " + t.query_word : t.query_word;
    }
  }
  return q;
}

/* --- filter card ---------------------------------------------------------- */
let brandTimer = null, lastBrandOpts = "";
function renderFilters() {
  const f = $("filters");
  const tChips = facets.types.map((t) =>
    `<button class="chip${state.typeKey === t.key ? " on" : ""}" data-type="${esc(t.key)}">${esc(t.label)} <span class="cnt">${t.count.toLocaleString("en-IN")}</span></button>`).join("");
  const sBtns = facets.sizes.map((s) =>
    `<button class="chip${state.size === s.key ? " on" : ""}" data-size="${esc(s.key)}">${esc(s.label)}</button>`).join("");
  const pBtns = facets.price_buckets.map((b) =>
    `<button class="chip${state.priceKey === b.key ? " on" : ""}" data-price="${esc(b.key)}">${esc(b.label)}</button>`).join("");
  const bOpts = facets.brands.map((b) => `<option value="${esc(b.name)}">${esc(capBrand(b.name))} (${b.count.toLocaleString("en-IN")})</option>`).join("");
  lastBrandOpts = bOpts;
  f.innerHTML =
    `<div class="frow"><span class="flabel">${ICO.type}Type</span>${tChips}</div>` +
    `<div class="frow"><span class="flabel">${ICO.brand}Brand</span>` +
    `<input id="brandInput" list="brandList" placeholder="Any brand — type to search all 35k" value="${esc(state.brand)}">` +
    `<datalist id="brandList">${bOpts}</datalist>` +
    `<span class="flabel" style="width:auto;margin-left:10px">${ICO.size}Size</span>${sBtns}</div>` +
    `<div class="frow"><span class="flabel">${ICO.price}Price</span>${pBtns}` +
    `<span class="flabel" style="width:auto;margin-left:10px">${ICO.star}Rating</span>` +
    `<button class="chip${state.rating ? " on" : ""}" id="ratingChip">4★ &amp; up</button>` +
    `<button class="clearf" id="clearF">✕ clear all filters</button></div>`;

  const bi = $("brandInput");
  bi.addEventListener("change", () => { state.brand = bi.value.trim().toLowerCase(); syncFilterUI(); doSearch(); });
  bi.addEventListener("keydown", (e) => { if (e.key === "Enter") { state.brand = bi.value.trim().toLowerCase(); syncFilterUI(); doSearch(); } });
  bi.addEventListener("input", () => {
    clearTimeout(brandTimer);
    const v = bi.value.trim();
    if (v.length < 2) return;
    brandTimer = setTimeout(() => updateBrandList(v), 300);
  });
  f.querySelectorAll("[data-type]").forEach((b) => b.addEventListener("click", () => {
    state.typeKey = state.typeKey === b.dataset.type ? "" : b.dataset.type; syncFilterUI(); doSearch();
  }));
  f.querySelectorAll("[data-size]").forEach((b) => b.addEventListener("click", () => {
    state.size = state.size === b.dataset.size ? "" : b.dataset.size; syncFilterUI(); doSearch();
  }));
  f.querySelectorAll("[data-price]").forEach((b) => b.addEventListener("click", () => {
    state.priceKey = state.priceKey === b.dataset.price ? "" : b.dataset.price; syncFilterUI(); doSearch();
  }));
  $("ratingChip").addEventListener("click", () => {
    state.rating = state.rating ? 0 : 4; syncFilterUI(); doSearch();
  });
  $("clearF").addEventListener("click", clearAllFilters);
}

// Search the full brand catalog (top-40 dropdown can't hold 35k brands).
// Falls back to the facet list when the server finds nothing.
function updateBrandList(q) {
  fetch("/api/brands?q=" + encodeURIComponent(q))
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      const names = d && d.brands && d.brands.length ? d.brands : (facets ? facets.brands.map((b) => b.name) : []);
      const html = names.map((b) => `<option value="${esc(b)}">`).join("");
      if (html !== lastBrandOpts) { $("brandList").innerHTML = html; lastBrandOpts = html; }
    })
    .catch(() => {});
}

function clearAllFilters() {
  state.brand = state.size = state.priceKey = state.typeKey = "";
  state.rating = 0;
  syncFilterUI(); doSearch();
}

// Active-filter bar: every applied filter as a removable chip.
function renderActiveChips() {
  const chips = [];
  if (state.typeKey && facets) chips.push(["type", facets.types.find((x) => x.key === state.typeKey)?.label || state.typeKey]);
  if (state.brand) chips.push(["brand", capBrand(state.brand)]);
  if (state.size && facets) chips.push(["size", facets.sizes.find((x) => x.key === state.size)?.label || state.size]);
  if (state.priceKey && facets) chips.push(["price", facets.price_buckets.find((x) => x.key === state.priceKey)?.label || state.priceKey]);
  if (state.rating) chips.push(["rating", state.rating + "★ & up"]);
  const box = $("activeChips");
  box.innerHTML = chips.length
    ? chips.map(([k, label]) => `<span class="achip">${esc(label)}<button data-clear="${k}" title="Remove">✕</button></span>`).join("")
      + `<button class="aclear" id="clearAll">clear all</button>`
    : "";
  box.querySelectorAll("[data-clear]").forEach((b) => b.addEventListener("click", () => {
    const k = b.dataset.clear;
    if (k === "type") state.typeKey = "";
    if (k === "brand") state.brand = "";
    if (k === "size") state.size = "";
    if (k === "price") state.priceKey = "";
    if (k === "rating") state.rating = 0;
    syncFilterUI(); doSearch();
  }));
  const ca = $("clearAll");
  if (ca) ca.addEventListener("click", clearAllFilters);
}

function syncFilterUI() { renderFilters(); renderActiveChips(); }

// /api/facets blocks until the engine is warm, so the first fetch may take
// ~60s — retry until it lands (the suggest box works meanwhile).
async function loadFacets() {
  try {
    const r = await fetch("/api/facets");
    if (!r.ok) throw new Error(String(r.status));
    facets = await r.json();
    USD_INR = facets.usd_inr || 83;
    renderFilters();
  } catch (_) { setTimeout(loadFacets, 5000); }
}

/* --- autocomplete (works even before the engine warms up) ---------------- */
let suggTimer = null, suggList = [], suggIdx = -1;
function hideSugg() { $("sugg").classList.remove("open"); suggIdx = -1; }
function renderSugg() {
  const box = $("sugg");
  if (!suggList.length) { hideSugg(); return; }
  const p = $("query").value.trim().toLowerCase();
  box.innerHTML = suggList.map((s, i) =>
    `<div class="sugg-item${i === suggIdx ? " sel" : ""}" data-i="${i}">` +
    (p && s.toLowerCase().startsWith(p)
      ? `<b>${esc(s.slice(0, p.length))}</b>${esc(s.slice(p.length))}` : esc(s)) +
    `</div>`).join("");
  box.classList.add("open");
  box.querySelectorAll(".sugg-item").forEach((el) =>
    el.addEventListener("mousedown", (e) => { e.preventDefault(); pickSugg(Number(el.dataset.i)); }));
}
function pickSugg(i) { $("query").value = suggList[i]; hideSugg(); doSearch(); }
$("query").addEventListener("input", () => {
  clearTimeout(suggTimer);
  const q = $("query").value.trim();
  if (!q) { hideSugg(); return; }
  suggTimer = setTimeout(async () => {
    try {
      const r = await fetch("/api/suggest?q=" + encodeURIComponent(q.slice(0, 30)));
      const d = await r.json();
      suggList = d.suggestions || []; suggIdx = -1; renderSugg();
    } catch (_) { hideSugg(); }
  }, 200);
});
$("query").addEventListener("keydown", (e) => {
  const open = $("sugg").classList.contains("open");
  if (e.key === "Escape") { hideSugg(); return; }
  if (open && e.key === "ArrowDown") { e.preventDefault(); suggIdx = (suggIdx + 1) % suggList.length; renderSugg(); }
  else if (open && e.key === "ArrowUp") { e.preventDefault(); suggIdx = (suggIdx - 1 + suggList.length) % suggList.length; renderSugg(); }
  else if (e.key === "Enter") { if (open && suggIdx >= 0) pickSugg(suggIdx); else doSearch(); }
});
document.addEventListener("click", (e) => { if (!e.target.closest(".search-box")) hideSugg(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "/" && !/input|select|textarea/i.test(document.activeElement.tagName)) {
    e.preventDefault(); $("query").focus();
  }
});

/* --- rendering ------------------------------------------------------------ */
function stars(rating, n) {
  if (rating == null) return "";
  const full = Math.round(rating);
  return "★".repeat(full) + "☆".repeat(5 - full) +
    ` <small>${rating.toFixed(1)}${n != null ? " (" + fmtCount(n) + ")" : ""}</small>`;
}
function bar(label, cls, value) {
  return `<div class="bar-row"><span>${label}</span><div class="bar-track"><div class="bar-fill ${cls}" style="width:${(value * 100).toFixed(1)}%"></div></div></div>`;
}
function priceText(p) {
  if (p.min != null && p.max != null) return inr(p.min_inr) + " – " + inr(p.max_inr);
  if (p.max != null) return "under " + inr(p.max_inr);
  return inr(p.min_inr) + " or more";
}

function render(data) {
  const st = $("status");
  st.className = "status";
  const shown = data.corrected_query || data.query;
  let s = `${data.items.length} picks for “${shown}” in ${data.elapsed_s}s`;
  if (data.price) s += ` · budget ${priceText(data.price)}` +
    (data.price.relaxed ? " (relaxed — most items here have no listed price)" : "");
  st.textContent = s;

  let items = data.items.slice();
  if (state.sort === "price") items.sort((a, b) => (a.price_inr ?? Infinity) - (b.price_inr ?? Infinity));
  else if (state.sort === "rating") items.sort((a, b) => (b.rating ?? -1) - (a.rating ?? -1));

  let bhtml = "";
  if (data.corrections && data.corrections.length) {
    bhtml += `<div class="banner info">✍️ Fixed a typo: ${data.corrections.map((c) =>
      `<b>${esc(c.from)}</b> → ${esc(c.to)}`).join(" · ")} — searched as “${esc(data.corrected_query)}”</div>`;
  }
  if (data.note) {
    if (data.note.fallback) bhtml += `<div class="banner info">${esc(data.note.message || "")}</div>`;
    if (data.note.brand_relaxed) bhtml += `<div class="banner warn">${esc(data.note.brand_relaxed)}</div>`;
    if (data.note.size_relaxed) bhtml += `<div class="banner warn">${esc(data.note.size_relaxed)}</div>`;
    if (data.note.rating_relaxed) bhtml += `<div class="banner warn">${esc(data.note.rating_relaxed)}</div>`;
  }
  $("banners").innerHTML = bhtml;

  $("grid").innerHTML = items.map((it, i) => `
    <div class="card">
      <div class="thumb"><span class="rank">${i + 1}</span>
        ${it.image_url ? `<img src="${esc(it.image_url)}" alt="" onerror="this.style.display='none'">` : ""}
      </div>
      <div class="body">
        <div class="title">${esc(it.title || it.parent_asin)}</div>
        ${it.brand ? `<div class="brand">${esc(capBrand(it.brand))}</div>` : ""}
        <div class="meta">
          <span class="price">${it.price_inr != null ? inr(it.price_inr) : '<span class="nopr">price not listed</span>'}</span>
          <span class="rating">${stars(it.rating, it.n_ratings)}</span>
        </div>
        <div class="bars">
          ${bar("content", "c", it.content_score)}
          ${bar("embedding", "e", it.embed_score)}
          ${bar("sentiment", "s", it.sentiment_score)}
          ${bar("co-buy", "co", it.collab_score)}
        </div>
        <div class="fused-row"><span>hybrid score</span><span class="fused-score">${it.fused_score.toFixed(4)}</span></div>
        ${data.price && data.price.relaxed ? `<div class="tag ${it.in_budget ? "ok" : "no"}">${it.in_budget ? "✓ in budget" : "over budget"}</div>` : ""}
        ${state.brand && it.in_brand === false ? `<div class="tag no">different brand</div>` : ""}
        ${state.size && it.in_size === false ? `<div class="tag no">size not listed</div>` : ""}
        ${state.rating && it.in_rating === false ? `<div class="tag no">below ${state.rating}★</div>` : ""}
        <button class="also-btn" onclick="toggleAlso('${it.parent_asin}', this)">People also bought ▾</button>
        <div class="also-strip" id="also-${it.parent_asin}"></div>
      </div>
    </div>`).join("");
  renderQuick();
}

async function toggleAlso(asin, btn) {
  const strip = $("also-" + asin);
  if (strip.style.display === "block") {
    strip.style.display = "none";
    btn.textContent = "People also bought ▾";
    return;
  }
  btn.textContent = "People also bought ▴";
  if (!strip.dataset.loaded) {
    strip.textContent = "loading...";
    try {
      const res = await fetch("/api/collab/" + asin);
      const data = await res.json();
      strip.innerHTML = data.items.length
        ? data.items.map((p) => `<div class="also-item"><a href="https://www.amazon.com/dp/${esc(p.parent_asin)}" target="_blank">${esc(p.title || p.parent_asin)}</a><small>${p.rating != null ? "★" + p.rating.toFixed(1) + " " : ""}${p.price != null ? inr(p.price * USD_INR) : ""}</small></div>`).join("")
        : "<div class='also-item'>No co-purchase data for this item</div>";
      strip.dataset.loaded = "1";
    } catch (e) { strip.textContent = "failed to load"; }
  }
  strip.style.display = "block";
}

async function attachPab(asin) {
  let strip = $("pab");
  if (!strip) {
    strip = document.createElement("div");
    strip.id = "pab";
    strip.className = "pab";
    $("grid").after(strip);
  }
  if (!asin) { strip.innerHTML = ""; return; }
  try {
    const res = await fetch("/api/collab/" + asin);
    const data = await res.json();
    if (!data.items || !data.items.length) { strip.innerHTML = ""; return; }
    strip.innerHTML = "<h3>People also bought</h3><div class=\"pab-row\">" +
      data.items.map((it) =>
        `<div class="pab-card">` +
        (it.image_url ? `<img src="${esc(it.image_url)}" alt="" onerror="this.style.display='none'">` : "") +
        `<div class="t">${esc((it.title || it.parent_asin).slice(0, 60))}</div>` +
        `<div class="p">${it.price != null ? inr(it.price * USD_INR) : ""}</div>` +
        `</div>`).join("") +
      "</div>";
  } catch (err) { strip.innerHTML = ""; }
}

/* --- quick chips (popular + recent, shown when no results) ---------------- */
const POPULAR = ["black shirt", "white polo", "slim fit tshirt", "linen shirt", "black hoodie", "cotton tshirt"];
const qchip = (q) => `<button class="qchip" data-q="${esc(q)}">${esc(q)}</button>`;
function renderQuick() {
  const box = $("quickChips");
  if ($("grid").children.length) { box.innerHTML = ""; return; }
  const parts = [];
  if (recent.length) parts.push(`<span class="qlabel">Recent:</span>` + recent.map(qchip).join(""));
  parts.push(`<span class="qlabel">Popular:</span>` + POPULAR.map(qchip).join(""));
  box.innerHTML = parts.join("");
  box.querySelectorAll(".qchip").forEach((el) => el.addEventListener("click", () => {
    $("query").value = el.dataset.q; doSearch();
  }));
}
function saveRecent(q) {
  if (!q) return;
  recent = [q, ...recent.filter((x) => x !== q)].slice(0, 5);
  try { localStorage.setItem("ss_recent", JSON.stringify(recent)); } catch (_) {}
  renderQuick();
}

/* --- search ---------------------------------------------------------------- */
function showSkeleton() {
  $("grid").innerHTML = Array.from({ length: 6 }, () =>
    `<div class="sk"><div class="sk-thumb"></div><div class="sk-line" style="width:70%"></div><div class="sk-line" style="width:45%"></div><div class="sk-line" style="width:85%"></div></div>`).join("");
}

async function doSearch() {
  hideSugg();
  const input = $("query").value.trim();
  if (input) { state.baseQuery = input; state.typeKey = ""; }  // fresh text drops the type chip
  const q = composeQuery();
  if (!q) {
    $("status").textContent = "Type what you want (e.g. “black shirt”) or tap a suggestion.";
    return;
  }
  // syncFilterUI (not just the active chips): the filter bar's type chip
  // must lose its highlight too, since state.typeKey was just cleared.
  syncFilterUI();
  const bucket = state.priceKey && facets ? facets.price_buckets.find((b) => b.key === state.priceKey) : null;
  const seq = ++searchSeq;
  const go = $("go");
  go.disabled = true;
  // /health is instant; only warn about the slow first load when it will happen.
  let ready = false;
  try { ready = (await fetch("/health").then((r) => r.json())).ready === true; } catch (_) {}
  if (ready) {
    $("status").className = "status";
    $("status").textContent = "Searching…";
    showSkeleton();
  } else {
    $("status").className = "status big";
    $("status").textContent = "First search loads the models (1-2 min) — please wait…";
    $("progress").classList.add("on");
  }
  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: q,
        top_k: 12,
        w_content: parseFloat($("wContent").value),
        w_embed: parseFloat($("wEmbed").value),
        w_sentiment: parseFloat($("wSentiment").value),
        w_collab: parseFloat($("wCollab").value),
        brand: state.brand || null,
        size: state.size || null,
        min_rating: state.rating || null,
        min_price_inr: bucket ? (bucket.min_inr || null) : null,
        max_price_inr: bucket ? (bucket.max_inr || null) : null,
      }),
    });
    if (!res.ok) {
      let msg = "server error " + res.status;
      try { msg = (await res.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    const data = await res.json();
    if (seq !== searchSeq) return;  // a newer search superseded this one
    if (data.corrected_query) {     // show the user exactly what was searched
      $("query").value = data.corrected_query;
      state.baseQuery = data.corrected_query;
    }
    lastData = data;
    render(data);
    saveRecent(data.corrected_query || q);
    attachPab(data.items && data.items.length ? data.items[0].parent_asin : null);
  } catch (err) {
    if (seq !== searchSeq) return;
    lastData = null;
    $("grid").innerHTML = "";
    $("status").className = "status";
    $("status").textContent = "";
    $("banners").innerHTML =
      `<div class="banner err">Something went wrong: ${esc(err.message)} <button id="retryBtn">Try again</button></div>`;
    $("retryBtn").addEventListener("click", doSearch);
    renderQuick();
  } finally {
    $("progress").classList.remove("on");
    if (seq === searchSeq) go.disabled = false;
  }
}

/* --- weights: sliders + localStorage persistence --------------------------- */
const WEIGHT_IDS = [["wContent", "vContent", "content"], ["wEmbed", "vEmbed", "embed"],
                    ["wSentiment", "vSentiment", "sentiment"], ["wCollab", "vCollab", "collab"]];
function saveWeights() {
  try {
    localStorage.setItem("ss_weights", JSON.stringify({
      content: parseFloat($("wContent").value), embed: parseFloat($("wEmbed").value),
      sentiment: parseFloat($("wSentiment").value), collab: parseFloat($("wCollab").value) }));
  } catch (_) {}
}
function loadWeights() {
  try {
    const w = JSON.parse(localStorage.getItem("ss_weights") || "null");
    if (!w) return;
    WEIGHT_IDS.forEach(([a, b, k]) => {
      if (typeof w[k] === "number" && w[k] >= 0 && w[k] <= 1) {
        $(a).value = w[k]; $(b).textContent = w[k].toFixed(2);
      }
    });
  } catch (_) {}
}

$("go").addEventListener("click", doSearch);
$("sortSel").addEventListener("change", (e) => { state.sort = e.target.value; if (lastData) render(lastData); });
WEIGHT_IDS.forEach(([a, b]) => $(a).addEventListener("input", () => { $(b).textContent = parseFloat($(a).value).toFixed(2); saveWeights(); }));
loadWeights();
loadFacets();
renderQuick();
renderActiveChips();
