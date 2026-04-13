
async function fetchJson(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return await r.json();
}

function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x).toFixed(digits);
}

function byId(id) {
  return document.getElementById(id);
}

function setStatus(msg, isError = false) {
  const el = byId("status");
  if (!el) return;
  el.textContent = msg || "";
  el.className = isError ? "status error" : "status";
}

function makeCards(cards) {
  const el = byId("cards");
  const items = [
    ["Market VW ICC", cards.market_vw_icc],
    ["Value ICC", cards.value_icc],
    ["Growth ICC", cards.growth_icc],
    ["IVP (B/M)", cards.ivp_bm],
  ];
  el.innerHTML = items
    .map(([label, value]) => `<div class="card"><div class="card-label">${label}</div><div class="card-value">${fmt(value)}</div></div>`)
    .join("");
}

function renderTable(id, rows, columns) {
  const el = byId(id);
  if (!el) return;
  if (!rows || !rows.length) {
    el.innerHTML = `<div class="empty">No data</div>`;
    return;
  }
  const thead = `<thead><tr>${columns.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => `<tr>${columns.map(c => {
    const v = r[c[0]];
    return `<td>${typeof c[2] === "function" ? c[2](v, r) : (v ?? "")}</td>`;
  }).join("")}</tr>`).join("");
  el.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

function renderDownloads(items) {
  const el = byId("downloads");
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }
  el.innerHTML = `<ul class="download-list">${items.map(x => `<li><a href="${x.path}" target="_blank" rel="noopener">${x.label}</a></li>`).join("")}</ul>`;
}

(async function () {
  try {
    const overview = await fetchJson("./data/overview.json");
    byId("asof").textContent = overview.asof_date ? `As of ${overview.asof_date}` : "";
    makeCards(overview.cards || {});
    renderDownloads(overview.downloads || []);

    const market = await fetchJson("./data/market_icc.json");
    renderTable("market-table", (market.history || []).slice().reverse(), [
      ["date", "Date"],
      ["vw_icc", "VW ICC", fmt],
      ["ew_icc", "EW ICC", fmt],
      ["n_firms", "N Firms"],
    ]);

    const value = await fetchJson("./data/value_icc_bm.json");
    renderTable("value-table", (value.history || []).slice().reverse(), [
      ["date", "Date"],
      ["value_icc", "Value ICC", fmt],
      ["growth_icc", "Growth ICC", fmt],
      ["ivp_bm", "IVP (B/M)", fmt],
      ["n_firms", "N Firms"],
    ]);

    const industry = await fetchJson("./data/industry_icc.json");
    renderTable("industry-table", industry.latest || [], [
      ["sector", "Industry"],
      ["vw_icc", "VW ICC", fmt],
      ["ew_icc", "EW ICC", fmt],
      ["n_firms", "N Firms"],
    ]);

    const etf = await fetchJson("./data/etf_icc.json");
    renderTable("etf-table", etf.latest || [], [
      ["ticker", "Ticker"],
      ["label", "Label"],
      ["category", "Category"],
      ["vw_icc", "ETF ICC", fmt],
      ["coverage_weight", "Coverage", x => fmt(x, 3)],
      ["n_holdings", "Holdings"],
      ["status", "Status"],
    ]);

    const country = await fetchJson("./data/country_icc.json");
    renderTable("country-table", country.latest || [], [
      ["country", "Country"],
      ["ticker", "Proxy ETF"],
      ["vw_icc", "Country ICC", fmt],
      ["coverage_weight", "Coverage", x => fmt(x, 3)],
      ["n_holdings", "Holdings"],
      ["status", "Status"],
    ]);

    const indexData = await fetchJson("./data/index_icc.json");
    renderTable("index-latest-table", indexData.latest || [], [
      ["universe", "Index"],
      ["date", "Date"],
      ["vw_icc", "VW ICC", fmt],
      ["ew_icc", "EW ICC", fmt],
      ["n_firms", "N Firms"],
    ]);

    const ym = await fetchJson("./data/year_month_manifest.json");
    renderTable("month-table", ym.rows || [], [
      ["yyyymm", "Year-Month"],
      ["n_files", "Files"],
      ["universes", "Universes"],
      ["download_folder", "Folder", x => `<a href="${x}" target="_blank" rel="noopener">${x}</a>`],
    ]);

    const allf = await fetchJson("./data/all_snapshot_files.json");
    renderTable("all-files-table", (allf.rows || []).slice().reverse(), [
      ["date", "Date"],
      ["universe", "Universe"],
      ["n_firms", "N Firms"],
      ["download_path", "Download", x => `<a href="${x}" target="_blank" rel="noopener">csv</a>`],
    ]);

    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus("Failed to load site data. Check docs/data/*.json deployment.", true);
  }
})();
