
async function fetchJson(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return await r.json();
}
function byId(id) { return document.getElementById(id); }
function text(id, value) { const node = byId(id); if (node) node.textContent = value ?? ""; }
function status(msg, isError = false) {
  const node = byId("status");
  if (!node) return;
  node.textContent = msg || "";
  node.className = isError ? "status error" : "status";
}
function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return `${(Number(x) * 100).toFixed(digits)}%`;
}
function fmtInt(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x).toLocaleString();
}
function parseDateSafe(s) {
  if (!s) return null;
  const d = new Date(`${s}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}
function filterLastDays(rows, nDays = 31) {
  if (!rows || !rows.length) return [];
  const dated = rows.map(r => ({...r, __d: parseDateSafe(r.date)})).filter(r => r.__d);
  if (!dated.length) return rows;
  const maxDate = dated.reduce((a,b)=>a.__d > b.__d ? a : b).__d;
  const cutoff = new Date(maxDate);
  cutoff.setDate(cutoff.getDate() - nDays);
  return dated.filter(r => r.__d >= cutoff).sort((a,b)=>b.__d-a.__d).map(({__d,...rest})=>rest);
}
function renderCards(id, items) {
  const node = byId(id);
  if (!node) return;
  node.innerHTML = items.map(it => `<div class="card"><div class="card-label">${it.label}</div><div class="card-value">${it.value}</div></div>`).join("");
}
function renderTable(id, rows, columns) {
  const node = byId(id);
  if (!node) return;
  if (!rows || !rows.length) {
    node.innerHTML = `<div class="empty">No data</div>`;
    return;
  }
  const thead = `<thead><tr>${columns.map(c => `<th>${c.label}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => `<tr>${columns.map(c => {
    const v = r[c.key];
    return `<td>${c.render ? c.render(v, r) : (v ?? "")}</td>`;
  }).join("")}</tr>`).join("");
  node.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}
function renderMonthlySummary(id, block) {
  const node = byId(id);
  if (!node) return;
  if (!block || !block.columns || !block.rows || !block.rows.length) {
    node.innerHTML = `<div class="empty">No monthly data</div>`;
    return;
  }
  const cols = block.columns;
  const thead = `<thead><tr><th>Series</th>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
  const tbody = block.rows.map(r => {
    return `<tr><td>${r.label}</td>${(r.values || []).map(v => `<td>${fmtPct(v)}</td>`).join("")}</tr>`;
  }).join("");
  node.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}
function renderDownloadGroups(id, groups) {
  const node = byId(id);
  if (!node) return;
  const entries = Object.entries(groups || {});
  if (!entries.length) {
    node.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }
  node.innerHTML = entries.map(([family, items]) => `
    <div class="download-card">
      <h3>${family}</h3>
      <ul class="download-list">
        ${(items || []).map(x => `<li><a class="btn-link" href="${x.path}" target="_blank" rel="noopener">${x.label}</a></li>`).join("")}
      </ul>
    </div>
  `).join("");
}
function groupTreeRows(rows) {
  const out = {};
  for (const r of rows || []) {
    const yyyymm = String(r.yyyymm || "");
    const year = yyyymm.slice(0, 4);
    const month = yyyymm.slice(4, 6);
    if (!year || !month) continue;
    if (!out[year]) out[year] = {};
    if (!out[year][month]) out[year][month] = [];
    out[year][month].push(r);
  }
  return out;
}
function buildTreeHtml(rows) {
  const tree = groupTreeRows(rows);
  const years = Object.keys(tree).sort().reverse();
  if (!years.length) return `<div class="empty">No raw snapshot files</div>`;
  return years.map(year => {
    const months = Object.keys(tree[year]).sort().reverse();
    const yearDownload = rows.length && rows[0].year_download ? rows[0].year_download[year] : null;
    return `<details class="tree-level">
      <summary><strong>${year}</strong>${yearDownload ? ` <a class="btn-link small" href="${yearDownload}" target="_blank" rel="noopener">Download all</a>` : ""}</summary>
      <div class="tree-body">
        ${months.map(month => {
          const monthRows = tree[year][month].slice().sort((a,b)=>String(b.date).localeCompare(String(a.date)));
          const monthDownload = monthRows[0] && monthRows[0].month_download ? monthRows[0].month_download : null;
          return `<details class="tree-level nested">
            <summary>${year}-${month}${monthDownload ? ` <a class="btn-link small" href="${monthDownload}" target="_blank" rel="noopener">Download all</a>` : ""}</summary>
            <div class="tree-body">
              ${monthRows.map(r => `<div class="tree-leaf"><span>${r.date} | ${r.universe} | ${fmtInt(r.n_firms)} firms</span><a class="btn-link small" href="${r.download_path}" target="_blank" rel="noopener">Download</a></div>`).join("")}
            </div>
          </details>`;
        }).join("")}
      </div>
    </details>`;
  }).join("");
}
function activateRawTabs(families) {
  const treeNode = byId("download-tree");
  if (!treeNode) return;
  const btns = Array.from(document.querySelectorAll("[data-raw-tab]"));
  function render(tab) {
    btns.forEach(b => b.classList.toggle("active", b.dataset.rawTab === tab));
    treeNode.innerHTML = buildTreeHtml((families || {})[tab] || []);
  }
  btns.forEach(btn => btn.addEventListener("click", () => render(btn.dataset.rawTab)));
  render("usall");
}
async function safeLoad(path, label, warnings) {
  try {
    return await fetchJson(path);
  } catch (err) {
    console.error(`[${label}]`, err);
    warnings.push(`${label} failed`);
    return null;
  }
}
async function renderOverviewPage(warnings) {
  const data = await safeLoad("./data/overview.json", "overview", warnings);
  if (!data) return;
  text("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  const d = data.latest_daily || {};
  renderCards("overview-cards", [
    { label: "All market", value: fmtPct(d.all_market_vw_icc) },
    { label: "S&P 500", value: fmtPct(d.sp500_vw_icc) },
    { label: "Value ICC", value: fmtPct(d.value_icc) },
    { label: "Growth ICC", value: fmtPct(d.growth_icc) },
    { label: "IVP", value: fmtPct(d.ivp_bm) },
    { label: "Industry median", value: fmtPct(d.industry_median_icc) },
    { label: "ETF median", value: fmtPct(d.etf_median_icc) },
    { label: "Country median", value: fmtPct(d.country_median_icc) },
  ]);
  renderMonthlySummary("overview-monthly", data.three_months);
  renderDownloadGroups("overview-downloads", data.downloads);
}
async function renderMarketwidePage(warnings) {
  const data = await safeLoad("./data/marketwide.json", "marketwide", warnings);
  if (!data) return;
  text("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  renderMonthlySummary("marketwide-monthly", data.three_months);
  renderTable("marketwide-all-table", filterLastDays(data.all_history || []), [
    { key: "date", label: "Date" },
    { key: "vw_icc", label: "VW ICC", render: v => fmtPct(v) },
    { key: "ew_icc", label: "EW ICC", render: v => fmtPct(v) },
    { key: "n_firms", label: "N Firms", render: v => fmtInt(v) },
  ]);
  renderTable("marketwide-sp500-table", filterLastDays(data.sp500_history || []), [
    { key: "date", label: "Date" },
    { key: "vw_icc", label: "VW ICC", render: v => fmtPct(v) },
    { key: "ew_icc", label: "EW ICC", render: v => fmtPct(v) },
    { key: "n_firms", label: "N Firms", render: v => fmtInt(v) },
  ]);
}
async function renderValuePage(warnings) {
  const data = await safeLoad("./data/value.json", "value", warnings);
  if (!data) return;
  text("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  renderMonthlySummary("value-monthly", data.three_months);
  renderTable("value-daily-table", filterLastDays(data.history || []), [
    { key: "date", label: "Date" },
    { key: "value_icc", label: "Value ICC", render: v => fmtPct(v) },
    { key: "growth_icc", label: "Growth ICC", render: v => fmtPct(v) },
    { key: "ivp_bm", label: "IVP", render: v => fmtPct(v) },
    { key: "n_firms", label: "N Firms", render: v => fmtInt(v) },
  ]);
}
async function renderFamilyOverviewPage(jsonPath, latestTableId, monthlyId, dailyId, warnings, latestCols, dailyCols) {
  const data = await safeLoad(jsonPath, jsonPath, warnings);
  if (!data) return;
  text("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  renderMonthlySummary(monthlyId, data.three_months);
  renderTable(latestTableId, data.latest || [], latestCols);
  renderTable(dailyId, filterLastDays(data.overview_history || []), dailyCols);
}
async function renderIndicesPage(warnings) {
  const data = await safeLoad("./data/indices.json", "indices", warnings);
  if (!data) return;
  text("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  const latest = (data.latest || []).filter(x => x.vw_icc !== null || x.ew_icc !== null);
  renderTable("indices-latest-table", latest, [
    { key: "universe", label: "Index" },
    { key: "date", label: "Date" },
    { key: "vw_icc", label: "VW ICC", render: v => fmtPct(v) },
    { key: "ew_icc", label: "EW ICC", render: v => fmtPct(v) },
    { key: "n_firms", label: "N Firms", render: v => fmtInt(v) },
  ]);
  renderTable("indices-history-table", filterLastDays(data.history || []), [
    { key: "date", label: "Date" },
    { key: "universe", label: "Index" },
    { key: "vw_icc", label: "VW ICC", render: v => fmtPct(v) },
    { key: "ew_icc", label: "EW ICC", render: v => fmtPct(v) },
    { key: "n_firms", label: "N Firms", render: v => fmtInt(v) },
  ]);
}
async function renderDownloadsPage(warnings) {
  const data = await safeLoad("./data/downloads_catalog.json", "downloads_catalog", warnings);
  if (!data) return;
  text("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  renderDownloadGroups("aggregate-downloads", data.aggregate || {});
  activateRawTabs(data.families || {});
}
(async function main() {
  const warnings = [];
  const page = document.body.dataset.page || "overview";
  if (page === "overview") await renderOverviewPage(warnings);
  else if (page === "marketwide") await renderMarketwidePage(warnings);
  else if (page === "value") await renderValuePage(warnings);
  else if (page === "industry") {
    await renderFamilyOverviewPage(
      "./data/industry.json",
      "industry-latest-table",
      "industry-monthly",
      "industry-daily-table",
      warnings,
      [
        { key: "sector", label: "Industry" },
        { key: "vw_icc", label: "VW ICC", render: v => fmtPct(v) },
        { key: "ew_icc", label: "EW ICC", render: v => fmtPct(v) },
        { key: "n_firms", label: "N Firms", render: v => fmtInt(v) },
      ],
      [
        { key: "date", label: "Date" },
        { key: "median_icc", label: "Median ICC", render: v => fmtPct(v) },
        { key: "mean_icc", label: "Mean ICC", render: v => fmtPct(v) },
        { key: "n_groups", label: "Groups", render: v => fmtInt(v) },
      ]
    );
  } else if (page === "etf") {
    await renderFamilyOverviewPage(
      "./data/etf.json",
      "etf-latest-table",
      "etf-monthly",
      "etf-daily-table",
      warnings,
      [
        { key: "ticker", label: "ETF" },
        { key: "label", label: "Label" },
        { key: "vw_icc", label: "ETF ICC", render: v => fmtPct(v) },
        { key: "coverage_weight", label: "Coverage", render: v => fmtPct(v) },
        { key: "status", label: "Status" },
      ],
      [
        { key: "date", label: "Date" },
        { key: "median_icc", label: "Median ICC", render: v => fmtPct(v) },
        { key: "mean_icc", label: "Mean ICC", render: v => fmtPct(v) },
        { key: "n_groups", label: "Groups", render: v => fmtInt(v) },
      ]
    );
  } else if (page === "country") {
    await renderFamilyOverviewPage(
      "./data/country.json",
      "country-latest-table",
      "country-monthly",
      "country-daily-table",
      warnings,
      [
        { key: "country", label: "Country" },
        { key: "ticker", label: "Proxy ETF" },
        { key: "vw_icc", label: "Country ICC", render: v => fmtPct(v) },
        { key: "coverage_weight", label: "Coverage", render: v => fmtPct(v) },
        { key: "status", label: "Status" },
      ],
      [
        { key: "date", label: "Date" },
        { key: "median_icc", label: "Median ICC", render: v => fmtPct(v) },
        { key: "mean_icc", label: "Mean ICC", render: v => fmtPct(v) },
        { key: "n_groups", label: "Groups", render: v => fmtInt(v) },
      ]
    );
  } else if (page === "indices") await renderIndicesPage(warnings);
  else if (page === "downloads") await renderDownloadsPage(warnings);
  if (warnings.length) status(`Partial load: ${warnings.join(" | ")}`, true);
  else status("");
})();
