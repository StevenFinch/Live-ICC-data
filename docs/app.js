
async function fetchJson(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return await r.json();
}

function el(id) { return document.getElementById(id); }
function text(id, value) { const node = el(id); if (node) node.textContent = value ?? ""; }

function status(msg, isError = false) {
  const node = el("status");
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

function renderCards(id, items) {
  const node = el(id);
  if (!node) return;
  node.innerHTML = items.map(it => `
    <div class="card">
      <div class="card-label">${it.label}</div>
      <div class="card-value">${it.value}</div>
    </div>
  `).join("");
}

function renderTable(id, rows, columns) {
  const node = el(id);
  if (!node) return;
  if (!rows || !rows.length) {
    node.innerHTML = `<div class="empty">No data</div>`;
    return;
  }
  const thead = `<thead><tr>${columns.map(c => `<th>${c.label}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => `
    <tr>
      ${columns.map(c => {
        const raw = r[c.key];
        const out = c.render ? c.render(raw, r) : (raw ?? "");
        return `<td>${out}</td>`;
      }).join("")}
    </tr>
  `).join("");
  node.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

function parseDateSafe(s) {
  if (!s) return null;
  const d = new Date(`${s}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function filterLastDays(rows, days = 31) {
  if (!rows || !rows.length) return [];
  const dated = rows.map(r => ({ ...r, __d: parseDateSafe(r.date) })).filter(r => r.__d);
  if (!dated.length) return rows;
  const maxD = dated.reduce((a, b) => a.__d > b.__d ? a : b).__d;
  const cutoff = new Date(maxD);
  cutoff.setDate(cutoff.getDate() - days);
  return dated.filter(r => r.__d >= cutoff).sort((a, b) => b.__d - a.__d).map(({__d, ...rest}) => rest);
}

function renderMonthlySummary(id, payload) {
  const node = el(id);
  if (!node) return;
  if (!payload || !payload.columns || !payload.rows || !payload.rows.length) {
    node.innerHTML = `<div class="empty">No monthly summary</div>`;
    return;
  }
  const thead = `<thead><tr><th>Series</th>${payload.columns.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
  const tbody = payload.rows.map(r => {
    return `<tr><td>${r.label}</td>${(r.values || []).map(v => `<td>${fmtPct(v)}</td>`).join("")}</tr>`;
  }).join("");
  node.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

function renderDownloadGroups(id, groups) {
  const node = el(id);
  if (!node) return;
  const families = Object.keys(groups || {});
  if (!families.length) {
    node.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }
  node.innerHTML = families.map(fam => `
    <div class="download-group">
      <div class="download-group-title">${fam.replaceAll("_", " ")}</div>
      <div class="download-links">
        ${(groups[fam] || []).map(x => `<a class="pill-link" href="${x.path}" target="_blank" rel="noopener">${x.label}</a>`).join("")}
      </div>
    </div>
  `).join("");
}

function renderDownloadTree(id, families) {
  const node = el(id);
  if (!node) return;
  if (!families || !Object.keys(families).length) {
    node.innerHTML = `<div class="empty">No daily download tree</div>`;
    return;
  }
  const order = ["marketwide", "value", "industry", "etf", "country", "indices"];
  const keys = order.filter(k => families[k]).concat(Object.keys(families).filter(k => !order.includes(k)));
  node.innerHTML = keys.map(k => {
    const fam = families[k];
    const years = fam.years || [];
    return `
      <details class="tree-family" open>
        <summary>${k.replaceAll("_", " ")}</summary>
        <div class="tree-body">
          ${years.map(y => `
            <details class="tree-year">
              <summary>
                <span>${y.year}</span>
                <a class="pill-link" href="${y.download_all}" target="_blank" rel="noopener">Download all ${y.year}</a>
              </summary>
              <div class="tree-body nested">
                ${(y.months || []).map(m => `
                  <details class="tree-month">
                    <summary>
                      <span>${y.year}-${m.month}</span>
                      <a class="pill-link" href="${m.download_all}" target="_blank" rel="noopener">Download all ${m.month}</a>
                    </summary>
                    <div class="tree-body nested">
                      ${(m.days || []).map(d => `
                        <div class="tree-leaf">
                          <span>${d.date}</span>
                          <a class="pill-link" href="${d.download_path}" target="_blank" rel="noopener">${d.label}</a>
                        </div>
                      `).join("")}
                    </div>
                  </details>
                `).join("")}
              </div>
            </details>
          `).join("")}
        </div>
      </details>
    `;
  }).join("");
}

async function safeLoad(path, label, warnings) {
  try { return await fetchJson(path); }
  catch (err) { console.error(label, err); warnings.push(`${label} failed`); return null; }
}

async function renderOverviewPage(warnings) {
  const data = await safeLoad("./data/overview.json", "overview", warnings);
  if (!data) return;
  text("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  const d = data.latest_daily || {};
  renderCards("daily-cards", [
    { label: "All U.S. market", value: fmtPct(d.all_market_vw_icc) },
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
  renderDownloadTree("download-tree", data.families || {});
}

(async function main() {
  const warnings = [];
  const page = document.body.dataset.page || "overview";
  if (page === "overview") await renderOverviewPage(warnings);
  else if (page === "marketwide") await renderMarketwidePage(warnings);
  else if (page === "value") await renderValuePage(warnings);
  else if (page === "industry") {
    await renderFamilyOverviewPage(
      "./data/industry.json", "industry-latest-table", "industry-monthly", "industry-daily-table", warnings,
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
      ],
    );
  } else if (page === "etf") {
    await renderFamilyOverviewPage(
      "./data/etf.json", "etf-latest-table", "etf-monthly", "etf-daily-table", warnings,
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
      ],
    );
  } else if (page === "country") {
    await renderFamilyOverviewPage(
      "./data/country.json", "country-latest-table", "country-monthly", "country-daily-table", warnings,
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
      ],
    );
  } else if (page === "indices") await renderIndicesPage(warnings);
  else if (page === "downloads") await renderDownloadsPage(warnings);

  if (warnings.length) status(`Partial load: ${warnings.join(" | ")}`, true);
  else status("");
})();
