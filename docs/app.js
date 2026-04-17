async function fetchJson(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return await r.json();
}

function byId(id) { return document.getElementById(id); }
function setStatus(msg, isError = false) {
  const el = byId("status");
  if (!el) return;
  el.textContent = msg || "";
  el.className = isError ? "status error" : "status";
}
function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return `${(Number(x) * 100).toFixed(digits)}%`;
}
function fmtInt(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x).toLocaleString();
}
function esc(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
function renderTable(id, rows, columns) {
  const el = byId(id);
  if (!el) return;
  if (!rows || !rows.length) {
    el.innerHTML = `<div class="empty">No data</div>`;
    return;
  }
  const thead = `<thead><tr>${columns.map(c => `<th>${esc(c.label)}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => `<tr>${columns.map(c => {
    const raw = r[c.key];
    const value = c.render ? c.render(raw, r) : esc(raw ?? "");
    return `<td>${value}</td>`;
  }).join("")}</tr>`).join("");
  el.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}
function renderLinkGrid(id, items) {
  const el = byId(id);
  if (!el) return;
  el.innerHTML = `<div class="link-grid">${items.map(x => `
    <div class="link-card"><a href="${x.href}">${esc(x.title)}</a></div>
  `).join("")}</div>`;
}
function renderDownloadCards(id, familyMap) {
  const el = byId(id);
  if (!el) return;
  const entries = Object.entries(familyMap || {});
  if (!entries.length) {
    el.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }
  el.innerHTML = `<div class="download-grid">${entries.map(([family, items]) => `
    <div class="download-card">
      <div><strong>${esc(family)}</strong></div>
      <div class="download-inline">${(items || []).map(x => `<a href="${x.path}" target="_blank" rel="noopener">${esc(x.label)}</a>`).join("")}</div>
    </div>
  `).join("")}</div>`;
}
function renderRawTree(id, years) {
  const el = byId(id);
  if (!el) return;
  if (!years || !years.length) {
    el.innerHTML = `<div class="empty">No raw snapshot files</div>`;
    return;
  }
  el.innerHTML = years.map(year => `
    <details class="tree-level">
      <summary>${esc(year.year)}</summary>
      <div class="tree-body">
        <div class="download-inline"><a href="${year.download_all_zip}" target="_blank" rel="noopener">Download all ${esc(year.year)} (.zip)</a></div>
        ${year.months.map(month => `
          <details class="tree-level">
            <summary>${esc(year.year)}-${esc(month.month)}</summary>
            <div class="tree-body">
              <div class="download-inline"><a href="${month.download_all_zip}" target="_blank" rel="noopener">Download all ${esc(month.yyyymm)} (.zip)</a></div>
              ${month.files.map(f => `
                <div class="tree-leaf">
                  <span>${esc(f.date)} | ${esc(f.universe)} | ${fmtInt(f.n_items)} items</span>
                  <a href="${f.download_path}" target="_blank" rel="noopener">Download day</a>
                </div>
              `).join("")}
            </div>
          </details>
        `).join("")}
      </div>
    </details>
  `).join("");
}
function parseDateSafe(s) {
  if (!s) return null;
  const d = new Date(`${s}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}
function filterRecentDaily(rows, days = 31) {
  if (!rows || !rows.length) return [];
  const dated = rows.map(r => ({ ...r, __d: parseDateSafe(r.date) })).filter(r => r.__d !== null);
  if (!dated.length) return rows;
  const maxDate = dated.reduce((a, b) => a.__d > b.__d ? a : b).__d;
  const cutoff = new Date(maxDate);
  cutoff.setDate(cutoff.getDate() - days);
  return dated.filter(r => r.__d >= cutoff).sort((a, b) => b.__d - a.__d).map(({ __d, ...rest }) => rest);
}
function onlyLastThreeMonthly(rows) {
  if (!rows || !rows.length) return [];
  return rows.slice().sort((a, b) => String(b.date).localeCompare(String(a.date))).slice(0, 3);
}
function attachTabs() {
  const btns = Array.from(document.querySelectorAll(".tabbtn"));
  if (!btns.length) return;
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      document.querySelectorAll(".rawtab").forEach(x => x.classList.remove("active"));
      const target = byId(`raw-tab-${tab}`);
      if (target) target.classList.add("active");
    });
  });
}
async function safeLoad(path, label, warnings) {
  try {
    return await fetchJson(path);
  } catch (err) {
    console.error(label, err);
    warnings.push(`${label} failed`);
    return null;
  }
}

async function renderOverviewPage(warnings) {
  const data = await safeLoad("./data/overview.json", "overview.json", warnings);
  if (!data) return;
  if (byId("asof")) byId("asof").textContent = data.asof_date ? `As of ${data.asof_date}` : "";
  renderTable("overview-table", data.rows || [], [
    { key: "family", label: "Family" },
    { key: "latest_daily", label: "Latest Daily", render: x => fmtPct(x) },
    { key: "m1_date", label: "Month 1" },
    { key: "m1_value", label: "Value 1", render: x => fmtPct(x) },
    { key: "m2_date", label: "Month 2" },
    { key: "m2_value", label: "Value 2", render: x => fmtPct(x) },
    { key: "m3_date", label: "Month 3" },
    { key: "m3_value", label: "Value 3", render: x => fmtPct(x) },
  ]);
  renderLinkGrid("overview-links", [
    { title: "Marketwide", href: "./marketwide.html" },
    { title: "Value Premium", href: "./value.html" },
    { title: "Industry", href: "./industry.html" },
    { title: "ETF", href: "./etf.html" },
    { title: "Country", href: "./country.html" },
    { title: "Indices", href: "./indices.html" },
    { title: "Downloads", href: "./downloads.html" },
  ]);
}

async function renderFamilyPage(page, warnings) {
  const data = await safeLoad(`./data/${page}.json`, `${page}.json`, warnings);
  if (!data) return;
  if (byId("asof")) byId("asof").textContent = data.asof_date ? `As of ${data.asof_date}` : "";

  const latestCols = {
    marketwide: [
      { key: "family", label: "Family" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "ew_icc", label: "EW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "N Items", render: x => fmtInt(x) },
    ],
    value: [
      { key: "bucket", label: "Bucket" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "N Items", render: x => fmtInt(x) },
    ],
    industry: [
      { key: "sector", label: "Sector" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "N Items", render: x => fmtInt(x) },
    ],
    etf: [
      { key: "ticker", label: "Ticker" },
      { key: "label", label: "Label" },
      { key: "category", label: "Category" },
      { key: "vw_icc", label: "ETF ICC", render: x => fmtPct(x) },
      { key: "source", label: "Source" },
      { key: "status", label: "Status" },
    ],
    country: [
      { key: "country", label: "Country" },
      { key: "ticker", label: "Ticker" },
      { key: "label", label: "Label" },
      { key: "vw_icc", label: "Country ICC", render: x => fmtPct(x) },
      { key: "source", label: "Source" },
      { key: "status", label: "Status" },
    ],
    indices: [
      { key: "universe", label: "Index" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "ew_icc", label: "EW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "N Items", render: x => fmtInt(x) },
    ],
  };
  const dailyCols = {
    marketwide: [
      { key: "date", label: "Date" },
      { key: "family", label: "Family" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "ew_icc", label: "EW ICC", render: x => fmtPct(x) },
    ],
    value: [
      { key: "date", label: "Date" },
      { key: "value_icc", label: "Value ICC", render: x => fmtPct(x) },
      { key: "growth_icc", label: "Growth ICC", render: x => fmtPct(x) },
      { key: "ivp", label: "IVP", render: x => fmtPct(x) },
    ],
    industry: [
      { key: "date", label: "Date" },
      { key: "summary_icc", label: "Summary ICC", render: x => fmtPct(x) },
      { key: "n_groups", label: "N Groups", render: x => fmtInt(x) },
    ],
    etf: [
      { key: "date", label: "Date" },
      { key: "ticker", label: "Ticker" },
      { key: "vw_icc", label: "ETF ICC", render: x => fmtPct(x) },
      { key: "status", label: "Status" },
    ],
    country: [
      { key: "date", label: "Date" },
      { key: "country", label: "Country" },
      { key: "vw_icc", label: "Country ICC", render: x => fmtPct(x) },
      { key: "status", label: "Status" },
    ],
    indices: [
      { key: "date", label: "Date" },
      { key: "universe", label: "Index" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
    ],
  };
  const monthlyCols = {
    marketwide: [
      { key: "date", label: "Date" },
      { key: "family", label: "Family" },
      { key: "vw_icc", label: "Monthly ICC", render: x => fmtPct(x) },
    ],
    value: [
      { key: "date", label: "Date" },
      { key: "ivp", label: "Monthly IVP", render: x => fmtPct(x) },
    ],
    industry: [
      { key: "date", label: "Date" },
      { key: "summary_icc", label: "Monthly Summary", render: x => fmtPct(x) },
    ],
    etf: [
      { key: "date", label: "Date" },
      { key: "ticker", label: "Ticker" },
      { key: "vw_icc", label: "Monthly ICC", render: x => fmtPct(x) },
    ],
    country: [
      { key: "date", label: "Date" },
      { key: "ticker", label: "Ticker" },
      { key: "vw_icc", label: "Monthly ICC", render: x => fmtPct(x) },
    ],
    indices: [
      { key: "date", label: "Date" },
      { key: "universe", label: "Index" },
      { key: "vw_icc", label: "Monthly ICC", render: x => fmtPct(x) },
    ],
  };

  renderTable("latest-table", data.latest || [], latestCols[page]);
  renderTable("daily-table", filterRecentDaily(data.daily || []), dailyCols[page]);
  renderTable("monthly-table", onlyLastThreeMonthly(data.monthly || []), monthlyCols[page]);
  renderDownloadCards("family-downloads", { [page]: data.downloads || [] });
}

async function renderDownloadsPage(warnings) {
  const overview = await safeLoad("./data/overview.json", "overview.json", warnings);
  const catalog = await safeLoad("./data/downloads_catalog.json", "downloads_catalog.json", warnings);
  if (overview && byId("asof")) byId("asof").textContent = overview.asof_date ? `As of ${overview.asof_date}` : "";
  if (!catalog) return;
  renderDownloadCards("category-downloads", catalog.families || {});
  attachTabs();
  const rawTabs = catalog.raw_tabs || {};
  renderRawTree("raw-tab-usall", rawTabs.usall || []);
  renderRawTree("raw-tab-sp500", rawTabs.sp500 || []);
  renderRawTree("raw-tab-other_indices", rawTabs.other_indices || []);
}

(async function main() {
  const page = document.body.dataset.page;
  const warnings = [];
  if (page === "overview") {
    await renderOverviewPage(warnings);
  } else if (page === "downloads") {
    await renderDownloadsPage(warnings);
  } else {
    await renderFamilyPage(page, warnings);
  }
  if (warnings.length) setStatus(`Partial load: ${warnings.join(" | ")}`, true);
  else setStatus("");
})();
