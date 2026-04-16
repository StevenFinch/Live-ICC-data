async function fetchJson(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return await r.json();
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

function fmtPct(x, digits = 2) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return `${(Number(x) * 100).toFixed(digits)}%`;
}

function fmtInt(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x).toLocaleString();
}

function escapeHtml(s) {
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
  const thead = `<thead><tr>${columns.map(c => `<th>${c.label}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => `<tr>${columns.map(c => `<td>${c.render ? c.render(r[c.key], r) : escapeHtml(r[c.key] ?? "")}</td>`).join("")}</tr>`).join("");
  el.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

function renderCardsGrid(id, items) {
  const el = byId(id);
  if (!el) return;
  el.innerHTML = items.map(x => `
    <a class="cardlink" href="${x.href}">
      <div class="cardlink-title">${escapeHtml(x.title)}</div>
      <div class="cardlink-text">${escapeHtml(x.text)}</div>
    </a>
  `).join("");
}

function renderDownloadsList(id, items) {
  const el = byId(id);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }
  el.innerHTML = `<div class="download-list">${items.map(x => `
    <a class="btn-link" href="${x.path}" target="_blank" rel="noopener">${escapeHtml(x.label)}</a>
  `).join("")}</div>`;
}

function parseDateSafe(s) {
  if (!s) return null;
  const d = new Date(`${s}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function latestThreeByFamily(rows) {
  const byFamily = {};
  for (const r of rows || []) {
    const fam = String(r.family || "");
    if (!byFamily[fam]) byFamily[fam] = [];
    byFamily[fam].push(r);
  }
  for (const k of Object.keys(byFamily)) {
    byFamily[k] = byFamily[k]
      .slice()
      .sort((a, b) => String(b.date).localeCompare(String(a.date)))
      .slice(0, 3);
  }
  return byFamily;
}

function filterRecentDaily(rows, days = 31) {
  if (!rows || !rows.length) return [];
  const dated = rows.map(r => ({ ...r, __d: parseDateSafe(r.date) })).filter(r => r.__d !== null);
  if (!dated.length) return rows;
  const maxDate = dated.reduce((a, b) => (a.__d > b.__d ? a : b)).__d;
  const cutoff = new Date(maxDate);
  cutoff.setDate(cutoff.getDate() - days);
  return dated.filter(r => r.__d >= cutoff).sort((a, b) => b.__d - a.__d).map(({ __d, ...rest }) => rest);
}

function buildTreeByYearMonth(rows) {
  const tree = {};
  for (const r of rows || []) {
    const ym = String(r.yyyymm || "");
    const year = ym.slice(0, 4);
    const month = ym.slice(4, 6);
    if (!tree[year]) tree[year] = {};
    if (!tree[year][month]) tree[year][month] = [];
    tree[year][month].push(r);
  }
  return tree;
}

function renderRawTree(id, rows) {
  const el = byId(id);
  if (!el) return;
  if (!rows || !rows.length) {
    el.innerHTML = `<div class="empty">No raw snapshot files</div>`;
    return;
  }
  const tree = buildTreeByYearMonth(rows);
  const years = Object.keys(tree).sort().reverse();

  el.innerHTML = years.map(year => {
    const months = Object.keys(tree[year]).sort().reverse();
    const yearFiles = months.flatMap(m => tree[year][m]);
    return `
      <details class="tree-level">
        <summary>
          <span>${year}</span>
          <a class="btn-link small" href="#" onclick="downloadGroup(${JSON.stringify(yearFiles).replaceAll('"', '&quot;')}); return false;">Download all</a>
        </summary>
        <div class="tree-body">
          ${months.map(month => {
            const monthFiles = tree[year][month].slice().sort((a,b)=>String(b.date).localeCompare(String(a.date)));
            return `
              <details class="tree-level nested">
                <summary>
                  <span>${year}-${month}</span>
                  <a class="btn-link small" href="#" onclick="downloadGroup(${JSON.stringify(monthFiles).replaceAll('"', '&quot;')}); return false;">Download all</a>
                </summary>
                <div class="tree-body">
                  ${monthFiles.map(r => `
                    <div class="tree-leaf">
                      <span>${escapeHtml(r.date)} | ${escapeHtml(r.universe)} | ${fmtInt(r.n_items)} items</span>
                      <a class="btn-link small" href="${r.download_path}" target="_blank" rel="noopener">Download</a>
                    </div>
                  `).join("")}
                </div>
              </details>
            `;
          }).join("")}
        </div>
      </details>
    `;
  }).join("");
}

window.downloadGroup = function(files) {
  for (const f of files) {
    window.open(f.download_path, "_blank", "noopener");
  }
};

async function safeLoad(path, label, warnings) {
  try {
    return await fetchJson(path);
  } catch (err) {
    console.error(`[${label}]`, err);
    warnings.push(`${label} failed`);
    return null;
  }
}

function attachTabBehavior() {
  const btns = Array.from(document.querySelectorAll(".tabbtn"));
  if (!btns.length) return;
  btns.forEach(btn => {
    btn.addEventListener("click", () => {
      btns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const tab = btn.dataset.tab;
      document.querySelectorAll(".rawtab").forEach(x => x.classList.remove("active"));
      const target = byId(`raw-tab-${tab}`);
      if (target) target.classList.add("active");
    });
  });
}

async function renderOverview(warnings) {
  const data = await safeLoad("./data/overview.json", "overview.json", warnings);
  if (!data) return;
  byId("asof") && (byId("asof").textContent = data.asof_date ? `As of ${data.asof_date}` : "");
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
  renderCardsGrid("overview-links", [
    { title: "Marketwide", text: "All U.S. and S&P 500 history and downloads", href: "./marketwide.html" },
    { title: "Value Premium", text: "Value, Growth, and IVP series", href: "./value.html" },
    { title: "Industry", text: "Sector-level history and latest cross-section", href: "./industry.html" },
    { title: "ETF", text: "Online holdings-based ETF ICC", href: "./etf.html" },
    { title: "Country", text: "Country ETF proxy ICC", href: "./country.html" },
    { title: "Indices", text: "Additional index ICC series", href: "./indices.html" },
    { title: "Downloads", text: "Family-specific and raw snapshot downloads", href: "./downloads.html" },
  ]);
}

async function renderFamilyPage(page, warnings) {
  const data = await safeLoad(`./data/${page}.json`, `${page}.json`, warnings);
  const catalog = await safeLoad("./data/downloads_catalog.json", "downloads_catalog.json", warnings);
  if (!data) return;
  byId("asof") && (byId("asof").textContent = data.asof_date ? `As of ${data.asof_date}` : "");

  if (page === "marketwide") {
    renderTable("marketwide-latest", data.latest || [], [
      { key: "family", label: "Family" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "ew_icc", label: "EW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Items", render: x => fmtInt(x) },
    ]);
    renderTable("marketwide-monthly", data.monthly || [], [
      { key: "family", label: "Family" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "ew_icc", label: "EW ICC", render: x => fmtPct(x) },
    ]);
    renderTable("marketwide-daily", filterRecentDaily(data.daily || []), [
      { key: "family", label: "Family" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "ew_icc", label: "EW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Items", render: x => fmtInt(x) },
    ]);
  }

  if (page === "value") {
    renderTable("value-latest", data.latest || [], [
      { key: "date", label: "Date" },
      { key: "value_icc", label: "Value ICC", render: x => fmtPct(x) },
      { key: "growth_icc", label: "Growth ICC", render: x => fmtPct(x) },
      { key: "ivp_bm", label: "IVP", render: x => fmtPct(x) },
    ]);
    renderTable("value-monthly", data.monthly || [], [
      { key: "date", label: "Date" },
      { key: "value_icc", label: "Value ICC", render: x => fmtPct(x) },
      { key: "growth_icc", label: "Growth ICC", render: x => fmtPct(x) },
      { key: "ivp_bm", label: "IVP", render: x => fmtPct(x) },
    ]);
    renderTable("value-daily", filterRecentDaily(data.daily || []), [
      { key: "date", label: "Date" },
      { key: "value_icc", label: "Value ICC", render: x => fmtPct(x) },
      { key: "growth_icc", label: "Growth ICC", render: x => fmtPct(x) },
      { key: "ivp_bm", label: "IVP", render: x => fmtPct(x) },
    ]);
  }

  if (page === "industry") {
    renderTable("industry-summary-latest", data.summary_latest || [], [
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "Industry-wide ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Sectors", render: x => fmtInt(x) },
    ]);
    renderTable("industry-summary-monthly", data.summary_monthly || [], [
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "Industry-wide ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Sectors", render: x => fmtInt(x) },
    ]);
    renderTable("industry-latest", data.latest || [], [
      { key: "sector", label: "Sector" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "ew_icc", label: "EW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Items", render: x => fmtInt(x) },
    ]);
  }

  if (page === "etf") {
    renderTable("etf-summary-latest", data.summary_latest || [], [
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "ETF ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "ETFs", render: x => fmtInt(x) },
    ]);
    renderTable("etf-summary-monthly", data.summary_monthly || [], [
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "ETF ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "ETFs", render: x => fmtInt(x) },
    ]);
    renderTable("etf-latest", data.latest || [], [
      { key: "ticker", label: "Ticker" },
      { key: "label", label: "ETF" },
      { key: "category", label: "Category" },
      { key: "vw_icc", label: "ICC", render: x => fmtPct(x) },
      { key: "coverage_weight", label: "Coverage", render: x => fmtPct(x) },
      { key: "source", label: "Source" },
      { key: "status", label: "Status" },
    ]);
  }

  if (page === "country") {
    renderTable("country-summary-latest", data.summary_latest || [], [
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "Country ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Countries", render: x => fmtInt(x) },
    ]);
    renderTable("country-summary-monthly", data.summary_monthly || [], [
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "Country ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Countries", render: x => fmtInt(x) },
    ]);
    renderTable("country-latest", data.latest || [], [
      { key: "country", label: "Country" },
      { key: "ticker", label: "Proxy ETF" },
      { key: "vw_icc", label: "ICC", render: x => fmtPct(x) },
      { key: "coverage_weight", label: "Coverage", render: x => fmtPct(x) },
      { key: "source", label: "Source" },
      { key: "status", label: "Status" },
    ]);
  }

  if (page === "indices") {
    renderTable("indices-latest", data.latest || [], [
      { key: "family", label: "Index" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Items", render: x => fmtInt(x) },
    ]);
    renderTable("indices-monthly", data.monthly || [], [
      { key: "family", label: "Index" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
    ]);
    renderTable("indices-daily", filterRecentDaily(data.daily || []), [
      { key: "family", label: "Index" },
      { key: "date", label: "Date" },
      { key: "vw_icc", label: "VW ICC", render: x => fmtPct(x) },
      { key: "n_items", label: "Items", render: x => fmtInt(x) },
    ]);
  }

  if (catalog) {
    const key = page === "value" ? "value" : page;
    renderDownloadsList(`${page}-downloads`, catalog.families?.[key] || []);
  }
}

async function renderDownloadsPage(warnings) {
  const catalog = await safeLoad("./data/downloads_catalog.json", "downloads_catalog.json", warnings);
  const overview = await safeLoad("./data/overview.json", "overview.json", warnings);
  if (overview) {
    byId("asof") && (byId("asof").textContent = overview.asof_date ? `As of ${overview.asof_date}` : "");
  }
  if (!catalog) return;
  const sections = byId("downloads-sections");
  if (sections) {
    const order = [
      ["marketwide", "Marketwide"],
      ["value", "Value Premium"],
      ["industry", "Industry"],
      ["etf", "ETF"],
      ["country", "Country"],
      ["indices", "Indices"],
    ];
    sections.innerHTML = order.map(([key, title]) => `
      <section class="section family-block">
        <h3>${title}</h3>
        <div class="panel"><div id="dl-${key}"></div></div>
      </section>
    `).join("");
    for (const [key] of order) {
      renderDownloadsList(`dl-${key}`, catalog.families?.[key] || []);
    }
  }
  renderRawTree("raw-tab-usall", catalog.raw_tabs?.usall || []);
  renderRawTree("raw-tab-sp500", catalog.raw_tabs?.sp500 || []);
  renderRawTree("raw-tab-other_indices", catalog.raw_tabs?.other_indices || []);
  attachTabBehavior();
}

(async function main() {
  const warnings = [];
  const page = document.body.dataset.page || "overview";
  if (page === "overview") {
    await renderOverview(warnings);
  } else if (page === "downloads") {
    await renderDownloadsPage(warnings);
  } else {
    await renderFamilyPage(page, warnings);
  }
  if (warnings.length) {
    setStatus(`Partial load: ${warnings.join(" | ")}`, true);
  } else {
    setStatus("");
  }
})();
