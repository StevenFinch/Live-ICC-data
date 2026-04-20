
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

function setAsof(v) {
  const el = byId("asof");
  if (!el) return;
  el.textContent = v ? `As of ${v}` : "";
}

function esc(v) {
  return String(v ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmtPct(v, digits = 2) {
  const x = Number(v);
  if (!Number.isFinite(x)) return "";
  return `${(x * 100).toFixed(digits)}%`;
}

function fmtInt(v) {
  const x = Number(v);
  if (!Number.isFinite(x)) return "";
  return x.toLocaleString();
}

function badgeClass(text) {
  const t = String(text || "").toLowerCase();
  if (t.includes("unavailable")) return "badge bad";
  if (t.includes("estimate")) return "badge warn";
  return "badge ok";
}

function renderTable(rows, columns) {
  if (!rows || !rows.length) return `<div class="empty">No data</div>`;
  const thead = `<thead><tr>${columns.map(c => `<th>${esc(c.label)}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(row => {
    return `<tr>${columns.map(c => {
      const raw = row[c.key];
      const val = c.render ? c.render(raw, row) : esc(raw ?? "");
      return `<td>${val}</td>`;
    }).join("")}</tr>`;
  }).join("");
  return `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

function renderCards(items) {
  return `<div class="cards">${items.map(item => `
    <div class="card">
      <div class="card-label">${esc(item.label)}</div>
      <div class="card-value">${esc(item.value)}</div>
    </div>
  `).join("")}</div>`;
}

function renderDownloadLinks(downloads) {
  if (!downloads) return `<div class="empty">No downloads</div>`;
  return `
    <div class="download-links">
      <a href="${downloads.latest_csv}" target="_blank" rel="noopener">Latest CSV</a>
      <a href="${downloads.daily_history_csv}" target="_blank" rel="noopener">Daily history CSV</a>
      <a href="${downloads.monthly_history_csv}" target="_blank" rel="noopener">Monthly history CSV</a>
    </div>
  `;
}

function renderYearTree(years) {
  if (!years || !years.length) return `<div class="empty">No archived files</div>`;
  return years.map(year => `
    <details class="tree-level">
      <summary>${esc(year.year)}</summary>
      <div class="tree-toolbar">
        <a class="button-link" href="${year.download_all}" target="_blank" rel="noopener">Download all ${esc(year.year)}</a>
      </div>
      <div class="tree-body">
        ${(year.months || []).map(month => `
          <details class="tree-level">
            <summary>${esc(month.yyyymm)}</summary>
            <div class="tree-toolbar">
              <a class="button-link" href="${month.download_all}" target="_blank" rel="noopener">Download all ${esc(month.yyyymm)}</a>
            </div>
            <div class="tree-body">
              ${(month.files || []).map(file => `
                <div class="tree-leaf">
                  <div>
                    <div><strong>${esc(file.date)}</strong></div>
                    ${file.universe ? `<div class="note">${esc(file.universe)}${file.n_firms ? ` · ${fmtInt(file.n_firms)} firms` : ""}</div>` : ""}
                  </div>
                  <a class="button-link" href="${file.download_path}" target="_blank" rel="noopener">Download day</a>
                </div>
              `).join("")}
            </div>
          </details>
        `).join("")}
      </div>
    </details>
  `).join("");
}

function renderTabs(tabDefs, groupId) {
  const nav = tabDefs.map((tab, i) => `
    <button class="tab-btn ${i === 0 ? "active" : ""}" data-tab-group="${groupId}" data-target="${groupId}_${i}">
      ${esc(tab.label)}
    </button>
  `).join("");

  const panels = tabDefs.map((tab, i) => `
    <div id="${groupId}_${i}" class="tab-panel ${i === 0 ? "active" : ""}">
      ${tab.html}
    </div>
  `).join("");

  return `<div class="tab-strip">${nav}</div>${panels}`;
}

function activateTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const group = btn.dataset.tabGroup;
      const target = btn.dataset.target;
      document.querySelectorAll(`.tab-btn[data-tab-group="${group}"]`).forEach(x => x.classList.remove("active"));
      document.querySelectorAll(`.tab-panel[id^="${group}_"]`).forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      const panel = document.getElementById(target);
      if (panel) panel.classList.add("active");
    });
  });
}

function lastNByFamily(rows, n = 3) {
  const out = {};
  for (const row of rows || []) {
    const fam = row.family || row.ticker || row.label;
    if (!out[fam]) out[fam] = [];
    out[fam].push(row);
  }
  Object.keys(out).forEach(k => {
    out[k] = out[k]
      .slice()
      .sort((a, b) => String(b.date).localeCompare(String(a.date)))
      .slice(0, n);
  });
  return out;
}

function recentDaily(rows, n = 30) {
  return (rows || []).slice().sort((a, b) => {
    if (String(a.family || a.ticker).localeCompare(String(b.family || b.ticker)) !== 0) {
      return String(a.family || a.ticker).localeCompare(String(b.family || b.ticker));
    }
    return String(b.date).localeCompare(String(a.date));
  }).slice(0, n * 50);
}

function methodCell(v) {
  if (!v) return "";
  return `<span class="${badgeClass(v)}">${esc(v)}</span>`;
}

function renderFamilyBlocks(monthlyGroups, familyLabel = "Series") {
  const families = Object.keys(monthlyGroups || {}).sort();
  if (!families.length) return `<div class="empty">No monthly data</div>`;
  return `<div class="family-stack">${families.map(family => `
    <div class="family-block">
      <h3 class="family-heading">${esc(family)}</h3>
      ${renderTable(monthlyGroups[family], [
        { key: "date", label: "Date" },
        { key: "value", label: "Monthly value", render: v => fmtPct(v) },
        { key: "method", label: "Method", render: v => methodCell(v) },
        { key: "source", label: "Source" },
      ])}
    </div>
  `).join("")}</div>`;
}

function setActiveNav(page) {
  document.querySelectorAll("[data-nav]").forEach(el => {
    if (el.dataset.nav === page) el.classList.add("active");
  });
}

async function loadData() {
  const paths = {
    overview: "./data/overview.json",
    marketwide: "./data/marketwide.json",
    value: "./data/value.json",
    industry: "./data/industry.json",
    etf: "./data/etf.json",
    country: "./data/country.json",
    indices: "./data/indices.json",
    downloads: "./data/downloads_catalog.json",
  };
  const out = {};
  const warnings = [];
  for (const [key, path] of Object.entries(paths)) {
    try {
      out[key] = await fetchJson(path);
    } catch (e) {
      console.error(e);
      warnings.push(`${key} failed`);
    }
  }
  return { out, warnings };
}

function renderOverview(overview) {
  const app = byId("app");
  const rows = (overview.rows || []).slice().sort((a, b) => {
    if (String(a.family_key).localeCompare(String(b.family_key)) !== 0) {
      return String(a.family_key).localeCompare(String(b.family_key));
    }
    return String(a.family).localeCompare(String(b.family));
  });

  const html = `
    <section class="panel">
      <h2 class="section-title">Overview</h2>
      <div class="note">Latest daily values and the last three monthly observations are shown for each series.</div>
      ${renderTable(rows, [
        { key: "family_key", label: "Family" },
        { key: "family", label: "Series" },
        { key: "label", label: "Label" },
        { key: "latest_daily", label: "Latest daily", render: v => fmtPct(v) },
        { key: "method", label: "Method", render: v => methodCell(v) },
        { key: "m1_date", label: "Month 1 date" },
        { key: "m1_value", label: "Month 1", render: v => fmtPct(v) },
        { key: "m2_date", label: "Month 2 date" },
        { key: "m2_value", label: "Month 2", render: v => fmtPct(v) },
        { key: "m3_date", label: "Month 3 date" },
        { key: "m3_value", label: "Month 3", render: v => fmtPct(v) },
      ])}
    </section>
  `;
  app.innerHTML = html;
}

function renderFamilyPage(data, options = {}) {
  const app = byId("app");
  const latestRows = (data.latest || []).slice().sort((a, b) => String(a.family || a.ticker).localeCompare(String(b.family || b.ticker)));
  const monthlyGroups = data.monthly_groups || lastNByFamily(data.monthly || [], 3);

  const cards = [];
  if (latestRows.length) {
    const first = latestRows[0];
    cards.push({ label: "Last update", value: first.date || "" });
    cards.push({ label: "Series", value: String(latestRows.length) });
    const available = latestRows.filter(x => String(x.method).toLowerCase().includes("icc calculation")).length;
    cards.push({ label: "ICC calculation rows", value: String(available) });
  }

  let html = `
    <section class="panel">
      <h2 class="section-title">${esc(options.title || "Data")}</h2>
      <div class="note">${esc(options.note || "Method is stated explicitly for every row.")}</div>
      ${cards.length ? renderCards(cards) : ""}
    </section>

    <section class="panel">
      <h2 class="section-title">Latest daily</h2>
      ${renderTable(latestRows, [
        { key: "family", label: options.familyLabel || "Series" },
        { key: "label", label: "Label" },
        { key: "ticker", label: "Ticker" },
        { key: "date", label: "Date" },
        { key: "value", label: "Value", render: v => fmtPct(v) },
        { key: "method", label: "Method", render: v => methodCell(v) },
        { key: "source", label: "Source" },
        { key: "coverage_weight", label: "Coverage", render: v => fmtPct(v) },
        { key: "n_items", label: "Items", render: v => fmtInt(v) },
        { key: "status", label: "Status" },
      ])}
    </section>

    <section class="panel">
      <h2 class="section-title">Last three monthly observations</h2>
      ${renderFamilyBlocks(monthlyGroups, options.familyLabel || "Series")}
    </section>

    <section class="panel">
      <h2 class="section-title">Downloads</h2>
      ${renderDownloadLinks(data.downloads)}
    </section>

    <section class="panel">
      <h2 class="section-title">Archived daily files</h2>
      ${renderYearTree((data.downloads || {}).years || [])}
    </section>
  `;
  app.innerHTML = html;
}

function renderDownloads(catalog) {
  const app = byId("app");
  const familyTabs = Object.entries(catalog.families || {}).map(([key, value], i) => ({
    label: value.title,
    html: `
      <section class="panel">
        <h2 class="section-title">${esc(value.title)} downloads</h2>
        ${renderDownloadLinks(value)}
      </section>
      <section class="panel">
        <h2 class="section-title">Archived daily files</h2>
        ${renderYearTree(value.years || [])}
      </section>
    `,
  }));

  const rawTabs = [
    { label: "usall", html: `<section class="panel"><h2 class="section-title">Raw usall snapshots</h2>${renderYearTree((catalog.raw || {}).usall || [])}</section>` },
    { label: "sp500", html: `<section class="panel"><h2 class="section-title">Raw S&P 500 snapshots</h2>${renderYearTree((catalog.raw || {}).sp500 || [])}</section>` },
    { label: "other indices", html: `<section class="panel"><h2 class="section-title">Raw other index snapshots</h2>${renderYearTree((catalog.raw || {}).other_indices || [])}</section>` },
  ];

  const html = `
    <section class="panel">
      <h2 class="section-title">Family downloads</h2>
      ${renderTabs(familyTabs, "family_tabs")}
    </section>
    <section class="panel">
      <h2 class="section-title">Raw snapshot downloads</h2>
      ${renderTabs(rawTabs, "raw_tabs")}
    </section>
  `;
  app.innerHTML = html;
}

(async function main() {
  const page = document.body.dataset.page || "index";
  setActiveNav(page);

  const { out, warnings } = await loadData();
  setAsof(out.overview?.asof_date || "");

  if (page === "index") {
    renderOverview(out.overview || { rows: [] });
  } else if (page === "marketwide") {
    renderFamilyPage(out.marketwide || { latest: [], monthly_groups: {}, downloads: { years: [] } }, {
      title: "Marketwide",
      familyLabel: "Series",
      note: "Both All market and S&P 500 show separate last-three-month blocks and download trees.",
    });
  } else if (page === "value") {
    renderFamilyPage(out.value || { latest: [], monthly_groups: {}, downloads: { years: [] } }, {
      title: "Value Premium",
      familyLabel: "Series",
      note: "Value ICC, Growth ICC, and IVP are all calculated from constituent ICC aggregation.",
    });
  } else if (page === "industry") {
    renderFamilyPage(out.industry || { latest: [], monthly_groups: {}, downloads: { years: [] } }, {
      title: "Industry",
      familyLabel: "Industry",
      note: "Industry values are market-cap weighted aggregations of constituent ICC.",
    });
  } else if (page === "etf") {
    renderFamilyPage(out.etf || { latest: [], monthly_groups: {}, downloads: { years: [] } }, {
      title: "ETF",
      familyLabel: "ETF",
      note: "Method is shown explicitly as ICC calculation or P/E estimate.",
    });
  } else if (page === "country") {
    renderFamilyPage(out.country || { latest: [], monthly_groups: {}, downloads: { years: [] } }, {
      title: "Country",
      familyLabel: "Country",
      note: "Country values use country ETF proxies and are labeled as ICC calculation or P/E estimate.",
    });
  } else if (page === "indices") {
    renderFamilyPage(out.indices || { latest: [], monthly_groups: {}, downloads: { years: [] } }, {
      title: "Indices",
      familyLabel: "Index",
      note: "Index values are built from index constituent universes.",
    });
  } else if (page === "downloads") {
    renderDownloads(out.downloads || { families: {}, raw: {} });
  }

  if (warnings.length) {
    setStatus(`Partial load: ${warnings.join(" | ")}`, true);
  } else {
    setStatus("");
  }

  activateTabs();
})();
