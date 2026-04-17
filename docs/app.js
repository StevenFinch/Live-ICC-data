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

function renderTable(rows, columns) {
  if (!rows || !rows.length) return `<div class="empty">No data</div>`;
  const thead = `<thead><tr>${columns.map(c => `<th>${esc(c.label)}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(row => {
    return `<tr>${columns.map(c => {
      const raw = row[c.key];
      const value = c.render ? c.render(raw, row) : esc(raw ?? "");
      return `<td>${value}</td>`;
    }).join("")}</tr>`;
  }).join("");
  return `<div class="table-wrap"><table class="data">${thead}<tbody>${tbody}</tbody></table></div>`;
}

function section(title, inner, note = "") {
  return `
    <section class="section">
      <h2>${esc(title)}</h2>
      ${note ? `<div class="note">${note}</div>` : ""}
      ${inner}
    </section>
  `;
}

function renderDownloadBox(downloads) {
  return `
    <div class="link-list">
      <div class="link-row">
        <a class="link-btn" href="${downloads.latest_csv}" target="_blank" rel="noopener">Latest CSV</a>
        <a class="link-btn" href="${downloads.daily_history_csv}" target="_blank" rel="noopener">Daily history CSV</a>
        <a class="link-btn" href="${downloads.monthly_history_csv}" target="_blank" rel="noopener">Monthly history CSV</a>
      </div>
    </div>
  `;
}

function renderYearTree(years) {
  if (!years || !years.length) return `<div class="empty">No archived files</div>`;
  return years.map(year => `
    <details class="tree">
      <summary>${esc(year.year)} &nbsp; <a class="link-btn" href="${year.download_all}" target="_blank" rel="noopener">Download all ${esc(year.year)}</a></summary>
      <div class="tree-body">
        ${year.months.map(month => `
          <details class="tree">
            <summary>${esc(month.yyyymm)} &nbsp; <a class="link-btn" href="${month.download_all}" target="_blank" rel="noopener">Download all ${esc(month.yyyymm)}</a></summary>
            <div class="tree-body">
              ${month.files.map(file => `
                <div class="tree-leaf">
                  <span>${esc(file.date)}</span>
                  <a class="link-btn" href="${file.download_path}" target="_blank" rel="noopener">Download day</a>
                </div>
              `).join("")}
            </div>
          </details>
        `).join("")}
      </div>
    </details>
  `).join("");
}

function renderTabs(tabDefs) {
  const id = `tabs_${Math.random().toString(36).slice(2)}`;
  const nav = tabDefs.map((tab, i) => `<button class="tab-btn ${i === 0 ? "active" : ""}" data-tab-group="${id}" data-target="${id}_${i}">${esc(tab.label)}</button>`).join("");
  const panels = tabDefs.map((tab, i) => `<div class="tab-panel ${i === 0 ? "active" : ""}" id="${id}_${i}">${tab.html}</div>`).join("");
  return `<div><div class="tabs">${nav}</div>${panels}</div>`;
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
    const fam = row.family || row.ticker;
    if (!out[fam]) out[fam] = [];
    out[fam].push(row);
  }
  Object.keys(out).forEach(k => {
    out[k] = out[k].slice().sort((a, b) => String(b.date).localeCompare(String(a.date))).slice(0, n);
  });
  return out;
}

function renderFamilyPage(data, options = {}) {
  const app = byId("app");
  const monthlyGroups = data.monthly_groups || lastNByFamily(data.monthly || [], 3);
  const dailyRecent = (data.daily || []).slice().sort((a, b) => String(b.date).localeCompare(String(a.date))).slice(0, 30);

  let html = "";

  html += section(
    "Latest daily",
    renderTable((data.latest || []).slice().sort((a, b) => String(a.family || a.ticker).localeCompare(String(b.family || b.ticker))), [
      { key: "family", label: options.familyLabel || "Series", render: (v, row) => esc(v || row.ticker) },
      { key: "label", label: "Label" },
      { key: "date", label: "Date" },
      { key: "value", label: "Value", render: v => fmtPct(v) },
      { key: "method", label: "Method" },
      { key: "source", label: "Source" },
    ]),
    "Method is shown explicitly as ICC calculation or P/E estimate."
  );

  const monthlySections = Object.entries(monthlyGroups).map(([family, rows]) => `
    <div>
      <h3>${esc(family)}</h3>
      ${renderTable(rows, [
        { key: "date", label: "Date" },
        { key: "value", label: "Monthly value", render: v => fmtPct(v) },
        { key: "method", label: "Method" },
      ])}
    </div>
  `).join("");

  html += section("Last three monthly observations", monthlySections);

  html += section(
    "Recent daily history",
    renderTable(dailyRecent, [
      { key: "family", label: options.familyLabel || "Series", render: (v, row) => esc(v || row.ticker) },
      { key: "label", label: "Label" },
      { key: "date", label: "Date" },
      { key: "value", label: "Daily value", render: v => fmtPct(v) },
      { key: "method", label: "Method" },
    ])
  );

  html += section(
    "Downloads",
    `${renderDownloadBox(data.downloads)}<h3>Archived daily files</h3>${renderYearTree(data.downloads.years)}`
  );

  app.innerHTML = html;
}

function renderOverview(dataMap, overview) {
  const app = byId("app");
  const rows = (overview.rows || []).slice().sort((a, b) => {
    const k1 = `${a.family_key}_${a.family}`;
    const k2 = `${b.family_key}_${b.family}`;
    return k1.localeCompare(k2);
  });
  const html = renderTable(rows, [
    { key: "family_key", label: "Family", render: v => esc(v) },
    { key: "family", label: "Series" },
    { key: "latest_daily", label: "Latest daily", render: v => fmtPct(v) },
    { key: "method", label: "Method" },
    { key: "m1_date", label: "Month 1 date" },
    { key: "m1_value", label: "Month 1", render: v => fmtPct(v) },
    { key: "m2_date", label: "Month 2 date" },
    { key: "m2_value", label: "Month 2", render: v => fmtPct(v) },
    { key: "m3_date", label: "Month 3 date" },
    { key: "m3_value", label: "Month 3", render: v => fmtPct(v) },
  ]);
  app.innerHTML = section(
    "Overview",
    html,
    "Latest daily and the last three monthly observations are shown for each family/series."
  );
}

function renderDownloads(catalog) {
  const app = byId("app");
  const familyTabs = Object.entries(catalog.families || {}).map(([key, v]) => ({
    label: v.title,
    html: `${renderDownloadBox(v)}<h3>Archived daily files</h3>${renderYearTree(v.years)}`,
  }));

  const rawTabs = [
    { label: "usall", html: renderYearTree((catalog.raw || {}).usall || []) },
    { label: "sp500", html: renderYearTree((catalog.raw || {}).sp500 || []) },
    { label: "other indices", html: renderYearTree((catalog.raw || {}).other_indices || []) },
  ];

  app.innerHTML =
    section("Family downloads", renderTabs(familyTabs), "Each family has latest, daily history, monthly history, and archived daily files.") +
    section("Raw snapshot downloads", renderTabs(rawTabs), "Raw snapshots are grouped into usall, sp500, and other indices.");
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
  for (const [k, p] of Object.entries(paths)) {
    try {
      out[k] = await fetchJson(p);
    } catch (e) {
      console.error(e);
      warnings.push(`${k} failed`);
    }
  }
  return { out, warnings };
}

function setActiveNav(page) {
  document.querySelectorAll("[data-nav]").forEach(el => {
    if (el.dataset.nav === page) el.classList.add("active");
  });
}

(async function main() {
  const page = document.body.dataset.page;
  setActiveNav(page);

  const { out, warnings } = await loadData();

  const asof = out.overview?.asof_date || out.marketwide?.latest?.[0]?.date || "";
  byId("asof").textContent = asof ? `As of ${asof}` : "";

  if (page === "index") {
    renderOverview(out, out.overview || { rows: [] });
  } else if (page === "marketwide") {
    renderFamilyPage(out.marketwide || { latest: [], daily: [], monthly: [], downloads: { years: [] } }, { familyLabel: "Series" });
  } else if (page === "value") {
    renderFamilyPage(out.value || { latest: [], daily: [], monthly: [], downloads: { years: [] } }, { familyLabel: "Series" });
  } else if (page === "industry") {
    renderFamilyPage(out.industry || { latest: [], daily: [], monthly: [], downloads: { years: [] } }, { familyLabel: "Industry" });
  } else if (page === "etf") {
    renderFamilyPage(out.etf || { latest: [], daily: [], monthly: [], downloads: { years: [] } }, { familyLabel: "Ticker" });
  } else if (page === "country") {
    renderFamilyPage(out.country || { latest: [], daily: [], monthly: [], downloads: { years: [] } }, { familyLabel: "Ticker" });
  } else if (page === "indices") {
    renderFamilyPage(out.indices || { latest: [], daily: [], monthly: [], downloads: { years: [] } }, { familyLabel: "Index" });
  } else if (page === "downloads") {
    renderDownloads(out.downloads || { families: {}, raw: {} });
  }

  if (warnings.length) {
    setStatus(`Partial load: ${warnings.join(" | ")}`, true);
  }

  activateTabs();
})();
