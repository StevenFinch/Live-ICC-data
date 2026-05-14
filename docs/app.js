async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${path}`);
  return await response.json();
}

function $(id) {
  return document.getElementById(id);
}

function pageName() {
  return document.body.dataset.page || "overview";
}

function setStatus(message, isError = false) {
  const el = $("status");
  if (!el) return;
  el.textContent = message || "";
  el.className = isError ? "status error" : "status";
}

function fmtPct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function fmtNum(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(value).toLocaleString();
}

function fmtText(value) {
  if (value === null || value === undefined) return "";
  return String(value);
}

const COUNTRY_ALIASES = {
  "uk": "United Kingdom",
  "u.k.": "United Kingdom",
  "u.k": "United Kingdom",
  "great britain": "United Kingdom",
  "britain": "United Kingdom",
  "united kingdom": "United Kingdom",
  "united kingdom of great britain and northern ireland": "United Kingdom",
  "england": "United Kingdom",
  "south korea": "South Korea",
  "korea, south": "South Korea",
  "republic of korea": "South Korea",
  "korea republic of": "South Korea",
  "korea": "South Korea",
  "people's republic of china": "China",
  "prc": "China",
  "china mainland": "China",
  "mainland china": "China",
  "hong kong sar": "Hong Kong",
  "hong kong sar china": "Hong Kong",
  "hong kong, china": "Hong Kong",
  "taiwan province of china": "Taiwan",
  "taiwan, china": "Taiwan",
  "russian federation": "Russia",
  "viet nam": "Vietnam",
  "u.a.e.": "United Arab Emirates",
  "uae": "United Arab Emirates",
  "czech republic": "Czechia",
  "slovak republic": "Slovakia",
  "macau": "Macao",
  "macao sar china": "Macao"
};

function normalizeCountryRegionName(value) {
  const text = fmtText(value).trim();
  if (!text) return "";
  const key = text.replace(/_/g, " ").replace(/\s+/g, " ").toLowerCase();
  return COUNTRY_ALIASES[key] || text;
}

function isUnavailableCountryRow(row) {
  const method = fmtText(row.method || row.Method || row.status).toLowerCase();
  const icc = row.icc ?? row.daily_icc ?? row["ICC"] ?? row["Monthly ICC"];
  return method.includes("unavailable") || icc === null || icc === undefined || Number.isNaN(Number(icc));
}

function countryRowScore(row) {
  const method = fmtText(row.method || row.Method || row.status).toLowerCase();
  const icc = row.icc ?? row.daily_icc ?? row["ICC"] ?? row["Monthly ICC"];
  const n = Number(row.n_icc_available ?? row.n_available ?? row.n_selected ?? row["Available ADRs"] ?? 0) || 0;
  const coverage = Number(row.coverage_mktcap ?? row.coverage_weight ?? row.coverage ?? 0) || 0;
  let score = 0;
  if (icc !== null && icc !== undefined && !Number.isNaN(Number(icc))) score += 1000;
  if (!method.includes("unavailable")) score += 500;
  if (method.includes("adr top-10") || method.includes("icc calculation")) score += 100;
  if (method.includes("partial")) score += 50;
  score += Math.min(Math.max(n, 0), 100);
  score += Math.min(Math.max(coverage, 0), 1) * 10;
  return score;
}

function cleanCountryRows(rows) {
  if (!rows || !Array.isArray(rows)) return [];
  const cleaned = rows
    .map(row => ({ ...row, country: normalizeCountryRegionName(row.country ?? row.country_region ?? row["Country / Region"] ?? row.Country) }))
    .filter(row => row.country && !isUnavailableCountryRow(row));

  const best = new Map();
  for (const row of cleaned) {
    const date = row.date || row.month_end_date || row.asof_date || "";
    const key = `${row.country}||${date}`;
    const existing = best.get(key);
    if (!existing || countryRowScore(row) > countryRowScore(existing)) best.set(key, row);
  }

  return Array.from(best.values()).sort((a, b) => {
    const c = String(a.country).localeCompare(String(b.country));
    if (c !== 0) return c;
    return String(b.date || b.month_end_date || "").localeCompare(String(a.date || a.month_end_date || ""));
  });
}

function table(id, rows, columns) {
  const el = $(id);
  if (!el) return;
  if (!rows || rows.length === 0) {
    el.innerHTML = `<div class="empty">No data available.</div>`;
    return;
  }
  const head = `<thead><tr>${columns.map(c => `<th>${c.label}</th>`).join("")}</tr></thead>`;
  const body = rows.map(row => `<tr>${columns.map(c => {
    const raw = row[c.key];
    const value = c.format ? c.format(raw, row) : fmtText(raw);
    return `<td>${value}</td>`;
  }).join("")}</tr>`).join("");
  el.innerHTML = `<table>${head}<tbody>${body}</tbody></table>`;
}

function methodBadge(method) {
  const m = fmtText(method);
  if (!m) return "";
  const cls = m.includes("P/E") ? "badge estimate" : m.includes("Partial") ? "badge partial" : m.includes("Unavailable") ? "badge unavailable" : "badge icc";
  return `<span class="${cls}">${m}</span>`;
}

function downloadCards(id, downloads) {
  const el = $(id);
  if (!el) return;
  if (!downloads) {
    el.innerHTML = `<div class="empty">No downloads available.</div>`;
    return;
  }
  const cards = [
    ["Latest", downloads.latest_csv, "Latest observation CSV"],
    ["Daily history", downloads.daily_history_csv, "All daily observations"],
    ["Monthly history", downloads.monthly_history_csv, "Month-end observations"],
    ["All standard files", downloads.all_zip, "ZIP: latest + daily + monthly"],
  ];
  el.innerHTML = `<div class="download-grid">${cards.map(([title, url, desc]) => {
    if (!url) return "";
    return `<a class="download-card" href="${url}"><strong>${title}</strong><span>${desc}</span></a>`;
  }).join("")}</div>`;
}

function archiveTree(id, tree) {
  const el = $(id);
  if (!el) return;
  if (!tree || tree.length === 0) {
    el.innerHTML = `<div class="empty">No archived files yet.</div>`;
    return;
  }
  el.innerHTML = tree.map(year => `
    <details class="archive-level">
      <summary><strong>${year.year}</strong> <a href="${year.zip || '#'}">Download year ZIP</a></summary>
      ${(year.months || []).map(month => `
        <details class="archive-level nested">
          <summary>${month.month} <a href="${month.zip || '#'}">Download month ZIP</a></summary>
          ${(month.days || []).map(day => `
            <div class="archive-file">
              <span>${day.date || ""} · ${fmtNum(day.n_rows || day.n_firms)} rows</span>
              <a href="${day.csv || day.path || '#'}">CSV</a>
            </div>
          `).join("")}
        </details>
      `).join("")}
    </details>
  `).join("");
}

function rawTree(id, rawGroup) {
  const el = $(id);
  if (!el) return;
  if (!rawGroup || !rawGroup.years || rawGroup.years.length === 0) {
    el.innerHTML = `<div class="empty">No raw snapshots in this group.</div>`;
    return;
  }
  el.innerHTML = rawGroup.years.map(year => `
    <details class="archive-level">
      <summary><strong>${year.year}</strong> <a href="${year.zip || '#'}">Download year ZIP</a></summary>
      ${(year.months || []).map(month => `
        <details class="archive-level nested">
          <summary>${month.month} <a href="${month.zip || '#'}">Download month ZIP</a></summary>
          ${(month.days || []).map(day => `
            <div class="archive-file">
              <span>${day.date} · ${day.universe} · ${fmtNum(day.n_firms)} firms</span>
              <a href="${day.csv || day.download_path || '#'}">CSV</a>
            </div>
          `).join("")}
        </details>
      `).join("")}
    </details>
  `).join("");
}

function familyColumns(family, monthly = false) {
  const dateKey = monthly ? "month_end_date" : "date";
  if (family === "marketwide") {
    return [
      { key: "family", label: "Series" },
      { key: dateKey, label: monthly ? "Month-end date" : "Date" },
      { key: "daily_icc", label: "ICC", format: fmtPct },
      { key: "ew_icc", label: "EW ICC", format: fmtPct },
      { key: "n_firms", label: "N firms", format: fmtNum },
      { key: "method", label: "Method", format: methodBadge },
    ];
  }
  if (family === "value") {
    return [
      { key: dateKey, label: monthly ? "Month-end date" : "Date" },
      { key: "value_icc", label: "Value ICC", format: fmtPct },
      { key: "growth_icc", label: "Growth ICC", format: fmtPct },
      { key: "ivp", label: "IVP", format: fmtPct },
      { key: "n_firms", label: "N firms", format: fmtNum },
      { key: "method", label: "Method", format: methodBadge },
    ];
  }
  if (family === "industry") {
    return [
      { key: "group", label: "Industry" },
      { key: dateKey, label: monthly ? "Month-end date" : "Date" },
      { key: "daily_icc", label: "ICC", format: fmtPct },
      { key: "ew_icc", label: "EW ICC", format: fmtPct },
      { key: "n_firms", label: "N firms", format: fmtNum },
      { key: "method", label: "Method", format: methodBadge },
    ];
  }
  if (family === "indices") {
    return [
      { key: "family", label: "Index" },
      { key: dateKey, label: monthly ? "Month-end date" : "Date" },
      { key: "daily_icc", label: "ICC", format: fmtPct },
      { key: "ew_icc", label: "EW ICC", format: fmtPct },
      { key: "n_firms", label: "N firms", format: fmtNum },
      { key: "method", label: "Method", format: methodBadge },
    ];
  }
  if (family === "etf") {
    return [
      { key: "ticker", label: "ETF" },
      { key: "label", label: "Name" },
      { key: dateKey, label: monthly ? "Month-end date" : "Date" },
      { key: "icc", label: "ICC / estimate", format: fmtPct },
      { key: "coverage_weight", label: "Coverage", format: fmtPct },
      { key: "method", label: "Method", format: methodBadge },
      { key: "holding_source", label: "Source" },
    ];
  }
  if (family === "country") {
    return [
      { key: "country", label: "Country / Region" },
      { key: dateKey, label: monthly ? "Month-end date" : "Date" },
      { key: "icc", label: "ICC / estimate", format: fmtPct },
      { key: "n_icc_available", label: "Available ADRs", format: fmtNum },
      { key: "coverage_mktcap", label: "Coverage", format: fmtPct },
      { key: "method", label: "Method", format: methodBadge },
    ];
  }
  return [];
}

function renderFamilyDownloadPanel(data) {
  downloadCards("family-downloads", data.downloads);
  archiveTree("family-archive-tree", data.downloads ? data.downloads.tree : []);
}

async function renderOverview() {
  const data = await fetchJson("./data/overview.json");
  table("overview-table", data.rows || [], [
    { key: "dataset", label: "Dataset" },
    { key: "latest_daily", label: "Latest daily", format: fmtPct },
    { key: "method", label: "Method", format: methodBadge },
    { key: "month_1", label: "Latest month", format: fmtPct },
    { key: "month_2", label: "Previous month", format: fmtPct },
    { key: "month_3", label: "Third month", format: fmtPct },
  ]);
  const el = $("overview-downloads");
  if (el && data.families) {
    el.innerHTML = Object.keys(data.families).map(key => {
      const d = data.families[key];
      return `<a class="download-card" href="${d.all_zip}"><strong>${d.label}</strong><span>Download all standard files</span></a>`;
    }).join("");
  }
}

async function renderFamily(family) {
  const data = await fetchJson(`./data/${family}.json`);
  const title = $("family-title");
  const note = $("family-note");
  if (title) title.textContent = data.label || "";
  if (note) note.textContent = data.note || "";

  let latestRows = data.latest || [];
  let monthlyRows = data.monthly || [];
  if (family === "country") {
    latestRows = cleanCountryRows(latestRows);
    monthlyRows = cleanCountryRows(monthlyRows);
  }

  table("latest-table", latestRows, familyColumns(family, false));
  table("monthly-table", monthlyRows, familyColumns(family, true));
  renderFamilyDownloadPanel(data);
}

async function renderDownloads() {
  const data = await fetchJson("./data/downloads_catalog.json");
  const familyBox = $("category-downloads");
  if (familyBox) {
    const families = data.families || {};
    familyBox.innerHTML = Object.keys(families).map(key => {
      const d = families[key];
      return `
        <section class="download-section">
          <h3>${d.label}</h3>
          <p>${d.note || ""}</p>
          <div class="download-grid">
            <a class="download-card" href="${d.latest_csv || '#'}"><strong>Latest</strong><span>Latest CSV</span></a>
            <a class="download-card" href="${d.daily_history_csv || '#'}"><strong>Daily history</strong><span>Full daily CSV</span></a>
            <a class="download-card" href="${d.monthly_history_csv || '#'}"><strong>Monthly history</strong><span>Month-end CSV</span></a>
            <a class="download-card" href="${d.all_zip || '#'}"><strong>Download all</strong><span>ZIP package</span></a>
          </div>
          <h4>Open year/month/day archive</h4>
          ${renderArchiveTreeHtml(d.tree || [])}
        </section>`;
    }).join("");
  }
  setupRawTabs(data.raw_snapshots || {});
}

function renderArchiveTreeHtml(tree) {
  if (!tree.length) return `<div class="empty">No archive yet.</div>`;
  return tree.map(year => `
    <details class="archive-level">
      <summary><strong>${year.year}</strong> <a href="${year.zip || '#'}">Download year ZIP</a></summary>
      ${(year.months || []).map(month => `
        <details class="archive-level nested">
          <summary>${month.month} <a href="${month.zip || '#'}">Download month ZIP</a></summary>
          ${(month.days || []).map(day => `
            <div class="archive-file">
              <span>${day.date} · ${fmtNum(day.n_rows)} rows</span>
              <a href="${day.csv || day.path || '#'}">CSV</a>
            </div>`).join("")}
        </details>`).join("")}
    </details>`).join("");
}

function setupRawTabs(raw) {
  const buttons = document.querySelectorAll("[data-raw-tab]");
  const draw = group => rawTree("raw-tree", raw[group]);
  buttons.forEach(button => {
    button.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      button.classList.add("active");
      draw(button.dataset.rawTab);
    });
  });
  draw("usall");
}

(async function main() {
  try {
    const page = pageName();
    if (page === "overview") await renderOverview();
    else if (page === "downloads") await renderDownloads();
    else await renderFamily(page);
    setStatus("");
  } catch (error) {
    console.error(error);
    setStatus(`Failed to load page data: ${error.message}`, true);
  }
})();
