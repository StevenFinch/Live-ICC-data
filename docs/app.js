async function fetchJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${path}`);
  return await response.json();
}

function byId(id) { return document.getElementById(id); }
function pageName() { return document.body.dataset.page || "overview"; }

function setStatus(message, isError = false) {
  const el = byId("status");
  if (!el) return;
  el.textContent = message || "";
  el.className = isError ? "status error" : "status";
}

function setAsof(value) {
  const el = byId("asof-date");
  if (el) el.textContent = value || "—";
}

function pct(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function integer(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "";
  return Number(value).toLocaleString();
}

function text(value) { return value === null || value === undefined ? "" : String(value); }

function escapeHtml(value) {
  return text(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function methodBadge(value) {
  const label = text(value);
  if (!label) return "";
  const lower = label.toLowerCase();
  let cls = "method-pill";
  if (lower.includes("partial")) cls += " partial";
  if (lower.includes("p/e") || lower.includes("estimate")) cls += " proxy";
  if (lower.includes("unavailable")) cls += " unavailable";
  return `<span class="${cls}">${escapeHtml(label)}</span>`;
}

function dataTable(targetId, rows, columns) {
  const target = byId(targetId);
  if (!target) return;
  if (!rows || rows.length === 0) {
    target.innerHTML = `<div class="empty">No data available.</div>`;
    return;
  }
  const header = columns.map(col => `<th>${escapeHtml(col.label)}</th>`).join("");
  const body = rows.map(row => {
    const cells = columns.map(col => {
      const raw = row[col.key];
      const rendered = col.format ? col.format(raw, row) : escapeHtml(raw);
      return `<td>${rendered}</td>`;
    }).join("");
    return `<tr>${cells}</tr>`;
  }).join("");
  target.innerHTML = `<table class="data-table"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function chooseRows(data) {
  if (!data) return [];
  if (Array.isArray(data.rows)) return data.rows;
  return [];
}

function lastThreeByGroup(rows, groupKey) {
  if (!rows || rows.length === 0) return [];
  if (!groupKey) return rows.slice().sort((a, b) => text(b.month_end_date).localeCompare(text(a.month_end_date))).slice(0, 3);
  const groups = new Map();
  for (const row of rows) {
    const key = text(row[groupKey]) || "Series";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  const output = [];
  for (const key of Array.from(groups.keys()).sort()) {
    const selected = groups.get(key)
      .slice()
      .sort((a, b) => text(b.month_end_date).localeCompare(text(a.month_end_date)))
      .slice(0, 3);
    output.push(...selected);
  }
  return output;
}

function firstExisting(obj, keys) {
  for (const key of keys) {
    if (obj && obj[key] !== undefined && obj[key] !== null) return obj[key];
  }
  return null;
}

function normalizeFamilyDownloads(family) {
  if (!family) return null;
  return {
    latest: family.latest_csv || family.latest || family.latest_path,
    daily: family.daily_history_csv || family.daily_history || family.daily,
    monthly: family.monthly_history_csv || family.monthly_history || family.monthly,
    allZip: family.all_zip || family.download_all_zip || family.zip,
    archive: family.archive || family.files || [],
    label: family.label || family.family || "Dataset",
    note: family.note || ""
  };
}

function cleanLink(path) {
  return path || "#";
}

function downloadAction(label, path, meta = "") {
  if (!path) return `<div class="download-action disabled"><span class="download-title">${escapeHtml(label)}</span><span class="download-meta">Not available</span></div>`;
  return `<a class="download-action" href="${cleanLink(path)}" target="_blank" rel="noopener"><span class="download-title">${escapeHtml(label)}</span><span class="download-meta">${escapeHtml(meta || "Download")}</span></a>`;
}

function renderFamilyDownloadBox(targetId, family) {
  const target = byId(targetId);
  if (!target) return;
  const d = normalizeFamilyDownloads(family);
  if (!d) {
    target.innerHTML = `<div class="empty">No downloads available.</div>`;
    return;
  }
  target.innerHTML = `<div class="download-actions-grid">
    ${downloadAction("Latest CSV", d.latest, "Most recent daily file")}
    ${downloadAction("Daily history CSV", d.daily, "All available daily observations")}
    ${downloadAction("Monthly history CSV", d.monthly, "Monthly summary series")}
    ${downloadAction("Download all ZIP", d.allZip, "Latest, daily, monthly, and archive files")}
  </div>`;
}

function renderArchiveTree(targetId, files) {
  const target = byId(targetId);
  if (!target) return;
  const rows = Array.isArray(files) ? files : [];
  if (rows.length === 0) {
    target.innerHTML = `<div class="empty">No archived files available.</div>`;
    return;
  }
  const tree = new Map();
  for (const row of rows) {
    const year = text(row.year || row.yyyy || (row.date || "").slice(0, 4));
    const month = text(row.month || row.mm || (row.date || "").slice(5, 7));
    if (!year || !month) continue;
    if (!tree.has(year)) tree.set(year, new Map());
    if (!tree.get(year).has(month)) tree.get(year).set(month, []);
    tree.get(year).get(month).push(row);
  }
  const html = Array.from(tree.keys()).sort().reverse().map(year => {
    const months = tree.get(year);
    const yearZip = rows.find(r => text(r.year) === year && (r.year_zip || r.zip_year))?.year_zip || rows.find(r => text(r.year) === year)?.zip_year;
    const monthHtml = Array.from(months.keys()).sort().reverse().map(month => {
      const monthRows = months.get(month).slice().sort((a, b) => text(b.date).localeCompare(text(a.date)));
      const monthZip = monthRows.find(r => r.month_zip || r.zip_month)?.month_zip || monthRows.find(r => r.month_zip || r.zip_month)?.zip_month;
      return `<details class="archive-month"><summary>${year}-${month}${monthZip ? ` <span class="archive-summary-actions"><a href="${monthZip}" target="_blank" rel="noopener">Download month ZIP</a></span>` : ""}</summary><div class="archive-body">
        ${monthRows.map(r => {
          const date = text(r.date || r.asof_date || `${year}-${month}`);
          const label = text(r.label || r.family || r.universe || r.ticker || r.country || "File");
          const path = r.path || r.download_path || r.csv || r.file;
          return `<div class="archive-row"><span>${escapeHtml(date)} · ${escapeHtml(label)}${r.n_firms ? ` · ${integer(r.n_firms)} rows` : ""}</span>${path ? `<a href="${path}" target="_blank" rel="noopener">CSV</a>` : ""}</div>`;
        }).join("")}
      </div></details>`;
    }).join("");
    return `<details class="archive-year"><summary>${year}${yearZip ? ` <span class="archive-summary-actions"><a href="${yearZip}" target="_blank" rel="noopener">Download year ZIP</a></span>` : ""}</summary><div class="archive-body">${monthHtml}</div></details>`;
  }).join("");
  target.innerHTML = html || `<div class="empty">No archived files available.</div>`;
}

function overviewColumns() {
  return [
    { key: "dataset", label: "Dataset" },
    { key: "latest_daily", label: "Latest daily", format: pct },
    { key: "method", label: "Method", format: methodBadge },
    { key: "month_1", label: "Latest month", format: pct },
    { key: "month_2", label: "Previous month", format: pct },
    { key: "month_3", label: "Third month", format: pct },
  ];
}

async function renderOverview() {
  const overview = await fetchJson("./data/overview.json");
  setAsof(overview.asof_date || overview.date || "Latest available");
  const rows = overview.rows || [];
  dataTable("overview-table", rows, overviewColumns());
  const families = overview.families || {};
  const cards = Object.keys(families).map(key => {
    const fam = normalizeFamilyDownloads(families[key]);
    if (!fam) return "";
    return `<a class="download-card" href="./${key === "country" ? "country" : key}.html"><span class="download-title">${escapeHtml(fam.label)}</span><span class="download-meta">Open dataset page</span></a>`;
  }).join("");
  const target = byId("overview-downloads");
  if (target) target.innerHTML = cards || `<div class="empty">No download packages available.</div>`;
}

function genericColumns(name, monthly = false) {
  const dateKey = monthly ? "month_end_date" : "date";
  if (name === "value") return [
    { key: dateKey, label: monthly ? "Month-end date" : "Date" },
    { key: "value_icc", label: "Value ICC", format: pct },
    { key: "growth_icc", label: "Growth ICC", format: pct },
    { key: "ivp", label: "IVP", format: pct },
    { key: "method", label: "Method", format: methodBadge },
  ];
  if (name === "etf") return [
    { key: dateKey, label: monthly ? "Month-end date" : "Date" },
    { key: "ticker", label: "ETF" },
    { key: "label", label: "Name" },
    { key: monthly ? "daily_icc" : "icc", label: "ICC", format: pct },
    { key: "coverage_weight", label: "Coverage", format: pct },
    { key: "method", label: "Method", format: methodBadge },
    { key: "holding_source", label: "Source" },
  ];
  if (name === "country") return [
    { key: dateKey, label: monthly ? "Month-end date" : "Date" },
    { key: "country", label: "Country / Region" },
    { key: monthly ? "daily_icc" : "icc", label: "ICC", format: pct },
    { key: "n_icc_available", label: "Available ADRs", format: integer },
    { key: "coverage_mktcap", label: "Coverage", format: pct },
    { key: "method", label: "Method", format: methodBadge },
  ];
  return [
    { key: dateKey, label: monthly ? "Month-end date" : "Date" },
    { key: name === "industry" ? "group" : "family", label: name === "industry" ? "Group" : "Family" },
    { key: "daily_icc", label: "ICC", format: pct },
    { key: "n_firms", label: "N firms", format: integer },
    { key: "method", label: "Method", format: methodBadge },
  ];
}

async function renderFamily(name) {
  const data = await fetchJson(`./data/${name}.json`);
  setAsof(data.asof_date || data.date || "Latest available");
  renderFamilyDownloadBox("family-downloads", data.downloads);
  renderArchiveTree("family-archive", data.downloads?.archive || data.archive || []);
  if (name === "marketwide") {
    dataTable("latest-table", data.latest || [], [
      { key: "family", label: "Series" },
      { key: "date", label: "Date" },
      { key: "daily_icc", label: "ICC", format: pct },
      { key: "ew_icc", label: "EW ICC", format: pct },
      { key: "n_firms", label: "N firms", format: integer },
      { key: "method", label: "Method", format: methodBadge },
    ]);
    dataTable("monthly-table", lastThreeByGroup(data.monthly || [], "family"), [
      { key: "family", label: "Series" },
      { key: "month_end_date", label: "Month-end date" },
      { key: "daily_icc", label: "Monthly ICC", format: pct },
      { key: "n_firms", label: "N firms", format: integer },
      { key: "method", label: "Method", format: methodBadge },
    ]);
    return;
  }
  const groupKey = name === "industry" ? "group" : name === "indices" ? "family" : name === "etf" ? "ticker" : name === "country" ? "country" : null;
  dataTable("latest-table", data.latest || [], genericColumns(name, false));
  dataTable("monthly-table", lastThreeByGroup(data.monthly || [], groupKey), genericColumns(name, true));
}

function renderDownloadFamilies(catalog) {
  const target = byId("category-downloads");
  if (!target) return;
  const families = catalog.families || {};
  const order = ["marketwide", "value", "industry", "indices", "etf", "country"];
  target.innerHTML = order.filter(key => families[key]).map(key => {
    const family = normalizeFamilyDownloads(families[key]);
    const page = key === "country" ? "country" : key;
    return `<article class="download-family-card"><h4>${escapeHtml(family.label)}</h4><p class="note">${escapeHtml(family.note)}</p><div class="download-actions-grid">
      ${downloadAction("Latest CSV", family.latest, "Most recent daily file")}
      ${downloadAction("Daily history", family.daily, "All daily observations")}
      ${downloadAction("Monthly history", family.monthly, "Monthly summary")}
      ${downloadAction("All data ZIP", family.allZip, "Complete package")}
    </div><p class="note"><a class="inline-link" href="./${page}.html">Open dataset tab</a></p></article>`;
  }).join("") || `<div class="empty">No dataset packages available.</div>`;
}

function renderRawSnapshots(catalog, group) {
  const target = byId("raw-tree");
  if (!target) return;
  const rows = (catalog.raw_snapshots || []).filter(row => (row.raw_group || row.group) === group);
  renderArchiveTree("raw-tree", rows);
}

async function renderDownloads() {
  const catalog = await fetchJson("./data/downloads_catalog.json");
  setAsof(catalog.asof_date || catalog.date || "Latest available");
  renderDownloadFamilies(catalog);
  const buttons = Array.from(document.querySelectorAll("[data-raw-tab]"));
  let current = "usall";
  const redraw = () => renderRawSnapshots(catalog, current);
  for (const button of buttons) {
    button.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      button.classList.add("active");
      current = button.dataset.rawTab;
      redraw();
    });
  }
  redraw();
}

(async function main() {
  try {
    const page = pageName();
    if (page === "overview") await renderOverview();
    else if (page === "downloads") await renderDownloads();
    else await renderFamily(page);
    setStatus("");
  } catch (err) {
    console.error(err);
    setStatus(`Failed to load page data: ${err.message}`, true);
  }
})();
