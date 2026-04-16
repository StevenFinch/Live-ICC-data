async function fetchJson(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${path}`);
  return await r.json();
}

function byId(id) {
  return document.getElementById(id);
}

function setText(id, text) {
  const el = byId(id);
  if (el) el.textContent = text ?? "";
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

function renderCards(id, items) {
  const el = byId(id);
  if (!el) return;
  el.innerHTML = items.map(item => `
    <div class="card">
      <div class="card-label">${item.label}</div>
      <div class="card-value">${item.value}</div>
    </div>
  `).join("");
}

function renderTable(id, rows, columns) {
  const el = byId(id);
  if (!el) return;
  if (!rows || !rows.length) {
    el.innerHTML = `<div class="empty">No data</div>`;
    return;
  }
  const thead = `<thead><tr>${columns.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => {
    return `<tr>${columns.map(c => {
      const v = r[c[0]];
      return `<td>${typeof c[2] === "function" ? c[2](v, r) : (v ?? "")}</td>`;
    }).join("")}</tr>`;
  }).join("");
  el.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

function renderDownloadList(id, items) {
  const el = byId(id);
  if (!el) return;
  if (!items || !items.length) {
    el.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }
  el.innerHTML = items.map(x => `
    <div class="download-item">
      <a class="btn-link" href="${x.path}" target="_blank" rel="noopener">${x.label}</a>
    </div>
  `).join("");
}

function renderDownloadsSections(id, familyDownloads) {
  const el = byId(id);
  if (!el) return;
  const entries = Object.entries(familyDownloads || {});
  if (!entries.length) {
    el.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }
  el.innerHTML = entries.map(([family, files]) => `
    <section class="download-family">
      <h3>${family}</h3>
      <div>${files.map(x => `<div class="download-item"><a class="btn-link" href="${x.path}" target="_blank" rel="noopener">${x.label}</a></div>`).join("")}</div>
    </section>
  `).join("");
}

function renderRawTree(id, years) {
  const el = byId(id);
  if (!el) return;
  if (!years || !years.length) {
    el.innerHTML = `<div class="empty">No raw snapshot files</div>`;
    return;
  }
  el.innerHTML = years.map(yearObj => `
    <details class="tree-level">
      <summary><strong>${yearObj.year}</strong> <a class="inline-link" href="${yearObj.download_all}" target="_blank" rel="noopener">Download all</a></summary>
      <div class="tree-body">
        ${yearObj.months.map(monthObj => `
          <details class="tree-level nested">
            <summary>${monthObj.yyyymm} <a class="inline-link" href="${monthObj.download_all}" target="_blank" rel="noopener">Download all</a></summary>
            <div class="tree-body">
              ${monthObj.days.map(day => `
                <div class="tree-leaf">
                  <span>${day.date} | ${day.universe} | ${fmtInt(day.n_firms)} firms</span>
                  <a class="btn-link small" href="${day.path}" target="_blank" rel="noopener">Download</a>
                </div>
              `).join("")}
            </div>
          </details>
        `).join("")}
      </div>
    </details>
  `).join("");
}

function renderOverviewLatest(rows) {
  renderTable("overview-latest-table", rows, [
    ["label", "Family"],
    ["latest_date", "Latest date"],
    ["latest_value", "Latest daily", x => fmtPct(x)],
  ]);
}

function renderOverviewMonthly(rows) {
  renderTable("overview-monthly-table", rows, [
    ["label", "Family"],
    ["m1_label", "Month - 1"],
    ["m1_value", "Value", x => fmtPct(x)],
    ["m2_label", "Month - 2"],
    ["m2_value", "Value", x => fmtPct(x)],
    ["m3_label", "Month - 3"],
    ["m3_value", "Value", x => fmtPct(x)],
  ]);
}

function renderOverviewDownloads(id, familyDownloads) {
  const el = byId(id);
  if (!el) return;
  const entries = Object.entries(familyDownloads || {});
  el.innerHTML = entries.map(([family, files]) => `
    <div class="panel compact-panel">
      <div class="panel-title">${family}</div>
      ${files.map(x => `<div class="download-item"><a class="btn-link" href="${x.path}" target="_blank" rel="noopener">${x.label}</a></div>`).join("")}
    </div>
  `).join("");
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

function attachRawTabBehavior() {
  const buttons = document.querySelectorAll(".tabbtn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tabbtn").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".rawtab").forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      const target = byId(`raw-tab-${btn.dataset.tab}`);
      if (target) target.classList.add("active");
    });
  });
}

async function renderOverviewPage(warnings) {
  const data = await safeLoad("./data/overview.json", "overview.json", warnings);
  if (!data) return;
  setText("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  renderOverviewLatest(data.overview_rows || []);
  renderOverviewMonthly(data.overview_rows || []);
  renderOverviewDownloads("overview-downloads", data.family_downloads || {});
}

async function renderFamilyPage(page, warnings) {
  const data = await safeLoad(`./data/${page}.json`, `${page}.json`, warnings);
  if (!data) return;
  setText("asof", data.asof_date ? `As of ${data.asof_date}` : "");

  if (page === "marketwide") {
    renderCards("marketwide-latest-cards", [
      { label: "VW ICC", value: fmtPct(data.latest_daily?.vw_icc) },
      { label: "EW ICC", value: fmtPct(data.latest_daily?.ew_icc) },
      { label: "N Firms", value: fmtInt(data.latest_daily?.n_firms) },
    ]);
    renderTable("marketwide-monthly-table", data.last_three_months || [], [
      ["yyyymm", "Month"],
      ["date", "Date"],
      ["vw_icc", "VW ICC", x => fmtPct(x)],
      ["ew_icc", "EW ICC", x => fmtPct(x)],
      ["n_firms", "N Firms", x => fmtInt(x)],
    ]);
    renderTable("marketwide-daily-table", (data.daily_history || []).slice().reverse(), [
      ["date", "Date"],
      ["vw_icc", "VW ICC", x => fmtPct(x)],
      ["ew_icc", "EW ICC", x => fmtPct(x)],
      ["n_firms", "N Firms", x => fmtInt(x)],
    ]);
    renderDownloadList("marketwide-downloads", data.downloads || []);
  }

  if (page === "value") {
    renderCards("value-latest-cards", [
      { label: "Value ICC", value: fmtPct(data.latest_daily?.value_icc) },
      { label: "Growth ICC", value: fmtPct(data.latest_daily?.growth_icc) },
      { label: "IVP (B/M)", value: fmtPct(data.latest_daily?.ivp_bm) },
    ]);
    renderTable("value-monthly-table", data.last_three_months || [], [
      ["yyyymm", "Month"],
      ["date", "Date"],
      ["value_icc", "Value ICC", x => fmtPct(x)],
      ["growth_icc", "Growth ICC", x => fmtPct(x)],
      ["ivp_bm", "IVP (B/M)", x => fmtPct(x)],
    ]);
    renderTable("value-daily-table", (data.daily_history || []).slice().reverse(), [
      ["date", "Date"],
      ["value_icc", "Value ICC", x => fmtPct(x)],
      ["growth_icc", "Growth ICC", x => fmtPct(x)],
      ["ivp_bm", "IVP (B/M)", x => fmtPct(x)],
    ]);
    renderDownloadList("value-downloads", data.downloads || []);
  }

  if (page === "industry") {
    renderCards("industry-latest-cards", [
      { label: "Industry ICC (EW sectors)", value: fmtPct(data.latest_summary?.industry_icc_eq_sector) },
      { label: "Industry ICC (cap-weighted sectors)", value: fmtPct(data.latest_summary?.industry_icc_cap_sector) },
      { label: "N Sectors", value: fmtInt(data.latest_summary?.n_sectors) },
    ]);
    renderTable("industry-monthly-table", data.last_three_months || [], [
      ["yyyymm", "Month"],
      ["date", "Date"],
      ["industry_icc_eq_sector", "EW sectors", x => fmtPct(x)],
      ["industry_icc_cap_sector", "Cap-weighted sectors", x => fmtPct(x)],
      ["n_sectors", "N Sectors", x => fmtInt(x)],
    ]);
    renderTable("industry-latest-table", data.latest_table || [], [
      ["sector", "Sector"],
      ["vw_icc", "VW ICC", x => fmtPct(x)],
      ["ew_icc", "EW ICC", x => fmtPct(x)],
      ["n_firms", "N Firms", x => fmtInt(x)],
    ]);
    renderDownloadList("industry-downloads", data.downloads || []);
  }

  if (page === "etf") {
    renderCards("etf-latest-cards", [
      { label: "ETF ICC", value: fmtPct(data.latest_summary?.etf_icc) },
      { label: "N ETFs", value: fmtInt(data.latest_summary?.n_items) },
    ]);
    setText("etf-note", data.note || "");
    renderTable("etf-monthly-table", data.last_three_months || [], [
      ["yyyymm", "Month"],
      ["date", "Date"],
      ["etf_icc", "ETF ICC", x => fmtPct(x)],
      ["n_items", "N ETFs", x => fmtInt(x)],
    ]);
    renderTable("etf-latest-table", data.latest_table || [], [
      ["ticker", "Ticker"],
      ["label", "Label"],
      ["category", "Category"],
      ["vw_icc", "ICC", x => fmtPct(x)],
      ["coverage_weight", "Coverage", x => fmtPct(x)],
      ["status", "Status"],
    ]);
    renderDownloadList("etf-downloads", data.downloads || []);
  }

  if (page === "country") {
    renderCards("country-latest-cards", [
      { label: "Country ICC", value: fmtPct(data.latest_summary?.country_icc) },
      { label: "N Country proxies", value: fmtInt(data.latest_summary?.n_items) },
    ]);
    setText("country-note", data.note || "");
    renderTable("country-monthly-table", data.last_three_months || [], [
      ["yyyymm", "Month"],
      ["date", "Date"],
      ["country_icc", "Country ICC", x => fmtPct(x)],
      ["n_items", "N Proxies", x => fmtInt(x)],
    ]);
    renderTable("country-latest-table", data.latest_table || [], [
      ["country", "Country"],
      ["ticker", "Proxy ETF"],
      ["label", "Label"],
      ["vw_icc", "ICC", x => fmtPct(x)],
      ["coverage_weight", "Coverage", x => fmtPct(x)],
      ["status", "Status"],
    ]);
    renderDownloadList("country-downloads", data.downloads || []);
  }

  if (page === "indices") {
    renderCards("indices-latest-cards", [
      { label: "Index ICC", value: fmtPct(data.latest_summary?.index_icc) },
      { label: "N Indices", value: fmtInt(data.latest_summary?.n_indices) },
    ]);
    renderTable("indices-monthly-table", data.last_three_months || [], [
      ["yyyymm", "Month"],
      ["date", "Date"],
      ["index_icc", "Index ICC", x => fmtPct(x)],
      ["n_indices", "N Indices", x => fmtInt(x)],
    ]);
    renderTable("indices-latest-table", data.latest_table || [], [
      ["universe", "Index"],
      ["date", "Date"],
      ["vw_icc", "VW ICC", x => fmtPct(x)],
      ["ew_icc", "EW ICC", x => fmtPct(x)],
      ["n_firms", "N Firms", x => fmtInt(x)],
    ]);
    renderDownloadList("indices-downloads", data.downloads || []);
  }
}

async function renderDownloadsPage(warnings) {
  const data = await safeLoad("./data/downloads_catalog.json", "downloads_catalog.json", warnings);
  if (!data) return;
  setText("asof", data.asof_date ? `As of ${data.asof_date}` : "");
  renderDownloadsSections("downloads-sections", data.family_downloads || {});
  const tabs = data.raw_catalog?.tabs || {};
  renderRawTree("raw-tab-usall", tabs.usall?.years || []);
  renderRawTree("raw-tab-sp500", tabs.sp500?.years || []);
  renderRawTree("raw-tab-other_indices", tabs.other_indices?.years || []);
  attachRawTabBehavior();
}

(async function () {
  const warnings = [];
  const page = document.body?.dataset?.page || "overview";

  if (page === "overview") {
    await renderOverviewPage(warnings);
  } else if (["marketwide", "value", "industry", "etf", "country", "indices"].includes(page)) {
    await renderFamilyPage(page, warnings);
  } else if (page === "downloads") {
    await renderDownloadsPage(warnings);
  }

  if (warnings.length) {
    setStatus(`Partial load: ${warnings.join(" | ")}`, true);
  } else {
    setStatus("");
  }
})();
