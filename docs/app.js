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

function table(id, rows, columns) {
  const el = $(id);
  if (!el) return;
  if (!rows || rows.length === 0) {
    el.innerHTML = `<div class="empty">No data available.</div>`;
    return;
  }
  const head = `<thead><tr>${columns.map(c => `<th>${c.label}</th>`).join("")}</tr></thead>`;
  const body = rows.map(row => {
    return `<tr>${columns.map(c => {
      const raw = row[c.key];
      const value = c.format ? c.format(raw, row) : fmtText(raw);
      return `<td>${value}</td>`;
    }).join("")}</tr>`;
  }).join("");
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
    return `<a class="download-card" href="${url}" download><strong>${title}</strong><span>${desc}</span></a>`;
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
    <details class="tree" open>
      <summary>
        <span>${year.year}</span>
        <a class="small-link" href="${year.download_all_zip}" download>Download year ZIP</a>
      </summary>
      <div class="tree-body">
        ${(year.months || []).map(month => `
          <details class="tree nested">
            <summary>
              <span>${month.month}</span>
              <a class="small-link" href="${month.download_all_zip}" download>Download month ZIP</a>
            </summary>
            <div class="tree-body">
              ${(month.days || []).map(day => `
                <div class="leaf">
                  <span>${day.date || ""} · ${fmtNum(day.n_rows || day.n_firms)} rows</span>
                  <a href="${day.csv}" download>CSV</a>
                </div>
              `).join("")}
            </div>
          </details>
        `).join("")}
      </div>
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
    <details class="tree" open>
      <summary>
        <span>${year.year}</span>
        <a class="small-link" href="${year.download_all_zip}" download>Download year ZIP</a>
      </summary>
      <div class="tree-body">
        ${(year.months || []).map(month => `
          <details class="tree nested">
            <summary>
              <span>${month.month}</span>
              <a class="small-link" href="${month.download_all_zip}" download>Download month ZIP</a>
            </summary>
            <div class="tree-body">
              ${(month.days || []).map(day => `
                <div class="leaf">
                  <span>${day.date} · ${day.universe} · ${fmtNum(day.n_firms)} firms</span>
                  <a href="${day.csv}" download>CSV</a>
                </div>
              `).join("")}
            </div>
          </details>
        `).join("")}
      </div>
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
      { key: monthly ? "icc" : "icc", label: "ICC / estimate", format: fmtPct },
      { key: "coverage_weight", label: "Coverage", format: fmtPct },
      { key: "method", label: "Method", format: methodBadge },
      { key: "holding_source", label: "Source" },
    ];
  }
  if (family === "country") {
    return [
      { key: "country", label: "Country" },
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
      return `<a class="download-card" href="${d.all_zip}" download><strong>${d.label}</strong><span>Download all standard files</span></a>`;
    }).join("");
  }
}

async function renderFamily(family) {
  const data = await fetchJson(`./data/${family}.json`);
  const title = $("family-title");
  const note = $("family-note");
  if (title) title.textContent = data.label || "";
  if (note) note.textContent = data.note || "";
  table("latest-table", data.latest || [], familyColumns(family, false));
  table("monthly-table", data.monthly || [], familyColumns(family, true));
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
          <p class="note">${d.note || ""}</p>
          <div class="download-grid">
            <a class="download-card" href="${d.latest_csv}" download><strong>Latest</strong><span>Latest CSV</span></a>
            <a class="download-card" href="${d.daily_history_csv}" download><strong>Daily history</strong><span>Full daily CSV</span></a>
            <a class="download-card" href="${d.monthly_history_csv}" download><strong>Monthly history</strong><span>Month-end CSV</span></a>
            <a class="download-card primary" href="${d.all_zip}" download><strong>Download all</strong><span>ZIP package</span></a>
          </div>
          <details class="tree compact">
            <summary>Open year/month/day archive</summary>
            <div class="tree-body">${renderArchiveTreeHtml(d.tree || [])}</div>
          </details>
        </section>
      `;
    }).join("");
  }
  setupRawTabs(data.raw_snapshots || {});
}

function renderArchiveTreeHtml(tree) {
  if (!tree.length) return `<div class="empty">No archive yet.</div>`;
  return tree.map(year => `
    <details class="tree" open>
      <summary><span>${year.year}</span><a class="small-link" href="${year.download_all_zip}" download>Download year ZIP</a></summary>
      <div class="tree-body">
        ${(year.months || []).map(month => `
          <details class="tree nested">
            <summary><span>${month.month}</span><a class="small-link" href="${month.download_all_zip}" download>Download month ZIP</a></summary>
            <div class="tree-body">
              ${(month.days || []).map(day => `<div class="leaf"><span>${day.date} · ${fmtNum(day.n_rows)} rows</span><a href="${day.csv}" download>CSV</a></div>`).join("")}
            </div>
          </details>
        `).join("")}
      </div>
    </details>
  `).join("");
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
