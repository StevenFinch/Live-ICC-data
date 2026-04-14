
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
    .replaceAll('"', "&quot;");
}

function renderNav(navItems) {
  const page = document.body.dataset.page;
  const el = byId("nav");
  if (!el) return;
  el.innerHTML = (navItems || []).map(x => {
    const href = x.href || "#";
    const label = x.label || href;
    const active =
      (page === "home" && href === "index.html") ||
      (page === "marketwide" && href === "marketwide.html") ||
      (page === "value" && href === "value.html") ||
      (page === "indices" && href === "indices.html") ||
      (page === "industry" && href === "industry.html") ||
      (page === "downloads" && href === "downloads.html");
    return `<a href="${href}" class="${active ? "active" : ""}">${escapeHtml(label)}</a>`;
  }).join("");
}

function renderSimpleTable(targetId, rows, columns) {
  const el = byId(targetId);
  if (!el) return;
  if (!rows || !rows.length) {
    el.innerHTML = `<div class="empty">No data</div>`;
    return;
  }

  const thead = `<thead><tr>${columns.map(c => `<th>${escapeHtml(c.label)}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => `<tr>${
    columns.map(c => {
      const raw = r[c.key];
      const val = c.format ? c.format(raw, r) : escapeHtml(raw ?? "");
      return `<td>${val}</td>`;
    }).join("")
  }</tr>`).join("");

  el.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

function renderMonthlyMatrix(targetId, months, rows) {
  const el = byId(targetId);
  if (!el) return;
  if (!rows || !rows.length || !months || !months.length) {
    el.innerHTML = `<div class="empty">No data</div>`;
    return;
  }

  const thead = `<thead><tr><th>Series</th>${months.map(m => `<th>${escapeHtml(m)}</th>`).join("")}</tr></thead>`;
  const tbody = rows.map(r => {
    const tds = months.map(m => `<td>${fmtPct(r[m])}</td>`).join("");
    return `<tr><td>${escapeHtml(r.series)}</td>${tds}</tr>`;
  }).join("");

  el.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
}

function buildDownloadTabs(families) {
  const el = byId("download-tabs");
  if (!el) return;
  el.innerHTML = families.map((f, idx) =>
    `<button class="${idx === 0 ? "active" : ""}" data-family="${escapeHtml(f.family)}">${escapeHtml(f.label)}</button>`
  ).join("");
}

function renderDownloadTreeFamily(family) {
  const el = byId("download-tree");
  if (!el) return;
  if (!family || !family.years || !family.years.length) {
    el.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }

  const html = `
    <div class="tree">
      ${family.years.map(year => `
        <details>
          <summary>${escapeHtml(year.year)}</summary>
          <div class="actions">
            <a href="${year.download_all_href}" download>${escapeHtml(year.download_all_label)}</a>
          </div>
          ${year.months.map(month => `
            <details>
              <summary>${escapeHtml(month.month)}</summary>
              <div class="actions">
                <a href="${month.download_all_href}" download>${escapeHtml(month.download_all_label)}</a>
              </div>
              ${month.days.map(day => `
                <div class="day-row">
                  <a class="day-link" href="${day.href}" download>${escapeHtml(day.label)}</a>
                </div>
              `).join("")}
            </details>
          `).join("")}
        </details>
      `).join("")}
    </div>
  `;
  el.innerHTML = html;
}

async function boot() {
  const warnings = [];
  const page = document.body.dataset.page;

  let overview = null;
  try {
    overview = await fetchJson("./data/overview.json");
    byId("asof").textContent = overview.asof_date ? `As of ${overview.asof_date}` : "";
    renderNav(overview.nav || []);
  } catch (err) {
    warnings.push("overview.json failed");
    console.error(err);
  }

  try {
    if (page === "home") {
      const home = await fetchJson("./data/home_monthly.json");
      renderMonthlyMatrix("home-core-table", home.core?.months || [], home.core?.rows || []);
      renderMonthlyMatrix("home-index-table", home.index?.months || [], home.index?.rows || []);
      renderMonthlyMatrix("home-industry-table", home.industry?.months || [], home.industry?.rows || []);
    }

    if (page === "marketwide") {
      const d = await fetchJson("./data/marketwide_page.json");
      renderSimpleTable("marketwide-recent-table", d.recent_daily || [], [
        { key: "date", label: "Date" },
        { key: "series", label: "Group" },
        { key: "vw_icc", label: "VW ICC", format: x => fmtPct(x) },
        { key: "ew_icc", label: "EW ICC", format: x => fmtPct(x) },
        { key: "n_firms", label: "N Firms", format: x => fmtInt(x) },
      ]);
      renderMonthlyMatrix("marketwide-monthly-table", d.monthly?.months || [], (d.monthly?.rows || []).filter(x =>
        ["All market VW ICC", "All market EW ICC", "S&P 500 VW ICC", "S&P 500 EW ICC"].includes(x.series)
      ));
    }

    if (page === "value") {
      const d = await fetchJson("./data/value_page.json");
      renderSimpleTable("value-recent-table", d.recent_daily || [], [
        { key: "date", label: "Date" },
        { key: "value_icc", label: "Value ICC", format: x => fmtPct(x) },
        { key: "growth_icc", label: "Growth ICC", format: x => fmtPct(x) },
        { key: "ivp_bm", label: "IVP (B/M)", format: x => fmtPct(x) },
        { key: "n_firms", label: "N Firms", format: x => fmtInt(x) },
      ]);
      renderMonthlyMatrix("value-monthly-table", d.months || [], d.monthly_rows || []);
    }

    if (page === "indices") {
      const d = await fetchJson("./data/indices_page.json");
      renderSimpleTable("indices-latest-table", d.latest || [], [
        { key: "universe", label: "Index" },
        { key: "date", label: "Date" },
        { key: "vw_icc", label: "VW ICC", format: x => fmtPct(x) },
        { key: "ew_icc", label: "EW ICC", format: x => fmtPct(x) },
        { key: "n_firms", label: "N Firms", format: x => fmtInt(x) },
      ]);
      renderSimpleTable("indices-recent-table", d.recent_daily || [], [
        { key: "date", label: "Date" },
        { key: "universe", label: "Index" },
        { key: "vw_icc", label: "VW ICC", format: x => fmtPct(x) },
        { key: "ew_icc", label: "EW ICC", format: x => fmtPct(x) },
        { key: "n_firms", label: "N Firms", format: x => fmtInt(x) },
      ]);
      renderMonthlyMatrix("indices-monthly-table", d.months || [], d.monthly_rows || []);
    }

    if (page === "industry") {
      const d = await fetchJson("./data/industry_page.json");
      renderSimpleTable("industry-latest-table", d.latest || [], [
        { key: "date", label: "Date" },
        { key: "sector", label: "Industry" },
        { key: "vw_icc", label: "VW ICC", format: x => fmtPct(x) },
        { key: "ew_icc", label: "EW ICC", format: x => fmtPct(x) },
        { key: "n_firms", label: "N Firms", format: x => fmtInt(x) },
      ]);
      renderSimpleTable("industry-recent-table", d.recent_daily || [], [
        { key: "date", label: "Date" },
        { key: "sector", label: "Industry" },
        { key: "vw_icc", label: "VW ICC", format: x => fmtPct(x) },
        { key: "ew_icc", label: "EW ICC", format: x => fmtPct(x) },
        { key: "n_firms", label: "N Firms", format: x => fmtInt(x) },
      ]);
      renderMonthlyMatrix("industry-monthly-table", d.months || [], d.monthly_rows || []);
    }

    if (page === "downloads") {
      const d = await fetchJson("./data/download_tree.json");
      const families = d.families || [];
      buildDownloadTabs(families);
      if (families.length) {
        renderDownloadTreeFamily(families[0]);
      }
      const tabEl = byId("download-tabs");
      if (tabEl) {
        tabEl.addEventListener("click", (evt) => {
          const btn = evt.target.closest("button[data-family]");
          if (!btn) return;
          const fam = btn.getAttribute("data-family");
          tabEl.querySelectorAll("button").forEach(x => x.classList.remove("active"));
          btn.classList.add("active");
          const found = families.find(x => x.family === fam);
          renderDownloadTreeFamily(found);
        });
      }
    }
  } catch (err) {
    warnings.push(`${page} page data failed`);
    console.error(err);
  }

  if (warnings.length) {
    setStatus(`Partial load: ${warnings.join(" | ")}`, true);
  } else {
    setStatus("");
  }
}

boot();
