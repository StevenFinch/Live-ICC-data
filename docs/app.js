
async function fetchJson(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${path}`);
  return await r.json();
}

function $(id) {
  return document.getElementById(id);
}

function pageName() {
  return document.body.dataset.page || "overview";
}

function status(msg, isError = false) {
  const el = $("status");
  if (!el) return;
  el.textContent = msg || "";
  el.className = isError ? "status error" : "status";
}

function pct(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return `${(Number(x) * 100).toFixed(2)}%`;
}

function num(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x).toLocaleString();
}

function clean(x) {
  return x === null || x === undefined ? "" : String(x);
}

function table(id, rows, cols) {
  const el = $(id);
  if (!el) return;
  if (!rows || !rows.length) {
    el.innerHTML = `<div class="empty">No data available.</div>`;
    return;
  }
  const head = `<thead><tr>${cols.map(c => `<th>${c[1]}</th>`).join("")}</tr></thead>`;
  const body = rows.map(r => `<tr>${cols.map(c => {
    const v = r[c[0]];
    return `<td>${typeof c[2] === "function" ? c[2](v, r) : clean(v)}</td>`;
  }).join("")}</tr>`).join("");
  el.innerHTML = `<table>${head}<tbody>${body}</tbody></table>`;
}

function downloadsBox(id, d) {
  const el = $(id);
  if (!el) return;
  if (!d) {
    el.innerHTML = `<div class="empty">No downloads available.</div>`;
    return;
  }
  el.innerHTML = `
    <div class="download-grid">
      <a class="download-card" href="${d.latest}" target="_blank">Latest CSV</a>
      <a class="download-card" href="${d.daily}" target="_blank">Daily history CSV</a>
      <a class="download-card" href="${d.monthly}" target="_blank">Monthly history CSV</a>
      <a class="download-card primary" href="${d.zip}" target="_blank">Download all ZIP</a>
    </div>`;
}

function lastThreeMonthly(rows, groupKey) {
  if (!rows) return [];
  if (!groupKey) return rows.slice().sort((a,b)=>clean(b.month_end_date).localeCompare(clean(a.month_end_date))).slice(0,3);
  const out = [];
  const groups = {};
  for (const r of rows) {
    const g = clean(r[groupKey]);
    if (!groups[g]) groups[g] = [];
    groups[g].push(r);
  }
  for (const g of Object.keys(groups).sort()) {
    out.push(...groups[g].sort((a,b)=>clean(b.month_end_date).localeCompare(clean(a.month_end_date))).slice(0,3));
  }
  return out;
}

async function renderOverview() {
  const [m, v, i, e, c, idx] = await Promise.all([
    fetchJson("./data/marketwide.json"),
    fetchJson("./data/value.json"),
    fetchJson("./data/industry.json"),
    fetchJson("./data/etf.json"),
    fetchJson("./data/country.json"),
    fetchJson("./data/indices.json"),
  ]);

  const rows = [];
  function add(family, latestRows, monthlyRows, latestKey="daily_icc", groupKey=null) {
    const latest = latestRows && latestRows.length ? latestRows[0] : {};
    const monthly = lastThreeMonthly(monthlyRows || [], groupKey);
    rows.push({
      family,
      latest_daily: latest[latestKey],
      method: latest.method || "",
      m1: monthly[0] ? (monthly[0][latestKey] ?? monthly[0].daily_icc ?? monthly[0].ivp) : null,
      m2: monthly[1] ? (monthly[1][latestKey] ?? monthly[1].daily_icc ?? monthly[1].ivp) : null,
      m3: monthly[2] ? (monthly[2][latestKey] ?? monthly[2].daily_icc ?? monthly[2].ivp) : null,
    });
  }

  const allLatest = (m.latest || []).filter(x => x.family === "all_market");
  const spLatest = (m.latest || []).filter(x => x.family === "sp500");
  add("All market", allLatest, (m.monthly || []).filter(x => x.family === "all_market"), "daily_icc");
  add("S&P 500", spLatest, (m.monthly || []).filter(x => x.family === "sp500"), "daily_icc");
  add("Value premium", v.latest || [], v.monthly || [], "ivp");
  add("Industry", i.latest || [], i.monthly || [], "daily_icc", "group");
  add("ETF", e.latest || [], e.monthly || [], "icc", "ticker");
  add("Country ADR", c.latest || [], c.monthly || [], "icc", "country");
  add("Indices", idx.latest || [], idx.monthly || [], "daily_icc", "family");

  table("overview-table", rows, [
    ["family", "Dataset"],
    ["latest_daily", "Latest daily", pct],
    ["method", "Method"],
    ["m1", "Latest month", pct],
    ["m2", "Previous month", pct],
    ["m3", "Third month", pct],
  ]);
}

async function renderFamily(name) {
  const data = await fetchJson(`./data/${name}.json`);
  downloadsBox("family-downloads", data.downloads);

  if (name === "marketwide") {
    const allLatest = (data.latest || []).filter(x => x.family === "all_market");
    const spLatest = (data.latest || []).filter(x => x.family === "sp500");
    table("latest-table", [...allLatest, ...spLatest], [
      ["family", "Family"],
      ["date", "Date"],
      ["daily_icc", "Daily ICC", pct],
      ["ew_icc", "EW ICC", pct],
      ["n_firms", "N firms", num],
      ["method", "Method"],
    ]);
    const rows = [
      ...lastThreeMonthly((data.monthly || []).filter(x => x.family === "all_market"), "family"),
      ...lastThreeMonthly((data.monthly || []).filter(x => x.family === "sp500"), "family"),
    ];
    table("monthly-table", rows, [
      ["family", "Family"],
      ["month_end_date", "Month-end date"],
      ["daily_icc", "Monthly ICC", pct],
      ["n_firms", "N firms", num],
      ["method", "Method"],
    ]);
    return;
  }

  const groupKey = name === "industry" ? "group" : name === "indices" ? "family" : name === "etf" ? "ticker" : name === "country" ? "country" : null;
  table("latest-table", data.latest || [], genericCols(name, false));
  table("monthly-table", lastThreeMonthly(data.monthly || [], groupKey), genericCols(name, true));
}

function genericCols(name, monthly) {
  if (name === "value") {
    return [
      [monthly ? "month_end_date" : "date", monthly ? "Month-end date" : "Date"],
      ["value_icc", "Value ICC", pct],
      ["growth_icc", "Growth ICC", pct],
      ["ivp", "IVP", pct],
      ["method", "Method"],
    ];
  }
  if (name === "etf") {
    return [
      [monthly ? "month_end_date" : "date", monthly ? "Month-end date" : "Date"],
      ["ticker", "ETF"],
      ["label", "Name"],
      [monthly ? "daily_icc" : "icc", "ICC", pct],
      ["coverage_weight", "Coverage", pct],
      ["method", "Method"],
      ["holding_source", "Source"],
    ];
  }
  if (name === "country") {
    return [
      [monthly ? "month_end_date" : "date", monthly ? "Month-end date" : "Date"],
      ["country", "Country"],
      [monthly ? "daily_icc" : "icc", "ICC", pct],
      ["n_icc_available", "Available ADRs", num],
      ["coverage_mktcap", "Coverage", pct],
      ["method", "Method"],
    ];
  }
  return [
    [monthly ? "month_end_date" : "date", monthly ? "Month-end date" : "Date"],
    [name === "industry" ? "group" : "family", name === "industry" ? "Group" : "Family"],
    ["daily_icc", "ICC", pct],
    ["n_firms", "N firms", num],
    ["method", "Method"],
  ];
}

function renderRawTree(rows, group) {
  const el = $("raw-tree");
  if (!el) return;
  const filtered = (rows || []).filter(r => r.raw_group === group);
  if (!filtered.length) {
    el.innerHTML = `<div class="empty">No raw snapshots in this group.</div>`;
    return;
  }
  const byYear = {};
  for (const r of filtered) {
    if (!byYear[r.year]) byYear[r.year] = {};
    if (!byYear[r.year][r.month]) byYear[r.year][r.month] = [];
    byYear[r.year][r.month].push(r);
  }
  el.innerHTML = Object.keys(byYear).sort().reverse().map(year => `
    <details class="tree" open>
      <summary>${year}</summary>
      ${Object.keys(byYear[year]).sort().reverse().map(month => `
        <details class="tree nested">
          <summary>${year}-${month}</summary>
          ${byYear[year][month].sort((a,b)=>clean(b.date).localeCompare(clean(a.date))).map(r => `
            <div class="leaf">
              <span>${r.date} · ${r.universe} · ${num(r.n_firms)} firms</span>
              <a href="${r.download_path}" target="_blank">CSV</a>
            </div>
          `).join("")}
        </details>`).join("")}
    </details>`).join("");
}

async function renderDownloads() {
  const data = await fetchJson("./data/downloads_catalog.json");
  const families = data.families || {};
  $("category-downloads").innerHTML = Object.keys(families).map(k => `
    <div class="download-section">
      <h3>${k}</h3>
      <div class="download-grid">
        <a class="download-card" href="${families[k].latest}" target="_blank">Latest CSV</a>
        <a class="download-card" href="${families[k].daily}" target="_blank">Daily history CSV</a>
        <a class="download-card" href="${families[k].monthly}" target="_blank">Monthly history CSV</a>
        <a class="download-card primary" href="${families[k].zip}" target="_blank">Download all ZIP</a>
      </div>
    </div>
  `).join("");
  const buttons = document.querySelectorAll("[data-raw-tab]");
  let current = "usall";
  const draw = () => renderRawTree(data.raw_snapshots || [], current);
  buttons.forEach(b => b.addEventListener("click", () => {
    buttons.forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    current = b.dataset.rawTab;
    draw();
  }));
  draw();
}

(async function main() {
  try {
    const page = pageName();
    if (page === "overview") await renderOverview();
    else if (page === "downloads") await renderDownloads();
    else await renderFamily(page);
    status("");
  } catch (e) {
    console.error(e);
    status(`Failed to load page data: ${e.message}`, true);
  }
})();
