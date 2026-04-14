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

function parseDateSafe(s) {
  if (!s) return null;
  const d = new Date(`${s}T00:00:00`);
  return Number.isNaN(d.getTime()) ? null : d;
}

function filterLastMonth(rows) {
  if (!rows || !rows.length) return [];
  const dated = rows
    .map(r => ({ ...r, __d: parseDateSafe(r.date) }))
    .filter(r => r.__d !== null);

  if (!dated.length) return rows;

  const maxDate = dated.reduce((a, b) => (a.__d > b.__d ? a : b)).__d;
  const cutoff = new Date(maxDate);
  cutoff.setDate(cutoff.getDate() - 31);

  return dated
    .filter(r => r.__d >= cutoff)
    .sort((a, b) => b.__d - a.__d)
    .map(({ __d, ...rest }) => rest);
}

function renderDownloadList(id, items) {
  const el = byId(id);
  if (!el) return;

  if (!items || !items.length) {
    el.innerHTML = `<div class="empty">No download files</div>`;
    return;
  }

  el.innerHTML = `
    <div class="download-list">
      ${items.map(x => `
        <div class="download-item">
          <a class="btn-link" href="${x.path}" target="_blank" rel="noopener">${x.label}</a>
        </div>
      `).join("")}
    </div>
  `;
}

function renderRawTree(id, rows) {
  const el = byId(id);
  if (!el) return;

  if (!rows || !rows.length) {
    el.innerHTML = `<div class="empty">No raw snapshot files</div>`;
    return;
  }

  const tree = {};
  for (const r of rows) {
    const yyyymm = String(r.yyyymm || "");
    const year = yyyymm.slice(0, 4);
    const month = yyyymm.slice(4, 6);
    if (!tree[year]) tree[year] = {};
    if (!tree[year][month]) tree[year][month] = [];
    tree[year][month].push(r);
  }

  const yearKeys = Object.keys(tree).sort().reverse();

  el.innerHTML = yearKeys.map(year => {
    const monthKeys = Object.keys(tree[year]).sort().reverse();

    return `
      <details class="tree-level">
        <summary><strong>${year}</strong></summary>
        <div class="tree-body">
          ${monthKeys.map(month => {
            const rowsMonth = tree[year][month]
              .slice()
              .sort((a, b) => String(b.date).localeCompare(String(a.date)));

            return `
              <details class="tree-level nested">
                <summary>${year}-${month}</summary>
                <div class="tree-body">
                  ${rowsMonth.map(r => `
                    <div class="tree-leaf">
                      <span>${r.date} | ${r.universe} | ${fmtInt(r.n_firms)} firms</span>
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

async function safeLoad(path, label, warnings) {
  try {
    return await fetchJson(path);
  } catch (err) {
    console.error(`[${label}]`, err);
    warnings.push(`${label} failed`);
    return null;
  }
}

async function renderHome(warnings) {
  const overviewData = await safeLoad("./data/overview.json", "overview.json", warnings);
  const marketData = await safeLoad("./data/market_icc.json", "market_icc.json", warnings);
  const valueData = await safeLoad("./data/value_icc_bm.json", "value_icc_bm.json", warnings);
  const indexData = await safeLoad("./data/index_icc.json", "index_icc.json", warnings);
  const industryData = await safeLoad("./data/industry_icc.json", "industry_icc.json", warnings);

  if (overviewData) {
    setText("asof", overviewData.asof_date ? `As of ${overviewData.asof_date}` : "");
    const cards = overviewData.cards || {};

    if (byId("all-cards")) {
      renderCards("all-cards", [
        { label: "VW ICC", value: fmtPct(cards.all_market_vw_icc ?? cards.market_vw_icc) },
        { label: "EW ICC", value: fmtPct(cards.all_market_ew_icc ?? cards.market_ew_icc) },
        { label: "N Firms", value: fmtInt(cards.all_n_firms ?? cards.n_firms) },
      ]);
    }

    if (byId("sp500-cards")) {
      renderCards("sp500-cards", [
        { label: "VW ICC", value: fmtPct(cards.sp500_vw_icc) },
        { label: "EW ICC", value: fmtPct(cards.sp500_ew_icc) },
        { label: "N Firms", value: fmtInt(cards.sp500_n_firms) },
      ]);
    }

    if (byId("style-cards")) {
      renderCards("style-cards", [
        { label: "Value ICC", value: fmtPct(cards.value_icc) },
        { label: "Growth ICC", value: fmtPct(cards.growth_icc) },
        { label: "IVP (B/M)", value: fmtPct(cards.ivp_bm) },
      ]);
    }
  }

  if (marketData && byId("market-table")) {
    const recentAll = filterLastMonth(marketData.history || []);
    renderTable("market-table", recentAll, [
      ["date", "Date"],
      ["vw_icc", "VW ICC", x => fmtPct(x)],
      ["ew_icc", "EW ICC", x => fmtPct(x)],
      ["n_firms", "N Firms", x => fmtInt(x)],
    ]);
  }

  if (indexData) {
    if (byId("sp500-table")) {
      const sp500History = (indexData.history || []).filter(x => x.universe === "sp500");
      const recentSP500 = filterLastMonth(sp500History);
      renderTable("sp500-table", recentSP500, [
        ["date", "Date"],
        ["vw_icc", "VW ICC", x => fmtPct(x)],
        ["ew_icc", "EW ICC", x => fmtPct(x)],
        ["n_firms", "N Firms", x => fmtInt(x)],
      ]);
    }

    if (byId("index-latest-table")) {
      const latestOther = (indexData.latest || [])
        .filter(x => x.universe !== "sp500")
        .filter(x => x.vw_icc !== null || x.ew_icc !== null)
        .sort((a, b) => String(a.universe).localeCompare(String(b.universe)));

      renderTable("index-latest-table", latestOther, [
        ["universe", "Index"],
        ["date", "Date"],
        ["vw_icc", "VW ICC", x => fmtPct(x)],
        ["ew_icc", "EW ICC", x => fmtPct(x)],
        ["n_firms", "N Firms", x => fmtInt(x)],
      ]);
    }
  }

  if (valueData && byId("value-table")) {
    const recentValue = filterLastMonth(valueData.history || []);
    renderTable("value-table", recentValue, [
      ["date", "Date"],
      ["value_icc", "Value ICC", x => fmtPct(x)],
      ["growth_icc", "Growth ICC", x => fmtPct(x)],
      ["ivp_bm", "IVP (B/M)", x => fmtPct(x)],
      ["n_firms", "N Firms", x => fmtInt(x)],
    ]);
  }

  if (industryData && byId("industry-table")) {
    renderTable("industry-table", industryData.latest || [], [
      ["sector", "Industry"],
      ["vw_icc", "VW ICC", x => fmtPct(x)],
      ["ew_icc", "EW ICC", x => fmtPct(x)],
      ["n_firms", "N Firms", x => fmtInt(x)],
    ]);
  }
}

async function renderDownloadsPage(warnings) {
  const overviewData = await safeLoad("./data/overview.json", "overview.json", warnings);
  const allFilesData = await safeLoad("./data/all_snapshot_files.json", "all_snapshot_files.json", warnings);

  if (overviewData) {
    setText("asof", overviewData.asof_date ? `As of ${overviewData.asof_date}` : "");
    renderDownloadList("aggregate-downloads", overviewData.downloads || []);
  }

  if (allFilesData) {
    renderRawTree("raw-download-tree", allFilesData.rows || []);
  }
}

(async function () {
  const warnings = [];

  if (byId("aggregate-downloads") || byId("raw-download-tree")) {
    await renderDownloadsPage(warnings);
  } else {
    await renderHome(warnings);
  }

  if (warnings.length) {
    setStatus(`Partial load: ${warnings.join(" | ")}`, true);
  } else {
    setStatus("");
  }
})();
