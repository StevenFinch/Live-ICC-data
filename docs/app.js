
async function j(path) {
  const r = await fetch(path);
  return await r.json();
}

function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x).toFixed(digits);
}

function makeCards(cards) {
  const el = document.getElementById("cards");
  const items = [
    ["Market VW ICC", cards.market_vw_icc],
    ["Value ICC", cards.value_icc],
    ["Growth ICC", cards.growth_icc],
    ["IVP (B/M)", cards.ivp_bm],
  ];
  el.innerHTML = items.map(([label, value]) => `
    <div class="card">
      <div class="label">${label}</div>
      <div class="value">${fmt(value)}</div>
    </div>
  `).join("");
}

function renderTable(id, rows, columns) {
  const el = document.getElementById(id);
  if (!rows || !rows.length) {
    el.innerHTML = "<tr><td>No data</td></tr>";
    return;
  }
  const thead = `<tr>${columns.map(c => `<th>${c[1]}</th>`).join("")}</tr>`;
  const tbody = rows.map(r => `<tr>${
    columns.map(c => `<td>${typeof c[2] === "function" ? c[2](r[c[0]], r) : (r[c[0]] ?? "")}</td>`).join("")
  }</tr>`).join("");
  el.innerHTML = thead + tbody;
}

function wireTabs() {
  document.querySelectorAll(".tab").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      document.querySelectorAll(".panel").forEach(x => x.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.target).classList.add("active");
    });
  });
}

(async function () {
  wireTabs();

  const overview = await j("./data/overview.json");
  makeCards(overview.cards);

  const market = await j("./data/market_icc.json");
  renderTable("market-table", market.history.slice().reverse(), [
    ["date", "Date"],
    ["vw_icc", "VW ICC", fmt],
    ["ew_icc", "EW ICC", fmt],
    ["n_firms", "N Firms"],
  ]);

  const value = await j("./data/value_icc_bm.json");
  renderTable("value-table", value.history.slice().reverse(), [
    ["date", "Date"],
    ["value_icc", "Value ICC", fmt],
    ["growth_icc", "Growth ICC", fmt],
    ["ivp_bm", "IVP (B/M)", fmt],
    ["n_firms", "N Firms"],
  ]);

  const industry = await j("./data/industry_icc.json");
  renderTable("industry-table", industry.latest, [
    ["sector", "Industry"],
    ["vw_icc", "VW ICC", fmt],
    ["ew_icc", "EW ICC", fmt],
    ["n_firms", "N Firms"],
  ]);

  const etf = await j("./data/etf_icc.json");
  renderTable("etf-table", etf.latest, [
    ["ticker", "Ticker"],
    ["label", "Label"],
    ["category", "Category"],
    ["vw_icc", "ETF ICC", fmt],
    ["coverage_weight", "Coverage", x => fmt(x, 3)],
    ["n_holdings", "Holdings"],
    ["holding_source", "Source"],
    ["status", "Status"],
  ]);

  const country = await j("./data/country_icc.json");
  renderTable("country-table", country.latest, [
    ["country", "Country"],
    ["ticker", "Proxy ETF"],
    ["vw_icc", "Country ICC", fmt],
    ["coverage_weight", "Coverage", x => fmt(x, 3)],
    ["n_holdings", "Holdings"],
    ["status", "Status"],
  ]);
})();
