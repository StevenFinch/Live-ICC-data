async function j(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`Failed to fetch ${path}: ${r.status}`);
  return await r.json();
}
function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "";
  return Number(x).toFixed(digits);
}
function esc(x) {
  return String(x ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}
function makeCards(cards) {
  const el = document.getElementById("cards");
  const items = [["Market VW ICC", cards.market_vw_icc], ["Value ICC", cards.value_icc], ["Growth ICC", cards.growth_icc], ["IVP (B/M)", cards.ivp_bm]];
  el.innerHTML = items.map(([label, value]) => `<div class="card"><div class="card-label">${esc(label)}</div><div class="card-value">${fmt(value)}</div></div>`).join("");
}
function makeDownloads(items) {
  const el = document.getElementById("downloads");
  el.innerHTML = items.map(d => `<a class="download" href="${esc(d.path)}" target="_blank" rel="noopener">${esc(d.label)}</a>`).join("");
}
function renderTable(id, rows, columns) {
  const el = document.getElementById(id);
  if (!rows || !rows.length) { el.innerHTML = "<tbody><tr><td>No data</td></tr></tbody>"; return; }
  const thead = `<thead><tr>${columns.map(c => `<th>${esc(c[1])}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${rows.map(r => `<tr>${columns.map(c => `<td>${typeof c[2] === "function" ? c[2](r[c[0]], r) : esc(r[c[0]] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody>`;
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
  try {
    wireTabs();
    const overview = await j("./data/overview.json");
    document.getElementById("asof").textContent = `As of ${overview.asof_date}`;
    makeCards(overview.cards);
    makeDownloads(overview.downloads || []);
    const market = await j("./data/market_icc.json");
    renderTable("market-table", market.history.slice().reverse(), [["date", "Date"], ["vw_icc", "VW ICC", x => fmt(x)], ["ew_icc", "EW ICC", x => fmt(x)], ["n_firms", "N Firms"]]);
    const value = await j("./data/value_icc_bm.json");
    renderTable("value-table", value.history.slice().reverse(), [["date", "Date"], ["value_icc", "Value ICC", x => fmt(x)], ["growth_icc", "Growth ICC", x => fmt(x)], ["ivp_bm", "IVP (B/M)", x => fmt(x)], ["n_firms", "N Firms"]]);
    const industry = await j("./data/industry_icc.json");
    renderTable("industry-table", industry.latest, [["sector", "Industry"], ["vw_icc", "VW ICC", x => fmt(x)], ["ew_icc", "EW ICC", x => fmt(x)], ["n_firms", "N Firms"]]);
    const etf = await j("./data/etf_icc.json");
    renderTable("etf-table", etf.latest, [["ticker", "Ticker"], ["label", "Label"], ["category", "Category"], ["vw_icc", "ETF ICC", x => fmt(x)], ["coverage_weight", "Coverage", x => fmt(x, 3)], ["n_holdings", "Holdings"], ["n_matched", "Matched"], ["holding_source", "Source"], ["status", "Status"]]);
    const country = await j("./data/country_icc.json");
    renderTable("country-table", country.latest, [["country", "Country"], ["ticker", "Proxy ETF"], ["vw_icc", "Country ICC", x => fmt(x)], ["coverage_weight", "Coverage", x => fmt(x, 3)], ["n_holdings", "Holdings"], ["n_matched", "Matched"], ["status", "Status"]]);
  } catch (err) {
    console.error(err);
    document.getElementById("cards").innerHTML = `<div class="error">Failed to load site data. Check docs/data/*.json deployment.</div>`;
  }
})();
