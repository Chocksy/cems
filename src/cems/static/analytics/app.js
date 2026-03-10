/* CEMS Analytics Dashboard */
(function () {
  "use strict";

  let adminKey = sessionStorage.getItem("cems_admin_key") || "";
  let charts = {};

  const $ = (sel) => document.querySelector(sel);

  function esc(str) {
    const d = document.createElement("div");
    d.textContent = String(str ?? "");
    return d.innerHTML;
  }

  function pct(n, d) {
    if (!d) return "—";
    return (n / d * 100).toFixed(1) + "%";
  }

  function rateClass(rate) {
    if (rate == null || rate === "—") return "";
    const v = parseFloat(rate);
    if (v >= 50) return "rate-good";
    if (v >= 25) return "rate-mid";
    return "rate-bad";
  }

  // --- API ---
  const baseUrl = window.location.origin;

  async function apiFetch(path) {
    const res = await fetch(baseUrl + path, {
      headers: { Authorization: "Bearer " + adminKey },
    });
    if (res.status === 401 || res.status === 403) {
      sessionStorage.removeItem("cems_admin_key");
      adminKey = "";
      showLogin();
      throw new Error("Unauthorized");
    }
    return res.json();
  }

  // --- Views ---
  function showLogin() {
    $("#login-view").hidden = false;
    $("#dashboard").hidden = true;
  }

  function showDashboard() {
    $("#login-view").hidden = true;
    $("#dashboard").hidden = false;
    loadData();
  }

  // --- Login ---
  $("#login-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    adminKey = $("#admin-key-input").value.trim();
    if (!adminKey) return;
    try {
      const data = await apiFetch("/admin/analytics/data");
      if (data.success) {
        sessionStorage.setItem("cems_admin_key", adminKey);
        showDashboard();
      } else {
        throw new Error(data.error || "Failed");
      }
    } catch (err) {
      $("#login-error").textContent = err.message;
      $("#login-error").hidden = false;
    }
  });

  $("#logout-btn").addEventListener("click", () => {
    sessionStorage.removeItem("cems_admin_key");
    adminKey = "";
    showLogin();
  });

  $("#refresh-btn").addEventListener("click", loadData);

  // --- Chart colors ---
  const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const textColor = isDark ? "#e5e5e5" : "#111";
  const gridColor = isDark ? "#333" : "#ddd";
  const COLORS = ["#3b82f6", "#22c55e", "#f97316", "#a855f7", "#ef4444", "#06b6d4", "#ec4899", "#eab308"];

  function chartDefaults() {
    return {
      responsive: true,
      maintainAspectRatio: true,
      plugins: { legend: { labels: { color: textColor, font: { size: 11 } } } },
      scales: {
        x: { ticks: { color: textColor, font: { size: 10 } }, grid: { color: gridColor } },
        y: { ticks: { color: textColor, font: { size: 10 } }, grid: { color: gridColor } },
      },
    };
  }

  // --- Load and render ---
  async function loadData() {
    try {
      const res = await apiFetch("/admin/analytics/data");
      if (!res.success) return;
      const d = res.data;

      renderKPIs(d.overview);
      renderSourceChart(d.by_source);
      renderLengthChart(d.by_length);
      renderTrendChart(d.weekly_trend);
      renderCategoryTable(d.by_category);
      renderNeverShownTable(d.never_shown);
      renderTopTable(d.top_relevant, "top-table");
      renderNoisyTable(d.top_noisy, "noisy-table");

      $("#last-updated").textContent = "Updated " + new Date().toLocaleTimeString();
    } catch (e) {
      console.error("Failed to load analytics", e);
    }
  }

  // --- KPIs ---
  function renderKPIs(o) {
    if (!o) return;
    const totalShown = o.total_shown || 0;
    const totalRelevant = o.total_relevant || 0;
    const totalNoise = o.total_noise || 0;
    const relevanceRate = totalShown ? (totalRelevant / totalShown * 100).toFixed(1) : "—";
    const noiseRate = totalShown ? (totalNoise / totalShown * 100).toFixed(1) : "—";
    const feedbackRate = o.total_memories ? ((o.memories_found_relevant + o.memories_found_noisy) / o.total_memories * 100).toFixed(0) : "—";

    const rateNum = parseFloat(relevanceRate);
    const rateKlass = isNaN(rateNum) ? "kpi-neutral" : rateNum >= 40 ? "kpi-good" : rateNum >= 20 ? "kpi-warn" : "kpi-bad";

    $("#kpi-row").innerHTML = `
      <div class="kpi-card kpi-neutral">
        <div class="kpi-value">${o.total_memories}</div>
        <div class="kpi-label">Total Memories</div>
        <div class="kpi-sub">${o.memories_shown} ever shown</div>
      </div>
      <div class="kpi-card kpi-neutral">
        <div class="kpi-value">${totalShown}</div>
        <div class="kpi-label">Total Shows</div>
        <div class="kpi-sub">${totalRelevant} relevant, ${totalNoise} noise</div>
      </div>
      <div class="kpi-card ${rateKlass}">
        <div class="kpi-value">${relevanceRate}%</div>
        <div class="kpi-label">Relevance Rate</div>
        <div class="kpi-sub">relevant / shown</div>
      </div>
      <div class="kpi-card ${parseFloat(noiseRate) > 30 ? 'kpi-bad' : 'kpi-neutral'}">
        <div class="kpi-value">${noiseRate}%</div>
        <div class="kpi-label">Noise Rate</div>
        <div class="kpi-sub">noise / shown</div>
      </div>
      <div class="kpi-card kpi-neutral">
        <div class="kpi-value">${feedbackRate}%</div>
        <div class="kpi-label">Feedback Coverage</div>
        <div class="kpi-sub">memories with any feedback</div>
      </div>
    `;
  }

  // --- Source Chart ---
  function renderSourceChart(data) {
    if (!data || !data.length) return;

    if (charts.source) charts.source.destroy();
    const ctx = $("#source-chart").getContext("2d");

    charts.source = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.map((d) => d.source_type),
        datasets: [
          {
            label: "Relevant",
            data: data.map((d) => d.relevant || 0),
            backgroundColor: "#22c55e",
            borderRadius: 3,
          },
          {
            label: "Noise",
            data: data.map((d) => d.noise || 0),
            backgroundColor: "#ef4444",
            borderRadius: 3,
          },
        ],
      },
      options: {
        ...chartDefaults(),
        plugins: {
          ...chartDefaults().plugins,
          tooltip: {
            callbacks: {
              afterBody: (items) => {
                const idx = items[0].dataIndex;
                const row = data[idx];
                const rate = row.relevance_rate != null ? (row.relevance_rate * 100).toFixed(1) + "%" : "N/A";
                return `Total: ${row.total} | Shown: ${row.shown} | Rate: ${rate}`;
              },
            },
          },
        },
      },
    });
  }

  // --- Length Chart ---
  function renderLengthChart(data) {
    if (!data || !data.length) return;

    if (charts.length) charts.length.destroy();
    const ctx = $("#length-chart").getContext("2d");

    charts.length = new Chart(ctx, {
      type: "bar",
      data: {
        labels: data.map((d) => d.length_bucket),
        datasets: [
          {
            label: "Relevance Rate %",
            data: data.map((d) => d.relevance_rate != null ? (d.relevance_rate * 100).toFixed(1) : 0),
            backgroundColor: data.map((_, i) => COLORS[i % COLORS.length]),
            borderRadius: 3,
          },
        ],
      },
      options: {
        ...chartDefaults(),
        plugins: {
          ...chartDefaults().plugins,
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterBody: (items) => {
                const idx = items[0].dataIndex;
                const row = data[idx];
                return `Total: ${row.total} | Shown: ${row.shown} | Rel: ${row.relevant} | Noise: ${row.noise}`;
              },
            },
          },
        },
        scales: {
          ...chartDefaults().scales,
          y: { ...chartDefaults().scales.y, beginAtZero: true, max: 100, title: { display: true, text: "%", color: textColor } },
        },
      },
    });
  }

  // --- Trend Chart ---
  function renderTrendChart(data) {
    if (!data || !data.length) return;

    if (charts.trend) charts.trend.destroy();
    const ctx = $("#trend-chart").getContext("2d");

    charts.trend = new Chart(ctx, {
      type: "line",
      data: {
        labels: data.map((d) => d.week),
        datasets: [
          {
            label: "Created",
            data: data.map((d) => d.created),
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59,130,246,.1)",
            tension: 0.3,
            fill: true,
          },
          {
            label: "Shown",
            data: data.map((d) => d.shown),
            borderColor: "#a855f7",
            tension: 0.3,
          },
          {
            label: "Relevant",
            data: data.map((d) => d.relevant),
            borderColor: "#22c55e",
            tension: 0.3,
          },
          {
            label: "Noise",
            data: data.map((d) => d.noise),
            borderColor: "#ef4444",
            tension: 0.3,
          },
        ],
      },
      options: {
        ...chartDefaults(),
        scales: {
          ...chartDefaults().scales,
          y: { ...chartDefaults().scales.y, beginAtZero: true },
        },
      },
    });
  }

  // --- Category Table ---
  function renderCategoryTable(data) {
    if (!data) return;
    const tbody = $("#category-table tbody");
    tbody.innerHTML = data
      .map((r) => {
        const rate = pct(r.relevant, r.shown);
        const rc = rateClass(rate);
        return `<tr>
          <td class="mono">${esc(r.category)}</td>
          <td>${r.total}</td>
          <td>${r.shown}</td>
          <td>${r.relevant}</td>
          <td>${r.noise}</td>
          <td class="${rc}">${rate}</td>
          <td>${r.avg_content_length || "—"}</td>
        </tr>`;
      })
      .join("");
  }

  // --- Never Shown Table ---
  function renderNeverShownTable(data) {
    if (!data) return;
    const tbody = $("#never-shown-table tbody");
    tbody.innerHTML = data
      .map((r) => `<tr>
        <td class="mono">${esc(r.category)}</td>
        <td>${r.never_shown_count}</td>
        <td>${r.avg_length || "—"}</td>
        <td>${esc(r.oldest)}</td>
      </tr>`)
      .join("");
  }

  // --- Top/Noisy Tables ---
  function renderTopTable(data, tableId) {
    if (!data) return;
    const tbody = $(`#${tableId} tbody`);
    tbody.innerHTML = data
      .map((r) => {
        const rate = pct(r.relevant_count, r.shown_count);
        return `<tr>
          <td class="preview-cell" title="${esc(r.preview)}">${esc(r.preview)}</td>
          <td class="mono">${esc(r.category)}</td>
          <td>${r.shown_count}</td>
          <td>${r.relevant_count}</td>
          <td>${r.noise_count}</td>
          <td class="${rateClass(rate)}">${rate}</td>
        </tr>`;
      })
      .join("");
  }

  function renderNoisyTable(data, tableId) {
    if (!data) return;
    const tbody = $(`#${tableId} tbody`);
    tbody.innerHTML = data
      .map((r) => {
        const rate = pct(r.noise_count, r.shown_count);
        return `<tr>
          <td class="preview-cell" title="${esc(r.preview)}">${esc(r.preview)}</td>
          <td class="mono">${esc(r.category)}</td>
          <td>${r.shown_count}</td>
          <td>${r.relevant_count}</td>
          <td>${r.noise_count}</td>
          <td class="${rateClass(rate)}">${rate}</td>
        </tr>`;
      })
      .join("");
  }

  // --- Init ---
  if (adminKey) {
    showDashboard();
  } else {
    showLogin();
  }
})();
