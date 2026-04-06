/* CEMS Knowledge Engine Dashboard */
(function () {
  "use strict";

  // --- State ---
  let apiKey = sessionStorage.getItem("cems_api_key") || "";
  let currentView = "graph";
  let graphData = null;
  let simulation = null;

  // --- DOM refs ---
  const loginView = document.getElementById("login-view");
  const dashView = document.getElementById("dashboard-view");
  const loginForm = document.getElementById("login-form");
  const apiKeyInput = document.getElementById("api-key-input");
  const loginError = document.getElementById("login-error");
  const healthBadge = document.getElementById("health-badge");
  const statsPanel = document.getElementById("stats-panel");
  const graphView = document.getElementById("graph-view");
  const graphInfo = document.getElementById("graph-info");
  const detailPanel = document.getElementById("detail-panel");

  // --- API helpers ---
  const baseUrl = window.location.origin;

  async function apiFetch(path) {
    const res = await fetch(baseUrl + path, {
      headers: { Authorization: "Bearer " + apiKey },
    });
    if (res.status === 401) {
      sessionStorage.removeItem("cems_api_key");
      apiKey = "";
      showLogin();
      throw new Error("Unauthorized");
    }
    return res.json();
  }

  // --- Views ---
  function showLogin() {
    loginView.hidden = false;
    dashView.hidden = true;
    loginError.hidden = true;
  }

  function showDashboard() {
    loginView.hidden = true;
    dashView.hidden = false;
    loadStats();
    loadGraph();
  }

  // --- Login ---
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    apiKey = apiKeyInput.value.trim();
    if (!apiKey) return;
    try {
      const data = await apiFetch("/api/wiki/stats");
      if (data.success) {
        sessionStorage.setItem("cems_api_key", apiKey);
        showDashboard();
      } else {
        loginError.textContent = "Failed to connect";
        loginError.hidden = false;
      }
    } catch {
      loginError.textContent = "Invalid API key";
      loginError.hidden = false;
    }
  });

  // --- View Toggle ---
  document.querySelectorAll(".toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentView = btn.dataset.view;
      if (currentView === "stats") {
        statsPanel.classList.add("active");
        graphView.classList.add("hidden");
      } else {
        statsPanel.classList.remove("active");
        graphView.classList.remove("hidden");
      }
    });
  });

  // --- Stats ---
  async function loadStats() {
    try {
      const data = await apiFetch("/api/wiki/stats");
      if (!data.success) return;
      const s = data.stats;

      document.getElementById("stat-total").textContent = s.total_memories;
      document.getElementById("stat-relations").textContent = s.total_relations;
      document.getElementById("stat-connected").textContent = s.connected_memories;
      document.getElementById("stat-orphans").textContent = s.orphan_memories;
      document.getElementById("stat-conflicts").textContent = s.open_conflicts;
      document.getElementById("stat-avg-rel").textContent = s.avg_relations_per_doc;

      // Health badge
      healthBadge.textContent = s.health_score + "/100";
      healthBadge.className = "badge " + (
        s.health_score >= 80 ? "badge-good" :
        s.health_score >= 50 ? "badge-warn" : "badge-bad"
      );

      // Heat distribution bars
      renderBars("heat-bars", s.heat_tiers, {
        hot: "bar-hot", warm: "bar-warm", cool: "bar-cool", cold: "bar-cold",
      });

      // Category bars
      const catColors = {};
      const catKeys = Object.keys(s.categories);
      catKeys.forEach((k) => { catColors[k] = "background:var(--accent)"; });
      renderBarsDynamic("category-bars", s.categories);
    } catch (e) {
      console.error("Failed to load stats:", e);
    }
  }

  function renderBars(containerId, data, classMap) {
    const el = document.getElementById(containerId);
    const max = Math.max(...Object.values(data), 1);
    el.innerHTML = Object.entries(data).map(([key, val]) => {
      const pct = Math.max((val / max) * 100, 2);
      const cls = classMap[key] || "bar-cool";
      return `<div class="bar-row">
        <span class="bar-label">${escapeHtml(key)}</span>
        <div class="bar-fill ${cls}" style="width:${pct}%"></div>
        <span class="bar-count">${Number(val)}</span>
      </div>`;
    }).join("");
  }

  function renderBarsDynamic(containerId, data) {
    const el = document.getElementById(containerId);
    const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const max = Math.max(...entries.map(([, v]) => v), 1);
    el.innerHTML = entries.map(([key, val]) => {
      const pct = Math.max((val / max) * 100, 2);
      return `<div class="bar-row">
        <span class="bar-label">${escapeHtml(key)}</span>
        <div class="bar-fill bar-cool" style="width:${pct}%"></div>
        <span class="bar-count">${Number(val)}</span>
      </div>`;
    }).join("");
  }

  // --- Graph ---
  async function loadGraph() {
    try {
      const data = await apiFetch("/api/wiki/graph?limit=300");
      if (!data.success) return;

      graphData = data;
      graphInfo.textContent = `${data.node_count} nodes, ${data.edge_count} edges`;
      renderGraph(data.nodes, data.edges);
    } catch (e) {
      graphInfo.textContent = "Failed to load graph";
      console.error("Graph load error:", e);
    }
  }

  function getNodeColor(node) {
    const shown = node.shown_count || 0;
    if (shown >= 20) return "var(--hot)";
    if (shown >= 5) return "var(--warm)";
    if (shown >= 1) return "var(--cool)";
    return "var(--cold)";
  }

  function getNodeRadius(node) {
    const shown = node.shown_count || 0;
    if (shown >= 20) return 10;
    if (shown >= 5) return 7;
    if (shown >= 1) return 5;
    return 4;
  }

  function renderGraph(nodes, edges) {
    if (simulation) simulation.stop();
    const svg = d3.select("#graph-svg");
    svg.selectAll("*").remove();

    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;

    const g = svg.append("g");

    // Zoom
    const zoom = d3.zoom()
      .scaleExtent([0.2, 5])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    // Build node map for D3
    const nodeMap = new Map(nodes.map((n) => [n.id, { ...n }]));
    const d3Nodes = Array.from(nodeMap.values());

    // Build edges with object references
    const d3Edges = edges
      .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        similarity: e.similarity || 0.5,
      }));

    // Force simulation
    simulation = d3.forceSimulation(d3Nodes)
      .force("link", d3.forceLink(d3Edges).id((d) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d) => getNodeRadius(d) + 2));

    // Edges
    const link = g.append("g")
      .selectAll("line")
      .data(d3Edges)
      .join("line")
      .attr("class", "link")
      .attr("stroke-width", (d) => Math.max(0.5, (d.similarity || 0.5) * 3));

    // Nodes
    const node = g.append("g")
      .selectAll("g")
      .data(d3Nodes)
      .join("g")
      .attr("class", "node")
      .call(d3.drag()
        .on("start", dragStart)
        .on("drag", dragged)
        .on("end", dragEnd));

    node.append("circle")
      .attr("r", (d) => getNodeRadius(d))
      .attr("fill", (d) => getNodeColor(d))
      .on("click", (event, d) => showDetail(d));

    node.append("title")
      .text((d) => d.title || d.id.slice(0, 8));

    // Labels for hot nodes only (reduce clutter)
    node.filter((d) => (d.shown_count || 0) >= 5)
      .append("text")
      .attr("dy", (d) => getNodeRadius(d) + 12)
      .text((d) => (d.title || d.category || "").slice(0, 20));

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

    // Drag handlers
    function dragStart(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragEnd(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Recenter button
    document.getElementById("btn-recenter").onclick = () => {
      svg.transition().duration(500).call(
        zoom.transform,
        d3.zoomIdentity.translate(width / 2, height / 2).scale(0.8).translate(-width / 2, -height / 2)
      );
    };
  }

  // --- Detail Panel ---
  async function showDetail(node) {
    detailPanel.hidden = false;
    document.getElementById("detail-title").textContent = node.title || node.id.slice(0, 12);

    try {
      const data = await apiFetch(`/api/wiki/relations?id=${node.id}`);
      if (!data.success) return;

      const m = data.memory;
      document.getElementById("detail-content").innerHTML = `
        <div class="detail-meta">
          <strong>${escapeHtml(m.category || "general")}</strong>
          &middot; shown ${Number(m.shown_count) || 0}x
          &middot; ${escapeHtml(m.source_ref || "no project")}
          &middot; ${m.created_at ? new Date(m.created_at).toLocaleDateString() : ""}
        </div>
        <div class="detail-text">${escapeHtml(m.content)}</div>
      `;

      const rels = data.relations || [];
      document.getElementById("detail-relations").innerHTML = rels.length
        ? `<h3 style="margin:1rem 0 .5rem;font-size:.9rem;color:var(--fg2)">Related (${rels.length})</h3>` +
          rels.map((r) => `
            <div class="detail-relation" onclick="document.getElementById('detail-panel').hidden=true">
              <span class="relation-score">${r.similarity ? (r.similarity * 100).toFixed(0) + "%" : ""}</span>
              <div>${escapeHtml((r.content || "").slice(0, 150))}</div>
              <div class="relation-cat">${escapeHtml(r.category || "")} &middot; ${escapeHtml(r.relation_type || "similar")}</div>
            </div>
          `).join("")
        : `<p style="color:var(--fg2);font-size:.85rem;margin-top:1rem">No relations found</p>`;
    } catch (e) {
      document.getElementById("detail-content").textContent = "Failed to load details";
    }
  }

  document.getElementById("detail-close").addEventListener("click", () => {
    detailPanel.hidden = true;
  });

  // --- Logout ---
  document.getElementById("logout-btn").addEventListener("click", () => {
    sessionStorage.removeItem("cems_api_key");
    apiKey = "";
    if (simulation) simulation.stop();
    showLogin();
  });

  // --- Helpers ---
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str || "";
    return div.innerHTML;
  }

  // --- Init ---
  if (apiKey) {
    showDashboard();
  } else {
    showLogin();
  }
})();
