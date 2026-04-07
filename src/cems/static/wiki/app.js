/* CEMS Knowledge Engine Dashboard */
(function () {
  "use strict";

  // --- State ---
  let apiKey = sessionStorage.getItem("cems_api_key") || "";
  let currentView = "wiki";
  let graphData = null;
  let simulation = null;
  // Memory list state
  let memOffset = 0;
  let memLimit = 50;
  let memTotal = 0;
  let memCategory = "";
  let memSearch = "";
  let memScope = "";
  let memSearchTimeout = null;
  let editingId = null;
  let memCategories = {};

  // --- DOM refs ---
  const loginView = document.getElementById("login-view");
  const mainLayout = document.getElementById("main-layout");
  const loginForm = document.getElementById("login-form");
  const apiKeyInput = document.getElementById("api-key-input");
  const loginError = document.getElementById("login-error");
  const healthBadge = document.getElementById("health-badge");
  const graphInfo = document.getElementById("graph-info");
  const detailPanel = document.getElementById("detail-panel");

  // --- API helpers ---
  const baseUrl = window.location.origin;

  async function apiFetch(path, opts = {}) {
    const headers = { Authorization: "Bearer " + apiKey, ...opts.headers };
    const res = await fetch(baseUrl + path, { ...opts, headers });
    if (res.status === 401) {
      sessionStorage.removeItem("cems_api_key");
      apiKey = "";
      showLogin();
      throw new Error("Unauthorized");
    }
    return res.json();
  }

  // --- Lucide icons helper ---
  function refreshIcons() {
    if (typeof lucide !== "undefined") lucide.createIcons();
  }

  // --- Views ---
  function showLogin() {
    loginView.style.display = "";
    mainLayout.classList.remove("open");
    loginError.hidden = true;
  }

  function showDashboard() {
    loginView.style.display = "none";
    mainLayout.classList.add("open");
    refreshIcons();
    loadStats();
    // Respect hash route if present, otherwise default to wiki
    const hash = window.location.hash.slice(1).split("/")[0];
    const validViews = ["wiki", "memories", "graph", "stats", "health"];
    switchView(validViews.includes(hash) ? hash : "wiki");
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

  // --- Sidebar Navigation ---
  const sidebarNav = document.getElementById("sidebar-nav");
  const sidebarWiki = document.getElementById("sidebar-wiki");

  function setSidebarMode(mode) {
    if (mode === "wiki") {
      sidebarNav.style.display = "none";
      sidebarWiki.style.display = "flex";
    } else {
      sidebarNav.style.display = "";
      sidebarWiki.style.display = "none";
    }
    refreshIcons();
  }

  function switchView(viewName) {
    currentView = viewName;
    // Update sidebar active state
    document.querySelectorAll(".nav-link[data-view]").forEach((n) => {
      const isActive = n.dataset.view === viewName;
      n.classList.toggle("active", isActive);
      n.classList.toggle("bg-blue-600", isActive);
      n.classList.toggle("text-white", isActive);
      n.classList.toggle("text-gray-400", !isActive);
      n.classList.toggle("hover:bg-gray-800", !isActive);
    });
    // Hide all content views, show selected
    document.querySelectorAll(".content-view").forEach((v) => { v.classList.remove("active"); });
    const target = document.getElementById("view-" + viewName);
    if (target) target.classList.add("active");
    // Contextual sidebar — show wiki entity browser or main nav
    setSidebarMode(viewName === "wiki" ? "wiki" : "nav");
    // Update URL hash
    history.replaceState(null, "", "#" + viewName);
    // Load data for the view
    if (viewName === "wiki") loadEntities();
    if (viewName === "graph") loadGraph();
    if (viewName === "stats") loadStats();
    if (viewName === "health") { loadConflicts(); loadLintStats(); }
    if (viewName === "memories") { loadMemCategories(); loadMemories(); }
  }

  // Sidebar back button — return to main nav
  document.getElementById("sidebar-back")?.addEventListener("click", () => {
    setSidebarMode("nav");
  });

  document.querySelectorAll(".nav-link[data-view]").forEach((btn) => {
    btn.addEventListener("click", (e) => { e.preventDefault(); switchView(btn.dataset.view); });
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
      const badgeColor = s.health_score >= 80
        ? "bg-green-900/50 text-green-400"
        : s.health_score >= 50
        ? "bg-amber-900/50 text-amber-400"
        : "bg-red-900/50 text-red-400";
      healthBadge.className = "text-xs px-2 py-0.5 rounded-full font-medium " + badgeColor;

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

  const heatColorMap = { hot: "#ef4444", warm: "#f59e0b", cool: "#3b82f6", cold: "#6b7280" };

  function renderBars(containerId, data, classMap) {
    const el = document.getElementById(containerId);
    const max = Math.max(...Object.values(data), 1);
    el.innerHTML = Object.entries(data).map(([key, val]) => {
      const pct = Math.max((val / max) * 100, 2);
      const color = heatColorMap[key] || "#3b82f6";
      return `<div class="flex items-center gap-3 text-sm">
        <span class="w-20 text-gray-400 text-xs">${escapeHtml(key)}</span>
        <div class="bar-fill" style="width:${pct}%;background:${color}"></div>
        <span class="w-8 text-right font-semibold text-xs">${Number(val)}</span>
      </div>`;
    }).join("");
  }

  function renderBarsDynamic(containerId, data) {
    const el = document.getElementById(containerId);
    const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const max = Math.max(...entries.map(([, v]) => v), 1);
    el.innerHTML = entries.map(([key, val]) => {
      const pct = Math.max((val / max) * 100, 2);
      return `<div class="flex items-center gap-3 text-sm">
        <span class="w-20 text-gray-400 text-xs truncate" title="${escapeHtml(key)}">${escapeHtml(key)}</span>
        <div class="bar-fill" style="width:${pct}%;background:#3b82f6"></div>
        <span class="w-8 text-right font-semibold text-xs">${Number(val)}</span>
      </div>`;
    }).join("");
  }

  // --- Graph ---
  async function loadGraph() {
    try {
      const data = await apiFetch("/api/wiki/graph?limit=500");
      if (!data.success) return;

      graphData = data;
      graphInfo.textContent = `${data.node_count} nodes, ${data.edge_count} edges`;

      // Populate category filter
      const catFilter = document.getElementById("graph-filter-cat");
      if (catFilter && catFilter.options.length <= 1) {
        const cats = [...new Set(data.nodes.map((n) => n.category))].sort();
        cats.forEach((c) => {
          const opt = document.createElement("option");
          opt.value = c;
          opt.textContent = c;
          catFilter.appendChild(opt);
        });
      }

      applyGraphFilters();
    } catch (e) {
      graphInfo.textContent = "Failed to load graph";
      console.error("Graph load error:", e);
    }
  }

  function applyGraphFilters() {
    if (!graphData) return;
    const heatFilter = document.getElementById("graph-filter-heat")?.value || "";
    const catFilter = document.getElementById("graph-filter-cat")?.value || "";

    let nodes = graphData.nodes;
    let edges = graphData.edges;

    if (heatFilter) {
      nodes = nodes.filter((n) => {
        const s = n.shown_count || 0;
        if (heatFilter === "hot") return s >= 20;
        if (heatFilter === "warm") return s >= 5 && s < 20;
        if (heatFilter === "cool") return s >= 1 && s < 5;
        if (heatFilter === "cold") return s === 0;
        return true;
      });
    }
    if (catFilter) {
      nodes = nodes.filter((n) => n.category === catFilter);
    }

    const nodeIds = new Set(nodes.map((n) => n.id));
    edges = edges.filter((e) => nodeIds.has(e.source) && nodeIds.has(e.target));

    graphInfo.textContent = `${nodes.length} nodes, ${edges.length} edges` +
      (heatFilter || catFilter ? " (filtered)" : "");
    renderGraph(nodes, edges);
  }

  // Filter event listeners
  document.getElementById("graph-filter-heat")?.addEventListener("change", applyGraphFilters);
  document.getElementById("graph-filter-cat")?.addEventListener("change", applyGraphFilters);

  function getNodeColor(node) {
    const shown = node.shown_count || 0;
    if (shown >= 20) return "#ef4444";
    if (shown >= 5) return "#f59e0b";
    if (shown >= 1) return "#3b82f6";
    return "#6b7280";
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

    // Labels for hot/warm nodes (reduce clutter)
    node.filter((d) => (d.shown_count || 0) >= 5)
      .append("text")
      .attr("dy", (d) => getNodeRadius(d) + 12)
      .text((d) => {
        // Show project + category instead of raw title/session tag
        const proj = (d.source_ref || "").replace("project:", "").split("/").pop();
        const cat = d.category || "";
        if (proj) return `${proj}: ${cat}`.slice(0, 25);
        return (d.title || cat).slice(0, 25);
      });

    // Start zoomed out to show more of the graph
    const initialScale = Math.min(1, Math.max(0.3, 50 / Math.sqrt(d3Nodes.length)));
    svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(initialScale).translate(-width / 2, -height / 2));

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
        <div class="flex flex-wrap gap-2 text-xs text-gray-400 mb-3">
          <span class="font-semibold text-blue-400">${escapeHtml(m.category || "general")}</span>
          <span>&middot; shown ${Number(m.shown_count) || 0}x</span>
          <span>&middot; ${escapeHtml(m.source_ref || "no project")}</span>
          <span>&middot; ${m.created_at ? new Date(m.created_at).toLocaleDateString() : ""}</span>
        </div>
        <div class="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">${escapeHtml(m.content)}</div>
      `;

      const rels = data.relations || [];
      document.getElementById("detail-relations").innerHTML = rels.length
        ? `<h3 class="text-sm font-semibold text-gray-400 mt-4 mb-2">Related (${rels.length})</h3>` +
          rels.map((r) => `
            <div class="detail-relation bg-gray-800/50 border border-gray-700 rounded-lg p-3 mb-2 cursor-pointer hover:border-gray-600 transition-colors" data-id="${escapeHtml(r.id)}">
              <span class="text-xs text-blue-400 font-medium">${r.similarity ? (r.similarity * 100).toFixed(0) + "%" : ""}</span>
              <div class="text-sm text-gray-300 mt-1">${escapeHtml((r.content || "").slice(0, 150))}</div>
              <div class="text-xs text-gray-500 mt-1">${escapeHtml(r.category || "")} &middot; ${escapeHtml(r.relation_type || "similar")}</div>
            </div>
          `).join("")
        : `<p class="text-sm text-gray-500 mt-4">No relations found</p>`;

      // Click handler: navigate to related memory
      document.querySelectorAll("#detail-relations .detail-relation").forEach((card) => {
        card.addEventListener("click", () => {
          const relId = card.dataset.id;
          if (relId) showDetail({ id: relId, title: card.querySelector(".relation-cat")?.textContent || "" });
        });
      });
    } catch (e) {
      document.getElementById("detail-content").textContent = "Failed to load details";
    }
  }

  document.getElementById("detail-close").addEventListener("click", () => {
    detailPanel.hidden = true;
  });

  // --- Entities (Wikipedia view) ---
  let allEntities = [];

  async function loadEntities() {
    try {
      const data = await apiFetch("/api/wiki/entities?limit=100");
      if (!data.success) return;

      allEntities = data.entities || [];
      const navList = document.getElementById("entity-nav-list");
      const emptyEl = document.getElementById("entities-empty");

      if (allEntities.length === 0) {
        navList.innerHTML = "";
        emptyEl.hidden = false;
        return;
      }
      emptyEl.hidden = true;
      renderEntityNav(allEntities);

      // Auto-select first entity and mark it active
      if (allEntities.length > 0) {
        const firstNav = document.querySelector(".wiki-topic");
        if (firstNav) firstNav.classList.add("bg-blue-600/15", "ring-1", "ring-blue-500/30");
        loadEntityArticle(allEntities[0].id);
      }
    } catch (e) {
      console.error("Failed to load entities:", e);
    }
  }

  function renderEntityNav(entities) {
    const navList = document.getElementById("entity-nav-list");
    navList.innerHTML = entities.map((e) => {
      const shown = Number(e.shown_count) || 0;
      const heatColor = shown >= 20 ? "#ef4444" : shown >= 5 ? "#f59e0b" : shown >= 1 ? "#3b82f6" : "#6b7280";
      const project = (e.source_ref || "").replace("project:", "").split("/").pop() || "";
      return `<a class="wiki-topic group block px-3 py-2 rounded-lg cursor-pointer transition-colors hover:bg-gray-800/70" data-id="${escapeHtml(e.id)}">
          <div class="text-[13px] text-gray-300 group-hover:text-gray-100 leading-snug">${escapeHtml(e.title || "Untitled")}</div>
          ${project ? `<div class="text-[11px] text-gray-600 mt-0.5">${escapeHtml(project)} &middot; ${shown}x shown</div>` : `<div class="text-[11px] text-gray-600 mt-0.5">${shown}x shown</div>`}
      </a>`;
    }).join("");

    // Click handlers
    navList.querySelectorAll(".wiki-topic").forEach((item) => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        navList.querySelectorAll(".wiki-topic").forEach((i) => {
          i.classList.remove("bg-blue-600/15", "ring-1", "ring-blue-500/30");
        });
        item.classList.add("bg-blue-600/15", "ring-1", "ring-blue-500/30");
        loadEntityArticle(item.dataset.id);
      });
    });
  }

  // Search/filter sidebar
  const entitySearchInput = document.getElementById("entity-search");
  if (entitySearchInput) {
    entitySearchInput.addEventListener("input", () => {
      const q = entitySearchInput.value.toLowerCase();
      const filtered = allEntities.filter((e) =>
        (e.title || "").toLowerCase().includes(q) ||
        (e.source_ref || "").toLowerCase().includes(q)
      );
      renderEntityNav(filtered);
    });
  }

  async function loadEntityArticle(entityId) {
    const placeholder = document.getElementById("article-placeholder");
    const content = document.getElementById("article-content");
    placeholder.hidden = true;
    content.hidden = false;
    history.replaceState(null, "", "#wiki/" + entityId);

    try {
      const data = await apiFetch(`/api/wiki/entity?id=${entityId}`);
      if (!data.success) return;

      const e = data.entity;
      const shown = Number(e.shown_count) || 0;
      const heatPct = Math.min(shown * 5, 100);
      const heatColor = shown >= 20 ? "#ef4444" : shown >= 5 ? "#f59e0b" : shown >= 1 ? "#3b82f6" : "#6b7280";
      const heatLabel = shown >= 20 ? "HOT" : shown >= 5 ? "WARM" : shown >= 1 ? "COOL" : "COLD";

      document.getElementById("article-title").textContent = e.title || "Untitled";
      document.getElementById("article-meta").innerHTML =
        `<span>${escapeHtml(e.source_ref || "no project")}</span>` +
        ` &middot; ${e.cluster_size || 0} sources` +
        ` &middot; shown ${shown}x (${heatLabel})` +
        ` &middot; ${e.created_at ? new Date(e.created_at).toLocaleDateString() : ""}`;
      document.getElementById("article-heat-bar").innerHTML =
        `<div class="h-full rounded-full transition-all" style="width:${heatPct}%;background:${heatColor}"></div>`;

      // Render markdown content — strip duplicate title (first h1 matching article title)
      let articleMd = e.content || "";
      const titleLine = (e.title || "").trim();
      if (titleLine) {
        articleMd = articleMd.replace(new RegExp("^#\\s+" + titleLine.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "\\s*\\n*", "i"), "");
      }
      document.getElementById("article-body").innerHTML = renderMarkdown(articleMd);

      // Related entities
      const relEl = document.getElementById("article-related-entities");
      if (data.related_entities && data.related_entities.length > 0) {
        relEl.innerHTML = `<h3 class="text-lg font-semibold mb-3">Related Topics</h3>` +
          data.related_entities.map((r) =>
            `<span class="inline-block bg-gray-800 hover:bg-gray-700 text-blue-400 text-sm px-3 py-1 rounded-full cursor-pointer m-1 transition-colors" data-id="${escapeHtml(r.id)}">${escapeHtml(r.title || "?")}</span>`
          ).join("");
        relEl.hidden = false;
        relEl.querySelectorAll("[data-id]").forEach((link) => {
          link.addEventListener("click", () => {
            loadEntityArticle(link.dataset.id);
            // Update sidebar active state
            document.querySelectorAll(".wiki-topic").forEach((i) => {
              i.classList.remove("bg-blue-600/15", "ring-1", "ring-blue-500/30");
              if (i.dataset.id === link.dataset.id) i.classList.add("bg-blue-600/15", "ring-1", "ring-blue-500/30");
            });
          });
        });
      } else {
        relEl.innerHTML = "";
        relEl.hidden = true;
      }

      // Timeline section
      const timelineSection = document.getElementById("article-timeline");
      const timelineList = document.getElementById("timeline-list");
      const btnTimeline = document.getElementById("btn-toggle-timeline");
      if (timelineSection) {
        timelineSection.hidden = false;
        timelineList.innerHTML = "";
        btnTimeline.textContent = "Show";
        btnTimeline.onclick = async () => {
          if (timelineList.innerHTML) {
            timelineList.innerHTML = "";
            btnTimeline.textContent = "Show";
            return;
          }
          btnTimeline.textContent = "Loading...";
          try {
            const tData = await apiFetch(`/api/wiki/timeline?id=${entityId}&limit=30`);
            if (tData.success && tData.timeline.length > 0) {
              const dotColors = { hot: "#ef4444", warm: "#f59e0b", cool: "#3b82f6", cold: "#6b7280" };
              timelineList.innerHTML = tData.timeline.map((t) => `
                <div class="timeline-entry flex gap-3 py-2">
                  <div class="w-3 h-3 rounded-full mt-1 flex-shrink-0" style="background:${dotColors[t.heat] || "#6b7280"}"></div>
                  <div>
                    <div class="text-xs text-gray-500">${t.created_at ? new Date(t.created_at).toLocaleDateString("en-US", {year:"numeric",month:"short",day:"numeric"}) : ""}</div>
                    <div class="text-sm text-gray-300 mt-0.5">${escapeHtml(t.content)}</div>
                    <div class="text-xs text-gray-500 mt-0.5">${escapeHtml(t.category || "")} &middot; ${escapeHtml(t.source || "")} &middot; ${t.similarity ? (t.similarity * 100).toFixed(0) + "% match" : ""}</div>
                  </div>
                </div>
              `).join("");
              btnTimeline.textContent = "Hide";
            } else {
              timelineList.innerHTML = '<p class="text-sm text-gray-500">No timeline data available</p>';
              btnTimeline.textContent = "Show";
            }
          } catch (err) {
            timelineList.innerHTML = '<p class="text-sm text-gray-500">Failed to load timeline</p>';
            btnTimeline.textContent = "Show";
          }
        };
      }

      // Source memories
      const srcList = document.getElementById("source-memories-list");
      if (data.source_memories && data.source_memories.length > 0) {
        document.getElementById("article-sources").hidden = false;
        srcList.innerHTML = data.source_memories.map((m) => {
          const mShown = Number(m.shown_count) || 0;
          const heatColor = mShown >= 20 ? "bg-red-500" : mShown >= 5 ? "bg-amber-500" : mShown >= 1 ? "bg-blue-500" : "bg-gray-600";
          const matchPct = m.similarity ? (m.similarity * 100).toFixed(0) + "%" : "";
          return `<div class="flex items-start gap-3 p-3 bg-gray-800/50 border border-gray-800 rounded-lg">
            <div class="flex-shrink-0 mt-1.5 w-2 h-2 rounded-full ${heatColor}"></div>
            <div class="min-w-0 flex-1">
              <div class="text-sm text-gray-300 leading-relaxed">${escapeHtml(m.content || "")}</div>
              <div class="flex items-center gap-2 text-xs text-gray-500 mt-1.5">
                <span>${escapeHtml(m.category || "")}</span>
                ${matchPct ? `<span class="text-blue-400 font-medium">${matchPct} match</span>` : ""}
              </div>
            </div>
          </div>`;
        }).join("");
      } else {
        document.getElementById("article-sources").hidden = true;
      }
    } catch (e) {
      document.getElementById("article-body").textContent = "Failed to load entity";
      console.error("Entity load error:", e);
    }
  }

  function renderMarkdown(md) {
    // Use marked.js for proper markdown rendering (code blocks, tables, etc.)
    if (typeof marked !== "undefined") {
      marked.setOptions({
        breaks: true,
        gfm: true,
      });
      // Sanitize: marked doesn't execute scripts but we escape first just in case
      return marked.parse(md);
    }
    // Fallback: basic rendering if marked.js fails to load
    return md.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/\n/g, "<br>");
  }

  // (btn-compile removed — dead code, element doesn't exist in HTML)

  // --- Lint / Health ---
  async function loadConflicts() {
    try {
      const data = await apiFetch("/api/wiki/conflicts?limit=20");
      if (!data.success) return;

      const listEl = document.getElementById("conflicts-list");
      const emptyEl = document.getElementById("conflicts-empty");

      if (!data.conflicts || data.conflicts.length === 0) {
        listEl.innerHTML = "";
        emptyEl.hidden = false;
        return;
      }
      emptyEl.hidden = true;

      listEl.innerHTML = data.conflicts.map((c) => `
        <div class="bg-gray-900 border border-gray-800 rounded-lg p-4 border-l-4 border-l-amber-500">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-semibold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
              <i data-lucide="shield-alert" class="w-3.5 h-3.5"></i> Contradiction
            </span>
            <span class="text-xs text-gray-500">${c.created_at ? new Date(c.created_at).toLocaleDateString() : ""}</span>
          </div>
          <div class="text-sm text-gray-400 mb-3">${escapeHtml(c.explanation || "")}</div>
          <div class="grid grid-cols-2 gap-4 mb-3">
            <div>
              <div class="text-xs font-medium text-gray-500 mb-1">Memory A</div>
              <div class="conflict-doc text-sm text-gray-300 leading-relaxed">${escapeHtml(c.doc_a_content || "")}</div>
            </div>
            <div>
              <div class="text-xs font-medium text-gray-500 mb-1">Memory B</div>
              <div class="conflict-doc text-sm text-gray-300 leading-relaxed">${escapeHtml(c.doc_b_content || "")}</div>
            </div>
          </div>
          <div class="flex gap-2">
            <button class="resolve-btn text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded transition-colors" data-conflict-id="${escapeHtml(c.id)}" data-resolution="keep_a">Keep A</button>
            <button class="resolve-btn text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-3 py-1.5 rounded transition-colors" data-conflict-id="${escapeHtml(c.id)}" data-resolution="keep_b">Keep B</button>
            <button class="resolve-btn text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 px-3 py-1.5 rounded transition-colors" data-conflict-id="${escapeHtml(c.id)}" data-resolution="dismiss">Dismiss</button>
          </div>
        </div>
      `).join("");
      // Bind resolve handlers via addEventListener (fix XSS from inline onclick)
      listEl.querySelectorAll(".resolve-btn").forEach((btn) => {
        btn.addEventListener("click", () => resolveConflict(btn.dataset.conflictId, btn.dataset.resolution));
      });
      refreshIcons();
    } catch (e) {
      console.error("Failed to load conflicts:", e);
    }
  }

  // Generate entity pages (gap resolution) — run multiple compilation batches
  window.compileCategory = async function(category) {
    const btn = event?.target;
    if (btn) { btn.disabled = true; btn.textContent = "Generating..."; }
    try {
      // Run compilation with force=true to regenerate, and higher limit
      let totalCreated = 0;
      const res = await fetch(baseUrl + "/api/memory/maintenance", {
        method: "POST",
        headers: { "Authorization": "Bearer " + apiKey, "Content-Type": "application/json" },
        body: JSON.stringify({ job_type: "compilation", limit: 50, full_sweep: true }),
      });
      const data = await res.json();
      if (data.success) totalCreated = (data.results?.pages_created || 0) + (data.results?.pages_updated || 0);
      if (btn) btn.textContent = totalCreated > 0 ? `Created ${totalCreated}!` : "No new pages";
      // Re-run lint + reload entities
      setTimeout(() => {
        document.getElementById("btn-run-lint")?.click();
        if (btn) { btn.textContent = "Generate"; btn.disabled = false; }
      }, 2000);
    } catch (e) {
      console.error("Compile failed:", e);
      if (btn) { btn.textContent = "Error"; setTimeout(() => { btn.textContent = "Generate"; btn.disabled = false; }, 2000); }
    }
  };

  // Make resolveConflict global for onclick
  window.resolveConflict = async function(conflictId, resolution) {
    try {
      const res = await fetch(baseUrl + "/api/memory/conflict/resolve", {
        method: "POST",
        headers: {
          "Authorization": "Bearer " + apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ conflict_id: conflictId, resolution }),
      });
      const data = await res.json();
      if (data.success) {
        loadConflicts();
        loadStats();
      }
    } catch (e) {
      console.error("Resolve failed:", e);
    }
  };

  // Auto-load lint stats when Health tab opens
  async function loadLintStats() {
    const reportEl = document.getElementById("lint-report");
    const statsEl = document.getElementById("lint-stats");
    if (!reportEl || !statsEl) return;

    reportEl.hidden = false;
    statsEl.innerHTML = '<div class="bg-gray-900 border border-gray-800 rounded-lg p-4 text-center"><p class="text-xl font-bold text-gray-500">...</p><p class="text-xs text-gray-500 mt-1">Loading</p></div>';

    try {
      const data = await apiFetch("/api/wiki/stats");
      if (!data.success) return;
      const s = data.stats;
      const hsc = (val, label, color) => `<div class="bg-gray-900 border border-gray-800 rounded-xl p-5 text-center"><p class="text-2xl font-bold ${color}">${val}</p><p class="text-xs text-gray-500 uppercase tracking-wider mt-2">${label}</p></div>`;
      const scoreColor = s.health_score >= 80 ? "text-green-400" : s.health_score >= 50 ? "text-amber-400" : "text-red-400";
      statsEl.innerHTML = hsc(s.health_score, "Health Score", scoreColor) +
        hsc(s.open_conflicts, "Open Conflicts", s.open_conflicts > 0 ? "text-red-400" : "text-green-400") +
        hsc(s.orphan_memories, "Orphans", "text-amber-400") +
        hsc(`${s.connected_memories}/${s.total_memories}`, "Connected", "text-blue-400");
    } catch (e) { console.error("Lint stats failed:", e); }
  }

  // Legacy lint button handler (kept for backwards compat, button removed from UI)
  const btnLint = document.getElementById("btn-run-lint");
  if (btnLint) {
    btnLint.addEventListener("click", async () => {
      btnLint.disabled = true;
      btnLint.textContent = "Running...";
      try {
        const res = await fetch(baseUrl + "/api/wiki/lint", {
          method: "POST",
          headers: { "Authorization": "Bearer " + apiKey },
        });
        const data = await res.json();
        if (data.success && data.report) {
          const r = data.report;
          const reportEl = document.getElementById("lint-report");
          const statsEl = document.getElementById("lint-stats");
          reportEl.hidden = false;
          const sc = (val, label) => `<div class="bg-gray-900 border border-gray-800 rounded-lg p-4 text-center"><p class="text-xl font-bold text-blue-400">${val}</p><p class="text-xs text-gray-500 uppercase tracking-wider mt-1">${label}</p></div>`;
          statsEl.innerHTML = sc(r.health_score || 0, "Health Score") +
            sc(r.open_conflicts || 0, "Open Conflicts") +
            sc(r.orphan_count || 0, "Orphans") +
            sc(`${r.connected_memories || 0}/${r.total_memories || 0}`, "Connected");
          if (r.contradictions_found > 0) {
            statsEl.innerHTML += `<div class="bg-gray-900 border border-gray-800 rounded-lg p-4 text-center"><p class="text-xl font-bold text-amber-400">${r.contradictions_found}</p><p class="text-xs text-gray-500 uppercase tracking-wider mt-1">New This Run</p></div>`;
          }
          if (r.entity_page_count !== undefined) {
            statsEl.innerHTML += `<div class="bg-gray-900 border border-gray-800 rounded-lg p-4 text-center"><p class="text-xl font-bold text-blue-400">${r.entity_page_count}</p><p class="text-xs text-gray-500 uppercase tracking-wider mt-1">Entity Pages</p></div>`;
          }

          // Knowledge gaps
          const gapsSection = document.getElementById("gaps-section");
          const gapsList = document.getElementById("gaps-list");
          if (r.knowledge_gaps && r.knowledge_gaps.length > 0) {
            gapsSection.hidden = false;
            gapsList.innerHTML = r.knowledge_gaps.map((g) => `
              <div class="flex items-center justify-between bg-gray-900 border border-gray-800 border-l-4 border-l-blue-500 rounded-lg p-3">
                <div>
                  <div class="text-sm font-medium text-gray-200">${escapeHtml(g.category)}</div>
                  <div class="text-xs text-gray-500">${Number(g.count)} memories, no entity page</div>
                </div>
                <button class="gap-action text-xs bg-blue-600 hover:bg-blue-500 text-white px-3 py-1.5 rounded transition-colors" data-category="${escapeHtml(g.category)}">Generate</button>
              </div>
            `).join("");
            // Bind click handlers via data attributes (no inline JS)
            gapsList.querySelectorAll(".gap-action").forEach((btn) => {
              btn.addEventListener("click", () => compileCategory(btn.dataset.category));
            });
          } else {
            gapsSection.hidden = true;
          }

          // Top orphans
          const orphansSection = document.getElementById("orphans-section");
          const orphansList = document.getElementById("orphans-list");
          if (r.top_orphans && r.top_orphans.length > 0) {
            orphansSection.hidden = false;
            orphansList.innerHTML = r.top_orphans.map((o) => `
              <div class="bg-gray-900 border border-gray-800 rounded-lg p-3">
                <div class="text-sm text-gray-300">${escapeHtml(o.content)}</div>
                <div class="text-xs text-gray-500 mt-1">${escapeHtml(o.category)} &middot; shown ${o.shown_count}x</div>
              </div>
            `).join("");
          } else {
            orphansSection.hidden = true;
          }

          loadConflicts();
          loadStats(); // Refresh the header health badge
        }
      } catch (e) {
        console.error("Lint failed:", e);
      }
      btnLint.textContent = "Run Lint";
      btnLint.disabled = false;
    });
  }

  // =========================================================================
  // MEMORIES VIEW (ported from /dashboard)
  // =========================================================================

  async function loadMemCategories() {
    try {
      const data = await apiFetch("/api/memory/summary/personal");
      if (!data.success) return;
      memCategories = data.categories || {};
      renderMemFilters();
    } catch (e) { console.error("Failed to load categories", e); }
  }

  function renderMemFilters() {
    const el = document.getElementById("memory-filters");
    if (!el) return;
    const allTotal = Object.values(memCategories).reduce((s, n) => s + n, 0);
    const activeClass = "bg-blue-600 text-white";
    const inactiveClass = "bg-gray-800 text-gray-300 hover:bg-gray-700";
    let html = `<span class="cat-pill px-3 py-1 rounded-full text-sm cursor-pointer transition-colors ${memCategory === "" ? activeClass : inactiveClass}" data-category="">All (${allTotal})</span>`;
    for (const [cat, count] of Object.entries(memCategories).sort((a, b) => b[1] - a[1])) {
      html += `<span class="cat-pill px-3 py-1 rounded-full text-sm cursor-pointer transition-colors ${memCategory === cat ? activeClass : inactiveClass}" data-category="${escapeHtml(cat)}">${escapeHtml(cat)} (${count})</span>`;
    }
    el.innerHTML = html;
    el.querySelectorAll(".cat-pill").forEach((btn) => {
      btn.addEventListener("click", () => { memCategory = btn.dataset.category; memOffset = 0; loadMemories(); renderMemFilters(); });
    });
  }

  async function loadMemories() {
    const listEl = document.getElementById("memory-list");
    if (!listEl) return;
    listEl.innerHTML = '<div class="text-sm text-gray-500 text-center py-8">Loading...</div>';
    try {
      let params = `limit=${memLimit}&offset=${memOffset}`;
      if (memScope) params += `&scope=${encodeURIComponent(memScope)}`;
      if (memCategory) params += `&category=${encodeURIComponent(memCategory)}`;
      if (memSearch) params += `&q=${encodeURIComponent(memSearch)}`;
      const data = await apiFetch(`/api/memory/list?${params}`);
      if (!data.success) { listEl.innerHTML = '<div class="text-sm text-gray-500 text-center py-8">Error loading memories.</div>'; return; }
      // Filter out entity-page memories (they belong in Wiki view, not here)
      const filtered = (data.results || []).filter((m) => m.category !== "entity-page");
      memTotal = data.total || 0;
      renderMemList(filtered);
      renderMemPagination(data.mode === "search");
    } catch (e) { listEl.innerHTML = '<div class="text-sm text-gray-500 text-center py-8">Failed to load.</div>'; }
  }

  function renderMemList(memories) {
    const listEl = document.getElementById("memory-list");
    if (!memories.length) { listEl.innerHTML = '<div class="text-sm text-gray-500 text-center py-8">No memories found.</div>'; return; }
    listEl.innerHTML = memories.map((m) => {
      const tags = (m.tags || []).map((t) => `<span class="bg-gray-800 text-gray-400 px-2 py-0.5 rounded text-xs">#${escapeHtml(t)}</span>`).join(" ");
      const content = escapeHtml(m.content || "");
      const isShort = content.length < 300;
      const date = m.created_at ? new Date(m.created_at).toLocaleDateString() : "";
      const shown = m.shown_count ? `shown: ${m.shown_count}` : "";
      return `<div class="memory-card bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors" data-id="${escapeHtml(m.id)}">
        <div class="flex flex-wrap items-center gap-2 text-xs text-gray-400 mb-2">
          <span class="font-semibold text-blue-400">${escapeHtml(m.category || "general")}</span>
          ${tags}
          ${m.scope ? `<span class="bg-gray-800 px-2 py-0.5 rounded">${escapeHtml(m.scope)}</span>` : ""}
          ${m.source_ref ? `<span class="text-gray-500">${escapeHtml(m.source_ref)}</span>` : ""}
          ${date ? `<span>${date}</span>` : ""}
          ${shown ? `<span>${shown}</span>` : ""}
        </div>
        <div class="memory-content text-sm text-gray-300 leading-relaxed whitespace-pre-wrap break-words ${isShort ? "short" : ""}">${content}</div>
        <div class="flex gap-2 mt-3">
          <button class="btn-expand text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-gray-200 px-2.5 py-1 rounded cursor-pointer transition-colors" data-id="${escapeHtml(m.id)}">Expand</button>
          <button class="btn-edit text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-blue-400 px-2.5 py-1 rounded cursor-pointer transition-colors" data-id="${escapeHtml(m.id)}">Edit</button>
          <button class="btn-delete text-xs bg-gray-800 hover:bg-gray-700 text-gray-400 hover:text-red-400 px-2.5 py-1 rounded cursor-pointer transition-colors" data-id="${escapeHtml(m.id)}">Delete</button>
        </div>
      </div>`;
    }).join("");
    // Event handlers
    listEl.querySelectorAll(".btn-expand").forEach((btn) => {
      btn.addEventListener("click", () => {
        const el = btn.closest(".memory-card").querySelector(".memory-content");
        el.classList.toggle("expanded");
        btn.textContent = el.classList.contains("expanded") ? "Collapse" : "Expand";
      });
    });
    listEl.querySelectorAll(".memory-content").forEach((el) => {
      el.addEventListener("click", () => { el.classList.toggle("expanded"); });
    });
    listEl.querySelectorAll(".btn-edit").forEach((btn) => {
      btn.addEventListener("click", () => openMemEdit(btn.dataset.id));
    });
    listEl.querySelectorAll(".btn-delete").forEach((btn) => {
      btn.addEventListener("click", () => deleteMemory(btn.dataset.id));
    });
  }

  function renderMemPagination(isSearch) {
    const pagEl = document.getElementById("memory-pagination");
    if (!pagEl) return;
    if (isSearch) { pagEl.hidden = true; return; }
    pagEl.hidden = memTotal <= memLimit;
    document.getElementById("prev-btn").disabled = memOffset === 0;
    document.getElementById("next-btn").disabled = memOffset + memLimit >= memTotal;
    const page = Math.floor(memOffset / memLimit) + 1;
    const pages = Math.ceil(memTotal / memLimit);
    document.getElementById("page-info").textContent = `Page ${page} of ${pages} (${memTotal} total)`;
  }

  document.getElementById("prev-btn")?.addEventListener("click", () => { memOffset = Math.max(0, memOffset - memLimit); loadMemories(); });
  document.getElementById("next-btn")?.addEventListener("click", () => { memOffset += memLimit; loadMemories(); });

  // Search
  document.getElementById("memory-search")?.addEventListener("input", (e) => {
    clearTimeout(memSearchTimeout);
    memSearchTimeout = setTimeout(() => { memSearch = e.target.value.trim(); memOffset = 0; loadMemories(); }, 400);
  });

  // Scope toggle (fix: consistent class names for active state)
  document.querySelectorAll(".scope-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".scope-btn").forEach((b) => {
        b.classList.remove("active", "bg-blue-600", "text-white");
        b.classList.add("text-gray-400", "hover:bg-gray-800");
      });
      btn.classList.add("active", "bg-blue-600", "text-white");
      btn.classList.remove("text-gray-400", "hover:bg-gray-800");
      memScope = btn.dataset.scope;
      memOffset = 0;
      loadMemories();
    });
  });

  // Edit modal
  async function openMemEdit(id) {
    try {
      const data = await apiFetch(`/api/memory/get?id=${id}`);
      if (!data.success) return;
      const doc = data.document;
      editingId = id;
      document.getElementById("edit-textarea").value = doc.content || "";
      document.getElementById("edit-category").value = doc.category || "";
      document.getElementById("edit-tags").value = (doc.tags || []).join(", ");
      document.getElementById("edit-source-ref").value = doc.source_ref || "";
      document.getElementById("edit-modal").classList.remove("hidden");
      refreshIcons();
    } catch (e) { console.error("Edit failed", e); }
  }

  function closeMemEdit() { document.getElementById("edit-modal").classList.add("hidden"); editingId = null; }

  document.getElementById("edit-cancel")?.addEventListener("click", closeMemEdit);
  document.querySelector(".modal-close-btn")?.addEventListener("click", closeMemEdit);
  document.getElementById("modal-backdrop")?.addEventListener("click", closeMemEdit);

  // Escape key closes modal and detail panel
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!document.getElementById("edit-modal").classList.contains("hidden")) closeMemEdit();
      if (!detailPanel.hidden) detailPanel.hidden = true;
    }
  });

  document.getElementById("edit-save")?.addEventListener("click", async () => {
    if (!editingId) return;
    const body = { memory_id: editingId };
    const content = document.getElementById("edit-textarea").value.trim();
    const category = document.getElementById("edit-category").value.trim();
    const tagsStr = document.getElementById("edit-tags").value.trim();
    const sourceRef = document.getElementById("edit-source-ref").value.trim();
    if (content) body.content = content;
    if (category) body.category = category;
    if (tagsStr) body.tags = tagsStr.split(",").map((t) => t.trim()).filter(Boolean);
    if (sourceRef) body.source_ref = sourceRef;
    try {
      const data = await apiFetch("/api/memory/update", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      if (data.success) { closeMemEdit(); loadMemories(); loadMemCategories(); showToast("Memory updated."); }
    } catch (e) { showToast("Update failed."); }
  });

  // Delete with undo
  async function deleteMemory(id) {
    try {
      const data = await apiFetch("/api/memory/forget", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ memory_id: id }) });
      if (data.success) {
        const card = document.querySelector(`.memory-card[data-id="${id}"]`);
        if (card) { card.style.opacity = "0"; setTimeout(() => card.remove(), 200); }
        showToast("Memory deleted.", async () => {
          await apiFetch("/api/memory/restore", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ memory_id: id }) });
          loadMemories();
        });
      }
    } catch (e) { showToast("Delete failed."); }
  }

  // Toast
  function showToast(message, undoCallback) {
    const container = document.getElementById("toast-container");
    container.innerHTML = "";
    const toast = document.createElement("div");
    toast.className = "flex items-center gap-3 bg-gray-800 border border-gray-700 text-gray-200 text-sm px-4 py-3 rounded-lg shadow-xl";
    toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
    if (undoCallback) {
      const btn = document.createElement("button");
      btn.className = "text-blue-400 hover:text-blue-300 font-medium ml-2 transition-colors";
      btn.textContent = "Undo";
      btn.addEventListener("click", async () => { toast.remove(); try { await undoCallback(); } catch (e) {} });
      toast.appendChild(btn);
    }
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  }

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

  // --- URL Hash Routing ---
  function handleHashRoute() {
    const hash = window.location.hash.slice(1);
    if (!hash || !apiKey) return;

    const [view, entityId] = hash.split("/");
    const validViews = ["wiki", "memories", "graph", "stats", "health"];
    if (view && validViews.includes(view)) {
      switchView(view);
      if (entityId && view === "wiki") {
        setTimeout(() => loadEntityArticle(entityId), 500);
      }
    }
  }

  window.addEventListener("hashchange", handleHashRoute);

  // --- Init ---
  if (apiKey) {
    showDashboard();
    setTimeout(handleHashRoute, 300);
  } else {
    showLogin();
  }

  // Initialize Lucide icons
  refreshIcons();
})();
