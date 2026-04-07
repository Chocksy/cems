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

  // --- Views ---
  function showLogin() {
    loginView.hidden = false;
    mainLayout.hidden = true;
    loginError.hidden = true;
  }

  function showDashboard() {
    loginView.hidden = true;
    mainLayout.hidden = false;
    loadStats();
    switchView("wiki");
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
  function switchView(viewName) {
    currentView = viewName;
    // Update sidebar active state
    document.querySelectorAll("#sidebar .nav-item").forEach((n) => {
      n.classList.toggle("active", n.dataset.view === viewName);
    });
    // Hide all content views, show selected
    document.querySelectorAll(".content-view").forEach((v) => { v.hidden = true; });
    const target = document.getElementById("view-" + viewName);
    if (target) target.hidden = false;
    // Update URL hash
    history.replaceState(null, "", "#" + viewName);
    // Load data for the view
    if (viewName === "wiki") loadEntities();
    if (viewName === "graph") loadGraph();
    if (viewName === "stats") loadStats();
    if (viewName === "health") { loadConflicts(); loadLintStats(); }
    if (viewName === "memories") { loadMemCategories(); loadMemories(); }
  }

  document.querySelectorAll("#sidebar .nav-item[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
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
            <div class="detail-relation" data-id="${escapeHtml(r.id)}">
              <span class="relation-score">${r.similarity ? (r.similarity * 100).toFixed(0) + "%" : ""}</span>
              <div>${escapeHtml((r.content || "").slice(0, 150))}</div>
              <div class="relation-cat">${escapeHtml(r.category || "")} &middot; ${escapeHtml(r.relation_type || "similar")}</div>
            </div>
          `).join("")
        : `<p style="color:var(--fg2);font-size:.85rem;margin-top:1rem">No relations found</p>`;

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
        const firstNav = document.querySelector(".nav-item");
        if (firstNav) firstNav.classList.add("active");
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
      const heatColor = shown >= 20 ? "var(--hot)" : shown >= 5 ? "var(--warm)" : shown >= 1 ? "var(--cool)" : "var(--cold)";
      const project = (e.source_ref || "").replace("project:", "").split("/").pop() || "";
      return `<div class="nav-item" data-id="${escapeHtml(e.id)}">
        <div class="nav-item-title">
          <span class="heat-dot" style="background:${heatColor}"></span>
          ${escapeHtml(e.title || "Untitled")}
        </div>
        ${project ? `<div class="nav-item-project">${escapeHtml(project)}</div>` : ""}
      </div>`;
    }).join("");

    // Click handlers
    navList.querySelectorAll(".nav-item").forEach((item) => {
      item.addEventListener("click", () => {
        navList.querySelectorAll(".nav-item").forEach((i) => i.classList.remove("active"));
        item.classList.add("active");
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
      const heatColor = shown >= 20 ? "var(--hot)" : shown >= 5 ? "var(--warm)" : shown >= 1 ? "var(--cool)" : "var(--cold)";
      const heatLabel = shown >= 20 ? "HOT" : shown >= 5 ? "WARM" : shown >= 1 ? "COOL" : "COLD";

      document.getElementById("article-title").textContent = e.title || "Untitled";
      document.getElementById("article-meta").innerHTML =
        `<span>${escapeHtml(e.source_ref || "no project")}</span>` +
        ` &middot; ${e.cluster_size || 0} sources` +
        ` &middot; shown ${shown}x (${heatLabel})` +
        ` &middot; ${e.created_at ? new Date(e.created_at).toLocaleDateString() : ""}`;
      document.getElementById("article-heat-bar").innerHTML =
        `<div class="heat-fill" style="width:${heatPct}%;background:${heatColor}"></div>`;

      // Render markdown content (simple renderer)
      document.getElementById("article-body").innerHTML = renderMarkdown(e.content || "");

      // Related entities
      const relEl = document.getElementById("article-related-entities");
      if (data.related_entities && data.related_entities.length > 0) {
        relEl.innerHTML = `<h3>Related Topics</h3>` +
          data.related_entities.map((r) =>
            `<span class="related-entity-link" data-id="${escapeHtml(r.id)}">${escapeHtml(r.title || "?")}</span>`
          ).join("");
        relEl.hidden = false;
        relEl.querySelectorAll(".related-entity-link").forEach((link) => {
          link.addEventListener("click", () => {
            loadEntityArticle(link.dataset.id);
            // Update sidebar active state
            document.querySelectorAll(".nav-item").forEach((i) => {
              i.classList.toggle("active", i.dataset.id === link.dataset.id);
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
              timelineList.innerHTML = tData.timeline.map((t) => `
                <div class="timeline-entry">
                  <div class="timeline-dot ${escapeHtml(t.heat)}"></div>
                  <div class="timeline-body">
                    <div class="timeline-date">${t.created_at ? new Date(t.created_at).toLocaleDateString("en-US", {year:"numeric",month:"short",day:"numeric"}) : ""}</div>
                    <div class="timeline-text">${escapeHtml(t.content)}</div>
                    <div class="timeline-meta">${escapeHtml(t.category || "")} &middot; ${escapeHtml(t.source || "")} &middot; ${t.similarity ? (t.similarity * 100).toFixed(0) + "% match" : ""}</div>
                  </div>
                </div>
              `).join("");
              btnTimeline.textContent = "Hide";
            } else {
              timelineList.innerHTML = '<p class="empty-msg">No timeline data available</p>';
              btnTimeline.textContent = "Show";
            }
          } catch (err) {
            timelineList.innerHTML = '<p class="empty-msg">Failed to load timeline</p>';
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
          const icon = mShown >= 20 ? "&#128293;" : mShown >= 5 ? "&#127777;" : mShown >= 1 ? "&#10052;" : "&#9898;";
          return `<div class="source-memory">
            <span class="source-heat">${icon}</span>
            <div>
              <div class="source-content">${escapeHtml(m.content || "")}</div>
              <div class="source-meta">${escapeHtml(m.category || "")} &middot; ${m.similarity ? (m.similarity * 100).toFixed(0) + "% match" : ""}</div>
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

  // Compile button
  const btnCompile = document.getElementById("btn-compile");
  if (btnCompile) {
    btnCompile.addEventListener("click", async () => {
      btnCompile.disabled = true;
      btnCompile.textContent = "Compiling...";
      try {
        const res = await fetch(baseUrl + "/api/memory/maintenance", {
          method: "POST",
          headers: {
            "Authorization": "Bearer " + apiKey,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ job_type: "compilation", limit: 10 }),
        });
        const data = await res.json();
        if (data.success) {
          btnCompile.textContent = `Done! ${data.results.pages_created || 0} created`;
          setTimeout(() => { loadEntities(); btnCompile.textContent = "Compile New"; btnCompile.disabled = false; }, 2000);
        }
      } catch (e) {
        btnCompile.textContent = "Error";
        setTimeout(() => { btnCompile.textContent = "Compile New"; btnCompile.disabled = false; }, 2000);
      }
    });
  }

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
        <div class="conflict-card">
          <div class="conflict-header">
            <span class="conflict-label">Contradiction</span>
            <span class="conflict-date">${c.created_at ? new Date(c.created_at).toLocaleDateString() : ""}</span>
          </div>
          <div style="font-size:.8rem;color:var(--fg2);margin-bottom:.5rem">${escapeHtml(c.explanation || "")}</div>
          <div class="conflict-pair">
            <div>
              <div class="conflict-doc-label">Memory A</div>
              <div class="conflict-doc">${escapeHtml(c.doc_a_content || "")}</div>
            </div>
            <div>
              <div class="conflict-doc-label">Memory B</div>
              <div class="conflict-doc">${escapeHtml(c.doc_b_content || "")}</div>
            </div>
          </div>
          <div class="conflict-actions">
            <button onclick="resolveConflict('${escapeHtml(c.id)}','keep_a')">Keep A</button>
            <button onclick="resolveConflict('${escapeHtml(c.id)}','keep_b')">Keep B</button>
            <button onclick="resolveConflict('${escapeHtml(c.id)}','dismiss')" class="btn-resolve">Dismiss</button>
          </div>
        </div>
      `).join("");
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
    statsEl.innerHTML = '<div class="stat-card"><div class="stat-value">...</div><div class="stat-label">Loading</div></div>';

    try {
      const data = await apiFetch("/api/wiki/stats");
      if (!data.success) return;
      const s = data.stats;
      statsEl.innerHTML = `
        <div class="stat-card"><div class="stat-value">${s.health_score}</div><div class="stat-label">Health Score</div></div>
        <div class="stat-card"><div class="stat-value">${s.open_conflicts}</div><div class="stat-label">Open Conflicts</div></div>
        <div class="stat-card"><div class="stat-value">${s.orphan_memories}</div><div class="stat-label">Orphans</div></div>
        <div class="stat-card"><div class="stat-value">${s.connected_memories}/${s.total_memories}</div><div class="stat-label">Connected</div></div>
      `;
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
          statsEl.innerHTML = `
            <div class="stat-card"><div class="stat-value">${r.health_score || 0}</div><div class="stat-label">Health Score</div></div>
            <div class="stat-card"><div class="stat-value">${r.open_conflicts || 0}</div><div class="stat-label">Open Conflicts</div></div>
            <div class="stat-card"><div class="stat-value">${r.orphan_count || 0}</div><div class="stat-label">Orphans</div></div>
            <div class="stat-card"><div class="stat-value">${r.connected_memories || 0}/${r.total_memories || 0}</div><div class="stat-label">Connected</div></div>
          `;
          if (r.contradictions_found > 0) {
            statsEl.innerHTML += `<div class="stat-card"><div class="stat-value" style="color:var(--warning)">${r.contradictions_found}</div><div class="stat-label">New This Run</div></div>`;
          }
          if (r.entity_page_count !== undefined) {
            statsEl.innerHTML += `<div class="stat-card"><div class="stat-value">${r.entity_page_count}</div><div class="stat-label">Entity Pages</div></div>`;
          }

          // Knowledge gaps
          const gapsSection = document.getElementById("gaps-section");
          const gapsList = document.getElementById("gaps-list");
          if (r.knowledge_gaps && r.knowledge_gaps.length > 0) {
            gapsSection.hidden = false;
            gapsList.innerHTML = r.knowledge_gaps.map((g) => `
              <div class="gap-card">
                <div>
                  <div class="gap-info">${escapeHtml(g.category)}</div>
                  <div class="gap-count">${Number(g.count)} memories, no entity page</div>
                </div>
                <button class="gap-action" data-category="${escapeHtml(g.category)}">Generate</button>
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
              <div class="orphan-card">
                <div>${escapeHtml(o.content)}</div>
                <div class="orphan-meta">${escapeHtml(o.category)} &middot; shown ${o.shown_count}x</div>
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
    let html = `<button class="filter-pill ${memCategory === "" ? "active" : ""}" data-category="">All (${allTotal})</button>`;
    for (const [cat, count] of Object.entries(memCategories).sort((a, b) => b[1] - a[1])) {
      html += `<button class="filter-pill ${memCategory === cat ? "active" : ""}" data-category="${escapeHtml(cat)}">${escapeHtml(cat)} (${count})</button>`;
    }
    el.innerHTML = html;
    el.querySelectorAll(".filter-pill").forEach((btn) => {
      btn.addEventListener("click", () => { memCategory = btn.dataset.category; memOffset = 0; loadMemories(); renderMemFilters(); });
    });
  }

  async function loadMemories() {
    const listEl = document.getElementById("memory-list");
    if (!listEl) return;
    listEl.innerHTML = '<div class="loading">Loading...</div>';
    try {
      let params = `limit=${memLimit}&offset=${memOffset}`;
      if (memScope) params += `&scope=${encodeURIComponent(memScope)}`;
      if (memCategory) params += `&category=${encodeURIComponent(memCategory)}`;
      if (memSearch) params += `&q=${encodeURIComponent(memSearch)}`;
      const data = await apiFetch(`/api/memory/list?${params}`);
      if (!data.success) { listEl.innerHTML = '<div class="empty">Error loading memories.</div>'; return; }
      memTotal = data.total || 0;
      renderMemList(data.results || []);
      renderMemPagination(data.mode === "search");
    } catch (e) { listEl.innerHTML = '<div class="empty">Failed to load.</div>'; }
  }

  function renderMemList(memories) {
    const listEl = document.getElementById("memory-list");
    if (!memories.length) { listEl.innerHTML = '<div class="empty">No memories found.</div>'; return; }
    listEl.innerHTML = memories.map((m) => {
      const tags = (m.tags || []).map((t) => `<span class="tag">#${escapeHtml(t)}</span>`).join(" ");
      const content = escapeHtml(m.content || "");
      const isShort = content.length < 300;
      const date = m.created_at ? new Date(m.created_at).toLocaleDateString() : "";
      const shown = m.shown_count ? `shown: ${m.shown_count}` : "";
      return `<div class="memory-card" data-id="${escapeHtml(m.id)}">
        <div class="memory-meta">
          <span class="cat">${escapeHtml(m.category || "general")}</span>
          ${tags}
          ${m.scope ? `<span class="scope-badge scope-${escapeHtml(m.scope)}">${escapeHtml(m.scope)}</span>` : ""}
          ${m.source_ref ? `<span class="source-ref">${escapeHtml(m.source_ref)}</span>` : ""}
          ${date ? `<span>${date}</span>` : ""}
          ${shown ? `<span>${shown}</span>` : ""}
        </div>
        <div class="memory-content ${isShort ? "short" : ""}">${content}</div>
        <div class="memory-actions">
          <button class="btn-expand" data-id="${escapeHtml(m.id)}">Expand</button>
          <button class="btn-edit" data-id="${escapeHtml(m.id)}">Edit</button>
          <button class="btn-delete" data-id="${escapeHtml(m.id)}">Delete</button>
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

  // Scope toggle
  document.querySelectorAll(".scope-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".scope-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
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
      document.getElementById("edit-modal").classList.add("open");
    } catch (e) { console.error("Edit failed", e); }
  }

  function closeMemEdit() { document.getElementById("edit-modal").classList.remove("open"); editingId = null; }

  document.getElementById("edit-cancel")?.addEventListener("click", closeMemEdit);
  document.querySelector(".modal-close")?.addEventListener("click", closeMemEdit);
  document.getElementById("edit-modal")?.addEventListener("click", (e) => { if (e.target.classList.contains("modal")) closeMemEdit(); });

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
    toast.className = "toast";
    toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
    if (undoCallback) {
      const btn = document.createElement("button");
      btn.className = "undo-btn";
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
})();
