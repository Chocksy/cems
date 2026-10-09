/* CEMS Knowledge Engine Dashboard */
(function () {
  "use strict";

  // --- State ---
  let apiKey = sessionStorage.getItem("cems_api_key") || "";
  let currentView = "wiki";
  let graphData = null;
  let simulation = null;
  let lastRoute = null;
  const VIEWS = ["wiki", "memories", "graph", "stats", "health"];

  // Memories view state (filters mirror the URL hash)
  const mem = {
    q: "",
    scope: "",
    tags: [],
    src: "",
    cat: "",
    offset: 0,
    limit: 50,
    total: 0,
    rawLoaded: 0,
    mode: "browse",
    items: [],
    selectedId: null,
    editing: false,
    reqSeq: 0,
    searchTimer: null,
  };

  // Slack ID -> display name maps, built from facets
  let userNames = {};
  let channelNames = {};
  let namesPromise = null;

  // --- DOM refs ---
  const loginView = document.getElementById("login-view");
  const mainLayout = document.getElementById("main-layout");
  const loginForm = document.getElementById("login-form");
  const apiKeyInput = document.getElementById("api-key-input");
  const loginError = document.getElementById("login-error");
  const healthBadge = document.getElementById("health-badge");
  const graphInfo = document.getElementById("graph-info");
  const detailPanel = document.getElementById("detail-panel");
  const sidebarWiki = document.getElementById("sidebar-wiki");
  const sidebarSources = document.getElementById("sidebar-sources");
  const memListEl = document.getElementById("memory-list");
  const memReaderEl = document.getElementById("memory-reader");
  const popover = document.getElementById("filter-popover");
  const popoverSearch = document.getElementById("popover-search");
  const popoverList = document.getElementById("popover-list");

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
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      return { success: false, error: text || res.statusText };
    }
  }

  function postJson(path, body) {
    return apiFetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }

  // --- Lucide icons helper ---
  function refreshIcons() {
    if (typeof lucide !== "undefined") lucide.createIcons();
  }

  // --- Helpers ---
  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str == null ? "" : String(str);
    return div.innerHTML.replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  function parseDate(s) {
    if (!s) return null;
    const d = new Date(String(s).replace(" ", "T").replace(/(\.\d{3})\d+/, "$1"));
    return isNaN(d.getTime()) ? null : d;
  }

  function shortDate(s) {
    const d = parseDate(s);
    if (!d) return "";
    const opts = d.getFullYear() === new Date().getFullYear()
      ? { month: "short", day: "numeric" }
      : { month: "short", day: "numeric", year: "numeric" };
    return d.toLocaleDateString("en-US", opts);
  }

  function longDate(s) {
    const d = parseDate(s);
    if (!d) return "";
    return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" }) +
      ", " + d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  }

  const AVATAR_COLORS = ["#7c6cf2", "#0ea5a4", "#e0732b", "#d9467a", "#4f7cc9", "#3f9d5a", "#b8862b", "#9b5de5", "#1f9bb5", "#c2533a"];

  function hashString(s) {
    let h = 5381;
    for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
    return Math.abs(h);
  }

  function initials(name) {
    const words = String(name || "?").trim().split(/\s+/).filter(Boolean);
    if (words.length >= 2) return (words[0][0] + words[1][0]).toUpperCase();
    return String(name || "?").slice(0, 2).toUpperCase();
  }

  function avatarHtml(id, name, large) {
    const color = AVATAR_COLORS[hashString(id || name || "") % AVATAR_COLORS.length];
    return `<span class="avatar${large ? " avatar-lg" : ""}" style="background:${color}">${escapeHtml(initials(name))}</span>`;
  }

  function userLabel(id) { return userNames[id] || id; }
  function channelLabel(id) { return "#" + (channelNames[id] || id); }
  function projectLabel(ref) { return String(ref || "").replace(/^project:/, ""); }

  // Tag dimension: which filter dropdown a tag belongs to
  function tagDim(tag) {
    if (tag.startsWith("slack-user:")) return "person";
    if (tag.startsWith("slack-channel:")) return "channel";
    return "tag";
  }

  function authorOf(m) {
    const t = (m.tags || []).find((x) => x.startsWith("slack-user:"));
    if (!t) return null;
    const id = t.slice("slack-user:".length);
    return { id, name: userLabel(id), named: Boolean(userNames[id]) };
  }

  function authorHtml(m, large) {
    const a = authorOf(m);
    if (a) {
      return `<span class="flex items-center gap-2 min-w-0">${avatarHtml(a.id, a.name, large)}<span class="truncate font-medium text-fg ${a.named ? "" : "font-mono text-[12px]"}">${escapeHtml(a.name)}</span></span>`;
    }
    const cat = m.category || "general";
    return `<span class="flex items-center gap-2 min-w-0"><span class="avatar${large ? " avatar-lg" : ""}" style="background:#24242b;color:#a99ff7">${escapeHtml(cat.slice(0, 1).toUpperCase())}</span><span class="truncate font-medium text-accent-2">${escapeHtml(cat)}</span></span>`;
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
    loadSources();
    lastRoute = null;
    route();
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

  // --- Routing (hash: #view, #wiki/<id>, #memories?tag=a&src=..&cat=..&q=..&scope=..) ---
  function parseHash() {
    const raw = window.location.hash.slice(1);
    const qi = raw.indexOf("?");
    const path = qi >= 0 ? raw.slice(0, qi) : raw;
    const query = qi >= 0 ? raw.slice(qi + 1) : "";
    const slash = path.indexOf("/");
    const view = slash >= 0 ? path.slice(0, slash) : path;
    let sub = slash >= 0 ? path.slice(slash + 1) : "";
    try { sub = decodeURIComponent(sub); } catch { /* keep raw */ }
    return { view, sub, params: new URLSearchParams(query) };
  }

  function route() {
    if (!apiKey) return;
    const key = window.location.hash;
    if (key === lastRoute) return;
    lastRoute = key;
    const { view, sub, params } = parseHash();
    const v = VIEWS.includes(view) ? view : "wiki";
    if (v === "memories") readFilterParams(params);
    activateView(v, sub);
  }

  function setHash(hash, push) {
    if (window.location.hash !== hash) {
      history[push ? "pushState" : "replaceState"](null, "", hash);
    }
    lastRoute = window.location.hash;
  }

  window.addEventListener("popstate", route);
  window.addEventListener("hashchange", route);

  function navigate(view) {
    setHash(view === "memories" ? memHash() : "#" + view, true);
    activateView(view);
  }

  function activateView(viewName, sub) {
    currentView = viewName;
    document.querySelectorAll(".nav-link[data-view]").forEach((n) => {
      n.classList.toggle("active", n.dataset.view === viewName);
    });
    document.querySelectorAll(".content-view").forEach((v) => { v.classList.remove("active"); });
    const target = document.getElementById("view-" + viewName);
    if (target) target.classList.add("active");
    // Sidebar: nav is always visible; below it, wiki topics or sources
    sidebarWiki.hidden = viewName !== "wiki";
    sidebarSources.hidden = viewName === "wiki";
    closePopover();

    if (viewName === "wiki") loadEntities(sub);
    if (viewName === "graph") loadGraph();
    if (viewName === "stats") loadStats();
    if (viewName === "health") { loadConflicts(); loadLintStats(); }
    if (viewName === "memories") { syncFilterUI(); loadMemories(true); }
    refreshIcons();
  }

  document.querySelectorAll(".nav-link[data-view]").forEach((btn) => {
    btn.addEventListener("click", (e) => { e.preventDefault(); navigate(btn.dataset.view); });
  });

  // --- Stats ---
  async function loadStats() {
    try {
      const data = await apiFetch("/api/wiki/stats");
      if (!data.success) return;
      const s = data.stats;

      document.getElementById("stat-total").textContent = Number(s.total_memories).toLocaleString();
      document.getElementById("stat-relations").textContent = Number(s.total_relations).toLocaleString();
      document.getElementById("stat-connected").textContent = Number(s.connected_memories).toLocaleString();
      document.getElementById("stat-orphans").textContent = Number(s.orphan_memories).toLocaleString();
      document.getElementById("stat-conflicts").textContent = s.open_conflicts;
      document.getElementById("stat-avg-rel").textContent = s.avg_relations_per_doc;

      // Health badge
      healthBadge.textContent = s.health_score + "/100";
      const badgeColor = s.health_score >= 80
        ? "bg-green-400/10 text-green-400"
        : s.health_score >= 50
        ? "bg-amber-400/10 text-amber-400"
        : "bg-red-400/10 text-red-400";
      healthBadge.className = "text-[11px] px-2 py-0.5 rounded-full font-medium " + badgeColor;

      renderBars("heat-bars", s.heat_tiers || {});
      renderBarsDynamic("category-bars", s.categories || {});
    } catch (e) {
      console.error("Failed to load stats:", e);
    }
  }

  const heatColorMap = { hot: "#ef4444", warm: "#f59e0b", cool: "#3b82f6", cold: "#6b7280" };

  function barRow(key, val, pct, color) {
    return `<div class="flex items-center gap-3 text-sm">
        <span class="w-28 text-dim text-xs truncate" title="${escapeHtml(key)}">${escapeHtml(key)}</span>
        <div class="flex-1"><div class="bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <span class="w-12 text-right text-xs tabular-nums text-fg-2">${Number(val).toLocaleString()}</span>
      </div>`;
  }

  function renderBars(containerId, data) {
    const el = document.getElementById(containerId);
    const max = Math.max(...Object.values(data), 1);
    el.innerHTML = Object.entries(data).map(([key, val]) =>
      barRow(key, val, Math.max((val / max) * 100, 2), heatColorMap[key] || "#7c6cf2")
    ).join("");
  }

  function renderBarsDynamic(containerId, data) {
    const el = document.getElementById(containerId);
    const entries = Object.entries(data).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const max = Math.max(...entries.map(([, v]) => v), 1);
    el.innerHTML = entries.map(([key, val]) =>
      barRow(key, val, Math.max((val / max) * 100, 2), "#7c6cf2")
    ).join("");
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
    if (typeof d3 === "undefined") { graphInfo.textContent = "Graph library failed to load"; return; }
    if (simulation) simulation.stop();
    const svg = d3.select("#graph-svg");
    svg.selectAll("*").remove();

    const width = svg.node().getBoundingClientRect().width;
    const height = svg.node().getBoundingClientRect().height;

    const g = svg.append("g");

    const zoom = d3.zoom()
      .scaleExtent([0.2, 5])
      .on("zoom", (event) => g.attr("transform", event.transform));
    svg.call(zoom);

    const nodeMap = new Map(nodes.map((n) => [n.id, { ...n }]));
    const d3Nodes = Array.from(nodeMap.values());

    const d3Edges = edges
      .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
      .map((e) => ({
        source: e.source,
        target: e.target,
        similarity: e.similarity || 0.5,
      }));

    simulation = d3.forceSimulation(d3Nodes)
      .force("link", d3.forceLink(d3Edges).id((d) => d.id).distance(80))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide().radius((d) => getNodeRadius(d) + 2));

    const link = g.append("g")
      .selectAll("line")
      .data(d3Edges)
      .join("line")
      .attr("class", "link")
      .attr("stroke-width", (d) => Math.max(0.5, (d.similarity || 0.5) * 3));

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
        const proj = (d.source_ref || "").replace("project:", "").split("/").pop();
        const cat = d.category || "";
        if (proj) return `${proj}: ${cat}`.slice(0, 25);
        return (d.title || cat).slice(0, 25);
      });

    const initialScale = Math.min(1, Math.max(0.3, 50 / Math.sqrt(d3Nodes.length || 1)));
    svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, height / 2).scale(initialScale).translate(-width / 2, -height / 2));

    simulation.on("tick", () => {
      link
        .attr("x1", (d) => d.source.x)
        .attr("y1", (d) => d.source.y)
        .attr("x2", (d) => d.target.x)
        .attr("y2", (d) => d.target.y);
      node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });

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
      const data = await apiFetch(`/api/wiki/relations?id=${encodeURIComponent(node.id)}`);
      if (!data.success) return;

      const m = data.memory;
      document.getElementById("detail-content").innerHTML = `
        <div class="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-dim mb-3">
          <span class="chip chip-accent">${escapeHtml(m.category || "general")}</span>
          <span>shown ${Number(m.shown_count) || 0}&times;</span>
          <span>&middot;</span>
          <span class="font-mono">${escapeHtml(m.source_ref || "no project")}</span>
          <span>&middot;</span>
          <span>${escapeHtml(shortDate(m.created_at))}</span>
        </div>
        <div class="mem-content text-sm">${escapeHtml(m.content)}</div>
      `;

      const rels = data.relations || [];
      document.getElementById("detail-relations").innerHTML = rels.length
        ? `<h3 class="text-xs font-semibold text-dim uppercase tracking-wider mt-5 mb-2">Related (${rels.length})</h3>` +
          rels.map((r) => `
            <div class="detail-relation card p-3 mb-2 cursor-pointer hover:border-[#2c2c34] transition-colors" data-id="${escapeHtml(r.id)}">
              <span class="text-xs text-accent-2 font-medium">${r.similarity ? (r.similarity * 100).toFixed(0) + "%" : ""}</span>
              <div class="text-sm text-fg-2 mt-1">${escapeHtml((r.content || "").slice(0, 150))}</div>
              <div class="text-xs text-dim mt-1">${escapeHtml(r.category || "")} &middot; ${escapeHtml(r.relation_type || "similar")}</div>
            </div>
          `).join("")
        : `<p class="text-sm text-dim mt-4">No relations found</p>`;

      document.querySelectorAll("#detail-relations .detail-relation").forEach((card) => {
        card.addEventListener("click", () => {
          const relId = card.dataset.id;
          if (relId) showDetail({ id: relId, title: "" });
        });
      });
    } catch (e) {
      document.getElementById("detail-content").textContent = "Failed to load details";
    }
  }

  document.getElementById("detail-close").addEventListener("click", () => {
    detailPanel.hidden = true;
  });

  // --- Entities (Wiki view) ---
  let allEntities = [];
  let activeEntityId = null;

  function markActiveTopic(id) {
    activeEntityId = id;
    document.querySelectorAll(".wiki-topic").forEach((i) => {
      i.classList.toggle("is-active", i.dataset.id === id);
    });
  }

  async function loadEntities(selectId) {
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
      const q = (document.getElementById("entity-search")?.value || "").toLowerCase();
      renderEntityNav(q ? filterEntities(q) : allEntities);

      const pick = selectId || activeEntityId || allEntities[0].id;
      markActiveTopic(pick);
      loadEntityArticle(pick);
    } catch (e) {
      console.error("Failed to load entities:", e);
    }
  }

  function renderEntityNav(entities) {
    const navList = document.getElementById("entity-nav-list");
    navList.innerHTML = entities.map((e) => {
      const project = (e.source_ref || "").replace("project:", "").split("/").pop() || "";
      return `<a class="wiki-topic ${e.id === activeEntityId ? "is-active" : ""}" data-id="${escapeHtml(e.id)}">
          <div class="topic-title text-[13px] text-fg-2 leading-snug break-words">${escapeHtml(e.title || "Untitled")}</div>
          ${project ? `<div class="text-[11px] text-dim mt-0.5 font-mono">${escapeHtml(project)}</div>` : ""}
      </a>`;
    }).join("");

    navList.querySelectorAll(".wiki-topic").forEach((item) => {
      item.addEventListener("click", (e) => {
        e.preventDefault();
        markActiveTopic(item.dataset.id);
        loadEntityArticle(item.dataset.id);
      });
    });
  }

  function filterEntities(q) {
    return allEntities.filter((e) =>
      (e.title || "").toLowerCase().includes(q) ||
      (e.source_ref || "").toLowerCase().includes(q)
    );
  }

  const entitySearchInput = document.getElementById("entity-search");
  if (entitySearchInput) {
    entitySearchInput.addEventListener("input", () => {
      renderEntityNav(filterEntities(entitySearchInput.value.toLowerCase()));
    });
  }

  async function loadEntityArticle(entityId) {
    const placeholder = document.getElementById("article-placeholder");
    const content = document.getElementById("article-content");
    placeholder.hidden = true;
    content.hidden = false;
    setHash("#wiki/" + encodeURIComponent(entityId), false);

    try {
      const data = await apiFetch(`/api/wiki/entity?id=${encodeURIComponent(entityId)}`);
      if (!data.success) return;

      const e = data.entity;
      const shown = Number(e.shown_count) || 0;
      const heatPct = Math.min(shown * 5, 100);
      const heatColor = shown >= 20 ? "#ef4444" : shown >= 5 ? "#f59e0b" : shown >= 1 ? "#3b82f6" : "#6b7280";
      const heatLabel = shown >= 20 ? "hot" : shown >= 5 ? "warm" : shown >= 1 ? "cool" : "cold";

      document.getElementById("article-title").textContent = e.title || "Untitled";
      document.getElementById("article-meta").innerHTML =
        `<span class="font-mono">${escapeHtml(e.source_ref || "no project")}</span>` +
        ` &middot; ${Number(e.cluster_size) || 0} sources` +
        ` &middot; shown ${shown}&times; (${heatLabel})` +
        ` &middot; ${escapeHtml(shortDate(e.created_at))}`;
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
        relEl.innerHTML = `<h3 class="text-sm font-semibold text-fg mb-3">Related Topics</h3><div class="flex flex-wrap gap-1.5">` +
          data.related_entities.map((r) =>
            `<button class="chip text-[12.5px] py-0.5 px-2.5 hover:text-fg" data-id="${escapeHtml(r.id)}">${escapeHtml(r.title || "?")}</button>`
          ).join("") + `</div>`;
        relEl.hidden = false;
        relEl.querySelectorAll("[data-id]").forEach((link) => {
          link.addEventListener("click", () => {
            markActiveTopic(link.dataset.id);
            loadEntityArticle(link.dataset.id);
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
            const tData = await apiFetch(`/api/wiki/timeline?id=${encodeURIComponent(entityId)}&limit=30`);
            if (tData.success && tData.timeline.length > 0) {
              timelineList.innerHTML = tData.timeline.map((t) => `
                <div class="timeline-entry flex gap-3 py-2">
                  <div class="w-[11px] h-[11px] rounded-full mt-1 flex-shrink-0" style="background:${heatColorMap[t.heat] || "#6b7280"}"></div>
                  <div>
                    <div class="text-xs text-dim">${escapeHtml(shortDate(t.created_at))}</div>
                    <div class="text-sm text-fg-2 mt-0.5">${escapeHtml(t.content)}</div>
                    <div class="text-xs text-dim mt-0.5">${escapeHtml(t.category || "")} &middot; ${escapeHtml(t.source || "")} ${t.similarity ? "&middot; " + (t.similarity * 100).toFixed(0) + "% match" : ""}</div>
                  </div>
                </div>
              `).join("");
              btnTimeline.textContent = "Hide";
            } else {
              timelineList.innerHTML = '<p class="text-sm text-dim">No timeline data available</p>';
              btnTimeline.textContent = "Show";
            }
          } catch (err) {
            timelineList.innerHTML = '<p class="text-sm text-dim">Failed to load timeline</p>';
            btnTimeline.textContent = "Show";
          }
        };
      }

      // Source memories
      const srcList = document.getElementById("source-memories-list");
      if (data.source_memories && data.source_memories.length > 0) {
        document.getElementById("article-sources").hidden = false;
        srcList.innerHTML = data.source_memories.map((m) => {
          const matchPct = m.similarity ? (m.similarity * 100).toFixed(0) + "%" : "";
          return `<div class="card p-3">
            <div class="text-sm text-fg-2 leading-relaxed">${escapeHtml(m.content || "")}</div>
            <div class="flex items-center gap-2 text-xs mt-2">
              <span class="text-dim">${escapeHtml(m.category || "")}</span>
              ${matchPct ? `<span class="text-accent-2">${matchPct} match</span>` : ""}
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
    if (typeof marked !== "undefined") {
      marked.setOptions({ breaks: true, gfm: true });
      return marked.parse(md);
    }
    return escapeHtml(md).replace(/\n/g, "<br>");
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
        <div class="card p-4 border-l-2 border-l-amber-500">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-medium text-amber-400 flex items-center gap-1.5">
              <i data-lucide="shield-alert" class="w-3.5 h-3.5"></i> Contradiction
            </span>
            <span class="text-xs text-dim">${escapeHtml(shortDate(c.created_at))}</span>
          </div>
          <div class="text-sm text-dim mb-3">${escapeHtml(c.explanation || "")}</div>
          <div class="grid grid-cols-2 gap-4 mb-3">
            <div>
              <div class="text-xs font-medium text-dim mb-1">Memory A</div>
              <div class="conflict-doc text-sm text-fg-2 leading-relaxed">${escapeHtml(c.doc_a_content || "")}</div>
            </div>
            <div>
              <div class="text-xs font-medium text-dim mb-1">Memory B</div>
              <div class="conflict-doc text-sm text-fg-2 leading-relaxed">${escapeHtml(c.doc_b_content || "")}</div>
            </div>
          </div>
          <div class="flex gap-2">
            <button class="resolve-btn btn" data-conflict-id="${escapeHtml(c.id)}" data-resolution="keep_a">Keep A</button>
            <button class="resolve-btn btn" data-conflict-id="${escapeHtml(c.id)}" data-resolution="keep_b">Keep B</button>
            <button class="resolve-btn btn btn-ghost" data-conflict-id="${escapeHtml(c.id)}" data-resolution="dismiss">Dismiss</button>
          </div>
        </div>
      `).join("");
      listEl.querySelectorAll(".resolve-btn").forEach((btn) => {
        btn.addEventListener("click", () => resolveConflict(btn.dataset.conflictId, btn.dataset.resolution));
      });
      refreshIcons();
    } catch (e) {
      console.error("Failed to load conflicts:", e);
    }
  }

  // Generate entity pages (gap resolution)
  async function compileCategory(category, btn) {
    if (btn) { btn.disabled = true; btn.textContent = "Generating..."; }
    try {
      let totalCreated = 0;
      const data = await postJson("/api/memory/maintenance", { job_type: "compilation", limit: 50, full_sweep: true });
      if (data.success) totalCreated = (data.results?.pages_created || 0) + (data.results?.pages_updated || 0);
      if (btn) btn.textContent = totalCreated > 0 ? `Created ${totalCreated}` : "No new pages";
      setTimeout(() => {
        document.getElementById("btn-run-lint")?.click();
        if (btn) { btn.textContent = "Generate"; btn.disabled = false; }
      }, 2000);
    } catch (e) {
      console.error("Compile failed:", e);
      if (btn) { btn.textContent = "Error"; setTimeout(() => { btn.textContent = "Generate"; btn.disabled = false; }, 2000); }
    }
  }

  async function resolveConflict(conflictId, resolution) {
    try {
      const data = await postJson("/api/memory/conflict/resolve", { conflict_id: conflictId, resolution });
      if (data.success) {
        loadConflicts();
        loadStats();
      }
    } catch (e) {
      console.error("Resolve failed:", e);
    }
  }

  function statCard(val, label, color) {
    return `<div class="card p-5"><p class="text-xs text-dim mb-2">${escapeHtml(label)}</p><p class="text-2xl font-semibold tabular-nums ${color}">${escapeHtml(val)}</p></div>`;
  }

  async function loadLintStats() {
    const reportEl = document.getElementById("lint-report");
    const statsEl = document.getElementById("lint-stats");
    if (!reportEl || !statsEl) return;

    reportEl.hidden = false;
    statsEl.innerHTML = statCard("...", "Loading", "text-dim");

    try {
      const data = await apiFetch("/api/wiki/stats");
      if (!data.success) return;
      const s = data.stats;
      const scoreColor = s.health_score >= 80 ? "text-green-400" : s.health_score >= 50 ? "text-amber-400" : "text-red-400";
      statsEl.innerHTML = statCard(s.health_score, "Health score", scoreColor) +
        statCard(s.open_conflicts, "Open conflicts", s.open_conflicts > 0 ? "text-red-400" : "text-green-400") +
        statCard(s.orphan_memories, "Orphans", "text-amber-400") +
        statCard(`${s.connected_memories}/${s.total_memories}`, "Connected", "text-fg");
    } catch (e) { console.error("Lint stats failed:", e); }
  }

  // Legacy lint button handler (button removed from UI; kept for compileCategory refresh)
  const btnLint = document.getElementById("btn-run-lint");
  if (btnLint) {
    btnLint.addEventListener("click", async () => {
      btnLint.disabled = true;
      btnLint.textContent = "Running...";
      try {
        const data = await apiFetch("/api/wiki/lint", { method: "POST" });
        if (data.success && data.report) {
          const r = data.report;
          const statsEl = document.getElementById("lint-stats");
          document.getElementById("lint-report").hidden = false;
          statsEl.innerHTML = statCard(r.health_score || 0, "Health score", "text-fg") +
            statCard(r.open_conflicts || 0, "Open conflicts", "text-fg") +
            statCard(r.orphan_count || 0, "Orphans", "text-fg") +
            statCard(`${r.connected_memories || 0}/${r.total_memories || 0}`, "Connected", "text-fg");
          if (r.contradictions_found > 0) statsEl.innerHTML += statCard(r.contradictions_found, "New this run", "text-amber-400");
          if (r.entity_page_count !== undefined) statsEl.innerHTML += statCard(r.entity_page_count, "Entity pages", "text-fg");

          const gapsSection = document.getElementById("gaps-section");
          const gapsList = document.getElementById("gaps-list");
          if (r.knowledge_gaps && r.knowledge_gaps.length > 0) {
            gapsSection.hidden = false;
            gapsList.innerHTML = r.knowledge_gaps.map((g) => `
              <div class="card flex items-center justify-between p-3">
                <div>
                  <div class="text-sm font-medium text-fg">${escapeHtml(g.category)}</div>
                  <div class="text-xs text-dim">${Number(g.count)} memories, no entity page</div>
                </div>
                <button class="gap-action btn btn-primary" data-category="${escapeHtml(g.category)}">Generate</button>
              </div>
            `).join("");
            gapsList.querySelectorAll(".gap-action").forEach((btn) => {
              btn.addEventListener("click", () => compileCategory(btn.dataset.category, btn));
            });
          } else {
            gapsSection.hidden = true;
          }

          const orphansSection = document.getElementById("orphans-section");
          const orphansList = document.getElementById("orphans-list");
          if (r.top_orphans && r.top_orphans.length > 0) {
            orphansSection.hidden = false;
            orphansList.innerHTML = r.top_orphans.map((o) => `
              <div class="card p-3">
                <div class="text-sm text-fg-2">${escapeHtml(o.content)}</div>
                <div class="text-xs text-dim mt-1">${escapeHtml(o.category)} &middot; shown ${Number(o.shown_count) || 0}&times;</div>
              </div>
            `).join("");
          } else {
            orphansSection.hidden = true;
          }

          loadConflicts();
          loadStats();
        }
      } catch (e) {
        console.error("Lint failed:", e);
      }
      btnLint.textContent = "Run Lint";
      btnLint.disabled = false;
    });
  }

  // =========================================================================
  // FACETS (names, sidebar sources, filter dropdowns)
  // =========================================================================

  async function fetchFacets(field, prefix, limit, narrow) {
    const p = new URLSearchParams();
    p.set("field", field);
    p.set("limit", String(limit));
    if (prefix) p.set("prefix", prefix);
    if (narrow) {
      (narrow.tags || []).forEach((t) => p.append("tag", t));
      if (narrow.scope) p.set("scope", narrow.scope);
      if (narrow.category) p.set("category", narrow.category);
      if (narrow.src) p.set("source_ref_prefix", narrow.src);
    }
    try {
      const data = await apiFetch("/api/memory/facets?" + p.toString());
      return data && data.success ? (data.facets || []) : [];
    } catch (e) {
      if (e.message !== "Unauthorized") console.warn("Facets unavailable:", e);
      return [];
    }
  }

  // Parse "<prefix><Id>:<Name>" (name may contain colons and spaces)
  function buildNameMap(facets, prefix) {
    const map = {};
    for (const f of facets) {
      const rest = String(f.value || "").slice(prefix.length);
      const i = rest.indexOf(":");
      if (i <= 0) continue;
      const id = rest.slice(0, i);
      const name = rest.slice(i + 1).trim();
      if (name && !map[id]) map[id] = name; // facets sorted by count desc: most-used name wins
    }
    return map;
  }

  function loadNames() {
    if (!namesPromise) {
      namesPromise = Promise.all([
        fetchFacets("tag", "slack-user-name:", 500),
        fetchFacets("tag", "slack-channel-name:", 500),
      ]).then(([u, c]) => {
        userNames = buildNameMap(u, "slack-user-name:");
        channelNames = buildNameMap(c, "slack-channel-name:");
      });
    }
    return namesPromise;
  }

  async function loadSources() {
    await loadNames();
    const [people, channels, projects] = await Promise.all([
      fetchFacets("tag", "slack-user:", 5),
      fetchFacets("tag", "slack-channel:", 5),
      fetchFacets("source_ref", "project:", 5),
    ]);

    renderSourceGroup("sources-people", people, (f) => {
      const id = f.value.slice("slack-user:".length);
      return { glyph: avatarHtml(id, userLabel(id)), label: userNames[id] ? userNames[id].split(" ")[0] : id, title: userLabel(id), kind: "tag" };
    });
    renderSourceGroup("sources-channels", channels, (f) => {
      const id = f.value.slice("slack-channel:".length);
      return { glyph: `<span class="source-glyph">#</span>`, label: channelNames[id] || id, title: channelLabel(id), kind: "tag" };
    });
    renderSourceGroup("sources-projects", projects, (f) => {
      const full = projectLabel(f.value);
      return { glyph: `<span class="source-glyph"><i data-lucide="folder-git-2" class="w-3.5 h-3.5"></i></span>`, label: full.split("/").pop() || full, title: full, kind: "src" };
    });
    refreshIcons();
    // Names may have arrived after the first list render
    if (currentView === "memories") { renderMemList(); renderReader(); renderActiveFilters(); }
  }

  function renderSourceGroup(groupId, facets, describe) {
    const group = document.getElementById(groupId);
    const list = group.querySelector(".sources-list");
    if (!facets.length) { group.hidden = true; list.innerHTML = ""; return; }
    group.hidden = false;
    list.innerHTML = facets.map((f) => {
      const d = describe(f);
      return `<button class="source-link" data-kind="${d.kind}" data-value="${escapeHtml(f.value)}" title="${escapeHtml(d.title)}">
        ${d.glyph}<span class="truncate">${escapeHtml(d.label)}</span><span class="count">${Number(f.count).toLocaleString()}</span>
      </button>`;
    }).join("");
    list.querySelectorAll(".source-link").forEach((btn) => {
      btn.addEventListener("click", () => {
        const { kind, value } = btn.dataset;
        setFilters(() => {
          mem.tags = kind === "tag" ? [value] : [];
          mem.src = kind === "src" ? value : "";
          mem.cat = "";
          mem.q = "";
        });
      });
    });
  }

  // =========================================================================
  // MEMORIES VIEW (filter bar + split list/reader)
  // =========================================================================

  function readFilterParams(params) {
    mem.tags = params.getAll("tag").filter(Boolean);
    mem.src = params.get("src") || "";
    mem.cat = params.get("cat") || "";
    mem.q = params.get("q") || "";
    const scope = params.get("scope") || "";
    mem.scope = ["personal", "shared"].includes(scope) ? scope : "";
  }

  function memHash() {
    const p = new URLSearchParams();
    mem.tags.forEach((t) => p.append("tag", t));
    if (mem.src) p.set("src", mem.src);
    if (mem.cat) p.set("cat", mem.cat);
    if (mem.q) p.set("q", mem.q);
    if (mem.scope) p.set("scope", mem.scope);
    const s = p.toString();
    return "#memories" + (s ? "?" + s : "");
  }

  // Mutate filters, write them to the URL, then show/reload the list
  function setFilters(mutate, push = true) {
    mutate();
    setHash(memHash(), push);
    if (currentView !== "memories") {
      activateView("memories");
    } else {
      syncFilterUI();
      loadMemories(true);
    }
  }

  function hasFilters() {
    return mem.tags.length > 0 || Boolean(mem.src || mem.cat || mem.q);
  }

  function syncFilterUI() {
    const search = document.getElementById("memory-search");
    if (search && document.activeElement !== search) search.value = mem.q;
    document.querySelectorAll(".scope-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.scope === mem.scope);
    });
    document.querySelectorAll(".filter-btn").forEach((b) => {
      const dim = b.dataset.dim;
      const on = dim === "project" ? Boolean(mem.src)
        : dim === "category" ? Boolean(mem.cat)
        : mem.tags.some((t) => tagDim(t) === dim);
      b.classList.toggle("has-value", on);
    });
    renderActiveFilters();
  }

  function pillHtml(label, value, kind, raw) {
    return `<button class="chip chip-accent filter-pill" data-kind="${kind}" data-value="${escapeHtml(raw)}" title="Remove filter">
      <span class="opacity-70">${escapeHtml(label)}:</span><span class="${kind === "tag" && tagDim(raw) === "tag" ? "font-mono" : ""}">${escapeHtml(value)}</span><span class="ml-0.5 opacity-70">&#x2715;</span>
    </button>`;
  }

  function renderActiveFilters() {
    const el = document.getElementById("active-filters");
    if (!el) return;
    let html = "";
    for (const t of mem.tags) {
      const dim = tagDim(t);
      if (dim === "person") html += pillHtml("Person", userLabel(t.slice("slack-user:".length)), "tag", t);
      else if (dim === "channel") html += pillHtml("Channel", channelLabel(t.slice("slack-channel:".length)), "tag", t);
      else html += pillHtml("Tag", t, "tag", t);
    }
    if (mem.src) html += pillHtml("Project", projectLabel(mem.src), "src", mem.src);
    if (mem.cat) html += pillHtml("Category", mem.cat, "cat", mem.cat);
    el.innerHTML = html;
    el.querySelectorAll(".filter-pill").forEach((b) => {
      b.addEventListener("click", () => {
        const { kind, value } = b.dataset;
        setFilters(() => {
          if (kind === "tag") mem.tags = mem.tags.filter((t) => t !== value);
          if (kind === "src") mem.src = "";
          if (kind === "cat") mem.cat = "";
        });
      });
    });
    document.getElementById("clear-filters").hidden = !hasFilters();
  }

  function renderResultCount() {
    const el = document.getElementById("result-count");
    if (!el) return;
    if (mem.mode === "search") {
      el.textContent = `${mem.items.length.toLocaleString()} result${mem.items.length === 1 ? "" : "s"}`;
    } else {
      el.textContent = `${mem.total.toLocaleString()} memor${mem.total === 1 ? "y" : "ies"}`;
    }
  }

  function listParams() {
    const p = new URLSearchParams();
    p.set("limit", String(mem.limit));
    p.set("offset", String(mem.offset));
    if (mem.scope) p.set("scope", mem.scope);
    if (mem.cat) p.set("category", mem.cat);
    if (mem.q) p.set("q", mem.q);
    mem.tags.forEach((t) => p.append("tag", t));
    if (mem.src) p.set("source_ref_prefix", mem.src);
    return p.toString();
  }

  async function loadMemories(reset) {
    if (!memListEl) return;
    if (reset) {
      mem.offset = 0;
      mem.rawLoaded = 0;
      mem.editing = false;
      memListEl.innerHTML = '<div class="text-sm text-dim text-center py-10">Loading...</div>';
    }
    const seq = ++mem.reqSeq;
    try {
      const data = await apiFetch(`/api/memory/list?${listParams()}`);
      if (seq !== mem.reqSeq) return; // a newer request superseded this one
      if (!data.success) {
        memListEl.innerHTML = '<div class="text-sm text-dim text-center py-10">Error loading memories.</div>';
        document.getElementById("memory-more").hidden = true;
        return;
      }
      const raw = data.results || [];
      mem.mode = data.mode || "browse";
      mem.total = Number(data.total) || 0;
      mem.rawLoaded += raw.length;
      // entity-page memories belong in the Wiki view
      const items = raw.filter((m) => m.category !== "entity-page");
      mem.items = reset ? items : mem.items.concat(items);
      if (reset && !mem.items.some((m) => m.id === mem.selectedId)) {
        mem.selectedId = mem.items.length ? mem.items[0].id : null;
      }
      renderMemList();
      renderReader();
      renderMoreButton();
      renderResultCount();
    } catch (e) {
      if (seq !== mem.reqSeq) return;
      memListEl.innerHTML = '<div class="text-sm text-dim text-center py-10">Failed to load.</div>';
      console.error("Failed to load memories:", e);
    }
  }

  function renderMoreButton() {
    const more = document.getElementById("memory-more");
    const show = mem.mode !== "search" && mem.rawLoaded < mem.total;
    more.hidden = !show;
    if (show) {
      document.getElementById("load-more-btn").textContent =
        `Load more (${mem.rawLoaded.toLocaleString()} of ${mem.total.toLocaleString()})`;
    }
  }

  document.getElementById("load-more-btn")?.addEventListener("click", () => {
    mem.offset += mem.limit;
    loadMemories(false);
  });

  function renderMemList() {
    if (!memListEl) return;
    if (!mem.items.length) {
      memListEl.innerHTML = `<div class="text-sm text-dim text-center py-10 px-6">No memories match${hasFilters() ? " these filters" : ""}.</div>`;
      return;
    }
    memListEl.innerHTML = mem.items.map((m) => {
      const snippet = String(m.content || "").replace(/\s+/g, " ").trim().slice(0, 200);
      return `<button class="mem-item ${m.id === mem.selectedId ? "selected" : ""}" data-id="${escapeHtml(m.id)}">
        <div class="flex items-center gap-2 text-[12.5px]">
          ${authorHtml(m)}
          <span class="ml-auto flex-shrink-0 text-[11.5px] text-dim">${escapeHtml(shortDate(m.created_at))}</span>
        </div>
        <div class="snippet">${escapeHtml(snippet)}</div>
      </button>`;
    }).join("");
    memListEl.querySelectorAll(".mem-item").forEach((el) => {
      el.addEventListener("click", () => selectMemory(el.dataset.id));
    });
  }

  function selectMemory(id, scroll) {
    if (!id) return;
    mem.selectedId = id;
    mem.editing = false;
    memListEl.querySelectorAll(".mem-item").forEach((el) => {
      const on = el.dataset.id === id;
      el.classList.toggle("selected", on);
      if (on && scroll) el.scrollIntoView({ block: "nearest" });
    });
    renderReader();
  }

  function moveSelection(delta) {
    if (!mem.items.length) return;
    const idx = mem.items.findIndex((m) => m.id === mem.selectedId);
    const next = Math.min(mem.items.length - 1, Math.max(0, (idx < 0 ? 0 : idx + delta)));
    selectMemory(mem.items[next].id, true);
  }

  function selectedMemory() {
    return mem.items.find((m) => m.id === mem.selectedId) || null;
  }

  function renderReader() {
    if (!memReaderEl) return;
    const m = selectedMemory();
    if (!m) {
      memReaderEl.innerHTML = `<div class="text-dim text-sm text-center pt-24">${mem.items.length ? "Select a memory." : ""}</div>`;
      return;
    }
    if (mem.editing) return; // edit form owns the reader

    const src = m.source_ref || "";
    const srcHtml = src
      ? (src.startsWith("project:")
        ? `<button class="reader-src font-mono hover:text-fg" data-value="${escapeHtml(src)}" title="Filter by this project">${escapeHtml(src)}</button>`
        : `<span class="font-mono">${escapeHtml(src)}</span>`)
      : "";
    const meta = [
      srcHtml,
      escapeHtml(longDate(m.created_at)),
      `shown ${Number(m.shown_count) || 0}&times;`,
      m.scope ? escapeHtml(m.scope) : "",
    ].filter(Boolean).join('<span class="opacity-50">&middot;</span>');

    const chips = [
      m.category ? `<button class="chip chip-accent reader-cat" data-value="${escapeHtml(m.category)}" title="Filter by category">${escapeHtml(m.category)}</button>` : "",
      // *-name: tags are lookup metadata for display names; the ID tag already shows them
      ...(m.tags || []).filter((t) => !/^slack-(user|channel)-name:/.test(t)).map((t) => `<button class="chip font-mono reader-tag" data-value="${escapeHtml(t)}" title="Filter by this tag">${escapeHtml(t)}</button>`),
    ].join("");

    memReaderEl.innerHTML = `<article class="max-w-3xl px-8 py-6">
      <div class="flex items-center gap-2.5 text-[15px] mb-1.5">
        ${authorHtml(m, true)}
      </div>
      <div class="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-dim mb-5">${meta}</div>
      <div class="mem-content">${escapeHtml(m.content || "")}</div>
      <div class="flex flex-wrap gap-1.5 mt-6">${chips}</div>
      <div class="flex items-center gap-2 mt-6 pt-4 border-t border-line">
        <button id="reader-edit" class="btn"><i data-lucide="pencil" class="w-3.5 h-3.5"></i>Edit</button>
        <button id="reader-delete" class="btn btn-danger"><i data-lucide="trash-2" class="w-3.5 h-3.5"></i>Delete</button>
        <button id="reader-copy" class="btn"><i data-lucide="copy" class="w-3.5 h-3.5"></i>Copy ID</button>
        <span class="ml-auto text-[11px] text-dim flex items-center gap-1"><span class="kbd">j</span><span class="kbd">k</span> or <span class="kbd">&uarr;</span><span class="kbd">&darr;</span> to move</span>
      </div>
    </article>`;

    memReaderEl.querySelectorAll(".reader-tag").forEach((b) => {
      b.addEventListener("click", () => {
        const v = b.dataset.value;
        if (!mem.tags.includes(v)) setFilters(() => { mem.tags = mem.tags.concat([v]); });
      });
    });
    memReaderEl.querySelector(".reader-cat")?.addEventListener("click", (e) => {
      const v = e.currentTarget.dataset.value;
      setFilters(() => { mem.cat = v; });
    });
    memReaderEl.querySelector(".reader-src")?.addEventListener("click", (e) => {
      const v = e.currentTarget.dataset.value;
      setFilters(() => { mem.src = v; });
    });
    document.getElementById("reader-edit").addEventListener("click", () => startEdit(m.id));
    document.getElementById("reader-delete").addEventListener("click", () => deleteMemory(m.id));
    document.getElementById("reader-copy").addEventListener("click", () => copyText(m.id));
    refreshIcons();
  }

  // --- Inline edit in the reader ---
  async function startEdit(id) {
    let doc = mem.items.find((m) => m.id === id);
    if (!doc) return;
    try {
      const data = await apiFetch(`/api/memory/get?id=${encodeURIComponent(id)}`);
      if (data.success && data.document) doc = { ...doc, ...data.document };
    } catch (e) { /* fall back to the list copy */ }
    if (mem.selectedId !== id) return;
    mem.editing = true;
    memReaderEl.innerHTML = `<div class="max-w-3xl px-8 py-6 space-y-4">
      <div class="text-xs text-dim">Editing memory <span class="font-mono">${escapeHtml(id.slice(0, 8))}</span></div>
      <textarea id="edit-textarea" class="field w-full leading-relaxed" rows="14">${escapeHtml(doc.content || "")}</textarea>
      <div class="grid grid-cols-3 gap-3">
        <label class="block"><span class="block text-xs text-dim mb-1">Category</span>
          <input class="field w-full" type="text" id="edit-category" placeholder="e.g. testing" value="${escapeHtml(doc.category || "")}"></label>
        <label class="block"><span class="block text-xs text-dim mb-1">Tags</span>
          <input class="field w-full font-mono" type="text" id="edit-tags" placeholder="comma-separated" value="${escapeHtml((doc.tags || []).join(", "))}"></label>
        <label class="block"><span class="block text-xs text-dim mb-1">Source ref</span>
          <input class="field w-full font-mono" type="text" id="edit-source-ref" placeholder="project:org/repo" value="${escapeHtml(doc.source_ref || "")}"></label>
      </div>
      <div class="flex gap-2">
        <button id="edit-save" class="btn btn-primary">Save</button>
        <button id="edit-cancel" class="btn btn-ghost">Cancel</button>
      </div>
    </div>`;
    document.getElementById("edit-save").addEventListener("click", () => saveEdit(id));
    document.getElementById("edit-cancel").addEventListener("click", cancelEdit);
    document.getElementById("edit-textarea").focus();
  }

  function cancelEdit() {
    mem.editing = false;
    renderReader();
  }

  async function saveEdit(id) {
    const body = { memory_id: id };
    const content = document.getElementById("edit-textarea").value.trim();
    const category = document.getElementById("edit-category").value.trim();
    const tagsStr = document.getElementById("edit-tags").value.trim();
    const sourceRef = document.getElementById("edit-source-ref").value.trim();
    if (content) body.content = content;
    if (category) body.category = category;
    if (tagsStr) body.tags = tagsStr.split(",").map((t) => t.trim()).filter(Boolean);
    if (sourceRef) body.source_ref = sourceRef;
    const saveBtn = document.getElementById("edit-save");
    saveBtn.disabled = true;
    try {
      const data = await postJson("/api/memory/update", body);
      if (!data.success) { showToast("Update failed."); saveBtn.disabled = false; return; }
      const m = mem.items.find((x) => x.id === id);
      if (m) {
        if (body.content) m.content = body.content;
        if (body.category) m.category = body.category;
        if (body.tags) m.tags = body.tags;
        if (body.source_ref) m.source_ref = body.source_ref;
      }
      mem.editing = false;
      renderMemList();
      renderReader();
      showToast("Memory updated.");
    } catch (e) {
      showToast("Update failed.");
      saveBtn.disabled = false;
    }
  }

  // --- Delete with undo ---
  async function deleteMemory(id) {
    if (!window.confirm("Delete this memory? You can undo for a few seconds.")) return;
    try {
      const data = await postJson("/api/memory/forget", { memory_id: id });
      if (!data.success) { showToast("Delete failed."); return; }
      const idx = mem.items.findIndex((m) => m.id === id);
      mem.items = mem.items.filter((m) => m.id !== id);
      mem.total = Math.max(0, mem.total - 1);
      const next = mem.items[Math.min(Math.max(idx, 0), mem.items.length - 1)];
      mem.selectedId = next ? next.id : null;
      mem.editing = false;
      renderMemList();
      renderReader();
      renderResultCount();
      showToast("Memory deleted.", async () => {
        await postJson("/api/memory/restore", { memory_id: id });
        mem.selectedId = id;
        loadMemories(true);
      });
    } catch (e) { showToast("Delete failed."); }
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      ta.remove();
    }
    showToast("ID copied.");
  }

  // --- Search + scope ---
  document.getElementById("memory-search")?.addEventListener("input", (e) => {
    clearTimeout(mem.searchTimer);
    mem.searchTimer = setTimeout(() => {
      const q = e.target.value.trim();
      if (q === mem.q) return;
      setFilters(() => { mem.q = q; }, false);
    }, 350);
  });

  document.querySelectorAll(".scope-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (btn.dataset.scope === mem.scope) return;
      setFilters(() => { mem.scope = btn.dataset.scope; });
    });
  });

  document.getElementById("clear-filters")?.addEventListener("click", () => {
    setFilters(() => { mem.tags = []; mem.src = ""; mem.cat = ""; mem.q = ""; });
  });

  // --- Filter dropdowns (popover) ---
  const DIMS = {
    person: { field: "tag", prefix: "slack-user:", limit: 200 },
    channel: { field: "tag", prefix: "slack-channel:", limit: 200 },
    project: { field: "source_ref", prefix: "project:", limit: 200 },
    tag: { field: "tag", prefix: "", limit: 300 },
    category: { field: "category", prefix: "", limit: 100 },
  };
  let popDim = null;
  let popItems = [];

  // Narrow facet counts by the other active filters (not the dimension being picked)
  function narrowFor(dim) {
    return {
      tags: mem.tags.filter((t) => tagDim(t) !== dim),
      scope: mem.scope,
      category: dim === "category" ? "" : mem.cat,
      src: dim === "project" ? "" : mem.src,
    };
  }

  function describeFacet(dim, value) {
    if (dim === "person") {
      const id = value.slice("slack-user:".length);
      return { label: userLabel(id), search: (userLabel(id) + " " + id).toLowerCase(), glyph: avatarHtml(id, userLabel(id)), mono: !userNames[id] };
    }
    if (dim === "channel") {
      const id = value.slice("slack-channel:".length);
      return { label: channelLabel(id), search: (channelLabel(id) + " " + id).toLowerCase(), glyph: "", mono: !channelNames[id] };
    }
    if (dim === "project") return { label: projectLabel(value), search: value.toLowerCase(), glyph: "", mono: true };
    if (dim === "tag") return { label: value, search: value.toLowerCase(), glyph: "", mono: true };
    return { label: value, search: value.toLowerCase(), glyph: "", mono: false };
  }

  function isSelected(dim, value) {
    if (dim === "project") return mem.src === value;
    if (dim === "category") return mem.cat === value;
    return mem.tags.includes(value);
  }

  async function openPopover(dim, btn) {
    if (popDim === dim && !popover.hidden) { closePopover(); return; }
    popDim = dim;
    popItems = [];
    popover.style.left = btn.offsetLeft + "px";
    popover.style.top = (btn.offsetTop + btn.offsetHeight + 6) + "px";
    popover.hidden = false;
    popoverSearch.value = "";
    popoverSearch.placeholder = `Filter ${dim === "person" ? "people" : dim === "category" ? "categories" : dim + "s"}...`;
    popoverList.innerHTML = '<div class="text-xs text-dim px-2 py-3">Loading...</div>';
    popoverSearch.focus();

    const cfg = DIMS[dim];
    await loadNames();
    let facets = await fetchFacets(cfg.field, cfg.prefix, cfg.limit, narrowFor(dim));
    if (popDim !== dim) return;
    if (dim === "tag") facets = facets.filter((f) => !String(f.value).startsWith("slack-"));
    popItems = facets.map((f) => ({ value: String(f.value), count: Number(f.count) || 0, ...describeFacet(dim, String(f.value)) }));
    // Keep active values visible even when the narrowed facets omit them
    const active = dim === "project" ? (mem.src ? [mem.src] : [])
      : dim === "category" ? (mem.cat ? [mem.cat] : [])
      : mem.tags.filter((t) => tagDim(t) === dim);
    active.forEach((v) => {
      if (!popItems.some((p) => p.value === v)) popItems.unshift({ value: v, count: null, ...describeFacet(dim, v) });
    });
    renderPopoverList();
  }

  function renderPopoverList() {
    const q = popoverSearch.value.trim().toLowerCase();
    const items = q ? popItems.filter((p) => p.search.includes(q)) : popItems;
    if (!items.length) {
      const empty = popItems.length ? "No matches." : (popDim === "channel" ? "No channel data yet." : "No values.");
      popoverList.innerHTML = `<div class="text-xs text-dim px-2 py-3">${empty}</div>`;
      return;
    }
    popoverList.innerHTML = items.map((p) => `<button class="popover-item ${isSelected(popDim, p.value) ? "selected" : ""}" data-value="${escapeHtml(p.value)}">
        ${p.glyph}<span class="label ${p.mono ? "font-mono text-[12px]" : ""}">${escapeHtml(p.label)}</span>
        <span class="count">${p.count == null ? "" : p.count.toLocaleString()}</span>
      </button>`).join("");
    popoverList.querySelectorAll(".popover-item").forEach((b) => {
      b.addEventListener("click", () => pickFacet(b.dataset.value));
    });
  }

  function pickFacet(value) {
    const dim = popDim;
    closePopover();
    setFilters(() => {
      if (dim === "project") mem.src = mem.src === value ? "" : value;
      else if (dim === "category") mem.cat = mem.cat === value ? "" : value;
      else mem.tags = mem.tags.includes(value) ? mem.tags.filter((t) => t !== value) : mem.tags.concat([value]);
    });
  }

  function closePopover() {
    if (!popover) return;
    if (popover.contains(document.activeElement)) document.activeElement.blur();
    popover.hidden = true;
    popDim = null;
  }

  popoverSearch?.addEventListener("input", renderPopoverList);
  popoverSearch?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      const first = popoverList.querySelector(".popover-item");
      if (first) pickFacet(first.dataset.value);
    }
  });

  document.querySelectorAll(".filter-btn").forEach((btn) => {
    btn.addEventListener("click", (e) => { e.stopPropagation(); openPopover(btn.dataset.dim, btn); });
  });

  document.addEventListener("mousedown", (e) => {
    if (popover.hidden) return;
    if (popover.contains(e.target) || e.target.closest(".filter-btn")) return;
    closePopover();
  });

  // --- Keyboard: Escape closes overlays; j/k and arrows move the memory selection ---
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      if (!popover.hidden) { closePopover(); return; }
      if (mem.editing) { cancelEdit(); return; }
      if (!detailPanel.hidden) detailPanel.hidden = true;
      return;
    }
    if (currentView !== "memories" || mem.editing || !popover.hidden) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
    let delta = 0;
    if (e.key === "ArrowDown" || e.key === "j") delta = 1;
    if (e.key === "ArrowUp" || e.key === "k") delta = -1;
    if (!delta) return;
    e.preventDefault();
    moveSelection(delta);
  });

  // --- Toast ---
  function showToast(message, undoCallback) {
    const container = document.getElementById("toast-container");
    container.innerHTML = "";
    const toast = document.createElement("div");
    toast.className = "card flex items-center gap-3 text-fg-2 text-sm px-4 py-2.5 shadow-2xl";
    toast.innerHTML = `<span>${escapeHtml(message)}</span>`;
    if (undoCallback) {
      const btn = document.createElement("button");
      btn.className = "text-accent-2 hover:text-fg font-medium ml-1 transition-colors";
      btn.textContent = "Undo";
      btn.addEventListener("click", async () => {
        toast.remove();
        try { await undoCallback(); } catch (e) { showToast("Undo failed."); }
      });
      toast.appendChild(btn);
    }
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 5000);
  }

  // --- Logout ---
  document.getElementById("logout-btn").addEventListener("click", () => {
    sessionStorage.removeItem("cems_api_key");
    apiKey = "";
    lastRoute = null;
    namesPromise = null;
    if (simulation) simulation.stop();
    showLogin();
  });

  // --- Init ---
  if (apiKey) {
    showDashboard();
  } else {
    showLogin();
  }

  refreshIcons();
})();
