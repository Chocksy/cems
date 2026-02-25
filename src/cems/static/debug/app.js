/* CEMS Debug Dashboard */
(function () {
  "use strict";

  const views = {
    sessions: document.getElementById("view-sessions"),
    detail: document.getElementById("view-detail"),
    status: document.getElementById("view-status"),
  };

  // --- Navigation ---
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const view = btn.dataset.view;
      if (view === "sessions") navigate("#/");
      else if (view === "status") navigate("#/status");
    });
  });

  function setActiveNav(name) {
    document.querySelectorAll(".nav-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.view === name);
    });
  }

  function showView(name) {
    Object.entries(views).forEach(([k, el]) => {
      el.hidden = k !== name;
    });
  }

  // --- Hash-based Router ---
  function navigate(hash) {
    if (window.location.hash !== hash) {
      window.location.hash = hash;
    } else {
      // Hash didn't change, still need to handle it
      handleRoute();
    }
  }

  function handleRoute() {
    const hash = window.location.hash || "#/";

    // #/session/:sid  or  #/session/:sid/:offset
    const sessionMatch = hash.match(/^#\/session\/([^/]+?)(?:\/(\d+))?$/);
    if (sessionMatch) {
      const sid = decodeURIComponent(sessionMatch[1]);
      const offset = sessionMatch[2] ? parseInt(sessionMatch[2]) : 0;
      showDetail(sid, offset, /* pushHash */ false);
      return;
    }

    if (hash === "#/status") {
      showStatus();
      return;
    }

    // Default: sessions list
    showSessions();
  }

  window.addEventListener("hashchange", handleRoute);

  // --- API ---
  async function api(path) {
    try {
      const res = await fetch(path);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    } catch (err) {
      console.error("API error:", path, err);
      return { error: err.message };
    }
  }

  let _lastSessionsHash = "";

  // --- Sessions List ---
  async function showSessions() {
    setActiveNav("sessions");
    showView("sessions");
    if (!views.sessions.innerHTML || views.sessions.innerHTML.includes("view-")) {
      views.sessions.innerHTML = '<div class="loading">Loading sessions...</div>';
    }

    const sessions = await api("/api/sessions?limit=50");

    if (sessions.error) {
      views.sessions.innerHTML = `<div class="empty">Error: ${esc(sessions.error)}</div>`;
      return;
    }

    if (!sessions.length) {
      views.sessions.innerHTML = '<div class="empty">No hook events found.</div>';
      return;
    }

    // Skip re-render if data hasn't changed (prevents scroll jank on auto-refresh)
    const hash = JSON.stringify(sessions.map(s => s.session_id + s.last_ts + s.event_count));
    if (hash === _lastSessionsHash) return;
    _lastSessionsHash = hash;

    // Stats bar
    const totalEvents = sessions.reduce((s, x) => s + x.event_count, 0);
    const totalRetrievals = sessions.reduce((s, x) => s + x.retrieval_count, 0);
    const totalPrompts = sessions.reduce((s, x) => s + x.prompt_count, 0);

    let html = `
      <div class="stats-bar">
        <div class="stat">Sessions: <span class="stat-value">${sessions.length}</span></div>
        <div class="stat">Events: <span class="stat-value">${totalEvents.toLocaleString()}</span></div>
        <div class="stat">Prompts: <span class="stat-value">${totalPrompts}</span></div>
        <div class="stat">Retrievals: <span class="stat-value">${totalRetrievals}</span></div>
      </div>
      <table class="sessions-table">
        <thead><tr>
          <th>Session</th>
          <th>Project</th>
          <th>Started</th>
          <th>Last Active</th>
          <th>Prompts</th>
          <th>Retrievals</th>
          <th>Tools</th>
          <th>Gates</th>
        </tr></thead>
        <tbody>`;

    for (const s of sessions) {
      const scoreBar = s.retrieval_count > 0 ? `<span class="badge badge-purple">${s.retrieval_count}</span>` : "-";
      const gatesBadge = s.gate_triggers > 0 ? `<span class="badge badge-orange">${s.gate_triggers}</span>` : "-";

      html += `<tr data-sid="${esc(s.session_id)}">
        <td><span class="sid">${esc(s.session_id)}</span></td>
        <td><span class="project-name">${esc(s.project || "-")}</span></td>
        <td><span class="ts">${formatTs(s.first_ts)}</span></td>
        <td><span class="ts">${formatTs(s.last_ts)}</span></td>
        <td>${s.prompt_count || "-"}</td>
        <td>${scoreBar}</td>
        <td>${s.tool_count || "-"}</td>
        <td>${gatesBadge}</td>
      </tr>`;
    }

    html += "</tbody></table>";
    views.sessions.innerHTML = html;

    // Click handler for rows
    views.sessions.querySelectorAll("tr[data-sid]").forEach((row) => {
      row.addEventListener("click", () => navigate(`#/session/${encodeURIComponent(row.dataset.sid)}`));
    });
  }

  // --- Session Detail ---
  const PAGE_SIZE = 200;
  let _detailRefreshTimer = null;
  let _currentDetailSid = null;

  async function showDetail(sid, offset = 0, pushHash = true) {
    showView("detail");
    setActiveNav("");

    // Update URL hash (unless called from hashchange handler)
    if (pushHash) {
      const hash = offset > 0
        ? `#/session/${encodeURIComponent(sid)}/${offset}`
        : `#/session/${encodeURIComponent(sid)}`;
      if (window.location.hash !== hash) {
        window.location.hash = hash;
        return; // hashchange will call us back
      }
    }

    views.detail.innerHTML = '<div class="loading">Loading session...</div>';

    const [session, verbose] = await Promise.all([
      api(`/api/sessions/${sid}?offset=${offset}&limit=${PAGE_SIZE}`),
      api(`/api/sessions/${sid}/verbose`),
    ]);

    if (session.error) {
      views.detail.innerHTML = `<div class="empty">${esc(session.error)}</div>`;
      return;
    }

    const totalEvents = session.total_events || session.events.length;
    const currentOffset = session.offset || 0;

    // Build a map of verbose entries by ts+event+tool for context enrichment
    const verboseMap = {};
    for (const v of verbose) {
      const key = `${v.ts}|${v.event}|${v.tool_name || ""}`;
      verboseMap[key] = v;
    }

    let html = `
      <a class="back-link" href="#/">&#8592; Back to sessions</a>
      <div class="session-header">
        <div>
          <div class="session-title">${esc(sid)}</div>
          <div class="session-meta">
            <span>Project: <strong>${esc(session.project || "unknown")}</strong></span>
            <span>Source: ${esc(session.source || "-")}</span>
            <span>${formatTs(session.first_ts)} &mdash; ${formatTs(session.last_ts)}</span>
          </div>
        </div>
        <div class="session-meta">
          <span class="badge badge-green">${session.prompt_count} prompts</span>
          <span class="badge badge-purple">${session.retrieval_count} retrievals</span>
          <span class="badge badge-blue">${session.tool_count} tools</span>
          ${session.gate_triggers ? `<span class="badge badge-orange">${session.gate_triggers} gate triggers</span>` : ""}
        </div>
      </div>
      <div class="timeline">`;

    for (const evt of session.events) {
      const evtType = evt.event;
      const verboseKey = `${evt.ts}|${evtType}|${evt.tool || ""}`;
      const vData = verboseMap[verboseKey];

      const autoExpand = evtType === "MemoryRetrieval" || evtType === "SessionStartOutput" || evtType === "GateTriggered" || evtType === "UserPromptSubmitOutput";
      html += `<div class="event-card${autoExpand ? " expanded" : ""}" data-event="${esc(evtType)}">
        <div class="event-header">
          <div class="event-dot ${esc(evtType)}"></div>
          <span class="event-type">${esc(evtType)}</span>
          <span class="event-ts">${formatTime(evt.ts)}</span>`;

      if (evt.tool) {
        html += `<span class="event-tool">${esc(evt.tool)}</span>`;
      }

      // Show extra info inline
      if (evtType === "MemoryRetrieval") {
        const q = evt.extra.query || "";
        const top = evt.extra.top_score || 0;
        const scoreClass = top >= 0.7 ? "score-high" : top >= 0.5 ? "score-mid" : "score-low";
        html += `<span class="badge badge-purple">${evt.extra.result_count} results</span>`;
        html += `<span class="score ${scoreClass}">top: ${top.toFixed(2)}</span>`;
      }

      if (evtType === "UserPromptSubmit" && evt.extra.prompt_len) {
        html += `<span class="badge badge-gray">${evt.extra.prompt_len} chars</span>`;
      }

      if (evtType === "SessionStartOutput") {
        html += `<span class="badge badge-blue">${(evt.extra.context_len || 0).toLocaleString()} chars injected</span>`;
        if (evt.extra.has_profile) html += `<span class="badge badge-green">Profile</span>`;
        if (evt.extra.has_foundation) html += `<span class="badge badge-purple">Foundation</span>`;
      }

      if (evtType === "UserPromptSubmitOutput") {
        html += `<span class="badge badge-green">${(evt.extra.output_len || 0).toLocaleString()} chars</span>`;
        if (evt.extra.has_memories) html += `<span class="badge badge-purple">Memories</span>`;
        if (evt.extra.has_ultrathink) html += `<span class="badge badge-blue">Ultrathink</span>`;
      }

      if (evtType === "GateTriggered") {
        const action = evt.extra.gate_action || "unknown";
        const badge = action === "block" ? "badge-red" : "badge-orange";
        html += `<span class="badge ${badge}">${esc(action)}</span>`;
      }

      html += `<span class="event-expand">&#9654;</span>
        </div>
        <div class="event-body">`;

      // Event-specific detail body
      if (evtType === "MemoryRetrieval") {
        html += renderRetrieval(evt.extra);
      } else if (evtType === "SessionStart" && vData) {
        html += renderVerboseDetail(vData);
      } else if (evtType === "UserPromptSubmit" && vData) {
        html += renderVerboseDetail(vData);
      } else if ((evtType === "PreToolUse" || evtType === "PostToolUse") && vData) {
        html += renderToolDetail(vData);
      } else if (evtType === "SessionStartOutput") {
        html += renderSessionStartOutput(evt.extra);
      } else if (evtType === "UserPromptSubmitOutput") {
        html += renderHookOutput(evt.extra, vData);
      } else if (evtType === "GateTriggered") {
        html += renderGateTriggered(evt.extra);
      } else if (evtType === "Stop") {
        html += renderStopDetail(evt.extra);
      } else {
        html += `<pre class="context-block">${esc(JSON.stringify(evt.extra, null, 2))}</pre>`;
      }

      html += `</div></div>`;
    }

    html += "</div>";

    // Pagination controls
    if (totalEvents > PAGE_SIZE) {
      html += `<div class="pagination">`;
      html += `<span class="pagination-info">Showing ${currentOffset + 1}–${Math.min(currentOffset + PAGE_SIZE, totalEvents)} of ${totalEvents.toLocaleString()} events</span>`;
      if (currentOffset > 0) {
        const prevOffset = Math.max(0, currentOffset - PAGE_SIZE);
        html += `<a class="nav-btn page-btn" href="#/session/${encodeURIComponent(sid)}${prevOffset > 0 ? "/" + prevOffset : ""}">&#8592; Prev</a>`;
      }
      if (currentOffset + PAGE_SIZE < totalEvents) {
        const nextOffset = currentOffset + PAGE_SIZE;
        html += `<a class="nav-btn page-btn" href="#/session/${encodeURIComponent(sid)}/${nextOffset}">Next &#8594;</a>`;
      }
      html += `</div>`;
    }

    views.detail.innerHTML = html;

    // Toggle expand on click
    views.detail.querySelectorAll(".event-header").forEach((hdr) => {
      hdr.addEventListener("click", () => {
        hdr.parentElement.classList.toggle("expanded");
      });
    });

    // Auto-refresh for active sessions (every 10s)
    if (_detailRefreshTimer) clearInterval(_detailRefreshTimer);
    _currentDetailSid = sid;
    _detailRefreshTimer = setInterval(() => {
      if (!views.detail.hidden && _currentDetailSid === sid) {
        refreshDetail(sid, currentOffset);
      } else {
        clearInterval(_detailRefreshTimer);
      }
    }, 10000);
  }

  async function refreshDetail(sid, offset) {
    // Lightweight refresh: only update if event count changed
    const session = await api(`/api/sessions/${sid}?offset=${offset}&limit=${PAGE_SIZE}`);
    if (session.error || !session.events) return;

    const totalEvents = session.total_events || session.events.length;
    const existingEvents = views.detail.querySelectorAll(".event-card").length;

    // If new events arrived, re-render
    if (session.events.length !== existingEvents) {
      showDetail(sid, offset, false);
    }
  }

  // Expose for back link (legacy, now using href)
  window.__showSessions = () => navigate("#/");

  function renderRetrieval(extra) {
    const details = extra.details || [];
    let html = `<div style="margin-bottom:.5rem">
      <strong>Query:</strong> <span style="color:var(--fg2)">"${esc(extra.query || "")}"</span><br>
      <strong>Results:</strong> ${extra.result_count || 0} &nbsp;
      <strong>Avg:</strong> ${(extra.avg_score || 0).toFixed(3)} &nbsp;
      <strong>Top:</strong> ${(extra.top_score || 0).toFixed(3)}
    </div>`;

    if (details.length) {
      html += `<table class="retrieval-table">
        <thead><tr><th>#</th><th>Score</th><th>Category</th><th>Content</th><th>ID</th></tr></thead>
        <tbody>`;
      details.forEach((d, i) => {
        const sc = d.score || 0;
        const cls = sc >= 0.7 ? "score-high" : sc >= 0.5 ? "score-mid" : "score-low";
        const content = d.content || "";
        html += `<tr>
          <td>${i + 1}</td>
          <td><span class="score ${cls}">${sc.toFixed(3)}</span></td>
          <td>${esc(d.category || "-")}</td>
          <td class="memory-content">${esc(content) || '<span style="color:var(--fg3)">—</span>'}</td>
          <td style="color:var(--fg3)">${esc(d.id || "-")}</td>
        </tr>`;
      });
      html += "</tbody></table>";
    }
    return html;
  }

  function renderVerboseDetail(vData) {
    let html = "";
    if (vData.prompt) {
      html += `<div style="margin-bottom:.5rem"><strong>Prompt:</strong></div>
        <pre class="context-block">${esc(vData.prompt)}</pre>`;
    }
    if (vData.cwd) {
      html += `<div style="margin-top:.5rem"><strong>CWD:</strong> <span style="font-family:var(--mono);color:var(--fg2)">${esc(vData.cwd)}</span></div>`;
    }
    // Show any other interesting fields
    const skip = new Set(["ts", "event", "prompt", "cwd", "session_id", "hook_event_name",
      "custom_instructions", "transcript_path", "conversation_id", "generation_id"]);
    const extras = Object.entries(vData).filter(([k]) => !skip.has(k));
    if (extras.length) {
      html += `<details style="margin-top:.5rem"><summary style="cursor:pointer;color:var(--fg2);font-size:.78rem">Raw payload</summary>
        <pre class="context-block">${esc(JSON.stringify(Object.fromEntries(extras), null, 2))}</pre>
      </details>`;
    }
    return html;
  }

  function renderToolDetail(vData) {
    let html = "";
    if (vData.tool_name) {
      html += `<div><strong>Tool:</strong> ${esc(vData.tool_name)}</div>`;
    }
    if (vData.tool_input) {
      const input = typeof vData.tool_input === "string" ? vData.tool_input : JSON.stringify(vData.tool_input, null, 2);
      html += `<div style="margin-top:.4rem"><strong>Input:</strong></div>
        <pre class="context-block" style="max-height:200px">${esc(input)}</pre>`;
    }
    if (vData.tool_response) {
      const output = typeof vData.tool_response === "string" ? vData.tool_response : JSON.stringify(vData.tool_response, null, 2);
      html += `<div style="margin-top:.4rem"><strong>Output:</strong></div>
        <pre class="context-block" style="max-height:200px">${esc(output)}</pre>`;
    }
    return html;
  }

  function renderSessionStartOutput(extra) {
    let html = "";
    const preview = extra.context_preview || "";
    if (preview) {
      html += `<div style="margin-bottom:.5rem">
        <strong>Injected Context</strong>
        <span class="badge badge-blue">${(extra.context_len || 0).toLocaleString()} chars</span>
        ${extra.has_profile ? '<span class="badge badge-green">Profile</span>' : ""}
        ${extra.has_foundation ? '<span class="badge badge-purple">Foundation</span>' : ""}
      </div>
      <pre class="context-block">${esc(preview)}</pre>`;
    } else {
      html += `<div style="color:var(--fg3)">No context injected.</div>`;
    }
    return html;
  }

  function renderHookOutput(extra, vData) {
    let html = `<div style="margin-bottom:.5rem">
      <strong>Hook Output</strong>
      <span class="badge badge-green">${(extra.output_len || 0).toLocaleString()} chars</span>
      ${extra.has_memories ? '<span class="badge badge-purple">Memories</span>' : ""}
      ${extra.has_ultrathink ? '<span class="badge badge-blue">Ultrathink</span>' : ""}
    </div>`;
    if (vData && vData.output) {
      html += `<pre class="context-block">${esc(vData.output)}</pre>`;
    } else {
      html += `<div style="color:var(--fg3)">Output text not in verbose log.</div>`;
    }
    return html;
  }

  function renderGateTriggered(extra) {
    const action = extra.gate_action || "unknown";
    const badge = action === "block" ? "badge-red" : "badge-orange";
    return `<div>
      <span class="badge ${badge}">${esc(action.toUpperCase())}</span>
      <div style="margin-top:.4rem"><strong>Reason:</strong> ${esc(extra.reason || "")}</div>
      <div><strong>Pattern:</strong> <code>${esc(extra.pattern || "")}</code></div>
      <div><strong>Tool:</strong> ${esc(extra.tool || "")}</div>
    </div>`;
  }

  function renderStopDetail(extra) {
    let html = "";
    if (extra.cwd) {
      html += `<div><strong>CWD:</strong> <span style="font-family:var(--mono);color:var(--fg2)">${esc(extra.cwd)}</span></div>`;
    }
    return html || "<div>Session ended.</div>";
  }

  // --- Status ---
  async function showStatus() {
    setActiveNav("status");
    showView("status");
    views.status.innerHTML = '<div class="loading">Loading status...</div>';

    const data = await api("/api/status");

    let html = `
      <div class="status-section">
        <h3>Hook Events</h3>
        <div class="status-grid">
          <span class="status-key">Events file</span>
          <span class="status-val">${esc(data.events_file || "")}</span>
          <span class="status-key">File size</span>
          <span class="status-val">${esc(data.events_file_size || "0")}</span>
          <span class="status-key">Total events</span>
          <span class="status-val">${(data.events_count || 0).toLocaleString()}</span>
          <span class="status-key">Sessions tracked</span>
          <span class="status-val">${data.session_count || 0}</span>
          <span class="status-key">Memory retrievals</span>
          <span class="status-val">${data.retrieval_count || 0}</span>
          <span class="status-key">Verbose log files</span>
          <span class="status-val">${data.verbose_files || 0}</span>
        </div>
      </div>

      <div class="status-section">
        <h3>Observer Daemon</h3>
        <div class="status-grid">
          <span class="status-key">Running</span>
          <span class="status-val">${data.daemon_running ? '<span style="color:var(--green)">Yes</span>' : '<span style="color:var(--red)">No</span>'}</span>
          ${data.daemon_pid ? `<span class="status-key">PID</span><span class="status-val">${data.daemon_pid}</span>` : ""}
          <span class="status-key">Observer sessions</span>
          <span class="status-val">${data.observer_sessions || 0}</span>
        </div>
      </div>`;

    // Gate rules
    const gates = data.gate_rules || [];
    if (gates.length) {
      html += `<div class="status-section"><h3>Gate Rules Cache</h3><div class="status-grid">`;
      for (const g of gates) {
        html += `<span class="status-key">${esc(g.project)}</span>
          <span class="status-val">${g.rule_count} rule${g.rule_count !== 1 ? "s" : ""}</span>`;
      }
      html += `</div></div>`;
    }

    // Daemon log tail
    if (data.daemon_log_tail && data.daemon_log_tail.length) {
      html += `<div class="status-section">
        <h3>Observer Log (last 10 lines)</h3>
        <div class="log-tail">${esc(data.daemon_log_tail.join("\n"))}</div>
      </div>`;
    }

    views.status.innerHTML = html;
  }

  // --- Helpers ---
  function esc(str) {
    if (str == null) return "";
    const d = document.createElement("div");
    d.textContent = String(str);
    return d.innerHTML;
  }

  function formatTs(ts) {
    if (!ts) return "-";
    // Parse "2026-02-22T17:26:45+0200" -> readable
    try {
      const d = new Date(ts.replace(/(\d{2})(\d{2})$/, "$1:$2")); // Fix timezone format
      return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
    } catch {
      return ts;
    }
  }

  function formatTime(ts) {
    if (!ts) return "";
    try {
      const d = new Date(ts.replace(/(\d{2})(\d{2})$/, "$1:$2"));
      return d.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    } catch {
      return ts;
    }
  }

  // --- Auto-refresh (sessions list only, every 10s) ---
  setInterval(() => {
    if (!views.sessions.hidden) showSessions();
  }, 10000);

  // --- Init: route based on current hash ---
  handleRoute();
})();
