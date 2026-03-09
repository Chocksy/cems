/* CEMS Memory Dashboard */
(function () {
  "use strict";

  // --- State ---
  let apiKey = sessionStorage.getItem("cems_api_key") || "";
  let offset = 0;
  let limit = 50;
  let total = 0;
  let currentCategory = "";
  let searchQuery = "";
  let categories = {};
  let editingId = null;
  let searchTimeout = null;
  let projectChart = null;
  let currentView = "list";
  let currentScope = "";

  // --- DOM refs ---
  const loginView = document.getElementById("login-view");
  const dashView = document.getElementById("dashboard-view");
  const loginForm = document.getElementById("login-form");
  const apiKeyInput = document.getElementById("api-key-input");
  const loginError = document.getElementById("login-error");
  const totalBadge = document.getElementById("total-badge");
  const searchInput = document.getElementById("search-input");
  const logoutBtn = document.getElementById("logout-btn");
  const filtersEl = document.getElementById("filters");
  const listEl = document.getElementById("memory-list");
  const paginationEl = document.getElementById("pagination");
  const prevBtn = document.getElementById("prev-btn");
  const nextBtn = document.getElementById("next-btn");
  const pageInfo = document.getElementById("page-info");
  const editModal = document.getElementById("edit-modal");
  const editTextarea = document.getElementById("edit-textarea");
  const editSave = document.getElementById("edit-save");
  const editCancel = document.getElementById("edit-cancel");
  const modalClose = editModal.querySelector(".modal-close");
  const chartView = document.getElementById("chart-view");
  const toastContainer = document.getElementById("toast-container");

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
    dashView.hidden = true;
    loginError.hidden = true;
    apiKeyInput.value = "";
  }

  function showDashboard() {
    loginView.hidden = true;
    dashView.hidden = false;
    loadCategories();
    loadMemories();
  }

  // --- Login ---
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    apiKey = apiKeyInput.value.trim();
    if (!apiKey) return;
    try {
      const data = await apiFetch("/api/memory/summary/personal");
      if (data.success) {
        sessionStorage.setItem("cems_api_key", apiKey);
        showDashboard();
      } else {
        throw new Error(data.error || "Login failed");
      }
    } catch (err) {
      loginError.textContent = err.message;
      loginError.hidden = false;
    }
  });

  logoutBtn.addEventListener("click", () => {
    sessionStorage.removeItem("cems_api_key");
    apiKey = "";
    showLogin();
  });

  // --- View Toggle ---
  document.querySelectorAll(".toggle-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentView = btn.dataset.view;

      const isList = currentView === "list";
      listEl.hidden = !isList;
      filtersEl.hidden = !isList;
      paginationEl.hidden = !isList || total <= limit;
      searchInput.hidden = !isList;
      chartView.hidden = isList;

      if (!isList) loadProjectChart();
    });
  });

  // --- Scope Toggle ---
  document.querySelectorAll(".scope-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".scope-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentScope = btn.dataset.scope;
      offset = 0;
      loadMemories();
      loadCategories();
    });
  });

  // --- Categories ---
  async function loadCategories() {
    try {
      const data = await apiFetch("/api/memory/summary/personal");
      if (!data.success) return;
      categories = data.categories || {};
      renderFilters();
      totalBadge.textContent = data.total || 0;
    } catch (e) {
      console.error("Failed to load categories", e);
    }
  }

  function renderFilters() {
    const allTotal = Object.values(categories).reduce((s, n) => s + n, 0);
    let html = `<button class="filter-pill ${currentCategory === "" ? "active" : ""}" data-category="">All (${allTotal})</button>`;
    const sorted = Object.entries(categories).sort((a, b) => b[1] - a[1]);
    for (const [cat, count] of sorted) {
      const active = currentCategory === cat ? "active" : "";
      html += `<button class="filter-pill ${active}" data-category="${esc(cat)}">${esc(cat)} (${count})</button>`;
    }
    filtersEl.innerHTML = html;

    filtersEl.querySelectorAll(".filter-pill").forEach((btn) => {
      btn.addEventListener("click", () => {
        currentCategory = btn.dataset.category;
        offset = 0;
        searchQuery = "";
        searchInput.value = "";
        loadMemories();
        renderFilters();
      });
    });
  }

  // --- Memory list ---
  async function loadMemories() {
    listEl.innerHTML = '<div class="loading">Loading...</div>';

    try {
      let params = `limit=${limit}&offset=${offset}`;
      if (currentScope) params += `&scope=${encodeURIComponent(currentScope)}`;
      if (currentCategory) params += `&category=${encodeURIComponent(currentCategory)}`;
      if (searchQuery) params += `&q=${encodeURIComponent(searchQuery)}`;

      const data = await apiFetch(`/api/memory/list?${params}`);
      if (!data.success) {
        listEl.innerHTML = `<div class="empty">Error: ${esc(data.error || "Unknown")}</div>`;
        return;
      }

      total = data.total || 0;
      renderMemories(data.results || []);
      renderPagination(data.mode === "search");
    } catch (e) {
      listEl.innerHTML = `<div class="empty">Failed to load memories.</div>`;
    }
  }

  function renderMemories(memories) {
    if (!memories.length) {
      listEl.innerHTML = '<div class="empty">No memories found.</div>';
      return;
    }

    listEl.innerHTML = memories
      .map((m) => {
        const tags = (m.tags || []).map((t) => `<span class="tag">#${esc(t)}</span>`).join(" ");
        const content = esc(m.content || "");
        const isShort = content.length < 300;
        const date = m.created_at ? new Date(m.created_at).toLocaleDateString() : "";
        const shown = m.shown_count ? `shown: ${m.shown_count}` : "";

        return `<div class="memory-card" data-id="${esc(m.id)}">
          <div class="memory-meta">
            <span class="cat">${esc(m.category || "general")}</span>
            ${tags}
            ${m.scope ? `<span class="scope-badge scope-${esc(m.scope)}">${esc(m.scope)}</span>` : ""}
            ${m.source_ref ? `<span class="source-ref">${esc(m.source_ref)}</span>` : ""}
            ${date ? `<span>${date}</span>` : ""}
            ${shown ? `<span>${shown}</span>` : ""}
            ${m.score != null ? `<span>score: ${Number(m.score).toFixed(3)}</span>` : ""}
          </div>
          <div class="memory-content ${isShort ? "short" : ""}">${content}</div>
          <div class="memory-actions">
            <button class="btn-expand" data-id="${esc(m.id)}">Expand</button>
            <button class="btn-edit" data-id="${esc(m.id)}">Edit</button>
            ${m.scope === "personal" ? `<button class="btn-promote" data-id="${esc(m.id)}">Promote to Team</button>` : ""}
            <button class="btn-delete" data-id="${esc(m.id)}">Delete</button>
          </div>
        </div>`;
      })
      .join("");

    // Expand
    listEl.querySelectorAll(".btn-expand").forEach((btn) => {
      btn.addEventListener("click", () => {
        const card = btn.closest(".memory-card");
        const el = card.querySelector(".memory-content");
        el.classList.toggle("expanded");
        btn.textContent = el.classList.contains("expanded") ? "Collapse" : "Expand";
      });
    });

    // Also expand on content click
    listEl.querySelectorAll(".memory-content").forEach((el) => {
      el.addEventListener("click", () => {
        el.classList.toggle("expanded");
        const btn = el.parentElement.querySelector(".btn-expand");
        if (btn) btn.textContent = el.classList.contains("expanded") ? "Collapse" : "Expand";
      });
    });

    // Edit
    listEl.querySelectorAll(".btn-edit").forEach((btn) => {
      btn.addEventListener("click", () => openEdit(btn.dataset.id));
    });

    // Promote
    listEl.querySelectorAll(".btn-promote").forEach((btn) => {
      btn.addEventListener("click", () => promoteMemory(btn.dataset.id));
    });

    // Delete
    listEl.querySelectorAll(".btn-delete").forEach((btn) => {
      btn.addEventListener("click", () => deleteMemory(btn.dataset.id));
    });
  }

  function renderPagination(isSearch) {
    if (isSearch) {
      paginationEl.hidden = true;
      return;
    }
    paginationEl.hidden = total <= limit;
    prevBtn.disabled = offset === 0;
    nextBtn.disabled = offset + limit >= total;

    const page = Math.floor(offset / limit) + 1;
    const pages = Math.ceil(total / limit);
    pageInfo.textContent = `Page ${page} of ${pages} (${total} total)`;
  }

  prevBtn.addEventListener("click", () => {
    offset = Math.max(0, offset - limit);
    loadMemories();
  });

  nextBtn.addEventListener("click", () => {
    offset += limit;
    loadMemories();
  });

  // --- Search ---
  searchInput.addEventListener("input", () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => {
      searchQuery = searchInput.value.trim();
      offset = 0;
      loadMemories();
    }, 400);
  });

  // --- Edit ---
  async function openEdit(id) {
    try {
      const data = await apiFetch(`/api/memory/get?id=${id}`);
      if (!data.success) return;
      const doc = data.document;
      editingId = id;
      editTextarea.value = doc.content || "";
      document.getElementById("edit-category").value = doc.category || "";
      document.getElementById("edit-tags").value = (doc.tags || []).join(", ");
      document.getElementById("edit-source-ref").value = doc.source_ref || "";
      editModal.hidden = false;
    } catch (e) {
      // Fallback: grab content from the card
      const card = listEl.querySelector(`.memory-card[data-id="${id}"]`);
      if (!card) return;
      editingId = id;
      editTextarea.value = card.querySelector(".memory-content").textContent;
      document.getElementById("edit-category").value = "";
      document.getElementById("edit-tags").value = "";
      document.getElementById("edit-source-ref").value = "";
      editModal.hidden = false;
    }
  }

  function closeEdit() {
    editModal.hidden = true;
    editingId = null;
  }

  editCancel.addEventListener("click", closeEdit);
  modalClose.addEventListener("click", closeEdit);
  editModal.addEventListener("click", (e) => {
    if (e.target === editModal) closeEdit();
  });

  editSave.addEventListener("click", async () => {
    if (!editingId) return;
    const content = editTextarea.value.trim();
    const category = document.getElementById("edit-category").value.trim() || undefined;
    const tagsStr = document.getElementById("edit-tags").value.trim();
    const tags = tagsStr ? tagsStr.split(",").map((t) => t.trim()).filter(Boolean) : undefined;
    const sourceRef = document.getElementById("edit-source-ref").value.trim() || undefined;

    const body = { memory_id: editingId };
    if (content) body.content = content;
    if (category !== undefined) body.category = category;
    if (tags !== undefined) body.tags = tags;
    if (sourceRef !== undefined) body.source_ref = sourceRef;

    editSave.disabled = true;
    editSave.textContent = "Saving...";
    try {
      const data = await apiFetch("/api/memory/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (data.success) {
        closeEdit();
        loadMemories();
        loadCategories();
      } else {
        alert("Update failed: " + (data.error || "Unknown error"));
      }
    } catch (e) {
      alert("Update failed: " + e.message);
    } finally {
      editSave.disabled = false;
      editSave.textContent = "Save";
    }
  });

  // --- Promote to Team ---
  async function promoteMemory(id) {
    try {
      const data = await apiFetch("/api/memory/promote", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memory_id: id }),
      });
      if (data.success) {
        showToast("Memory promoted to team.");
        loadMemories();
        loadCategories();
      } else {
        showToast("Promote failed: " + (data.error || "Unknown error"));
      }
    } catch (e) {
      showToast("Promote failed: " + e.message);
    }
  }

  // --- Delete with Toast + Undo ---
  async function deleteMemory(id) {
    try {
      const data = await apiFetch("/api/memory/forget", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memory_id: id }),
      });
      if (data.success) {
        loadMemories();
        loadCategories();
        showToast("Memory deleted.", async () => {
          await apiFetch("/api/memory/restore", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ memory_id: id }),
          });
          loadMemories();
          loadCategories();
        });
      } else {
        alert("Delete failed: " + (data.error || "Unknown error"));
      }
    } catch (e) {
      alert("Delete failed: " + e.message);
    }
  }

  // --- Toast ---
  function showToast(message, undoCallback) {
    // Remove any existing toast
    toastContainer.innerHTML = "";

    const toast = document.createElement("div");
    toast.className = "toast";
    toast.innerHTML = `<span>${esc(message)}</span>`;

    if (undoCallback) {
      const undoBtn = document.createElement("button");
      undoBtn.className = "undo-btn";
      undoBtn.textContent = "Undo";
      undoBtn.addEventListener("click", async () => {
        toast.remove();
        try {
          await undoCallback();
        } catch (e) {
          alert("Undo failed: " + e.message);
        }
      });
      toast.appendChild(undoBtn);
    }

    toastContainer.appendChild(toast);

    // Auto-dismiss after 5 seconds
    setTimeout(() => toast.remove(), 5000);
  }

  // --- Project Chart ---
  async function loadProjectChart() {
    try {
      const data = await apiFetch("/api/memory/stats/projects");
      if (!data.success) return;

      const projects = data.projects || [];
      const labels = projects.map((p) =>
        p.project.startsWith("project:") ? p.project.replace("project:", "") : p.project
      );
      const values = projects.map((p) => p.count);

      const colors = [
        "#3b82f6", "#22c55e", "#a855f7", "#f97316", "#eab308",
        "#ef4444", "#06b6d4", "#ec4899", "#84cc16", "#6366f1",
        "#14b8a6", "#f43f5e", "#8b5cf6", "#f59e0b", "#10b981",
      ];

      const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
      const textColor = isDark ? "#e5e5e5" : "#111";
      const gridColor = isDark ? "#333" : "#ddd";
      const subtextColor = isDark ? "#999" : "#555";

      const ctx = document.getElementById("project-chart").getContext("2d");

      if (projectChart) projectChart.destroy();

      // Dynamic height based on number of projects
      const chartEl = document.getElementById("project-chart");
      chartEl.style.height = Math.max(300, projects.length * 32) + "px";

      projectChart = new Chart(ctx, {
        type: "bar",
        data: {
          labels: labels,
          datasets: [{
            label: "Memories",
            data: values,
            backgroundColor: labels.map((_, i) => colors[i % colors.length]),
            borderRadius: 4,
          }],
        },
        options: {
          indexAxis: "y",
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { display: false },
            title: { display: true, text: "Memories per Project", color: textColor, font: { size: 14 } },
          },
          scales: {
            x: { ticks: { color: subtextColor }, grid: { color: gridColor } },
            y: { ticks: { color: textColor, font: { family: "monospace", size: 11 } }, grid: { display: false } },
          },
        },
      });
    } catch (e) {
      console.error("Failed to load project chart", e);
    }
  }

  // --- Utils ---
  function esc(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }

  // --- Init ---
  if (apiKey) {
    showDashboard();
  } else {
    showLogin();
  }
})();
