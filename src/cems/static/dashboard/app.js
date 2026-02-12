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
            ${m.scope ? `<span>${esc(m.scope)}</span>` : ""}
            ${m.source_ref ? `<span class="source-ref">${esc(m.source_ref)}</span>` : ""}
            ${date ? `<span>${date}</span>` : ""}
            ${shown ? `<span>${shown}</span>` : ""}
            ${m.score != null ? `<span>score: ${Number(m.score).toFixed(3)}</span>` : ""}
          </div>
          <div class="memory-content ${isShort ? "short" : ""}">${content}</div>
          <div class="memory-actions">
            <button class="btn-expand" data-id="${esc(m.id)}">Expand</button>
            <button class="btn-edit" data-id="${esc(m.id)}">Edit</button>
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
  function openEdit(id) {
    const card = listEl.querySelector(`.memory-card[data-id="${id}"]`);
    if (!card) return;
    const content = card.querySelector(".memory-content").textContent;
    editingId = id;
    editTextarea.value = content;
    editModal.hidden = false;
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
    if (!content) return;

    editSave.disabled = true;
    editSave.textContent = "Saving...";
    try {
      const data = await apiFetch("/api/memory/update", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memory_id: editingId, content }),
      });
      if (data.success) {
        closeEdit();
        loadMemories();
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

  // --- Delete ---
  async function deleteMemory(id) {
    if (!confirm("Delete this memory? (soft-delete, can be recovered from DB)")) return;

    try {
      const data = await apiFetch("/api/memory/forget", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ memory_id: id }),
      });
      if (data.success) {
        loadMemories();
        loadCategories();
      } else {
        alert("Delete failed: " + (data.error || "Unknown error"));
      }
    } catch (e) {
      alert("Delete failed: " + e.message);
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
