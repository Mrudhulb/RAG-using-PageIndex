/* =========================================================
   PageIndex RAG Chat — Frontend Logic
   ========================================================= */

// ---------------------------------------------------------------------------
// Session ID (persisted in sessionStorage for the tab's lifetime)
// ---------------------------------------------------------------------------

let SESSION_ID = sessionStorage.getItem("pageindex_session_id");
if (!SESSION_ID) {
  SESSION_ID = crypto.randomUUID();
  sessionStorage.setItem("pageindex_session_id", SESSION_ID);
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let activeDocHandle = null;   // currently selected doc handle
let isSending = false;        // prevent double-sends

// ---------------------------------------------------------------------------
// DOM helpers
// ---------------------------------------------------------------------------

const $ = (id) => document.getElementById(id);

function scrollToBottom() {
  const msgs = $("messages");
  msgs.scrollTop = msgs.scrollHeight;
}

function setTyping(visible) {
  $("typingIndicator").style.display = visible ? "flex" : "none";
}

function setSendDisabled(disabled) {
  $("sendBtn").disabled = disabled;
  isSending = disabled;
}

function setUploadStatus(msg, type = "") {
  const el = $("uploadStatus");
  el.textContent = msg;
  el.className = "upload-status" + (type ? " " + type : "");
}

// ---------------------------------------------------------------------------
// Active document banner
// ---------------------------------------------------------------------------

function updateDocBanner(docName) {
  const bar = $("activeDocBar");
  if (docName) {
    $("activeDocName").textContent = "Active: " + docName;
    bar.style.display = "flex";
  } else {
    bar.style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// Markdown renderer (uses marked.js loaded from CDN)
// ---------------------------------------------------------------------------

function renderMarkdown(text) {
  if (typeof marked === "undefined") return escapeHtml(text);
  // Configure marked: no pedantic, no mangle, sanitize
  marked.setOptions({ breaks: true, gfm: true });
  return marked.parse(text);
}

// ---------------------------------------------------------------------------
// Render a message bubble
// ---------------------------------------------------------------------------

function renderMessage(role, content, route, sources) {
  const msgs = $("messages");

  // Remove welcome message on first real message
  const welcome = msgs.querySelector(".welcome-msg");
  if (welcome) welcome.remove();

  const row = document.createElement("div");
  row.className = "message-row " + role;

  // Avatar
  const avatar = document.createElement("div");
  avatar.className = "avatar " + (role === "user" ? "user-avatar" : "assistant-avatar");
  avatar.textContent = role === "user" ? "U" : "AI";

  // Bubble wrapper
  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "assistant") {
    bubble.classList.add("markdown-body");
    bubble.innerHTML = renderMarkdown(content);
  } else {
    // User messages: plain text, just preserve newlines
    bubble.textContent = content;
  }
  wrap.appendChild(bubble);

  if (role === "assistant") {
    // Route badge
    if (route) {
      const badge = document.createElement("span");
      badge.className = "route-badge " + route;
      badge.textContent = route === "document" ? "Document" : "General";
      wrap.appendChild(badge);
    }

    // Sources
    if (sources && sources.length > 0) {
      const details = document.createElement("details");
      details.className = "sources-details";
      const summary = document.createElement("summary");
      summary.textContent = "Source pages (" + sources.length + ")";
      details.appendChild(summary);

      sources.forEach((src) => {
        const page = document.createElement("div");
        page.className = "source-page";
        page.innerHTML =
          '<div class="source-page-header">Page ' + src.page_number +
          ' <span style="font-weight:400;color:#999;">— ' + src.source + ', score: ' +
          src.score + '</span></div>' +
          '<div class="source-page-text">' + escapeHtml(src.text) + '</div>';
        details.appendChild(page);
      });
      wrap.appendChild(details);
    }
  }

  row.appendChild(avatar);
  row.appendChild(wrap);

  msgs.appendChild(row);
  scrollToBottom();
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// ---------------------------------------------------------------------------
// Send a message
// ---------------------------------------------------------------------------

async function sendMessage() {
  if (isSending) return;

  const input = $("queryInput");
  const query = input.value.trim();
  if (!query) return;

  const model = $("modelSelector").value;

  input.value = "";
  input.style.height = "auto";

  renderMessage("user", query);
  setTyping(true);
  setSendDisabled(true);

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model, session_id: SESSION_ID }),
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }

    const data = await resp.json();
    renderMessage("assistant", data.answer, data.route, data.sources);
  } catch (err) {
    renderMessage("assistant", "Error: " + err.message, "general", []);
  } finally {
    setTyping(false);
    setSendDisabled(false);
    input.focus();
  }
}

// Auto-resize textarea
$("queryInput").addEventListener("input", function () {
  this.style.height = "auto";
  this.style.height = Math.min(this.scrollHeight, 140) + "px";
});

function handleKey(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

// ---------------------------------------------------------------------------
// Clear chat history
// ---------------------------------------------------------------------------

async function clearChat() {
  try {
    await fetch("/api/history?session_id=" + SESSION_ID, { method: "DELETE" });
  } catch (_) {}

  const msgs = $("messages");
  msgs.innerHTML =
    '<div class="welcome-msg"><h2>Welcome to PageIndex RAG Chat</h2>' +
    "<p>Upload a PDF in the sidebar and start asking questions, or chat freely without a document.</p></div>";
}

// ---------------------------------------------------------------------------
// Load documents list
// ---------------------------------------------------------------------------

async function loadDocuments() {
  try {
    const resp = await fetch("/api/documents");
    if (!resp.ok) return;
    const data = await resp.json();
    renderDocList(data.documents || []);
  } catch (_) {}
}

function renderDocList(docs) {
  const list = $("docList");
  list.innerHTML = "";

  if (!docs || docs.length === 0) {
    list.innerHTML = '<li class="doc-empty">No documents loaded.</li>';
    return;
  }

  docs.forEach((doc) => {
    const li = document.createElement("li");
    li.className = "doc-item" + (doc.doc_handle === activeDocHandle ? " active" : "");
    li.dataset.handle = doc.doc_handle;

    const info = document.createElement("div");
    info.className = "doc-info";
    info.innerHTML =
      '<div class="doc-name" title="' + escapeHtml(doc.filename) + '">' +
      escapeHtml(doc.filename) +
      "</div>" +
      '<div class="doc-meta">' + doc.pages + " pages · " + doc.mode + "</div>";

    const actions = document.createElement("div");
    actions.className = "doc-actions";

    const selectBtn = document.createElement("button");
    selectBtn.className = "btn-select" + (doc.doc_handle === activeDocHandle ? " selected" : "");
    selectBtn.textContent = doc.doc_handle === activeDocHandle ? "Active" : "Select";
    if (doc.doc_handle !== activeDocHandle) {
      selectBtn.onclick = () => selectDoc(doc.doc_handle, doc.filename);
    }

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "btn-delete";
    deleteBtn.textContent = "Del";
    deleteBtn.onclick = () => deleteDoc(doc.doc_handle);

    actions.appendChild(selectBtn);
    actions.appendChild(deleteBtn);

    li.appendChild(info);
    li.appendChild(actions);
    list.appendChild(li);
  });
}

// ---------------------------------------------------------------------------
// Select / Deselect document
// ---------------------------------------------------------------------------

async function selectDoc(handle, filename) {
  try {
    const resp = await fetch("/api/documents/" + handle + "/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
    if (!resp.ok) throw new Error("Failed to select document.");
    activeDocHandle = handle;
    updateDocBanner(filename);
    loadDocuments();
  } catch (err) {
    alert(err.message);
  }
}

async function deselectDoc() {
  try {
    await fetch("/api/documents/deselect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: SESSION_ID }),
    });
    activeDocHandle = null;
    updateDocBanner(null);
    loadDocuments();
  } catch (_) {}
}

// ---------------------------------------------------------------------------
// Upload document
// ---------------------------------------------------------------------------

$("fileInput").addEventListener("change", async function () {
  const file = this.files[0];
  if (!file) return;
  this.value = "";  // reset so same file can be re-selected

  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setUploadStatus("Only PDF files are accepted.", "error");
    return;
  }

  setUploadStatus("Uploading and ingesting… (this may take a moment)", "");
  $("btnUpload").disabled = true;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch("/api/documents", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    setUploadStatus(
      "Uploaded: " + data.filename + " (" + data.pages + " pages, " + data.mode + ")",
      "success"
    );
    loadDocuments();
  } catch (err) {
    setUploadStatus("Upload failed: " + err.message, "error");
  } finally {
    $("btnUpload").disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Delete document
// ---------------------------------------------------------------------------

async function deleteDoc(handle) {
  if (!confirm("Delete this document?")) return;
  try {
    const resp = await fetch("/api/documents/" + handle, { method: "DELETE" });
    if (!resp.ok) throw new Error("Failed to delete document.");
    if (handle === activeDocHandle) {
      activeDocHandle = null;
      updateDocBanner(null);
    }
    setUploadStatus("Document deleted.", "success");
    loadDocuments();
  } catch (err) {
    alert(err.message);
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  loadDocuments();
  $("queryInput").focus();
});
