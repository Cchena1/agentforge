const state = {
  history: [],
  textFiles: [],
  llmHistory: [
    {
      role: "system",
      content: "你是一个本地AI agent助手，回答要简洁，优先给出可执行步骤。",
    },
  ],
};

const el = (id) => document.getElementById(id);

function messageTemplate(role, content, time) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.innerHTML = `
    <div class="message-meta">${role === "user" ? "用户" : "agent"} · ${time || ""}</div>
    <div class="message-body"></div>
  `;
  node.querySelector(".message-body").textContent = content;
  return node;
}

function renderHistory(history) {
  const log = el("chatLog");
  log.innerHTML = "";
  for (const item of history) {
    if (!item || !item.role) continue;
    log.appendChild(messageTemplate(item.role, item.content || "", item.time || ""));
  }
  log.scrollTop = log.scrollHeight;
}

function renderFiles(files) {
  const list = el("fileList");
  list.innerHTML = "";
  if (!files.length) {
    list.textContent = "没有找到文本文件。";
    return;
  }
  for (const file of files) {
    const btn = document.createElement("button");
    btn.className = "file-item";
    btn.textContent = file;
    btn.addEventListener("click", () => {
      el("readPath").value = file;
      readFile();
    });
    list.appendChild(btn);
  }
}

function renderLlmHistory(history) {
  const log = el("llmLog");
  log.innerHTML = "";
  for (const item of history) {
    if (!item || !item.role || item.role === "system") continue;
    log.appendChild(messageTemplate(item.role === "assistant" ? "assistant" : "user", item.content || "", item.time || ""));
  }
  if (!log.children.length) {
    log.textContent = "在这里输入一条消息，然后点击发送到模型。";
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `${response.status} ${response.statusText}`);
  }
  return payload;
}

async function refreshState() {
  const payload = await requestJson("/api/state");
  state.history = payload.history || [];
  state.textFiles = payload.textFiles || [];
  renderHistory(state.history);
  renderFiles(state.textFiles);
  el("workspacePath").textContent = payload.workspaceRoot || "";
  el("serverStatus").textContent = "在线";
}

async function refreshLlmConfig() {
  const payload = await requestJson("/api/llm/config");
  const defaults = payload.defaults || {};
  if (!el("llmBaseUrl").value) el("llmBaseUrl").value = defaults.baseUrl || "";
  if (!el("llmModel").value) el("llmModel").value = defaults.model || "";
  if (!el("llmTemperature").value) el("llmTemperature").value = "0.2";
  const apiKeyState = payload.apiKeyConfigured ? "后端已配置 API Key" : "未预置 API Key，可在页面中填写";
  el("llmStatus").textContent = `${payload.notes || "模型接口已就绪。"} ${apiKeyState}`;
}

async function sendChat(event) {
  event.preventDefault();
  const input = el("chatInput");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  const payload = await requestJson("/api/chat", {
    method: "POST",
    body: JSON.stringify({ message }),
  });
  state.history = payload.history || [];
  renderHistory(state.history);
  if (payload.memory?.last_summary_path) {
    el("sampleTaskResult").textContent = payload.memory.last_summary_path;
  }
  if (payload.action?.path) {
    el("filePreview").textContent = `已生成文件：${payload.action.path}`;
  }
}

async function runSampleTask() {
  const payload = await requestJson("/api/sample-summary", { method: "POST", body: "{}" });
  const summary = payload.summary;
  el("sampleTaskResult").textContent = `已写入：${summary.path}`;
  el("filePreview").textContent = summary.text;
  await refreshState();
}

async function readFile() {
  const path = el("readPath").value.trim();
  if (!path) return;
  const payload = await requestJson(`/api/file?path=${encodeURIComponent(path)}`);
  el("filePreview").textContent = payload.file.content;
}

async function writeFile() {
  const path = el("writePath").value.trim();
  const content = el("writeContent").value;
  if (!path) return;
  const payload = await requestJson("/api/file", {
    method: "POST",
    body: JSON.stringify({ path, content }),
  });
  el("filePreview").textContent = `已写入：${payload.file.path}`;
  await refreshState();
}

function llmConfigFromForm() {
  const maxTokensRaw = el("llmMaxTokens").value.trim();
  return {
    baseUrl: el("llmBaseUrl").value.trim(),
    model: el("llmModel").value.trim(),
    apiKey: el("llmApiKey").value.trim(),
    temperature: Number(el("llmTemperature").value || "0.2"),
    maxTokens: maxTokensRaw ? Number(maxTokensRaw) : null,
  };
}

function appendLlmMessage(role, content) {
  state.llmHistory.push({
    role,
    content,
    time: new Date().toLocaleString(),
  });
  renderLlmHistory(state.llmHistory);
}

async function sendLlmMessage() {
  const text = el("llmInput").value.trim();
  if (!text) return;
  el("llmInput").value = "";
  appendLlmMessage("user", text);
  const payload = await requestJson("/api/llm/chat", {
    method: "POST",
    body: JSON.stringify({
      ...llmConfigFromForm(),
      messages: state.llmHistory.map(({ role, content }) => ({ role, content })),
    }),
  });
  appendLlmMessage("assistant", payload.reply || "");
  el("llmStatus").textContent = `模型：${payload.model || "unknown"}${payload.usage ? `｜用量：${JSON.stringify(payload.usage)}` : ""}`;
}

function resetLlmConversation() {
  state.llmHistory = [
    {
      role: "system",
      content: "你是一个本地AI agent助手，回答要简洁，优先给出可执行步骤。",
    },
  ];
  renderLlmHistory(state.llmHistory);
  el("llmStatus").textContent = "模型对话已清空。";
}

async function resetChat() {
  await requestJson("/api/reset", { method: "POST", body: "{}" });
  await refreshState();
  el("sampleTaskResult").textContent = "";
  el("filePreview").textContent = "";
}

function bind() {
  el("chatForm").addEventListener("submit", (event) => {
    sendChat(event).catch((error) => {
      el("serverStatus").textContent = `错误：${error.message}`;
    });
  });
  el("sampleTaskBtn").addEventListener("click", () => {
    runSampleTask().catch((error) => {
      el("sampleTaskResult").textContent = error.message;
    });
  });
  el("readBtn").addEventListener("click", () => {
    readFile().catch((error) => {
      el("filePreview").textContent = error.message;
    });
  });
  el("writeBtn").addEventListener("click", () => {
    writeFile().catch((error) => {
      el("filePreview").textContent = error.message;
    });
  });
  el("llmSendBtn").addEventListener("click", () => {
    sendLlmMessage().catch((error) => {
      el("llmStatus").textContent = error.message;
    });
  });
  el("llmResetBtn").addEventListener("click", () => {
    resetLlmConversation();
  });
  el("refreshStateBtn").addEventListener("click", () => {
    refreshState().catch((error) => {
      el("serverStatus").textContent = error.message;
    });
  });
  el("reloadFilesBtn").addEventListener("click", () => {
    refreshState().catch((error) => {
      el("serverStatus").textContent = error.message;
    });
  });
  el("resetBtn").addEventListener("click", () => {
    resetChat().catch((error) => {
      el("serverStatus").textContent = error.message;
    });
  });
}

async function init() {
  bind();
  try {
    await refreshState();
    await refreshLlmConfig();
    renderLlmHistory(state.llmHistory);
  } catch (error) {
    el("serverStatus").textContent = `离线：${error.message}`;
    renderHistory([{ role: "assistant", content: "服务器尚未启动，或浏览器暂时无法访问本地接口。", time: new Date().toLocaleString() }]);
  }
}

init();
