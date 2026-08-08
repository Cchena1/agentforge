const BACKEND_URL = "http://localhost:8000";

const state = {
  messages: [],
  sessionId: null,
};

const $ = (id) => document.getElementById(id);

function escapeText(value) {
  return value == null ? "" : String(value);
}

function renderMessages() {
  const chatWindow = $("chatWindow");
  chatWindow.innerHTML = "";

  for (const message of state.messages) {
    const bubble = document.createElement("div");
    bubble.className = `message ${message.role}`;

    const meta = document.createElement("div");
    meta.className = "message-meta";
    meta.textContent = message.role === "user" ? "用户" : "Agent";

    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = message.content || "";

    bubble.appendChild(meta);
    bubble.appendChild(body);

    if (message.role === "assistant" && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
      const tools = document.createElement("div");
      tools.className = "tool-calls";

      const header = document.createElement("div");
      header.textContent = "工具调用：";
      tools.appendChild(header);

      for (const call of message.tool_calls) {
        const item = document.createElement("div");
        item.className = "tool-item";
        const argsText = call.arguments ? ` ${escapeText(JSON.stringify(call.arguments))}` : "";
        const resultText = call.result ? ` -> ${escapeText(call.result)}` : "";
        const statusText = call.status ? ` [${call.status}]` : "";
        item.textContent = `${call.tool_name || call.name || "tool"}${statusText}${argsText}${resultText}`;
        tools.appendChild(item);
      }

      bubble.appendChild(tools);
    }

    chatWindow.appendChild(bubble);
  }

  chatWindow.scrollTop = chatWindow.scrollHeight;
  $("sessionState").textContent = `${state.messages.length} 条消息`;
}

function addMessage(role, content, tool_calls = []) {
  state.messages.push({ role, content, tool_calls });
  renderMessages();
}

function setToolLog(toolCalls) {
  const log = $("toolLog");
  if (!toolCalls || toolCalls.length === 0) {
    log.textContent = "暂无工具调用。";
    $("toolState").textContent = "无";
    return;
  }

  log.textContent = toolCalls
    .map((item) => {
      const args = item.arguments ? JSON.stringify(item.arguments) : "{}";
      const result = item.result ? `\n结果：${item.result}` : "";
      return `${item.tool_name || item.name}(${args}) [${item.status || "unknown"}]${result}`;
    })
    .join("\n\n");

  $("toolState").textContent = toolCalls[toolCalls.length - 1]?.tool_name || toolCalls[toolCalls.length - 1]?.name || "有";
}

async function fetchBackend(path, options = {}) {
  const response = await fetch(`${BACKEND_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || response.statusText);
  }
  return payload;
}

async function sendMessage(message) {
  const history = state.messages.map(({ role, content }) => ({ role, content }));
  const payload = await fetchBackend("/chat", {
    method: "POST",
    body: JSON.stringify({ message, history, session_id: state.sessionId }),
  });

  state.sessionId = payload.session_id || state.sessionId;
  addMessage("assistant", payload.reply || "", payload.tool_calls || []);
  setToolLog(payload.tool_calls || []);
}

async function refreshBackendStatus() {
  try {
    const payload = await fetchBackend("/health");
    $("statusText").textContent = payload.status === "ok" ? "Online" : "Unknown";
  } catch (error) {
    $("statusText").textContent = "Offline";
  }
}

function clearChat() {
  state.messages = [];
  state.sessionId = null;
  renderMessages();
  setToolLog([]);
}

function bindSamples() {
  document.querySelectorAll("[data-sample]").forEach((button) => {
    button.addEventListener("click", () => {
      $("chatInput").value = button.dataset.sample || "";
      $("chatInput").focus();
    });
  });
}

function bindForm() {
  const form = $("chatForm");
  const input = $("chatInput");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    input.value = "";
    addMessage("user", message);

    try {
      await sendMessage(message);
    } catch (error) {
      addMessage("assistant", `请求失败：${error.message}`);
      setToolLog([]);
    }
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      form.requestSubmit();
    }
  });
}

function bindControls() {
  $("clearBtn").addEventListener("click", clearChat);
}

async function init() {
  bindForm();
  bindControls();
  bindSamples();
  renderMessages();
  await refreshBackendStatus();
  setInterval(refreshBackendStatus, 5000);
}

init();
