"use strict";

const state = {
  data: null,
  selectedNodeId: null,
  traceFilter: "all",
  eventSource: null,
  eventChain: Promise.resolve(),
  streamTerminal: false,
  llmSettings: null,
  llmModelsRequest: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function compact(value, limit = 120) {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  if (!text) return "No content";
  return text.length > limit ? `${text.slice(0, limit)}…` : text;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function scoreText(value) {
  return typeof value === "number" ? value.toFixed(2) : "—";
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) return;
    const payload = await response.json();
    $("#inferenceLabel").textContent = payload.inference;
  } catch (_) {
    $("#inferenceLabel").textContent = "Runtime Offline";
  }
}

async function loadLLMSettings() {
  const response = await fetch("/api/llm/settings", { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "无法读取 LLM 配置");
  state.llmSettings = payload;
  renderLLMSettings();
  return payload;
}

function renderLLMSettings() {
  const settings = state.llmSettings;
  if (!settings) return;
  const select = $("#llmTargetSelect");
  const previous = select.value;
  select.innerHTML = settings.targets.map((target) => `
    <option value="${escapeHtml(target.name)}">
      ${escapeHtml(target.name)} · ${escapeHtml(target.model)}
    </option>
  `).join("");
  const preferred = settings.activeTarget || previous || settings.defaultTarget;
  if (settings.targets.some((target) => target.name === preferred)) select.value = preferred;
  updateLLMTargetSummary();
}

function selectedLLMTarget() {
  return state.llmSettings?.targets.find((target) => target.name === $("#llmTargetSelect").value) || null;
}

function updateLLMTargetSummary() {
  const target = selectedLLMTarget();
  if (!target) return;
  const credentialLabel = target.requiresApiKey
    ? (target.credentialConfigured ? `凭据已配置 · ${target.credentialSource}` : "尚未配置 API Key")
    : (target.credentialSource === "oauth_cli" ? "使用宿主 CLI OAuth 会话" : "无需 API Key");
  $("#llmTargetSummary").innerHTML = `
    <strong>${escapeHtml(target.service)} · ${escapeHtml(target.model)}</strong><br>
    Structured output: ${escapeHtml(target.structuredOutput)} · ${escapeHtml(credentialLabel)}
  `;
  const keyField = $("#llmApiKeyField");
  const keyInput = $("#llmApiKeyInput");
  keyInput.disabled = !target.requiresApiKey;
  keyField.classList.toggle("is-disabled", !target.requiresApiKey);
  keyInput.placeholder = target.credentialConfigured
    ? "留空则保留 llm.local.toml 中现有密钥"
    : "输入 Provider API Key";
}

async function loadLLMModels({ includeApiKey = false } = {}) {
  const target = selectedLLMTarget();
  if (!target) return;
  const requestId = ++state.llmModelsRequest;
  const select = $("#llmModelSelect");
  const refresh = $("#llmModelsRefresh");
  const save = $("#llmSettingsSave");
  const apiKey = $("#llmApiKeyInput").value.trim();
  select.disabled = true;
  save.disabled = true;
  select.innerHTML = "<option>正在探测 Provider…</option>";
  refresh.disabled = true;
  setLLMSettingsStatus(`正在获取 ${target.service} 实时模型列表…`, "running");
  try {
    const response = await fetch("/api/llm/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target: target.name,
        ...(includeApiKey && apiKey ? { apiKey } : {}),
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const message = response.status === 404
        ? "模型目录接口未加载，请重启 Research Web 服务"
        : (payload.error || "模型列表获取失败");
      throw new Error(message);
    }
    if (requestId !== state.llmModelsRequest) return;
    const models = Array.isArray(payload.models) ? payload.models : [];
    const visibleModels = models.length ? models : [payload.currentModel || target.model];
    select.innerHTML = visibleModels.map((model) => `
      <option value="${escapeHtml(model)}">${escapeHtml(model)}</option>
    `).join("");
    if (visibleModels.includes(payload.currentModel)) select.value = payload.currentModel;
    select.disabled = false;
    save.disabled = false;
    if (models.length) {
      const auth = payload.authMethod ? ` · ${payload.authMethod}` : "";
      if (payload.availability === "available") {
        setLLMSettingsStatus(`已实时获取 ${models.length} 个模型${auth}`, "success");
      } else {
        const diagnostics = payload.diagnostics?.join(" · ") || payload.availability;
        setLLMSettingsStatus(
          `已获取 ${models.length} 个模型，但连接状态为 ${payload.availability}：${diagnostics}`,
          "error",
        );
      }
    } else {
      const diagnostics = payload.diagnostics?.join(" · ") || "Provider 未提供模型目录";
      setLLMSettingsStatus(`未获取到实时列表：${diagnostics}`, "error");
    }
  } catch (error) {
    if (requestId !== state.llmModelsRequest) return;
    select.innerHTML = "<option>模型列表获取失败</option>";
    select.disabled = true;
    save.disabled = true;
    setLLMSettingsStatus(error.message || "模型列表获取失败", "error");
  } finally {
    if (requestId === state.llmModelsRequest) refresh.disabled = false;
  }
}

function setLLMSettingsStatus(message, kind = "") {
  const status = $("#llmSettingsStatus");
  status.textContent = message;
  status.className = `settings-status${kind ? ` is-${kind}` : ""}`;
}

async function openLLMSettings() {
  const dialog = $("#llmSettingsDialog");
  dialog.showModal();
  setLLMSettingsStatus("正在读取本地 LLM 配置…", "running");
  try {
    await loadLLMSettings();
    await loadLLMModels();
  } catch (error) {
    setLLMSettingsStatus(error.message || "无法读取 LLM 配置", "error");
  }
}

$("#llmSettingsButton").addEventListener("click", openLLMSettings);
$("#llmSettingsClose").addEventListener("click", () => $("#llmSettingsDialog").close());
$("#llmSettingsCancel").addEventListener("click", () => $("#llmSettingsDialog").close());
$("#llmTargetSelect").addEventListener("change", async () => {
  updateLLMTargetSummary();
  await loadLLMModels();
});
$("#llmModelsRefresh").addEventListener("click", async () => {
  await loadLLMModels({ includeApiKey: true });
});

$("#llmSettingsForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const target = selectedLLMTarget();
  if (!target) return;
  const saveButton = $("#llmSettingsSave");
  const apiKeyInput = $("#llmApiKeyInput");
  const apiKey = apiKeyInput.value.trim();
  const model = $("#llmModelSelect").value;
  if (!model) return;
  saveButton.disabled = true;
  setLLMSettingsStatus("正在保存本地配置…", "running");
  try {
    const settingsResponse = await fetch("/api/llm/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target: target.name,
        model,
        ...(apiKey ? { apiKey } : {}),
      }),
    });
    const settingsPayload = await settingsResponse.json();
    if (!settingsResponse.ok) throw new Error(settingsPayload.error || "LLM 配置保存失败");
    apiKeyInput.value = "";
    state.llmSettings = settingsPayload;
    renderLLMSettings();
    setLLMSettingsStatus("配置已保存，正在执行真实连接验证…", "running");

    const probeResponse = await fetch("/api/llm/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    const probe = await probeResponse.json();
    if (!probeResponse.ok) throw new Error(probe.error || "LLM 连接验证失败");
    if (probe.availability !== "available") {
      const diagnostics = probe.diagnostics?.join(" · ") || "未返回诊断信息";
      throw new Error(`连接不可用：${diagnostics}`);
    }
    $("#inferenceLabel").textContent = probe.inference;
    const authentication = probe.authMethod
      || (target.credentialSource === "oauth_cli" ? "CLI OAuth 会话" : "无需认证");
    setLLMSettingsStatus(
      `模型已切换并连接成功 · ${probe.model} · ${authentication}`,
      "success",
    );
    await loadLLMSettings();
  } catch (error) {
    setLLMSettingsStatus(error.message || "LLM 配置失败", "error");
  } finally {
    saveButton.disabled = false;
  }
});

$("#researchForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const task = $("#taskInput").value.trim();
  const mode = $("#researchMode").value;
  if (!task) return;
  setRunning(true);
  resetLiveRun();
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, mode }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Research run failed");
    connectProgressStream(payload);
  } catch (error) {
    failLiveRun(error.message || "Research run failed");
  }
});

$("#researchMode").addEventListener("change", () => {
  const hints = {
    auto: "自动模式会在 LLM 可用时优先执行模型研究流程。",
    llm_research: "Information Retrieval 与报告生成使用所选模型；结果不是实时行情或联网检索。",
    fixture_demo: "使用演示数据稳定触发 Context & Memory、Evaluation 与 Governance。",
  };
  $("#researchModeHint").textContent = hints[$("#researchMode").value];
});

function resetLiveRun() {
  if (state.eventSource) state.eventSource.close();
  state.eventSource = null;
  state.eventChain = Promise.resolve();
  state.streamTerminal = false;
  state.data = null;
  state.selectedNodeId = null;
  ["metricNodes", "metricEvents", "metricTools", "metricContext", "metricScore"].forEach((id) => {
    $(`#${id}`).textContent = "—";
  });
}

function connectProgressStream(session) {
  const source = new EventSource(session.eventsUrl);
  state.eventSource = source;
  source.onmessage = (message) => {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch (_) {
      failLiveRun("Runtime progress stream returned invalid data");
      return;
    }
    state.eventChain = state.eventChain.then(() => handleProgressEvent(event, session));
  };
  source.onerror = () => {
    if (!state.streamTerminal) {
      setMessage("实时连接正在恢复，Runtime 执行不会中断…", "running");
    }
  };
}

async function handleProgressEvent(event, session) {
  const { kind, payload = {} } = event;
  if (kind === "run.preparing") {
    $("#runId").textContent = `RUN ${String(event.runId).slice(0, 8)} · PREPARING`;
    setMessage(`正在准备 ${payload.company || "研究对象"} 的任务图…`, "running");
    return;
  }
  if (kind === "graph.initialized") {
    initializeLiveGraph(event);
    return;
  }
  if (!state.data) return;
  if (kind === "trace.recorded") {
    state.data.runtimeTrace.push(payload);
    $("#metricEvents").textContent = state.data.runtimeTrace.length;
    renderTimeline();
    return;
  }
  if (kind === "node.started") {
    const node = normalizeLiveNode(payload.node);
    node.status = "running";
    upsertLiveNode(node);
    state.selectedNodeId = node.id;
    renderLiveGraph();
    setMessage(`正在执行：${node.goal}`, "running");
    await presentationBeat();
    return;
  }
  if (kind === "node.completed" || kind === "node.failed") {
    const node = state.data.graph.nodes.find((item) => item.id === payload.nodeId);
    if (node) {
      node.status = kind === "node.completed" ? "completed" : "failed";
      node.output = payload.output;
      node.error = payload.error;
      renderLiveGraph();
      await presentationBeat();
    }
    return;
  }
  if (kind === "graph.node_added") {
    const node = normalizeLiveNode(payload.node);
    node.dynamic = true;
    upsertLiveNode(node);
    renderLiveGraph();
    setMessage(`任务图动态新增：${node.goal}`, "running");
    return;
  }
  if (kind === "graph.dependency_added") {
    const node = state.data.graph.nodes.find((item) => item.id === payload.nodeId);
    if (node && !node.dependencies.includes(payload.dependencyId)) {
      node.dependencies.push(payload.dependencyId);
      state.data.graph.version += 1;
      renderLiveGraph();
    }
    return;
  }
  if (kind === "runtime.completed") {
    $("#metricStatus").textContent = "FINALIZING";
    $("#metricStatus").className = "running";
    setMessage("Task Graph 已完成，正在生成 Evaluation 与最终快照…", "running");
    return;
  }
  if (kind === "runtime.failed") {
    const running = state.data.graph.nodes.find((node) => node.status === "running");
    if (running) {
      running.status = "failed";
      running.error = payload.error || "Runtime failed during node execution";
      renderLiveGraph();
    }
    $("#metricStatus").textContent = "FAILED";
    $("#metricStatus").className = "failed";
    return;
  }
  if (kind === "run.result_ready") {
    state.streamTerminal = true;
    state.eventSource?.close();
    await loadCompletedResult(session.resultUrl);
    return;
  }
  if (kind === "run.failed") {
    state.streamTerminal = true;
    state.eventSource?.close();
    failLiveRun(payload.error || "Research run failed");
  }
}

function initializeLiveGraph(event) {
  const rawGraph = event.payload.graph;
  const nodes = rawGraph.nodes.map(normalizeLiveNode);
  state.data = {
    meta: { runId: event.runId, status: "running", graphVersion: rawGraph.version },
    summary: {},
    graph: { id: rawGraph.graph_id, version: rawGraph.version, nodes },
    runtimeTrace: [],
    tools: [],
  };
  state.selectedNodeId = nodes[0]?.id ?? null;
  $("#runId").textContent = `RUN ${String(event.runId).slice(0, 8)} · GRAPH V${rawGraph.version}`;
  renderLiveGraph();
}

function normalizeLiveNode(node) {
  return {
    id: node.node_id || node.id,
    shortId: String(node.node_id || node.id).slice(0, 8),
    goal: node.goal,
    expectedOutput: node.expected_output || node.expectedOutput,
    strategy: node.strategy_id || node.strategy,
    status: node.status || "pending",
    dependencies: [...(node.dependencies || [])],
    dynamic: Boolean(node.dynamic),
    output: node.observation?.output ?? node.output ?? null,
    error: node.failure_reason ?? node.error ?? null,
  };
}

function upsertLiveNode(node) {
  const index = state.data.graph.nodes.findIndex((item) => item.id === node.id);
  if (index < 0) {
    state.data.graph.nodes.push(node);
  } else {
    const current = state.data.graph.nodes[index];
    state.data.graph.nodes[index] = { ...current, ...node, dynamic: current.dynamic || node.dynamic };
  }
}

function renderLiveGraph() {
  renderGraph();
  renderInspector();
  const nodes = state.data.graph.nodes;
  const completed = nodes.filter((node) => node.status === "completed").length;
  $("#metricNodes").textContent = `${completed}/${nodes.length}`;
  $("#metricStatus").textContent = nodes.some((node) => node.status === "failed") ? "FAILED" : "RUNNING";
  $("#metricStatus").className = nodes.some((node) => node.status === "failed") ? "failed" : "running";
}

function presentationBeat() {
  return new Promise((resolve) => window.setTimeout(resolve, 90));
}

async function loadCompletedResult(resultUrl) {
  try {
    const response = await fetch(resultUrl);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Result is unavailable");
    state.data = payload;
    state.selectedNodeId = payload.graph.nodes[0]?.id ?? null;
    renderAll();
    setMessage(`运行完成 · ${payload.meta.elapsedSeconds}s · ${payload.meta.inference}`, "success");
    setRunning(false);
  } catch (error) {
    failLiveRun(error.message || "Failed to load the completed result");
  }
}

function failLiveRun(message) {
  state.streamTerminal = true;
  state.eventSource?.close();
  setMessage(message, "error");
  $("#metricStatus").textContent = "FAILED";
  $("#metricStatus").className = "failed";
  setRunning(false);
}

function setRunning(running) {
  const button = $("#runButton");
  button.disabled = running;
  $("#researchMode").disabled = running;
  button.querySelector(".button-label").textContent = running ? "Runtime 执行中" : "开始研究";
  if (running) {
    setMessage("正在规划任务图、执行能力并组装研究结果…", "running");
    $("#metricStatus").textContent = "RUNNING";
    $("#metricStatus").className = "running";
  }
}

function setMessage(message, kind) {
  const element = $("#runMessage");
  element.textContent = message;
  element.className = `run-message is-${kind}`;
}

function renderAll() {
  renderMeta();
  renderGraph();
  renderTimeline();
  renderInspector();
  renderReport();
  renderContext();
  renderEvaluation();
}

function renderMeta() {
  const { meta, summary, evaluation } = state.data;
  $("#inferenceLabel").textContent = meta.inference;
  $("#runId").textContent = `RUN ${meta.runId.slice(0, 8)} · GRAPH V${meta.graphVersion}`;
  $("#metricNodes").textContent = `${summary.completedNodes}/${summary.nodes}`;
  $("#metricEvents").textContent = summary.runtimeEvents;
  $("#metricTools").textContent = summary.toolTraceEntries;
  $("#metricContext").textContent = summary.contextAssemblies;
  $("#metricScore").textContent = scoreText(evaluation.trajectory.score);
  $("#metricStatus").textContent = meta.status.toUpperCase();
  $("#metricStatus").className = meta.status;
}

function nodeDepths(nodes) {
  const byId = new Map(nodes.map((node) => [node.id, node]));
  const cache = new Map();
  function depth(node) {
    if (cache.has(node.id)) return cache.get(node.id);
    const value = node.dependencies.length
      ? 1 + Math.max(...node.dependencies.map((id) => depth(byId.get(id))))
      : 0;
    cache.set(node.id, value);
    return value;
  }
  nodes.forEach(depth);
  return cache;
}

function renderGraph() {
  const stage = $("#graphStage");
  const empty = $("#graphEmpty");
  empty.hidden = true;
  stage.querySelectorAll(".task-node").forEach((node) => node.remove());
  const nodes = state.data.graph.nodes;
  const depths = nodeDepths(nodes);
  const layers = new Map();
  nodes.forEach((node) => {
    const depth = depths.get(node.id);
    if (!layers.has(depth)) layers.set(depth, []);
    layers.get(depth).push(node);
  });
  const width = Math.max(stage.clientWidth, 720);
  const maxDepth = Math.max(...depths.values());
  const verticalStep = Math.min(112, (stage.clientHeight - 94) / Math.max(maxDepth, 1));

  [...layers.entries()].forEach(([depth, layer]) => {
    layer.forEach((node, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = [
        "task-node",
        "is-visible",
        node.dynamic ? "is-dynamic" : "",
        `is-${node.status}`,
        node.id === state.selectedNodeId ? "is-selected" : "",
      ].filter(Boolean).join(" ");
      button.dataset.nodeId = node.id;
      const x = ((index + 1) / (layer.length + 1)) * width - 87;
      const y = 24 + depth * verticalStep;
      button.style.left = `${Math.max(12, Math.min(width - 186, x))}px`;
      button.style.top = `${y}px`;
      button.innerHTML = `
        <span class="node-top">
          <span class="node-id">${escapeHtml(node.shortId)}</span>
          <span class="node-state">${node.dynamic ? "+ DYNAMIC · " : ""}${escapeHtml(node.status)}</span>
        </span>
        <strong>${escapeHtml(node.goal)}</strong>
        <small>${escapeHtml(node.strategy)} strategy</small>
      `;
      button.addEventListener("click", () => selectNode(node.id));
      stage.appendChild(button);
    });
  });
  requestAnimationFrame(drawGraphEdges);
}

function drawGraphEdges() {
  if (!state.data) return;
  const stage = $("#graphStage");
  const svg = $("#graphEdges");
  svg.replaceChildren();
  const stageBox = stage.getBoundingClientRect();
  const byId = new Map(state.data.graph.nodes.map((node) => [node.id, node]));
  state.data.graph.nodes.forEach((node) => {
    const target = stage.querySelector(`[data-node-id="${CSS.escape(node.id)}"]`);
    if (!target) return;
    const targetBox = target.getBoundingClientRect();
    node.dependencies.forEach((dependencyId) => {
      const source = stage.querySelector(`[data-node-id="${CSS.escape(dependencyId)}"]`);
      if (!source) return;
      const sourceBox = source.getBoundingClientRect();
      const x1 = sourceBox.left - stageBox.left + sourceBox.width / 2;
      const y1 = sourceBox.bottom - stageBox.top;
      const x2 = targetBox.left - stageBox.left + targetBox.width / 2;
      const y2 = targetBox.top - stageBox.top;
      const bend = Math.max(24, (y2 - y1) / 2);
      const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      path.setAttribute("d", `M ${x1} ${y1} C ${x1} ${y1 + bend}, ${x2} ${y2 - bend}, ${x2} ${y2}`);
      const dynamic = node.dynamic || byId.get(dependencyId)?.dynamic;
      path.setAttribute("class", `graph-edge${dynamic ? " dynamic" : ""}`);
      svg.appendChild(path);
    });
  });
}

function selectNode(nodeId) {
  state.selectedNodeId = nodeId;
  $$(".task-node").forEach((element) => {
    element.classList.toggle("is-selected", element.dataset.nodeId === nodeId);
  });
  renderInspector();
}

function renderInspector() {
  if (!state.data || !state.selectedNodeId) return;
  const node = state.data.graph.nodes.find((item) => item.id === state.selectedNodeId);
  if (!node) return;
  const dependencies = node.dependencies.map((id) => {
    const dependency = state.data.graph.nodes.find((item) => item.id === id);
    return dependency ? dependency.shortId : id.slice(0, 8);
  });
  const tools = state.data.tools.filter((item) => item.nodeId === node.id);
  $("#nodeStatus").textContent = node.status.toUpperCase();
  $("#nodeStatus").className = `status-pill ${node.status}`;
  $("#nodeInspector").className = "inspector-content";
  $("#nodeInspector").innerHTML = `
    <h3>${escapeHtml(node.goal)}</h3>
    <div class="detail-row"><label>Strategy</label><p>${escapeHtml(node.strategy)}</p></div>
    <div class="detail-row"><label>Expected output</label><p>${escapeHtml(node.expectedOutput)}</p></div>
    <div class="detail-row"><label>Dependencies</label><div class="chip-row">${
      dependencies.length ? dependencies.map((id) => `<span class="chip">${escapeHtml(id)}</span>`).join("") : '<span class="chip">ROOT NODE</span>'
    }</div></div>
    <div class="detail-row"><label>Tool capability</label><p>${
      tools.length ? tools.map((tool) => `${escapeHtml(tool.capability)} · ${escapeHtml(tool.provider)} · ${escapeHtml(tool.status)}`).join("<br>") : "No direct Tool observation"
    }</p></div>
    <div class="detail-row"><label>Observation output</label><pre>${escapeHtml(pretty(node.output))}</pre></div>
  `;
}

function renderTimeline() {
  const timeline = $("#timeline");
  const entries = state.data.runtimeTrace.filter((entry) => {
    return state.traceFilter === "all" || entry.kind.startsWith(state.traceFilter);
  });
  timeline.innerHTML = entries.map((entry) => `
    <button class="trace-item" data-kind="${escapeHtml(entry.kind)}" data-node="${escapeHtml(entry.nodeId || "")}" type="button">
      <strong>${String(entry.sequence).padStart(2, "0")} · ${escapeHtml(entry.kind)}</strong>
      <small>${escapeHtml(entry.source)}</small>
    </button>
  `).join("");
  timeline.querySelectorAll(".trace-item").forEach((item) => {
    item.addEventListener("click", () => {
      if (item.dataset.node) selectNode(item.dataset.node);
    });
  });
}

$("#traceFilters").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  state.traceFilter = button.dataset.filter;
  $$("#traceFilters button").forEach((item) => item.classList.toggle("is-active", item === button));
  if (state.data) renderTimeline();
});

function renderReport() {
  const report = state.data.report;
  const sections = report.sections.map((section) => `
    <section class="report-section">
      <h2>${escapeHtml(section.title)}</h2>
      <ul>${section.findings.map((finding) => `<li>${escapeHtml(finding)}</li>`).join("")}</ul>
    </section>
  `).join("");
  $("#reportContent").innerHTML = `
    <span class="report-kicker">STRUCTURED RESEARCH · ${escapeHtml(report.company.toUpperCase())}</span>
    <h1>${escapeHtml(report.company)} Investment Research</h1>
    <p class="report-summary">${escapeHtml(report.executive_summary)}</p>
    ${sections}
    <section class="report-section"><h2>Risk Review</h2><ul>${report.risk_factors.map((risk) => `<li>${escapeHtml(risk)}</li>`).join("")}</ul></section>
    <section class="report-section"><h2>Investment View</h2><p class="investment-view">${escapeHtml(report.investment_view)}</p></section>
  `;
  const evidence = state.data.graph.nodes.filter((node) => node.strategy !== "report");
  $("#evidenceList").innerHTML = evidence.map((node) => `
    <button class="evidence-card" type="button" data-evidence-node="${escapeHtml(node.id)}">
      <strong>${escapeHtml(node.shortId)} · ${escapeHtml(node.strategy)}</strong>
      <span>${escapeHtml(node.goal)}</span>
    </button>
  `).join("");
  $("#evidenceList").querySelectorAll("[data-evidence-node]").forEach((button) => {
    button.addEventListener("click", () => {
      activateView("overview");
      selectNode(button.dataset.evidenceNode);
    });
  });
  $("#fixtureNotice").textContent = state.data.meta.fixtureNotice;
}

function renderContext() {
  const units = state.data.context.units;
  const totalTokens = units.reduce((sum, unit) => sum + unit.tokens, 0);
  $("#contextTokenSummary").textContent = `${totalTokens} EST. TOKENS · ${state.data.context.assemblies.length} ASSEMBLIES`;
  ["active", "compressed", "archived"].forEach((status) => {
    const target = $(`#context${status[0].toUpperCase()}${status.slice(1)}`);
    const matches = units.filter((unit) => unit.state === status);
    target.innerHTML = matches.length ? matches.map(contextCard).join("") : '<p class="muted">No units</p>';
  });
  $("#memoryCount").textContent = state.data.memories.length;
  $("#memoryGrid").innerHTML = state.data.memories.length ? state.data.memories.map((memory) => `
    <article class="memory-card">
      <header><h3>${escapeHtml(memory.key)}</h3><span class="confidence">${Math.round(memory.confidence * 100)}%</span></header>
      <p>${escapeHtml(compact(memory.content, 180))}</p>
      <div class="memory-meta"><span>${escapeHtml(memory.status)}</span><span>REV ${memory.revision}</span><span>${memory.evidenceCount} EVIDENCE</span><span>${memory.conflictCount} CONFLICT</span></div>
    </article>
  `).join("") : '<p class="muted">No Memory retained.</p>';
}

function contextCard(unit) {
  return `
    <article class="context-card">
      <div class="context-top"><span>${escapeHtml(unit.shortId)}</span><span>${unit.tokens} T</span></div>
      <strong>${escapeHtml(unit.source)} · ${escapeHtml(unit.layer)}</strong>
      <p>${escapeHtml(compact(unit.conclusions.length ? unit.conclusions : unit.content, 100))}</p>
    </article>
  `;
}

function renderEvaluation() {
  const evaluation = state.data.evaluation;
  $("#overallScore").textContent = scoreText(evaluation.overallScore);
  const scores = [evaluation.outcome, evaluation.trajectory, ...evaluation.components];
  $("#scoreList").innerHTML = scores.map((item, index) => {
    const label = item.component || (index === 0 ? "task success" : "trajectory quality");
    const score = typeof item.score === "number" ? item.score : 0;
    return `
      <div class="score-row">
        <label>${escapeHtml(label.replaceAll("_", " "))}</label>
        <div class="score-track"><div class="score-fill" style="width:${score * 100}%"></div></div>
        <strong>${scoreText(item.score)}</strong>
      </div>
    `;
  }).join("");

  const failures = evaluation.failurePatterns.map((pattern) => `
    <article class="failure-card"><strong>${escapeHtml(pattern.component)} → ${escapeHtml(pattern.key)}</strong><p>${escapeHtml(pattern.rootCause)}</p></article>
  `).join("");
  const proposals = evaluation.optimizationProposals.length
    ? evaluation.optimizationProposals.map((proposal) => `<article class="failure-card optimization-note"><strong>${escapeHtml(proposal.changeKind)}</strong><p>${escapeHtml(proposal.benefit)}</p></article>`).join("")
    : `<article class="failure-card optimization-note"><strong>NO OPTIMIZATION PROPOSAL</strong><p>当前证据运行次数：${evaluation.historyRuns}。Runtime 不会基于不足证据自动演化。</p></article>`;
  $("#failureChain").innerHTML = failures + proposals;

  const governance = state.data.governance;
  const reviewCount = governance.filter((item) => item.reviewed).length;
  $("#reviewCount").textContent = `${reviewCount} REVIEW`;
  $("#governanceList").innerHTML = governance.length ? governance.map((record) => `
    <article class="governance-record">
      <span class="scenario">${escapeHtml(record.scenario)}</span>
      <div class="decision-flow">
        <span class="${escapeHtml(record.preliminary)}">${escapeHtml(record.preliminary)}</span>
        <span class="arrow">→</span>
        <span class="${escapeHtml(record.final)}">${escapeHtml(record.final)}</span>
      </div>
      <span class="${record.reviewed ? "review-tag" : "automatic-tag"}">${record.reviewed ? "HUMAN REVIEW" : "AUTOMATIC"}</span>
    </article>
  `).join("") : '<p class="muted">No governance decisions.</p>';

  const llm = state.data.llm || {};
  const candidateTools = llm.candidateTools || [];
  const toolIntentRecords = llm.toolIntentRecords || [];
  $("#toolIntentCount").textContent = `${toolIntentRecords.length} INTENT`;
  $("#toolIntentCatalog").innerHTML = candidateTools.length ? candidateTools.map((tool) => `
    <article class="tool-intent-card capability-card">
      <span class="intent-stage">RUNTIME CANDIDATE</span>
      <strong>${escapeHtml(tool.name)}</strong>
      <p>${escapeHtml(tool.description)}</p>
      <code>${escapeHtml(tool.capabilityId)}</code>
      <pre>${escapeHtml(JSON.stringify(tool.inputSchema, null, 2))}</pre>
    </article>
  `).join("") : '<p class="muted">Runtime 未向该模型暴露候选工具。</p>';
  $("#toolIntentList").innerHTML = toolIntentRecords.length ? toolIntentRecords.map((record) => `
    <article class="tool-intent-card">
      <div class="intent-flow">
        <span>LLM PROPOSAL</span><i>→</i><span>RUNTIME VALIDATION</span><i>→</i><span>${escapeHtml(record.provider)}</span><i>→</i><span class="${escapeHtml(record.status)}">${escapeHtml(record.status)}</span>
      </div>
      <strong>${escapeHtml(record.capabilityId)} · ${escapeHtml(record.callKey)}</strong>
      <p>参数：${escapeHtml(compact(record.arguments, 180))}</p>
      <p>Observation：${escapeHtml(compact(record.output || record.error || "无输出", 220))}</p>
    </article>
  `).join("") : `
    <p class="muted">模型本次未提出 Tool Intent。候选能力可用不代表 Runtime 会强制模型调用；模型可在已有证据充分时直接完成分析。</p>
  `;
}

function activateView(name) {
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.view === name;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $$(".view").forEach((view) => {
    const active = view.id === `view-${name}`;
    view.hidden = !active;
    view.classList.toggle("is-active", active);
  });
  if (name === "overview" && state.data) requestAnimationFrame(drawGraphEdges);
}

$$(".tab").forEach((tab) => tab.addEventListener("click", () => activateView(tab.dataset.view)));
window.addEventListener("resize", () => {
  if (state.data && !$("#view-overview").hidden) drawGraphEdges();
});

loadHealth();
