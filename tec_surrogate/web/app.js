const state = {
  meta: null,
  values: {},
  prediction: null,
};

const SVG_NS = "http://www.w3.org/2000/svg";

function formatNumber(value, precision = 2) {
  if (!Number.isFinite(value)) return "--";
  return Number(value).toFixed(precision);
}

function trimNumber(value, precision = 4) {
  if (!Number.isFinite(value)) return "--";
  return Number(value).toFixed(precision).replace(/\.?0+$/, "");
}

function setStateChip(id, text, className = "") {
  const element = document.getElementById(id);
  element.className = `state-chip ${className}`.trim();
  element.innerHTML = `<i></i>${text}`;
}

function makeSvg(name, attributes = {}, text = "") {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
  if (text) element.textContent = text;
  return element;
}

function buildInputs() {
  const list = document.getElementById("input-list");
  list.replaceChildren();

  state.meta.inputs.forEach((item) => {
    state.values[item.name] = item.default;

    const row = document.createElement("label");
    row.className = "input-field";
    row.htmlFor = `input-${item.name}`;

    const copy = document.createElement("span");
    copy.className = "input-copy";
    copy.innerHTML = `<span class="input-label">${item.label}</span><span class="input-range">训练范围 ${trimNumber(item.min)}–${trimNumber(item.max)}</span>`;

    const shell = document.createElement("span");
    shell.className = "number-shell";
    const input = document.createElement("input");
    input.id = `input-${item.name}`;
    input.name = item.name;
    input.type = "number";
    input.step = item.name === "n_pairs" ? "1" : String(item.step);
    input.value = String(item.default);
    input.inputMode = item.name === "n_pairs" ? "numeric" : "decimal";
    input.setAttribute("aria-describedby", `range-${item.name}`);
    input.addEventListener("input", () => {
      state.values[item.name] = Number(input.value);
      validateInput(item, input, false);
    });

    const unit = document.createElement("span");
    unit.textContent = item.unit;
    shell.append(input, unit);
    row.append(copy, shell);
    list.appendChild(row);
  });
}

function buildMetricCards() {
  const grid = document.getElementById("metric-grid");
  grid.replaceChildren();
  state.meta.scalar_outputs.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <span class="metric-label">${metric.label}</span>
      <div class="metric-value-row">
        <strong id="metric-${metric.name}" class="metric-value">--</strong>
        <span class="metric-unit">${metric.unit}</span>
      </div>`;
    grid.appendChild(card);
  });
}

function buildCurveOptions() {
  const select = document.getElementById("delta-t-select");
  select.replaceChildren();
  state.meta.delta_t_values.forEach((deltaT, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${trimNumber(deltaT)} K`;
    select.appendChild(option);
  });
  select.addEventListener("change", drawCurve);
}

function validateInput(item, input, showMessage = true) {
  const value = Number(input.value);
  let message = "";
  if (input.value.trim() === "" || !Number.isFinite(value)) {
    message = `${item.label}必须是有限数值`;
  } else if (item.name === "n_pairs" && !Number.isInteger(value)) {
    message = "PN 对数必须为整数";
  }
  input.setAttribute("aria-invalid", String(Boolean(message)));
  if (showMessage && message) document.getElementById("form-error").textContent = message;
  return !message;
}

function collectInputs() {
  const error = document.getElementById("form-error");
  error.textContent = "";
  let valid = true;

  state.meta.inputs.forEach((item) => {
    const input = document.getElementById(`input-${item.name}`);
    if (!validateInput(item, input, valid)) valid = false;
    state.values[item.name] = Number(input.value);
  });
  return valid;
}

function renderMetrics() {
  state.meta.scalar_outputs.forEach((metric) => {
    const value = state.prediction.scalars[metric.name];
    document.getElementById(`metric-${metric.name}`).textContent = formatNumber(value, metric.precision);
  });
}

function renderWarnings() {
  const panel = document.getElementById("warning-panel");
  const list = document.getElementById("warning-list");
  const warnings = state.prediction.outside_training_range || [];
  list.replaceChildren();
  panel.hidden = warnings.length === 0;

  warnings.forEach((warning) => {
    const item = document.createElement("li");
    item.textContent = `${warning.label} = ${trimNumber(warning.value)}，训练范围为 ${trimNumber(warning.min)}–${trimNumber(warning.max)}`;
    list.appendChild(item);
  });
}

function canvasContext(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(300, Math.round(rect.width));
  const height = Math.max(220, Math.round(rect.height));
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.font = '11px "Segoe UI", "Microsoft YaHei", sans-serif';
  context.textBaseline = "middle";
  return { context, width, height };
}

function heatColor(value, minimum, maximum) {
  const ratio = maximum > minimum ? Math.max(0, Math.min(1, (value - minimum) / (maximum - minimum))) : 0.5;
  const stops = [
    [216, 237, 240],
    [82, 160, 163],
    [246, 196, 93],
    [214, 83, 67],
  ];
  const scaled = ratio * (stops.length - 1);
  const index = Math.min(stops.length - 2, Math.floor(scaled));
  const local = scaled - index;
  const color = stops[index].map((channel, channelIndex) => Math.round(channel + (stops[index + 1][channelIndex] - channel) * local));
  return `rgb(${color.join(",")})`;
}

function drawHeatmap() {
  if (!state.prediction) return;
  const canvas = document.getElementById("heatmap-canvas");
  const { context, width, height } = canvasContext(canvas);
  const values = state.prediction.cop_surface;
  const flat = values.flat().filter(Number.isFinite);
  const minimum = Math.min(...flat);
  const maximum = Math.max(...flat);
  const currents = state.meta.currents;
  const deltaTs = state.meta.delta_t_values;
  const margin = { left: 58, right: 18, top: 22, bottom: 48 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const cellWidth = plotWidth / currents.length;
  const cellHeight = plotHeight / deltaTs.length;

  context.fillStyle = "#687681";
  context.textAlign = "right";
  deltaTs.forEach((deltaT, row) => {
    const y = margin.top + row * cellHeight;
    context.fillText(`${trimNumber(deltaT)} K`, margin.left - 9, y + cellHeight / 2);
    currents.forEach((current, column) => {
      const value = values[row][column];
      const x = margin.left + column * cellWidth;
      context.fillStyle = heatColor(value, minimum, maximum);
      context.fillRect(x + 1, y + 1, Math.max(1, cellWidth - 2), Math.max(1, cellHeight - 2));
      context.fillStyle = value > minimum + (maximum - minimum) * 0.67 ? "#fff" : "#17212b";
      context.textAlign = "center";
      context.font = '600 11px "Segoe UI", sans-serif';
      context.fillText(formatNumber(value, 2), x + cellWidth / 2, y + cellHeight / 2);
    });
  });

  context.font = '11px "Segoe UI", sans-serif';
  context.fillStyle = "#687681";
  context.textAlign = "center";
  currents.forEach((current, column) => {
    context.fillText(trimNumber(current, 2), margin.left + (column + 0.5) * cellWidth, height - 29);
  });
  context.fillText("工作电流 / A", margin.left + plotWidth / 2, height - 10);
  context.save();
  context.translate(13, margin.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("温差", 0, 0);
  context.restore();
}

function drawCurve() {
  if (!state.prediction) return;
  const canvas = document.getElementById("curve-canvas");
  const { context, width, height } = canvasContext(canvas);
  const row = Number(document.getElementById("delta-t-select").value || 0);
  const values = state.prediction.cop_surface[row];
  const currents = state.meta.currents;
  const margin = { left: 48, right: 18, top: 18, bottom: 43 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const minX = Math.min(...currents);
  const maxX = Math.max(...currents);
  const minY = Math.min(0, ...values);
  const maxY = Math.max(...values);
  const yPadding = Math.max((maxY - minY) * 0.12, 0.1);
  const yLow = minY - yPadding;
  const yHigh = maxY + yPadding;
  const xAt = (value) => margin.left + ((value - minX) / Math.max(maxX - minX, 1e-9)) * plotWidth;
  const yAt = (value) => margin.top + (1 - (value - yLow) / Math.max(yHigh - yLow, 1e-9)) * plotHeight;

  context.strokeStyle = "#e3e8ec";
  context.fillStyle = "#687681";
  context.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = yLow + (yHigh - yLow) * (tick / 4);
    const y = yAt(value);
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(width - margin.right, y);
    context.stroke();
    context.textAlign = "right";
    context.fillText(formatNumber(value, 1), margin.left - 8, y);
  }

  context.strokeStyle = "#0969da";
  context.lineWidth = 2.5;
  context.beginPath();
  values.forEach((value, index) => {
    const x = xAt(currents[index]);
    const y = yAt(value);
    if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
  });
  context.stroke();

  values.forEach((value, index) => {
    const x = xAt(currents[index]);
    const y = yAt(value);
    context.fillStyle = "#fff";
    context.strokeStyle = "#0969da";
    context.lineWidth = 2;
    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  });

  context.fillStyle = "#687681";
  context.textAlign = "center";
  currents.forEach((current) => context.fillText(trimNumber(current, 2), xAt(current), height - 27));
  context.fillText("工作电流 / A", margin.left + plotWidth / 2, height - 9);
  context.save();
  context.translate(13, margin.top + plotHeight / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("COP", 0, 0);
  context.restore();
}

function drawLatent() {
  if (!state.prediction) return;
  const svg = document.getElementById("latent-chart");
  svg.replaceChildren();
  const training = state.meta.training_latent;
  const current = state.prediction.latent_2d;
  const points = [...training, current];
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const bounds = { left: 46, right: 444, top: 18, bottom: 257 };
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const padX = Math.max((maxX - minX) * 0.08, 0.1);
  const padY = Math.max((maxY - minY) * 0.08, 0.1);
  const xAt = (value) => bounds.left + ((value - minX + padX) / (maxX - minX + 2 * padX)) * (bounds.right - bounds.left);
  const yAt = (value) => bounds.bottom - ((value - minY + padY) / (maxY - minY + 2 * padY)) * (bounds.bottom - bounds.top);

  for (let tick = 0; tick <= 4; tick += 1) {
    const x = bounds.left + (bounds.right - bounds.left) * (tick / 4);
    const y = bounds.top + (bounds.bottom - bounds.top) * (tick / 4);
    svg.appendChild(makeSvg("line", { x1: x, x2: x, y1: bounds.top, y2: bounds.bottom, stroke: "#e3e8ec" }));
    svg.appendChild(makeSvg("line", { x1: bounds.left, x2: bounds.right, y1: y, y2: y, stroke: "#e3e8ec" }));
  }

  training.forEach((point, index) => {
    const circle = makeSvg("circle", { cx: xAt(point[0]), cy: yAt(point[1]), r: 4.2, fill: "#8daeba", opacity: "0.78" });
    circle.appendChild(makeSvg("title", {}, `训练样本 ${index + 1} · PC1 ${formatNumber(point[0], 2)} · PC2 ${formatNumber(point[1], 2)}`));
    svg.appendChild(circle);
  });

  svg.appendChild(makeSvg("circle", { cx: xAt(current[0]), cy: yAt(current[1]), r: 9, fill: "none", stroke: "#f0aca3", "stroke-width": 4 }));
  const marker = makeSvg("circle", { cx: xAt(current[0]), cy: yAt(current[1]), r: 5.5, fill: "#d65343" });
  marker.appendChild(makeSvg("title", {}, `当前设计 · PC1 ${formatNumber(current[0], 2)} · PC2 ${formatNumber(current[1], 2)}`));
  svg.appendChild(marker);
  svg.appendChild(makeSvg("text", { x: (bounds.left + bounds.right) / 2, y: 289, "text-anchor": "middle", fill: "#687681", "font-size": 11 }, "主成分 PC1"));
  svg.appendChild(makeSvg("text", { x: 13, y: (bounds.top + bounds.bottom) / 2, transform: `rotate(-90 13 ${(bounds.top + bounds.bottom) / 2})`, "text-anchor": "middle", fill: "#687681", "font-size": 11 }, "主成分 PC2"));
}

function renderPrediction() {
  renderMetrics();
  renderWarnings();
  drawHeatmap();
  drawCurve();
  drawLatent();
  document.getElementById("last-updated").textContent = `更新于 ${new Date().toLocaleTimeString("zh-CN", { hour12: false })}`;
  setStateChip("prediction-state", "预测完成", "is-ready");
}

async function runPrediction(event) {
  if (event) event.preventDefault();
  if (!state.meta || !collectInputs()) return;
  const button = document.getElementById("predict-button");
  button.disabled = true;
  setStateChip("prediction-state", "计算中", "is-loading");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inputs: state.values }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "预测请求失败");
    state.prediction = payload;
    renderPrediction();
  } catch (error) {
    document.getElementById("form-error").textContent = error.message;
    setStateChip("prediction-state", "预测失败", "is-error");
  } finally {
    button.disabled = false;
  }
}

function resetInputs() {
  if (!state.meta) return;
  state.meta.inputs.forEach((item) => {
    state.values[item.name] = item.default;
    const input = document.getElementById(`input-${item.name}`);
    input.value = String(item.default);
    input.setAttribute("aria-invalid", "false");
  });
  document.getElementById("form-error").textContent = "";
  runPrediction();
}

function renderMeta() {
  const model = state.meta.model;
  document.getElementById("sample-fact").textContent = `${model.sample_count} 个样本`;
  document.getElementById("latent-fact").textContent = `潜维度 ${model.latent_dim}`;
  document.getElementById("variance-fact").textContent = `保留方差 ${(model.retained_variance * 100).toFixed(1)}%`;
  document.getElementById("cv-fact").textContent = `COP 交叉验证：MAE ${formatNumber(model.cop_cv_mae, 3)} · R² ${formatNumber(model.cop_cv_r2, 3)}`;
}

function redrawCharts() {
  if (!state.prediction) return;
  drawHeatmap();
  drawCurve();
  drawLatent();
}

async function initialize() {
  try {
    const response = await fetch("/api/meta");
    if (!response.ok) throw new Error("模型信息加载失败");
    state.meta = await response.json();
    buildInputs();
    buildMetricCards();
    buildCurveOptions();
    renderMeta();
    document.getElementById("design-form").addEventListener("submit", runPrediction);
    document.getElementById("reset-button").addEventListener("click", resetInputs);
    window.addEventListener("resize", redrawCharts);
    if (window.lucide) window.lucide.createIcons();
    setStateChip("model-state", "模型就绪", "is-ready");
    await runPrediction();
  } catch (error) {
    document.getElementById("form-error").textContent = error.message;
    setStateChip("model-state", "加载失败", "is-error");
    setStateChip("prediction-state", "服务不可用", "is-error");
  }
}

initialize();
