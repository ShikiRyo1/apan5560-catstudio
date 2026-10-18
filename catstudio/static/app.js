"use strict";

const byId = (id) => document.getElementById(id);
const form = byId("portrait-form");
const promptInput = byId("prompt");
const statusBox = byId("status");
let generating = false;
let lastRecord = null;
let currentImageURL = null;

function showStatus(message, kind = "") {
  statusBox.textContent = message;
  statusBox.className = `status ${kind}`;
}

function countCharacters() {
  byId("character-count").textContent = `${promptInput.value.length} / 500`;
}

async function refreshHealth() {
  if (generating) return;
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Health request failed");
    const { portrait } = await response.json();
    byId("generate-button").disabled = !portrait.base_files_present || portrait.status === "busy";
    if (!portrait.base_files_present) {
      showStatus("Model files are missing. Follow the README setup. / 尚未安装模型，请先按 README 配置。", "error");
    } else if (portrait.status === "busy") {
      showStatus("Another generation is running. / 另一个生成任务正在运行，请稍候。");
    } else if (portrait.last_error) {
      showStatus(`Last run failed: ${portrait.last_error}`, "error");
    } else {
      const modelLabel = portrait.model_loaded ? `Model loaded on ${portrait.device_loaded}` : "Model files found · loads on first use";
      const adapterLabel = portrait.adapter_files_present ? "LoRA files found" : "LoRA missing · select Base model";
      showStatus(`${modelLabel}. ${adapterLabel}.`);
    }
    const loraOption = byId("model-choice").querySelector('[value="lora"]');
    loraOption.disabled = !portrait.adapter_files_present;
    if (!portrait.adapter_files_present) byId("model-choice").value = "base";
  } catch {
    byId("generate-button").disabled = true;
    showStatus("Cannot reach the local server. Start it with the README command. / 无法连接本机服务。", "error");
  }
}

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => { promptInput.value = button.dataset.example; countCharacters(); });
});
promptInput.addEventListener("input", countCharacters);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (generating || !form.reportValidity()) return;
  const request = {
    prompt: promptInput.value.trim(),
    seed: Number(byId("seed").value),
    model_choice: byId("model-choice").value,
    num_steps: Number(byId("num-steps").value),
    guidance_scale: Number(byId("guidance-scale").value),
    adapter_scale: Number(byId("adapter-scale").value),
  };
  generating = true;
  byId("generate-button").disabled = true;
  byId("busy-overlay").hidden = false;
  byId("elapsed").textContent = "0 seconds";
  showStatus("Generating locally. This may take several minutes on CPU. / 正在本机生成，CPU 可能需要数分钟。");
  const started = performance.now();
  const timer = setInterval(() => { byId("elapsed").textContent = `${Math.floor((performance.now() - started) / 1000)} seconds`; }, 1000);
  try {
    const response = await fetch("/portraits", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
    const result = await response.json();
    if (!response.ok) {
      const detail = typeof result.detail === "string" ? result.detail : JSON.stringify(result.detail);
      throw new Error(detail || `HTTP ${response.status}`);
    }
    const bytes = Uint8Array.from(atob(result.image_base64), (character) => character.charCodeAt(0));
    const imageURL = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
    if (currentImageURL) URL.revokeObjectURL(currentImageURL);
    currentImageURL = imageURL;
    byId("result-image").src = imageURL;
    byId("result-image").alt = `Generated cat portrait: ${request.prompt}`;
    byId("result-image").hidden = false;
    byId("empty-state").hidden = true;
    byId("download").href = imageURL;
    byId("download").download = `catstudio-${request.model_choice}-seed${request.seed}.png`;
    byId("download").hidden = false;
    lastRecord = { request, result: result.metadata };
    byId("metadata").textContent = JSON.stringify(lastRecord, null, 2);
    byId("metadata-panel").hidden = false;
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    byId("result-summary").textContent = `${request.model_choice === "lora" ? "Trained LoRA" : "Base model"} · seed ${request.seed} · ${seconds}s`;
    const promptNote = result.metadata.prompt_was_truncated
      ? " The description exceeded the model's text limit. Shorten it for a complete prompt. / 描述超过模型的文本长度限制，请缩短后重试。"
      : " Inspect its details and try the other model with the same settings. / 可用相同参数比较另一模型。";
    showStatus(`Portrait generated. / 已生成。${promptNote}`, "success");
  } catch (error) {
    showStatus(`${error.message} / 本次未生成新图；如有旧图，保留供查看。`, "error");
  } finally {
    clearInterval(timer);
    generating = false;
    byId("busy-overlay").hidden = true;
    byId("generate-button").disabled = false;
  }
});

byId("download-record").addEventListener("click", () => {
  if (!lastRecord) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(lastRecord, null, 2)], { type: "application/json" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `catstudio-record-seed${lastRecord.request.seed}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

countCharacters();
refreshHealth();
setInterval(() => { if (!generating && !lastRecord) refreshHealth(); }, 15000);
