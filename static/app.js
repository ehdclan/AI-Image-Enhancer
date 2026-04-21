const form = document.querySelector("#enhance-form");
const input = document.querySelector("#image-input");
const dropzone = document.querySelector(".dropzone");
const button = document.querySelector("#enhance-button");
const engine = document.querySelector("#engine-select");
const preset = document.querySelector("#preset-select");
const status = document.querySelector("#status");
const engineStatus = document.querySelector("#engine-status");
const originalPreview = document.querySelector("#original-preview");
const enhancedPreview = document.querySelector("#enhanced-preview");
const enhancedEmpty = document.querySelector("#enhanced-empty");
const originalMeta = document.querySelector("#original-meta");
const enhancedMeta = document.querySelector("#enhanced-meta");

let selectedFile = null;
let originalObjectUrl = null;
let engines = {};

const setStatus = (message, isError = false) => {
  status.textContent = message;
  status.style.color = isError ? "#b3261e" : "";
};

const formatBytes = (bytes) => {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

const describeSelectedEngine = () => {
  const selected = engines[engine.value];
  if (!selected) {
    engineStatus.textContent = "Engine status unavailable.";
    return;
  }

  const availability = selected.available ? "Ready" : "Not ready";
  engineStatus.textContent = `${selected.label}: ${availability}. ${selected.detail}`;
};

const loadEngineStatus = async () => {
  try {
    const response = await fetch("/api/engines");
    engines = await response.json();
    describeSelectedEngine();

    if (engines.studio_product?.available) {
      engine.value = "studio_product";
      describeSelectedEngine();
    } else if (engines.realesrgan?.available) {
      engine.value = "realesrgan";
      describeSelectedEngine();
    } else {
      engine.value = "pillow_fallback";
      describeSelectedEngine();
    }
  } catch {
    engineStatus.textContent = "Engine status unavailable.";
  }
};

const setOriginalImage = (file) => {
  selectedFile = file;
  button.disabled = false;
  enhancedPreview.classList.add("hidden");
  enhancedPreview.removeAttribute("src");
  enhancedEmpty.classList.remove("hidden");
  enhancedMeta.textContent = "Ready after processing";

  if (originalObjectUrl) {
    URL.revokeObjectURL(originalObjectUrl);
  }

  originalObjectUrl = URL.createObjectURL(file);
  originalPreview.src = originalObjectUrl;
  originalPreview.classList.remove("sample");

  originalPreview.onload = () => {
    originalMeta.textContent = `${originalPreview.naturalWidth} x ${originalPreview.naturalHeight} · ${formatBytes(file.size)}`;
  };

  setStatus("Image loaded. Ready to enhance.");
};

input.addEventListener("change", () => {
  const [file] = input.files;
  if (!file) return;

  if (!file.type.startsWith("image/")) {
    setStatus("Choose a valid image file.", true);
    return;
  }

  setOriginalImage(file);
});

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragging");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragging");
  });
}

dropzone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (!file) return;
  if (!file.type.startsWith("image/")) {
    setStatus("Choose a valid image file.", true);
    return;
  }

  input.files = event.dataTransfer.files;
  setOriginalImage(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!selectedFile) {
    setStatus("Choose an image first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("engine", engine.value);
  formData.append("preset", preset.value);

  button.disabled = true;
  button.textContent = "Enhancing...";
  setStatus("Enhancing image. Larger files can take a moment.");

  try {
    const response = await fetch("/api/enhance", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Enhancement failed.");
    }

    enhancedPreview.src = payload.image;
    enhancedPreview.classList.remove("hidden");
    enhancedEmpty.classList.add("hidden");
    enhancedMeta.textContent = `${payload.width} x ${payload.height} · ${payload.engine_label || payload.engine}`;
    setStatus("Enhancement complete.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "Enhance";
  }
});

engine.addEventListener("change", describeSelectedEngine);

loadEngineStatus();
