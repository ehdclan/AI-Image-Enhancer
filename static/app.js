const form = document.querySelector("#enhance-form");
const input = document.querySelector("#image-input");
const dropzone = document.querySelector(".dropzone");
const button = document.querySelector("#enhance-button");
const apiKeyInput = document.querySelector("#api-key-input");
const engine = document.querySelector("#engine-select");
const preset = document.querySelector("#preset-select");
const status = document.querySelector("#status");
const engineStatus = document.querySelector("#engine-status");
const originalPreview = document.querySelector("#original-preview");
const originalEmpty = document.querySelector("#original-empty");
const enhancedPreview = document.querySelector("#enhanced-preview");
const enhancedEmpty = document.querySelector("#enhanced-empty");
const originalMeta = document.querySelector("#original-meta");
const enhancedMeta = document.querySelector("#enhanced-meta");

let selectedFile = null;
let originalObjectUrl = null;
let enhancedObjectUrl = null;
let engines = {};
const allowedMimeTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

const selectedEngineStatus = () => engines[engine.value];

const selectedEngineIsReady = () => {
  const selected = selectedEngineStatus();
  return selected ? selected.available : false;
};

const apiHeaders = () => {
  const headers = {};
  const apiKey = apiKeyInput.value.trim();
  if (apiKey) {
    headers["X-API-Key"] = apiKey;
  }
  return headers;
};

const isAllowedImage = (file) => allowedMimeTypes.has(file.type);

const updateSubmitState = () => {
  button.disabled = !selectedFile || !selectedEngineIsReady();
};

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

const resetEnhancedPreview = () => {
  if (enhancedObjectUrl) {
    URL.revokeObjectURL(enhancedObjectUrl);
    enhancedObjectUrl = null;
  }

  enhancedPreview.classList.add("hidden");
  enhancedPreview.removeAttribute("src");
  enhancedEmpty.classList.remove("hidden");
  enhancedMeta.textContent = "Ready after processing";
};

const describeSelectedEngine = () => {
  const selected = selectedEngineStatus();
  if (!selected) {
    engineStatus.textContent = "Engine status unavailable.";
    updateSubmitState();
    return;
  }

  const availability = selected.available ? "Ready" : "Not ready";
  engineStatus.textContent = `${selected.label}: ${availability}. ${selected.detail}`;
  updateSubmitState();
};

const loadEngineStatus = async () => {
  try {
    const response = await fetch("/api/engines", { headers: apiHeaders() });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Unable to load engine status.");
    }

    engines = payload;
    if (engines.studio_product?.available) {
      engine.value = "studio_product";
    } else if (engines.ultra_upscale?.available) {
      engine.value = "ultra_upscale";
    } else if (engines.studio_product_realesrgan?.available) {
      engine.value = "studio_product_realesrgan";
    } else if (engines.realesrgan?.available) {
      engine.value = "realesrgan";
    } else {
      engine.value = "pillow_fallback";
    }
    describeSelectedEngine();
  } catch (error) {
    engines = {};
    engineStatus.textContent = error.message || "Engine status unavailable.";
    updateSubmitState();
  }
};

const setOriginalImage = (file) => {
  selectedFile = file;
  updateSubmitState();
  resetEnhancedPreview();

  if (originalObjectUrl) {
    URL.revokeObjectURL(originalObjectUrl);
  }

  originalObjectUrl = URL.createObjectURL(file);
  originalPreview.src = originalObjectUrl;
  originalPreview.classList.remove("hidden");
  originalEmpty.classList.add("hidden");

  originalPreview.onload = () => {
    originalMeta.textContent = `${originalPreview.naturalWidth} x ${originalPreview.naturalHeight} · ${formatBytes(file.size)}`;
  };

  setStatus("Image loaded. Ready to enhance.");
};

const parseError = async (response) => {
  try {
    const payload = await response.json();
    return payload.detail || "Enhancement failed.";
  } catch {
    return "Enhancement failed.";
  }
};

input.addEventListener("change", () => {
  const [file] = input.files;
  if (!file) return;

  if (!isAllowedImage(file)) {
    setStatus("Choose a JPEG, PNG, or WebP image.", true);
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
  if (!isAllowedImage(file)) {
    setStatus("Choose a JPEG, PNG, or WebP image.", true);
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

  if (!selectedEngineIsReady()) {
    const selected = selectedEngineStatus();
    setStatus(selected?.detail || "Selected engine is not ready.", true);
    updateSubmitState();
    return;
  }

  button.disabled = true;
  button.textContent = "Enhancing...";
  setStatus("Enhancing image. Larger files can take a moment.");

  try {
    const query = new URLSearchParams({
      engine: engine.value,
      preset: preset.value,
    });

    const response = await fetch(`/api/enhance?${query.toString()}`, {
      method: "POST",
      headers: {
        ...apiHeaders(),
        "Content-Type": selectedFile.type,
      },
      body: selectedFile,
    });

    if (!response.ok) {
      throw new Error(await parseError(response));
    }

    const imageBlob = await response.blob();
    if (enhancedObjectUrl) {
      URL.revokeObjectURL(enhancedObjectUrl);
    }

    enhancedObjectUrl = URL.createObjectURL(imageBlob);
    enhancedPreview.src = enhancedObjectUrl;
    enhancedPreview.classList.remove("hidden");
    enhancedEmpty.classList.add("hidden");
    enhancedMeta.textContent = `${response.headers.get("x-image-width")} x ${response.headers.get("x-image-height")} · ${response.headers.get("x-enhancer-engine-label") || response.headers.get("x-enhancer-engine")}`;
    setStatus("Enhancement complete.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    updateSubmitState();
    button.textContent = "Enhance";
  }
});

engine.addEventListener("change", describeSelectedEngine);
apiKeyInput.addEventListener("change", loadEngineStatus);

loadEngineStatus();
