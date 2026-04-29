const ENGINE = "studio_product_focus";
const PRESET = "product_detail";
const DEFAULT_BEFORE_SRC = "/static/demo-assets/mj.JPG";
const DEFAULT_AFTER_SRC = "/static/demo-assets/mj-enhanced.jpg";

const form = document.querySelector("#demo-enhance-form");
const input = document.querySelector("#demo-image-input");
const dropzone = document.querySelector(".dropzone");
const enhanceButton = document.querySelector("#demo-enhance-button");
const originalMeta = document.querySelector("#demo-original-meta");
const enhancedMeta = document.querySelector("#demo-enhanced-meta");
const afterLayer = document.querySelector("#compare-after-layer");
const divider = document.querySelector("#compare-divider");
const handle = document.querySelector("#compare-handle");
const compareStage = document.querySelector("#compare-stage");
const beforeLabel = document.querySelector(".compare-label-before");
const afterLabel = document.querySelector(".compare-label-after");
const progressOverlay = document.querySelector("#demo-progress-overlay");
const progressRing = document.querySelector("#demo-progress-ring");
const progressValue = document.querySelector("#demo-progress-value");

let statusText = document.querySelector("#demo-status");
let beforeImage = document.querySelector("#compare-before-image");
let afterImage = document.querySelector("#compare-after-image");
let selectedFile = null;
let originalObjectUrl = null;
let enhancedObjectUrl = null;
let progressInterval = null;
let uploadProgress = 0;
let pointerActive = false;
let progressPhase = 0;
let compareFrame = 0;
let pendingCompareRatio = 0.5;

const allowedMimeTypes = new Set(["image/jpeg", "image/png", "image/webp"]);

function ensureStatusNode() {
  if (statusText || !form) {
    return;
  }

  statusText = document.createElement("p");
  statusText.id = "demo-status";
  statusText.className = "status-text";
  statusText.setAttribute("aria-live", "polite");
  form.append(statusText);
}

function ensureCompareImages() {
  if (!compareStage || !afterLayer) {
    return;
  }

  if (!beforeImage) {
    beforeImage = document.createElement("img");
    beforeImage.id = "compare-before-image";
    beforeImage.className = "compare-image";
    beforeImage.alt = "Original preview";
    compareStage.insertBefore(beforeImage, afterLayer);
  }

  if (!afterImage) {
    afterImage = document.createElement("img");
    afterImage.id = "compare-after-image";
    afterImage.className = "compare-image";
    afterImage.alt = "Enhanced preview";
    afterLayer.append(afterImage);
  }
}

function setStageState(state) {
  if (!compareStage) {
    return;
  }

  compareStage.classList.toggle("is-empty", state === "empty");
  compareStage.classList.toggle("is-original-only", state === "original");
  compareStage.classList.toggle("is-compare-ready", state === "enhanced");

  if (state !== "enhanced") {
    pointerActive = false;
  }
}

function setStatus(message, isError = false) {
  if (!statusText) {
    return;
  }

  statusText.textContent = message;
  statusText.style.color = isError ? "#ff8b8b" : "";
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function revokeUrl(url) {
  if (url) {
    URL.revokeObjectURL(url);
  }
}

function setCompareRatio(ratio) {
  if (!afterLayer || !divider) {
    return;
  }

  pendingCompareRatio = Math.max(0.05, Math.min(0.95, ratio));
  if (compareFrame) {
    return;
  }

  compareFrame = window.requestAnimationFrame(() => {
    compareFrame = 0;
    const percent = pendingCompareRatio * 100;
    afterLayer.style.clipPath = `inset(0 0 0 ${percent}%)`;
    divider.style.left = `${percent}%`;
  });
}

function pointerRatio(event) {
  if (!compareStage) {
    return 0.5;
  }

  const rect = compareStage.getBoundingClientRect();
  return (event.clientX - rect.left) / rect.width;
}

function updateProgress(value) {
  if (!progressValue || !progressRing) {
    return;
  }

  const progress = Math.max(0, Math.min(100, Math.round(value)));
  progressValue.textContent = `${progress}%`;
  progressRing.style.setProperty("--progress", `${progress}%`);
}

function showProgress() {
  if (!progressOverlay) {
    return;
  }

  progressOverlay.classList.remove("hidden");
  progressOverlay.setAttribute("aria-hidden", "false");
}

function hideProgress() {
  if (!progressOverlay) {
    return;
  }

  progressOverlay.classList.add("hidden");
  progressOverlay.setAttribute("aria-hidden", "true");
}

function stopProgressSimulation() {
  if (progressInterval) {
    window.clearInterval(progressInterval);
    progressInterval = null;
  }
}

function startProgressSimulation() {
  stopProgressSimulation();
  const startTime = performance.now();
  progressPhase = 0;

  progressInterval = window.setInterval(() => {
    const elapsed = (performance.now() - startTime) / 1000;
    let estimate = 0;
    if (elapsed < 1.5) {
      estimate = 8 + elapsed * 20;
    } else if (elapsed < 5) {
      estimate = 38 + (elapsed - 1.5) * 10;
    } else if (elapsed < 12) {
      estimate = 73 + (elapsed - 5) * 2.8;
    } else if (elapsed < 24) {
      estimate = 92.6 + (elapsed - 12) * 0.42;
    } else {
      estimate = 97.6 + Math.min(1.2, (elapsed - 24) * 0.08);
    }

    if (elapsed > 6 && progressPhase < 1) {
      setStatus("Enhancing image. This can take a little longer on shared public links.");
      progressPhase = 1;
    }

    if (elapsed > 14 && progressPhase < 2) {
      setStatus("Still working. We are finalizing the enhanced image now.");
      progressPhase = 2;
    }

    if (elapsed > 26 && progressPhase < 3) {
      setStatus("Almost there. The enhanced image is taking a bit longer than usual.");
      progressPhase = 3;
    }

    const floor = Math.max(uploadProgress, 10);
    updateProgress(Math.max(floor, Math.min(99, estimate)));
  }, 140);
}

function completeProgress() {
  stopProgressSimulation();
  updateProgress(100);
  window.setTimeout(() => {
    hideProgress();
    updateProgress(0);
  }, 380);
}

function resetEnhancedPreview() {
  revokeUrl(enhancedObjectUrl);
  enhancedObjectUrl = null;

  if (afterImage) {
    afterImage.removeAttribute("src");
  }
}

function parseErrorResponse(xhr) {
  try {
    return JSON.parse(xhr.responseText).detail || "Enhancement failed.";
  } catch {
    return "Enhancement failed.";
  }
}

function setOriginalMeta(text) {
  if (originalMeta) {
    originalMeta.textContent = text;
  }
}

function setEnhancedMeta(text) {
  if (enhancedMeta) {
    enhancedMeta.textContent = text;
  }
}

function setSelectedFile(file) {
  if (!beforeImage || !afterImage || !enhanceButton) {
    setStatus("Preview panel is not ready yet.", true);
    return;
  }

  if (!allowedMimeTypes.has(file.type)) {
    setStatus("Please use a JPEG, PNG, or WebP image.", true);
    return;
  }

  selectedFile = file;
  enhanceButton.disabled = false;
  resetEnhancedPreview();

  revokeUrl(originalObjectUrl);
  originalObjectUrl = URL.createObjectURL(file);
  beforeImage.src = originalObjectUrl;
  afterImage.removeAttribute("src");
  setStageState("original");

  beforeImage.onload = () => {
    setOriginalMeta(
      `${beforeImage.naturalWidth} x ${beforeImage.naturalHeight} · ${formatBytes(file.size)}`
    );
  };

  setEnhancedMeta("Awaiting enhancement");
  setCompareRatio(0.5);
  setStatus("Image loaded. Ready to enhance.");
}

function loadDefaultPreview() {
  if (!beforeImage || !afterImage) {
    return;
  }

  beforeImage.onload = () => {
    setOriginalMeta(`${beforeImage.naturalWidth} x ${beforeImage.naturalHeight} · Default sample`);
  };

  afterImage.onload = () => {
    setEnhancedMeta(`${afterImage.naturalWidth} x ${afterImage.naturalHeight} · Enhanced sample`);
  };

  beforeImage.src = DEFAULT_BEFORE_SRC;
  afterImage.src = DEFAULT_AFTER_SRC;
  setStageState("enhanced");
  setCompareRatio(0.5);
}

function enhanceImage() {
  if (!selectedFile || !enhanceButton || !afterImage) {
    setStatus("Choose an image first.", true);
    return;
  }

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/demo/api/enhance");
  xhr.responseType = "blob";
  xhr.timeout = 120000;
  xhr.setRequestHeader("Content-Type", selectedFile.type);

  xhr.upload.onprogress = (event) => {
    if (!event.lengthComputable) {
      return;
    }

    uploadProgress = Math.max(6, (event.loaded / event.total) * 18);
    updateProgress(uploadProgress);
  };

  xhr.onloadstart = () => {
    uploadProgress = 0;
    updateProgress(0);
    showProgress();
    startProgressSimulation();
    enhanceButton.disabled = true;
    enhanceButton.textContent = "Enhancing...";
    setStatus("Enhancing image. This progress indicator is an estimate while processing runs.");
  };

  xhr.onerror = () => {
    stopProgressSimulation();
    hideProgress();
    updateProgress(0);
    enhanceButton.disabled = false;
    enhanceButton.textContent = "Enhance image";
    setStatus("Network error while enhancing the image.", true);
  };

  xhr.onabort = () => {
    stopProgressSimulation();
    hideProgress();
    updateProgress(0);
    enhanceButton.disabled = false;
    enhanceButton.textContent = "Enhance image";
    setStatus("The request was interrupted before the enhancement finished.", true);
  };

  xhr.ontimeout = () => {
    stopProgressSimulation();
    hideProgress();
    updateProgress(0);
    enhanceButton.disabled = false;
    enhanceButton.textContent = "Enhance image";
    setStatus("Enhancement timed out. Please try again with the same image.", true);
  };

  xhr.onload = () => {
    if (xhr.status < 200 || xhr.status >= 300) {
      stopProgressSimulation();
      hideProgress();
      updateProgress(0);
      enhanceButton.disabled = false;
      enhanceButton.textContent = "Enhance image";
      setStatus(parseErrorResponse(xhr), true);
      return;
    }

    resetEnhancedPreview();
    enhancedObjectUrl = URL.createObjectURL(xhr.response);
    afterImage.src = enhancedObjectUrl;
    setStageState("enhanced");

    const width = xhr.getResponseHeader("x-image-width");
    const height = xhr.getResponseHeader("x-image-height");
    setEnhancedMeta(width && height ? `${width} x ${height}` : "Enhanced");
    setCompareRatio(0.5);
    setStatus("Enhancement complete.");
    completeProgress();
    enhanceButton.disabled = false;
    enhanceButton.textContent = "Enhance image";
  };

  xhr.send(selectedFile);
}

function onPointerMove(event) {
  if (!pointerActive || !compareStage?.classList.contains("is-compare-ready")) {
    return;
  }

  event.preventDefault();
  setCompareRatio(pointerRatio(event));
}

function beginPointerDrag(event) {
  if (!compareStage?.classList.contains("is-compare-ready")) {
    return;
  }

  pointerActive = true;
  event.preventDefault();
  if (event.target && typeof event.target.setPointerCapture === "function") {
    event.target.setPointerCapture(event.pointerId);
  }
  setCompareRatio(pointerRatio(event));
}

ensureStatusNode();
ensureCompareImages();
loadDefaultPreview();
if (beforeLabel) {
  beforeLabel.textContent = "Before";
}
if (afterLabel) {
  afterLabel.textContent = "After";
}
setStatus("Default sample loaded. Upload an image to test your own product.");

if (enhanceButton) {
  enhanceButton.disabled = true;
}

if (input) {
  input.addEventListener("change", () => {
    const [file] = input.files;
    if (file) {
      setSelectedFile(file);
    }
  });
}

if (dropzone) {
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
    if (!file) {
      return;
    }

    try {
      input.files = event.dataTransfer.files;
    } catch {
      // Some browsers keep this property read-only.
    }

    setSelectedFile(file);
  });
}

window.addEventListener("paste", (event) => {
  const imageItem = Array.from(event.clipboardData?.items || []).find((item) =>
    item.type.startsWith("image/")
  );

  if (imageItem) {
    const file = imageItem.getAsFile();
    if (file) {
      setSelectedFile(file);
      setStatus("Image pasted from clipboard. Ready to enhance.");
    }
  }
});

if (form) {
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    enhanceImage();
  });
}

if (handle) {
  handle.addEventListener("pointerdown", beginPointerDrag);
}

if (compareStage) {
  compareStage.addEventListener("pointerdown", beginPointerDrag);
}

window.addEventListener("pointermove", onPointerMove, { passive: false });

window.addEventListener("pointerup", () => {
  pointerActive = false;
});

window.addEventListener("pointercancel", () => {
  pointerActive = false;
});

setCompareRatio(0.5);
