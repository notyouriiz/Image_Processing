document.addEventListener("DOMContentLoaded", () => {
  initPage();
});

function initPage() {
  const sliders = Array.from(document.querySelectorAll(".slider"));
  const files = Array.from(document.querySelectorAll(".item")).map(
    (i) => i.dataset.filename
  );

  const fileInput = document.getElementById("fileInput");
  const uploadBtn = document.getElementById("uploadBtn");

  fileInput.addEventListener("change", () => {
    const count = fileInput.files.length;
    uploadBtn.textContent = `Upload (${count} image${count !== 1 ? "s" : ""})`;
  });

  // Show or hide empty state
  document.getElementById("emptyState").style.display = files.length
    ? "none"
    : "block";

  // attach slider listeners with debounce
  const timers = {};
  sliders.forEach((sl) => {
    // initialize displayed value
    const filename = sl.dataset.file;
    const type = sl.dataset.type;
    document.getElementById(`val-${type}-${filename}`).textContent = sl.value;

    sl.addEventListener("input", (e) => {
      document.getElementById(`val-${type}-${filename}`).textContent = sl.value;

      clearTimeout(timers[filename]);
      timers[filename] = setTimeout(() => processImage(filename), 200);
    });
  });

  // attach Download buttons
  const downloadBtns = document.querySelectorAll(".download-btn");
  downloadBtns.forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.preventDefault();
      const filename = btn.dataset.file;
      downloadCurrentImage(filename);
    });
  });

  // Fullscreen viewer
    document.addEventListener("click", function (e) {
        if (e.target.classList.contains("fullscreen-btn")) {
            const file = e.target.dataset.file;
            const imgSrc = document.getElementById("img-" + file).src;

            const div = document.createElement("div");
            div.className = "fullscreen-view";
            div.innerHTML = `
                <span class="fullscreen-close">&times;</span>
                <img src="${imgSrc}">
            `;
            document.body.appendChild(div);
        }

        if (e.target.classList.contains("fullscreen-close")) {
            e.target.parentElement.remove();
        }
    });
    
  // On page load, process each displayed image with current slider (initially zero)
  files.forEach((name) => {
    // ensure the initial displayed values are visible and trigger a single process to populate histogram
    setTimeout(() => processImage(name), 50);
  });

  // drag & drop UI (optional)
  initDragDrop();
}

function initDragDrop() {
  const fileInput = document.getElementById("fileInput");
  const label = document.querySelector(".file-label");
  ["dragenter", "dragover"].forEach((evt) => {
    label.addEventListener(evt, (e) => {
      e.preventDefault();
      label.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    label.addEventListener(evt, (e) => {
      e.preventDefault();
      label.classList.remove("dragover");
    });
  });
  label.addEventListener("drop", (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    // Put files into file input then submit form automatically
    fileInput.files = files;
    document.getElementById("uploadForm").submit();
  });
}

function processImage(filename) {
  const data = new FormData();
  data.append("filename", filename);
  data.append("clahe", getVal(filename, "clahe"));
  data.append("blur", getVal(filename, "blur"));
  data.append("sharp", getVal(filename, "sharp"));
  data.append("denoise", getVal(filename, "denoise"));

  fetch("/process", { method: "POST", body: data })
    .then((r) => r.json())
    .then((out) => {
      if (!out || !out.image) return;
      const imgEl = document.getElementById("img-" + filename);
      imgEl.src = "data:image/jpeg;base64," + out.image;

      // draw histogram: server may return single-channel array or dict with b/g/r arrays
      if (out.hist) {
        drawHistogramFlexible(filename, out.hist);
      }
    })
    .catch((err) => {
      console.error("process error", err);
    });
}

function getVal(file, type) {
  const node = document.querySelector(
    `.slider[data-file="${file}"][data-type="${type}"]`
  );
  return node ? node.value : 0;
}

function drawHistogramFlexible(filename, hist) {
  // accepts either:
  // - flat array hist[0..255] (grayscale)
  // - object {b:[], g:[], r:[]} (color)
  const canvas = document.getElementById("hist-" + filename);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // normalize helper
  function normalize(arr) {
    const max = Math.max(...arr, 1);
    return arr.map((v) => v / max);
  }

  if (Array.isArray(hist)) {
    const normalized = normalize(hist);
    ctx.beginPath();
    ctx.strokeStyle = "#00d1ff";
    for (let x = 0; x < 256; x++) {
      const nx = x * (canvas.width / 256);
      const h = normalized[x] * canvas.height;
      if (x === 0) ctx.moveTo(nx, canvas.height - h);
      else ctx.lineTo(nx, canvas.height - h);
    }
    ctx.stroke();
  } else if (hist.b && hist.g && hist.r) {
    const nb = normalize(hist.b);
    const ng = normalize(hist.g);
    const nr = normalize(hist.r);
    const step = canvas.width / 256;

    // draw thin filled areas with slight transparency for visibility
    [
      ["#3ea6ff", nb],
      ["#75e36f", ng],
      ["#ff8aa1", nr],
    ].forEach(([color, arr]) => {
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 1.0;
      for (let x = 0; x < 256; x++) {
        const nx = x * step;
        const h = arr[x] * canvas.height;
        if (x === 0) ctx.moveTo(nx, canvas.height - h);
        else ctx.lineTo(nx, canvas.height - h);
      }
      ctx.stroke();
    });
    ctx.globalAlpha = 1.0;
  } else {
    // fallback: if unexpected format, try to extract numbers
    const arr = Array.isArray(hist) ? hist : Object.values(hist)[0] || [];
    if (arr.length)
      drawHistogramFlexible(filename, Array.isArray(arr[0]) ? arr[0] : arr);
  }
}

function downloadCurrentImage(filename) {
  const imgEl = document.getElementById("img-" + filename);
  if (!imgEl || !imgEl.src) return;

  // If src is a data URL, download directly
  if (imgEl.src.startsWith("data:")) {
    const a = document.createElement("a");
    a.href = imgEl.src;
    a.download = `processed-${filename}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    return;
  }

  // otherwise fetch the data and download
  fetch(imgEl.src)
    .then((res) => res.blob())
    .then((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `processed-${filename}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
}
