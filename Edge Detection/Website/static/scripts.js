const toggleBtn = document.getElementById("toggleBtn"); 
const fullscreenBtn = document.getElementById("fullscreenBtn");
const layerSelect = document.getElementById("layerSelect");
const tableBody = document.querySelector("#dataTable tbody");
const videoFeed = document.getElementById("videoFeed");
const histogram = document.getElementById("histogram");
const histTime = document.getElementById("histTime");

const modeCameraBtn = document.getElementById("modeCamera");
const modeImageBtn = document.getElementById("modeImage");
const uploadForm = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const nextImageBtn = document.getElementById("nextImage");
const prevImageBtn = document.getElementById("prevImage");

let currentMode = "camera"; // frontend state

toggleBtn.addEventListener("click", () => {
  fetch("/toggle", { method: "POST" })
    .then(r => r.json())
    .then(data => {
      toggleBtn.textContent = data.running ? "⏸ Pause" : "▶ Continue";
    });
});

fullscreenBtn.addEventListener("click", () => {
  const video = document.getElementById("videoFeed");
  if (video.requestFullscreen) video.requestFullscreen();
  else if (video.webkitRequestFullscreen) video.webkitRequestFullscreen();
  else if (video.msRequestFullscreen) video.msRequestFullscreen();
});

layerSelect.addEventListener("change", () => {
  fetch("/set_layer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ layer: layerSelect.value })
  });
});

modeCameraBtn.addEventListener("click", () => setMode("camera"));
modeImageBtn.addEventListener("click", () => setMode("image"));


function setMode(m) {
  fetch("/set_mode", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode: m })
  }).then(r => r.json()).then(resp => {
    currentMode = resp.mode;
    if (currentMode === "camera") {
      videoFeed.src = "/video_feed";
    } else {
      // for image modes, poll /image_feed
      videoFeed.src = "/image_feed?ts=" + Date.now();
    }
  });
}

// Upload images
uploadForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const files = fileInput.files;
  if (!files.length) {
    alert("Select at least one image to upload.");
    return;
  }
  const fd = new FormData();
  for (let i = 0; i < files.length; i++) fd.append('files', files[i]);
  fetch("/upload", { method: "POST", body: fd })
    .then(r => r.json())
    .then(resp => {
      alert(`Saved ${resp.saved} files. Total images on server: ${resp.total_images}`);
      setMode("image");
    });
});

// next / prev buttons
nextImageBtn.addEventListener("click", () => {
  fetch("/next_image").then(r => r.json()).then(j => {
    videoFeed.src = "/image_feed?ts=" + Date.now();
  });
});
prevImageBtn.addEventListener("click", () => {
  fetch("/prev_image").then(r => r.json()).then(j => {
    videoFeed.src = "/image_feed?ts=" + Date.now();
  });
});

// Data table live update
setInterval(() => {
  fetch("/data")
    .then(r => r.json())
    .then(data => {
      tableBody.innerHTML = "";
      data.forEach(row => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${row.id}</td>
          <td>${row.width}</td>
          <td>${row.height}</td>
          <td>${row.volume}</td>
          <td>${row.error}</td>
          <td>${row.marker}</td>`;
        tableBody.appendChild(tr);
      });
    });
}, 1000);

// For camera mode, video_feed is a multipart MJPEG so img src works.
// For image mode, we poll the image endpoint periodically to refresh
setInterval(() => {
  if (currentMode !== "camera") {
    videoFeed.src = "/image_feed?ts=" + Date.now();
  }
  // refresh histogram
  histogram.src = "/histogram.jpg?ts=" + Date.now();
  histTime.textContent = new Date().toLocaleTimeString();
}, 900);

// initial mode
videoFeed.src = "/video_feed";
