const toggleBtn = document.getElementById("toggleBtn");
const fullscreenBtn = document.getElementById("fullscreenBtn");
const layerSelect = document.getElementById("layerSelect");
const tableBody = document.querySelector("#dataTable tbody");

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
