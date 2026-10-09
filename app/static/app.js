const form = document.querySelector("#mission-form");
const statusBox = document.querySelector("#status");
const result = document.querySelector("#result");
const comparison = document.querySelector("#comparison");

function payload() {
  return {
    age: Number(document.querySelector("#age").value),
    environment: document.querySelector("#environment").value.trim(),
    topic: document.querySelector("#topic").value.trim(),
    duration_minutes: Number(document.querySelector("#duration").value)
  };
}

function renderMission(data) {
  document.querySelector("#mission-title").textContent = data.mission;
  document.querySelector("#observe").textContent = data.observe;
  document.querySelector("#safety").textContent = data.safety;
  document.querySelector("#reflection").textContent = data.reflection;
  document.querySelector("#provider").textContent = data.provider;

  const q = document.querySelector("#questions");
  q.innerHTML = "";
  data.questions.forEach(item => {
    const li = document.createElement("li");
    li.textContent = item;
    q.appendChild(li);
  });

  result.classList.remove("hidden");
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed.");
  }
  return data;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  comparison.classList.add("hidden");
  statusBox.textContent = "Creating a short outdoor mission…";

  try {
    const data = await postJson("/api/mission", payload());
    renderMission(data);
    statusBox.textContent = "Mission ready. Read it, put the screen away, and go explore.";
  } catch (error) {
    statusBox.textContent = error.message;
  }
});

document.querySelector("#compare").addEventListener("click", async () => {
  result.classList.add("hidden");
  statusBox.textContent = "Running baseline and tuned-model comparison…";

  try {
    const data = await postJson("/api/compare", payload());
    comparison.innerHTML = `
      <h2>Baseline vs tuned</h2>
      <div class="comparison-grid">
        <div>
          <h3>${data.baseline.provider}</h3>
          <pre>${JSON.stringify(data.baseline, null, 2)}</pre>
        </div>
        <div>
          <h3>${data.tuned.provider}</h3>
          <pre>${JSON.stringify(data.tuned, null, 2)}</pre>
        </div>
      </div>
    `;
    comparison.classList.remove("hidden");
    statusBox.textContent = "Comparison complete.";
  } catch (error) {
    statusBox.textContent = error.message;
  }
});
