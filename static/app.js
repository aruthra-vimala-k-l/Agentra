let sessionId = null;

const chatLog = document.getElementById("chat-log");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const productsEl = document.getElementById("products");
const gateEl = document.getElementById("gate");
const trajectoryEl = document.getElementById("trajectory-json");
const statusEl = document.getElementById("aside-status");

function addMessage(text, who) {
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function renderProducts(products) {
  // Bug fix: this used to return early on an empty/undefined list, leaving
  // the PREVIOUS turn's product cards on screen -- most visibly after "new
  // search", where the backend had already cleared its shortlist but the UI
  // kept showing the stale one. Always clear first; show the empty hint when
  // there's nothing (yet) to show instead of silently doing nothing.
  productsEl.innerHTML = "";
  if (!products || products.length === 0) {
    productsEl.innerHTML = '<p class="empty-hint">Products will appear here once enough is known about your intent.</p>';
    return;
  }
  products.forEach((p) => {
    const card = document.createElement("div");
    card.className = "product-card";
    card.innerHTML = `
      <div class="row">
        <span class="name">${p.name}</span>
        <span class="price">₹${p.price.toLocaleString("en-IN")}</span>
      </div>
      <div class="justification">${p.justification}</div>
      <div class="meta">
        ${p.merchant}${p.is_new_merchant ? ' · <span class="new-merchant-flag">new merchant</span>' : ""}
        · ${p.delivery_days == null ? "delivery time not stated" : p.delivery_days === 0 ? "same-day delivery" : `${p.delivery_days}-day delivery`}
      </div>
      <button data-id="${p.id}">Buy ${p.id}</button>
    `;
    card.querySelector("button").addEventListener("click", () => {
      sendMessage(`buy ${p.id}`, true);
    });
    productsEl.appendChild(card);
  });
}

const PENDING_DECISIONS = new Set(["REQUIRE_HUMAN_APPROVAL", "REQUIRE_VERIFIABLE_APPROVAL"]);

function _gateCard(gate, label, productId) {
  const div = document.createElement("div");
  div.className = `gate-card ${gate.decision}`;
  const threatList = (gate.threats || []).length
    ? `<ul>${gate.threats.map((t) => `<li>${t}</li>`).join("")}</ul>` : "";
  const needsApproval = PENDING_DECISIONS.has(gate.decision) && productId;
  div.innerHTML = `
    ${label ? `<div class="gate-label">${label}</div>` : ""}
    <div class="decision">${gate.decision.replace(/_/g, " ")}</div>
    <div class="risk">Risk score: ${gate.risk_score}/100${gate.confidence !== undefined ? ` · confidence ${(gate.confidence * 100).toFixed(0)}%` : ""}</div>
    ${threatList}
    <div class="reasoning">${gate.reasoning}</div>
    ${gate.unreachable ? '<div class="reasoning">(ASIDE was unreachable — failed closed.)</div>' : ""}
    ${needsApproval ? `
      <div class="approval-actions">
        <button class="approve-btn" data-id="${productId}">Approve (simulate reviewer)</button>
        <button class="reject-btn" data-id="${productId}">Reject</button>
      </div>` : ""}
  `;
  if (needsApproval) {
    div.querySelector(".approve-btn").addEventListener("click", () => sendMessage(`approve ${productId}`, true));
    div.querySelector(".reject-btn").addEventListener("click", () => sendMessage(`reject ${productId}`, true));
  }
  return div;
}

// Renders every product evaluated this turn as its own card -- a shopper
// can check or buy several shortlist items in one message (e.g. "check
// j-104 and e-201"), and each gets its own decision shown, not just the
// last one.
function renderGates(gates, fallbackGate) {
  gateEl.innerHTML = "";
  if (gates && gates.length > 0) {
    gates.forEach((g) => gateEl.appendChild(_gateCard(g, g.product_name, g.product_id)));
  } else if (fallbackGate) {
    gateEl.appendChild(_gateCard(fallbackGate));
  }
}

function renderTrajectory(trajectory) {
  if (!trajectory) return;
  trajectoryEl.textContent = JSON.stringify(trajectory, null, 2);
}

function renderDualLayer(dualLayer) {
  if (!dualLayer || !dualLayer.consumer_view) return;
  const cv = dualLayer.consumer_view;
  const ev = dualLayer.enterprise_view || {};
  const div = document.createElement("div");
  div.className = "dual-layer";
  div.innerHTML = `
    <h3>Why this decision (consumer view)</h3>
    ${cv.top_reason ? `<div class="consumer-line">${cv.top_reason}</div>` : ""}
    ${cv.irreversibility_level ? `<div class="consumer-line">
        <span class="irr-badge ${cv.irreversibility_level}">${cv.irreversibility_level}</span>
        ${cv.irreversibility_note ? ` ${cv.irreversibility_note}` : ""}
      </div>` : ""}
    ${cv.promotion_disclosure ? `<div class="consumer-line">${cv.promotion_disclosure}</div>` : ""}
    <details>
      <summary>Enterprise behavioral profile (signal decomposition, bias audit, drift)</summary>
      <pre class="trajectory-json">${JSON.stringify(ev, null, 2)}</pre>
    </details>
  `;
  gateEl.appendChild(div);
}

async function startSession() {
  const res = await fetch("/session/start", { method: "POST" });
  const data = await res.json();
  sessionId = data.session_id;
  addMessage(data.greeting, "agent");
}

async function sendMessage(text, fromButton) {
  if (!text || !sessionId) return;
  if (!fromButton) addMessage(text, "user");
  chatInput.value = "";
  sendBtn.disabled = true;
  try {
    const res = await fetch("/session/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text }),
    });
    const data = await res.json();
    if (data.restarted) {
      // The backend just reset this session's slots/shortlist for a "new
      // search". Signal that visually too, instead of leaving the old
      // conversation and shortlist on screen looking like they still apply.
      chatLog.innerHTML = "";
      gateEl.innerHTML = "";
      trajectoryEl.textContent = "";
      addMessage("Starting a new search.", "agent");
    }
    addMessage(data.reply, "agent");
    renderProducts(data.products);
    renderGates(data.gates, data.gate);
    renderDualLayer(data.dual_layer);
    renderTrajectory(data.trajectory);
  } catch (e) {
    addMessage("Something went wrong talking to the agent service.", "agent");
  } finally {
    sendBtn.disabled = false;
  }
}

async function checkAsideStatus() {
  try {
    const res = await fetch("/system/aside-status");
    const data = await res.json();
    if (data.connected) {
      statusEl.textContent = "ASIDE security gate: connected";
      statusEl.className = "status ok";
    } else {
      statusEl.textContent = "ASIDE security gate: unreachable (evaluations will fail closed)";
      statusEl.className = "status down";
    }
  } catch (e) {
    statusEl.textContent = "ASIDE security gate: unreachable";
    statusEl.className = "status down";
  }
}

sendBtn.addEventListener("click", () => sendMessage(chatInput.value, false));
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage(chatInput.value, false);
});

checkAsideStatus();
startSession();
