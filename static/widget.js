/**
 * Agentra Consumer Widget -- the embeddable, checkout-time explanation
 * component from the original proposal. Deliberately separate from the
 * full chat UI (index.html): a real deployment embeds this on a
 * merchant's own checkout page, not the whole shopping agent.
 *
 * Usage:
 *   <div id="agentra-widget"></div>
 *   <script src="/static/widget.js"></script>
 *   <script>
 *     AgentraWidget.renderFromSession("agentra-widget", sessionId);
 *     // or, if you already have the payload:
 *     AgentraWidget.render("agentra-widget", decisionPayload);
 *   </script>
 */
(function (global) {
  const LEVEL_COLORS = {
    LOW: "#1a7f37",
    MEDIUM: "#9a6700",
    HIGH: "#bc4c00",
    CRITICAL: "#cf222e",
  };

  const DECISION_COLORS = {
    ALLOW: "#1a7f37",
    REQUIRE_HUMAN_APPROVAL: "#9a6700",
    REQUIRE_VERIFIABLE_APPROVAL: "#bc4c00",
    DENY: "#cf222e",
  };

  function el(tag, className, text) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (text !== undefined) e.textContent = text;
    return e;
  }

  function render(containerId, payload) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = "";
    container.className = "agentra-widget";

    const consumer = payload.consumer_view || {};
    const gate = payload.gate_decision || {};
    const selection = payload.final_selection || {};

    const card = el("div", "agentra-widget-card");

    const header = el("div", "agentra-widget-header");
    const badge = el("span", "agentra-widget-badge", (gate.decision || "").replace(/_/g, " "));
    badge.style.background = DECISION_COLORS[gate.decision] || "#57606a";
    header.appendChild(el("strong", null, selection.name || "This purchase"));
    header.appendChild(badge);
    card.appendChild(header);

    if (consumer.summary) card.appendChild(el("p", "agentra-widget-summary", consumer.summary));
    if (consumer.top_reason) card.appendChild(el("p", "agentra-widget-reason", consumer.top_reason));

    if (consumer.irreversibility_level) {
      const irr = el("div", "agentra-widget-irreversibility");
      const dot = el("span", "agentra-widget-dot");
      dot.style.background = LEVEL_COLORS[consumer.irreversibility_level] || "#57606a";
      irr.appendChild(dot);
      irr.appendChild(el("span", null,
        `Irreversibility: ${consumer.irreversibility_level}` +
        (consumer.irreversibility_note ? ` -- ${consumer.irreversibility_note}` : "")));
      card.appendChild(irr);
    }

    if (consumer.promotion_disclosure) {
      card.appendChild(el("p", "agentra-widget-disclosure", consumer.promotion_disclosure));
    }

    card.appendChild(el("div", "agentra-widget-footer", "Explained by Agentra"));
    container.appendChild(card);
  }

  async function renderFromSession(containerId, sessionId, baseUrl) {
    const base = baseUrl || "";
    try {
      const res = await fetch(`${base}/consumer/decision/${sessionId}`);
      if (!res.ok) throw new Error("no decision yet");
      const data = await res.json();
      render(containerId, data);
    } catch (e) {
      const container = document.getElementById(containerId);
      if (container) container.textContent = "No Agentra decision available for this session yet.";
    }
  }

  global.AgentraWidget = { render, renderFromSession };
})(window);
