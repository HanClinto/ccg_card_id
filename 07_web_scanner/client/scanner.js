/**
 * CCG Card Scanner — client-side logic
 *
 * Architecture:
 *   Scanner      — orchestrates camera, frame capture loop, API calls
 *   ScanBucket   — deduplication state machine (the "grocery scanner" model)
 *   UI helpers   — corner overlay drawing, card list management
 *
 * No frameworks.  Vanilla JS ES2020.
 */

"use strict";

// ---------------------------------------------------------------------------
// ScanBucket — deduplication state machine
// ---------------------------------------------------------------------------

/**
 * Accumulates per-frame identification results and emits a confirmed card
 * only after `fillAt` consecutive matching identifications.
 *
 * Mental model: like a grocery scanner — hold the card steady until it beeps.
 *
 * @param {object} opts
 * @param {number} opts.fillAt       — confirmations needed (default 3)
 * @param {number} opts.drainPerMiss — how much to decrement on a miss (default 1)
 * @param {number} opts.cooldownMs   — ms before same card can re-trigger (default 4000)
 */
class ScanBucket {
  constructor({ fillAt = 3, drainPerMiss = 1, cooldownMs = 4000 } = {}) {
    this.fillAt = fillAt;
    this.drainPerMiss = drainPerMiss;
    this.cooldownMs = cooldownMs;

    this._candidate = null;   // { scryfall_id, count, result }
    this._cooldowns = new Map(); // scryfall_id → expiry timestamp (ms)
  }

  /**
   * Feed one identification result into the bucket.
   *
   * @param {object|null} result  — API record response, or null for no-card
   * @returns {object|null}       — confirmed result dict, or null
   */
  push(result) {
    const now = Date.now();

    // Expire old cooldowns
    for (const [id, expiry] of this._cooldowns) {
      if (now >= expiry) this._cooldowns.delete(id);
    }

    // No card detected
    if (!result || !result.card_present || !result.scryfall_id) {
      if (this._candidate) {
        this._candidate.count = Math.max(0, this._candidate.count - this.drainPerMiss);
        if (this._candidate.count === 0) {
          this._candidate = null;
        }
      }
      return null;
    }

    const id = result.scryfall_id;

    // Card is in cooldown — ignore
    if (this._cooldowns.has(id)) {
      return null;
    }

    // Same card as current candidate
    if (this._candidate && this._candidate.scryfall_id === id) {
      this._candidate.count++;
      this._candidate.result = result; // keep freshest data

      if (this._candidate.count >= this.fillAt) {
        // CONFIRM
        const confirmed = this._candidate.result;
        this._cooldowns.set(id, now + this.cooldownMs);
        this._candidate = null;
        return confirmed;
      }
      return null;
    }

    // Different card — reset candidate
    this._candidate = { scryfall_id: id, count: 1, result };
    return null;
  }

  /** Return current bucket state for display. */
  getState() {
    return {
      candidate: this._candidate
        ? { scryfall_id: this._candidate.scryfall_id, count: this._candidate.count }
        : null,
      fillAt: this.fillAt,
      cooldowns: this._cooldowns.size,
    };
  }

  /** Update configuration without resetting state. */
  configure({ fillAt, drainPerMiss, cooldownMs }) {
    if (fillAt != null)      this.fillAt = fillAt;
    if (drainPerMiss != null) this.drainPerMiss = drainPerMiss;
    if (cooldownMs != null)  this.cooldownMs = cooldownMs;
  }
}


// ---------------------------------------------------------------------------
// Scanner — orchestrates camera + capture loop + API calls
// ---------------------------------------------------------------------------

class Scanner {
  constructor() {
    // DOM refs
    this.video     = document.getElementById("video");
    this.overlay   = document.getElementById("overlay");
    this.ctx       = this.overlay.getContext("2d");
    this.cardList  = document.getElementById("card-list");
    this.startBtn  = document.getElementById("start-btn");
    this.stopBtn   = document.getElementById("stop-btn");
    this.fpsSlider = document.getElementById("fps-slider");
    this.fpsValue  = document.getElementById("fps-value");
    this.serverUrl = document.getElementById("server-url");
    this.detSel    = document.getElementById("detector-select");
    this.idSel     = document.getElementById("identifier-select");
    this.fillAtEl  = document.getElementById("fill-at");
    this.cooldownEl= document.getElementById("cooldown-ms");
    this.statusDot = document.getElementById("status-indicator");
    this.bucketSt  = document.getElementById("bucket-status");
    this.confBadge = document.getElementById("confidence-badge");
    this.lastCard  = document.getElementById("last-card");
    this.lastName  = document.getElementById("last-card-name");
    this.lastMeta  = document.getElementById("last-card-meta");
    this.lastPrice = document.getElementById("last-card-price");

    this.bucket    = new ScanBucket();
    this._loop     = null;       // setInterval handle
    this._sending  = false;      // prevent overlapping requests
    this._csvHeader = false;     // whether header row has been written
    this._stream   = null;

    this._bindEvents();
  }

  // ------------------------------------------------------------------
  // Setup
  // ------------------------------------------------------------------

  _bindEvents() {
    this.startBtn.addEventListener("click", () => this.start());
    this.stopBtn.addEventListener("click",  () => this.stop());

    document.getElementById("copy-btn").addEventListener("click", () => {
      navigator.clipboard.writeText(this.cardList.value)
        .then(() => this._flash(document.getElementById("copy-btn"), "Copied!"))
        .catch(() => {});
    });

    document.getElementById("clear-btn").addEventListener("click", () => {
      this.cardList.value = "";
      this._csvHeader = false;
      this._setStatus("Cleared");
    });

    this.fpsSlider.addEventListener("input", () => {
      this.fpsValue.textContent = this.fpsSlider.value;
      if (this._loop) {
        clearInterval(this._loop);
        this._startLoop();
      }
    });

    this.fillAtEl.addEventListener("change", () => this._syncBucketConfig());
    this.cooldownEl.addEventListener("change", () => this._syncBucketConfig());

    // Fetch detectors + identifiers from server on load
    this.serverUrl.addEventListener("change", () => this._loadServerOptions());
    window.addEventListener("load", () => this._loadServerOptions());
  }

  async _loadServerOptions() {
    const base = this.serverUrl.value.replace(/\/$/, "");
    try {
      const [dResp, iResp, defResp] = await Promise.all([
        fetch(`${base}/v1/detectors`),
        fetch(`${base}/v1/identifiers`),
        fetch(`${base}/v1/defaults`),
      ]);
      const { detectors }   = await dResp.json();
      const { identifiers } = await iResp.json();
      const defaults        = await defResp.json();

      this._populateSelect(this.detSel, detectors, "name", "label", defaults.detector);
      this._populateSelect(this.idSel,  identifiers, "name", "label", defaults.identifier);
      this._setDotState("ready");
    } catch (_) {
      this._setDotState("error");
    }
  }

  _populateSelect(selectEl, items, valueKey, labelKey, defaultValue) {
    selectEl.innerHTML = "";
    for (const item of items) {
      const opt = document.createElement("option");
      opt.value = item[valueKey];
      opt.textContent = item[labelKey];
      if (item[valueKey] === defaultValue) opt.selected = true;
      selectEl.appendChild(opt);
    }
  }

  _syncBucketConfig() {
    this.bucket.configure({
      fillAt:    parseInt(this.fillAtEl.value, 10) || 3,
      cooldownMs: parseInt(this.cooldownEl.value, 10) || 4000,
    });
  }

  // ------------------------------------------------------------------
  // Camera
  // ------------------------------------------------------------------

  async init() {
    try {
      this._stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      this.video.srcObject = this._stream;
      await new Promise(resolve => { this.video.onloadedmetadata = resolve; });
      this.video.play();
      this._resizeOverlay();
      window.addEventListener("resize", () => this._resizeOverlay());
    } catch (err) {
      alert(`Camera error: ${err.message}`);
      throw err;
    }
  }

  _resizeOverlay() {
    // Size the canvas intrinsic resolution to match its CSS display size so that
    // canvas coordinates are CSS pixels — required for the cover-offset math below.
    this.overlay.width  = this.overlay.clientWidth  || this.video.videoWidth;
    this.overlay.height = this.overlay.clientHeight || this.video.videoHeight;
  }

  // ------------------------------------------------------------------
  // Capture loop
  // ------------------------------------------------------------------

  async start() {
    if (!this._stream) {
      await this.init();
    }
    this._syncBucketConfig();
    this._startLoop();
    this.startBtn.disabled = true;
    this.stopBtn.disabled  = false;
    this._setDotState("scanning");
  }

  stop() {
    if (this._loop) {
      clearInterval(this._loop);
      this._loop = null;
    }
    this.ctx.clearRect(0, 0, this.overlay.width, this.overlay.height);
    this.startBtn.disabled = false;
    this.stopBtn.disabled  = true;
    this._setDotState("ready");
    this._setStatus("Stopped");
  }

  _startLoop() {
    const fps = parseInt(this.fpsSlider.value, 10) || 3;
    this._loop = setInterval(() => this._captureFrame(), 1000 / fps);
  }

  async _captureFrame() {
    if (this._sending) return;  // skip frame if previous request still in flight
    if (!this.video.videoWidth) return;

    this._sending = true;
    try {
      // Draw current video frame to a temporary canvas to get JPEG bytes
      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width  = this.video.videoWidth;
      tmpCanvas.height = this.video.videoHeight;
      tmpCanvas.getContext("2d").drawImage(this.video, 0, 0);

      const blob = await new Promise(resolve =>
        tmpCanvas.toBlob(resolve, "image/jpeg", 0.85)
      );
      const arrayBuf = await blob.arrayBuffer();
      const b64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuf)));

      const base = this.serverUrl.value.replace(/\/$/, "");
      const body = {
        records: [{
          _base64: b64,
          detector:   this.detSel.value,
          identifier: this.idSel.value,
        }],
      };

      const resp = await fetch(`${base}/v1/identify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      const rec = data.records?.[0] ?? null;
      this._handleResult(rec);
    } catch (err) {
      this._setDotState("error");
      console.warn("identify error:", err);
    } finally {
      this._sending = false;
    }
  }

  // ------------------------------------------------------------------
  // Result handling
  // ------------------------------------------------------------------

  _handleResult(rec) {
    if (!rec) {
      this._drawCorners(null);
      this._updateBucketStatus(null);
      return;
    }

    this._setDotState("scanning");

    // Update overlay
    this._drawCorners(rec.card_present ? rec.corners : null);

    // Update confidence badge
    if (rec.card_present && rec.confidence != null) {
      this.confBadge.textContent = `${Math.round(rec.confidence * 100)}%`;
      this.confBadge.classList.remove("hidden");
    } else {
      this.confBadge.classList.add("hidden");
    }

    // Feed into bucket
    const confirmed = this.bucket.push(rec.card_present ? rec : null);
    this._updateBucketStatus(rec.card_present ? rec : null);

    if (confirmed) {
      this._addCardToList(confirmed);
      this._flashOverlay();
    }
  }

  _updateBucketStatus(rec) {
    const state = this.bucket.getState();
    if (!rec || !rec.card_present) {
      this._setStatus(state.candidate
        ? `Losing lock… [${state.candidate.count}/${state.fillAt}]`
        : "Scanning…");
      return;
    }
    if (state.candidate) {
      const name = rec.card_name || rec.scryfall_id || "card";
      this._setStatus(`Locking: ${name} [${state.candidate.count}/${state.fillAt}]`);
    } else {
      this._setStatus("Ready");
    }
  }

  // ------------------------------------------------------------------
  // Card list
  // ------------------------------------------------------------------

  _addCardToList(rec) {
    // Write CSV header on first card
    if (!this._csvHeader) {
      this.cardList.value = "card_name,set_code,set_name,scryfall_id,tcgplayer_id,price_usd\n";
      this._csvHeader = true;
    }

    const row = [
      this._csvEscape(rec.card_name   || ""),
      this._csvEscape(rec.set_code    || ""),
      this._csvEscape(rec.set_name    || ""),
      this._csvEscape(rec.scryfall_id || ""),
      rec.tcgplayer_id       != null ? String(rec.tcgplayer_id) : "",
      rec.tcgplayer_price_usd != null ? rec.tcgplayer_price_usd.toFixed(2) : "",
    ].join(",");

    this.cardList.value += row + "\n";
    this.cardList.scrollTop = this.cardList.scrollHeight;

    // Update the "last card" panel
    this.lastName.textContent = rec.card_name  || "Unknown card";
    this.lastMeta.textContent = `${rec.set_name || rec.set_code || "–"} · ${rec.scryfall_id || "–"}`;
    const price = rec.tcgplayer_price_usd != null
      ? `$${rec.tcgplayer_price_usd.toFixed(2)} USD`
      : "Price unavailable";
    this.lastPrice.textContent = price;
    this.lastCard.classList.remove("hidden");
    this.lastCard.classList.add("flash-in");
    setTimeout(() => this.lastCard.classList.remove("flash-in"), 600);
  }

  _csvEscape(val) {
    if (/[",\n]/.test(val)) return `"${val.replace(/"/g, '""')}"`;
    return val;
  }

  // ------------------------------------------------------------------
  // Corner overlay
  // ------------------------------------------------------------------

  _drawCorners(corners) {
    const cw = this.overlay.width;   // = CSS display width (see _resizeOverlay)
    const ch = this.overlay.height;  // = CSS display height
    this.ctx.clearRect(0, 0, cw, ch);

    if (!corners || corners.length !== 4) return;

    // The video uses object-fit:cover — it is uniformly scaled to fill (cw×ch),
    // with excess cropped symmetrically.  Map normalised corner coords through
    // the same transform so the overlay aligns with what the camera actually shows.
    const vw = this.video.videoWidth  || cw;
    const vh = this.video.videoHeight || ch;
    const coverScale = Math.max(cw / vw, ch / vh);
    const ox = (cw - vw * coverScale) / 2;  // negative = left/right crop offset
    const oy = (ch - vh * coverScale) / 2;  // negative = top/bottom crop offset

    const pts = corners.map(([x, y]) => [
      x * vw * coverScale + ox,
      y * vh * coverScale + oy,
    ]);

    this.ctx.beginPath();
    this.ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i++) this.ctx.lineTo(pts[i][0], pts[i][1]);
    this.ctx.closePath();

    this.ctx.strokeStyle = "rgba(0, 230, 120, 0.9)";
    this.ctx.lineWidth   = 3;
    this.ctx.stroke();

    // Corner dots
    this.ctx.fillStyle = "rgba(0, 230, 120, 0.9)";
    for (const [px, py] of pts) {
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  _flashOverlay() {
    this.overlay.classList.add("confirmed-flash");
    setTimeout(() => this.overlay.classList.remove("confirmed-flash"), 400);
  }

  // ------------------------------------------------------------------
  // UI state helpers
  // ------------------------------------------------------------------

  _setStatus(msg) {
    this.bucketSt.textContent = msg;
  }

  _setDotState(state) {
    this.statusDot.className = `status-dot ${state}`;
  }

  _flash(el, text) {
    const orig = el.textContent;
    el.textContent = text;
    setTimeout(() => { el.textContent = orig; }, 1500);
  }
}


// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

const scanner = new Scanner();
