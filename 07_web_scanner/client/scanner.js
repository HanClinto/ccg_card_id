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
    this.lastCrop      = document.getElementById("last-card-crop");
    this.cropPrev      = document.getElementById("crop-preview");
    this.heatmapToggle      = document.getElementById("heatmap-toggle");
    this.heatmapOpacity     = document.getElementById("heatmap-opacity");
    this._heatmapOpacityRow = document.getElementById("heatmap-opacity-row");
    this.sharpnessSlider      = document.getElementById("sharpness-threshold");
    this.sharpnessValue       = document.getElementById("sharpness-threshold-value");
    this._sharpnessRow        = document.getElementById("sharpness-row");
    this._mSharpness          = null;  // injected into metrics bar dynamically
    this._lastSharpness       = null;  // latest value for canvas overlay
    this.confidenceThreshold  = document.getElementById("confidence-threshold");
    this.confidenceValue      = document.getElementById("confidence-threshold-value");
    this.heatmapOpacityValue  = document.getElementById("heatmap-opacity-value");

    // Metrics display
    this._mDetect   = document.getElementById("m-detect");
    this._mIdentify = document.getElementById("m-identify");
    this._mServer   = document.getElementById("m-server");
    this._mRtt      = document.getElementById("m-rtt");
    this._mRam      = document.getElementById("m-ram");
    this._mVram     = document.getElementById("m-vram");
    this._memPoll   = null;

    this.bucket    = new ScanBucket();
    this._loop     = null;       // setInterval handle
    this._sending  = false;      // prevent overlapping requests
    this._csvHeader = false;     // whether header row has been written
    this._stream   = null;

    this._restoreSettings();
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
      if (this._loop) { clearInterval(this._loop); this._startLoop(); }
      this._saveSettings();
    });

    this.fillAtEl.addEventListener("change", () => { this._syncBucketConfig(); this._saveSettings(); });
    this.cooldownEl.addEventListener("change", () => { this._syncBucketConfig(); this._saveSettings(); });

    this.heatmapToggle.addEventListener("change", () => {
      this._heatmapOpacityRow.style.display = this.heatmapToggle.checked ? "" : "none";
      this._saveSettings();
    });
    this.heatmapOpacity.addEventListener("input", () => {
      this.heatmapOpacityValue.textContent = this.heatmapOpacity.value;
      this._saveSettings();
    });
    this.sharpnessSlider.addEventListener("input", () => {
      this.sharpnessValue.textContent = this.sharpnessSlider.value;
      this._saveSettings();
    });
    this.confidenceThreshold.addEventListener("input", () => {
      this.confidenceValue.textContent = this.confidenceThreshold.value;
      this._saveSettings();
    });

    // Hide sharpness slider when Canny is selected (no sharpness signal)
    this.detSel.addEventListener("change", () => this._syncDetectorUI());

    // Fetch detectors + identifiers from server on load
    this.serverUrl.addEventListener("change", () => { this._saveSettings(); this._loadServerOptions(); });
    this.detSel.addEventListener("change", () => this._saveSettings());
    this.idSel.addEventListener("change",  () => this._saveSettings());
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
    if (selectEl === this.detSel) {
      if (this._pendingDetector) {
        const opt = [...selectEl.options].find(o => o.value === this._pendingDetector);
        if (opt) { opt.selected = true; this._pendingDetector = null; }
      }
      this._syncDetectorUI();
    }
    if (selectEl === this.idSel && this._pendingIdentifier) {
      const opt = [...selectEl.options].find(o => o.value === this._pendingIdentifier);
      if (opt) { opt.selected = true; this._pendingIdentifier = null; }
    }
  }

  /** Show/hide sharpness controls depending on whether the detector supports it. */
  _syncDetectorUI() {
    const isNeural = this.detSel.value !== "canny";
    this._sharpnessRow.style.display = isNeural ? "" : "none";
    if (this._mSharpness) this._mSharpness.style.display = isNeural ? "" : "none";
  }

  _syncBucketConfig() {
    this.bucket.configure({
      fillAt:    parseInt(this.fillAtEl.value, 10) || 3,
      cooldownMs: parseInt(this.cooldownEl.value, 10) || 4000,
    });
  }

  // ------------------------------------------------------------------
  // Settings persistence (localStorage)
  // ------------------------------------------------------------------

  _settingsKey() { return "ccg-scanner-settings"; }

  _saveSettings() {
    const s = {
      serverUrl:          this.serverUrl.value,
      fps:                this.fpsSlider.value,
      detector:           this.detSel.value,
      identifier:         this.idSel.value,
      fillAt:             this.fillAtEl.value,
      cooldownMs:         this.cooldownEl.value,
      heatmapOn:          this.heatmapToggle.checked,
      heatmapOpacity:     this.heatmapOpacity.value,
      sharpnessThreshold: this.sharpnessSlider.value,
      confidenceThreshold: this.confidenceThreshold.value,
    };
    try { localStorage.setItem(this._settingsKey(), JSON.stringify(s)); } catch (_) {}
  }

  _restoreSettings() {
    let s;
    try { s = JSON.parse(localStorage.getItem(this._settingsKey()) ?? "null"); } catch (_) {}
    if (!s) return;

    if (s.serverUrl)      this.serverUrl.value  = s.serverUrl;
    if (s.fps != null) {
      this.fpsSlider.value = s.fps;
      this.fpsValue.textContent = s.fps;
    }
    if (s.fillAt != null)     this.fillAtEl.value   = s.fillAt;
    if (s.cooldownMs != null) this.cooldownEl.value  = s.cooldownMs;
    if (s.heatmapOn != null) {
      this.heatmapToggle.checked = s.heatmapOn;
      this._heatmapOpacityRow.style.display = s.heatmapOn ? "" : "none";
    }
    if (s.heatmapOpacity != null) {
      this.heatmapOpacity.value = s.heatmapOpacity;
      this.heatmapOpacityValue.textContent = s.heatmapOpacity;
    }
    if (s.sharpnessThreshold != null) {
      this.sharpnessSlider.value = s.sharpnessThreshold;
      this.sharpnessValue.textContent = s.sharpnessThreshold;
    }
    if (s.confidenceThreshold != null) {
      this.confidenceThreshold.value = s.confidenceThreshold;
      this.confidenceValue.textContent = s.confidenceThreshold;
    }
    // Detector and identifier selects are populated from the server — their
    // saved values are applied after _populateSelect runs (see below).
    this._pendingDetector   = s.detector   ?? null;
    this._pendingIdentifier = s.identifier ?? null;
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
    // Match canvas intrinsic size to its CSS display size so the coordinate
    // system is 1:1 with CSS pixels and no browser scaling occurs.
    this.overlay.width  = this.overlay.clientWidth  || this.video.clientWidth;
    this.overlay.height = this.overlay.clientHeight || this.video.clientHeight;
  }

  /**
   * Compute the source rectangle that object-fit:cover shows from the native
   * video frame, expressed in native video pixels.
   *
   * @returns {{ sx, sy, sw, sh }} — source crop in video pixels
   */
  _coverCrop() {
    const vw = this.video.videoWidth;
    const vh = this.video.videoHeight;
    const cssW = this.overlay.clientWidth;
    const cssH = this.overlay.clientHeight;

    // Uniform scale so the video fills (covers) the CSS box.
    const scale = Math.max(cssW / vw, cssH / vh);

    // Scaled video dimensions in CSS pixels
    const scaledW = vw * scale;
    const scaledH = vh * scale;

    // How many CSS px are cropped off each side
    const cropX = (scaledW - cssW) / 2;
    const cropY = (scaledH - cssH) / 2;

    // Convert crop back to native video pixels
    const sx = cropX / scale;
    const sy = cropY / scale;
    const sw = cssW  / scale;
    const sh = cssH  / scale;

    return { sx, sy, sw, sh };
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
    this._startMemoryPoll();
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
    this.cropPrev.classList.add("hidden");
    this._stopMemoryPoll();
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
      // Capture only the region that object-fit:cover actually displays so
      // that server-returned normalised corners map 1:1 to the overlay.
      const { sx, sy, sw, sh } = this._coverCrop();
      const cw = this.overlay.clientWidth  || this.video.clientWidth;
      const ch = this.overlay.clientHeight || this.video.clientHeight;

      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width  = cw;
      tmpCanvas.height = ch;
      // drawImage(video, sx, sy, sw, sh, dx, dy, dw, dh)
      tmpCanvas.getContext("2d").drawImage(this.video, sx, sy, sw, sh, 0, 0, cw, ch);

      const blob = await new Promise(resolve =>
        tmpCanvas.toBlob(resolve, "image/jpeg", 0.85)
      );
      const arrayBuf = await blob.arrayBuffer();
      const b64 = btoa(String.fromCharCode(...new Uint8Array(arrayBuf)));

      const base = this.serverUrl.value.replace(/\/$/, "");
      const body = {
        records: [{
          _base64:       b64,
          detector:      this.detSel.value,
          identifier:    this.idSel.value,
          heatmaps:       this.heatmapToggle?.checked ?? false,
          min_sharpness:  parseInt(this.sharpnessSlider?.value ?? "0", 10) / 100,
          min_confidence: parseInt(this.confidenceThreshold?.value ?? "0", 10) / 100,
        }],
      };

      const t0 = performance.now();
      const resp = await fetch(`${base}/v1/identify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await resp.json();
      const rtt = Math.round(performance.now() - t0);

      const rec = data.records?.[0] ?? null;
      this._handleResult(rec);
      this._updateTimingMetrics(rec?._timing ?? null, rtt);
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
      this._updateOverlay(null);
      this._updateBucketStatus(null);
      return;
    }
    this._updateSharpness(rec.sharpness ?? null);

    this._setDotState("scanning");

    // Update overlay (heatmaps behind corner outline)
    this._updateOverlay(rec);

    // Live crop preview
    if (rec.card_present && rec.crop_jpeg) {
      this.cropPrev.src = `data:image/jpeg;base64,${rec.crop_jpeg}`;
      this.cropPrev.classList.remove("hidden");
    } else if (!rec.card_present) {
      this.cropPrev.classList.add("hidden");
    }

    // Confidence badge is replaced by the canvas corner labels — nothing to do here.

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
    if (rec.crop_jpeg) {
      this.lastCrop.src = `data:image/jpeg;base64,${rec.crop_jpeg}`;
      this.lastCrop.style.display = "";
    } else {
      this.lastCrop.style.display = "none";
    }
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

  /** Clear overlay, draw optional heatmaps, then draw corner outline on top. */
  _updateOverlay(rec) {
    const cw = this.overlay.width;
    const ch = this.overlay.height;
    this.ctx.clearRect(0, 0, cw, ch);

    const corners = rec?.card_present ? rec.corners : null;
    const heatmaps = rec?.corner_heatmaps ?? null;

    if (heatmaps && this.heatmapToggle?.checked) {
      this._drawHeatmaps(heatmaps, cw, ch);
    }
    this._drawCornerLabels(rec, cw, ch);
    this._drawCornerOutline(corners, cw, ch);
  }

  /** Draw corner polygon + dots on the canvas (no clear). */
  _drawCornerOutline(corners, cw, ch) {
    if (!corners || corners.length !== 4) return;
    if (!cw || !ch) return;

    const pts = corners.map(([x, y]) => [x * cw, y * ch]);

    this.ctx.beginPath();
    this.ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i++) this.ctx.lineTo(pts[i][0], pts[i][1]);
    this.ctx.closePath();

    this.ctx.strokeStyle = "rgba(0, 230, 120, 0.9)";
    this.ctx.lineWidth   = 3;
    this.ctx.stroke();

    this.ctx.fillStyle = "rgba(0, 230, 120, 0.9)";
    for (const [px, py] of pts) {
      this.ctx.beginPath();
      this.ctx.arc(px, py, 5, 0, Math.PI * 2);
      this.ctx.fill();
    }
  }

  /**
   * Render SimCC heatmap blobs — one per corner, behind the corner outline.
   *
   * Each entry in cornerHeatmaps is a base64-encoded float32 array of length
   * 2 × num_bins: first half = X distribution, second half = Y distribution.
   * The 2D blob is the outer product of the X and Y distributions.
   *
   * Colors: TL=red, TR=cyan, BR=yellow, BL=magenta (40% max opacity).
   */
  _drawHeatmaps(cornerHeatmaps, cw, ch) {
    if (!cornerHeatmaps || cornerHeatmaps.length !== 4) return;
    if (!cw || !ch) return;

    const COLORS = [
      [255, 80,  80],   // TL — red
      [0,   200, 255],  // TR — cyan
      [255, 220, 0],    // BR — yellow
      [220, 80,  255],  // BL — magenta
    ];

    for (let ci = 0; ci < 4; ci++) {
      const dists = this._decodeF32(cornerHeatmaps[ci]);
      const numBins = dists.length >> 1;   // half for x, half for y
      const xDist = dists.subarray(0, numBins);
      const yDist = dists.subarray(numBins);

      // Peak value for normalization
      let maxX = 0, maxY = 0;
      for (let i = 0; i < numBins; i++) {
        if (xDist[i] > maxX) maxX = xDist[i];
        if (yDist[i] > maxY) maxY = yDist[i];
      }
      const maxVal = maxX * maxY;
      if (maxVal === 0) continue;

      const [r, g, b] = COLORS[ci];
      const opacity = (parseInt(this.heatmapOpacity?.value ?? "40", 10) / 100);

      // Render at bin resolution, then scale to canvas
      const offscreen = document.createElement("canvas");
      offscreen.width  = numBins;
      offscreen.height = numBins;
      const offCtx = offscreen.getContext("2d");
      const imgData = offCtx.createImageData(numBins, numBins);

      for (let py = 0; py < numBins; py++) {
        const yVal = yDist[py];
        for (let px = 0; px < numBins; px++) {
          const alpha = Math.round((xDist[px] * yVal / maxVal) * 255 * opacity);
          const idx = (py * numBins + px) << 2;
          imgData.data[idx]     = r;
          imgData.data[idx + 1] = g;
          imgData.data[idx + 2] = b;
          imgData.data[idx + 3] = alpha;
        }
      }

      offCtx.putImageData(imgData, 0, 0);
      this.ctx.drawImage(offscreen, 0, 0, cw, ch);
    }
  }

  /**
   * Draw stacked detector + identifier labels in the top-left corner.
   * Detector (Shp) always on top; identifier (Match) below.
   * Both green; red if below their respective thresholds.
   */
  _drawCornerLabels(rec, cw, ch) {
    if (!cw || !ch) return;

    const labels = [];

    if (this._lastSharpness != null) {
      const threshold = parseInt(this.sharpnessSlider?.value ?? "0", 10);
      const pct = (this._lastSharpness * 100).toFixed(1);
      const below = threshold > 0 && this._lastSharpness * 100 < threshold;
      labels.push({ text: `Shp ${pct}%`, below });
    }

    if (rec?.confidence != null) {
      const threshold = parseInt(this.confidenceThreshold?.value ?? "0", 10);
      const pct = Math.round(rec.confidence * 100);
      const below = threshold > 0 && pct < threshold;
      labels.push({ text: `Match ${pct}%`, below });
    }

    if (labels.length === 0) return;

    const pad = 6;
    const lineH = 20;
    this.ctx.font = "bold 12px monospace";
    this.ctx.textBaseline = "top";

    labels.forEach(({ text, below }, i) => {
      const tw = this.ctx.measureText(text).width;
      const x = pad;
      const y = pad + i * lineH;
      this.ctx.fillStyle = "rgba(0,0,0,0.6)";
      this.ctx.fillRect(x - 2, y - 1, tw + 8, 16);
      this.ctx.fillStyle = below ? "#e64a4a" : "#00e678";
      this.ctx.fillText(text, x + 2, y + 1);
    });
  }

  /** Decode a base64 string to a Float32Array. */
  _decodeF32(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Float32Array(bytes.buffer);
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

  // ------------------------------------------------------------------
  // Metrics
  // ------------------------------------------------------------------

  _updateTimingMetrics(timing, rttMs) {
    this._mRtt.textContent = `RTT ${rttMs}ms`;
    if (!timing) return;
    if (timing.detect_ms   != null) this._mDetect.textContent   = `Det ${timing.detect_ms}ms`;
    if (timing.identify_ms != null) this._mIdentify.textContent = `ID ${timing.identify_ms}ms`;
    if (timing.total_ms    != null) this._mServer.textContent   = `Srv ${timing.total_ms}ms`;
  }

  _updateSharpness(sharpness) {
    if (sharpness != null) this._lastSharpness = sharpness;
    if (sharpness == null) return;

    // Lazily create the Shp metric span the first time we get a sharpness value
    if (!this._mSharpness) {
      const bar = document.getElementById("metrics-bar");
      const sep = document.createElement("span");
      sep.className = "metric-sep";
      sep.textContent = "·";
      const span = document.createElement("span");
      span.className = "metric";
      span.id = "m-sharpness";
      span.title = "SimCC distribution sharpness (mean peak)";
      bar.insertBefore(sep, this._mRam);
      bar.insertBefore(span, this._mRam);
      this._mSharpness = span;
      this._syncDetectorUI();
    }

    const pct = (sharpness * 100).toFixed(1);
    const threshold = parseInt(this.sharpnessValue?.textContent ?? "0", 10);
    const below = threshold > 0 && sharpness * 100 < threshold;
    this._mSharpness.textContent = `Shp ${pct}%`;
    this._mSharpness.style.color = below ? "var(--danger)" : "var(--text-muted)";
  }

  _startMemoryPoll() {
    this._pollMemory();  // immediate first sample
    this._memPoll = setInterval(() => this._pollMemory(), 5000);
  }

  _stopMemoryPoll() {
    if (this._memPoll) {
      clearInterval(this._memPoll);
      this._memPoll = null;
    }
  }

  async _pollMemory() {
    const base = this.serverUrl.value.replace(/\/$/, "");
    try {
      const data = await fetch(`${base}/v1/memory`).then(r => r.json());
      if (data.process_rss_mb != null) {
        this._mRam.textContent = `RAM ${data.process_rss_mb}MB`;
      }
      const vramMb = data.device_driver_mb ?? data.device_allocated_mb;
      if (vramMb != null) {
        const label = data.device === "mps" ? "MPS" : "VRAM";
        this._mVram.textContent = ` ${label} ${vramMb}MB`;
        this._mVram.style.display = "";
      }
    } catch (_) {
      // memory endpoint optional — ignore errors
    }
  }
}


// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

const scanner = new Scanner();
