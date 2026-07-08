// zyf-WAN-Lora-Loader frontend extension
//
// A single-node paired LoRA loader for Wan 2.2:
//   * Folder-tree LoRA selector – navigate folders, pick a file → backend
//     auto-assigns first file as high, second as low (same folder).
//   * H / L weight boxes with [−] value [+] buttons (step = 0.05) and click-to-input.
//   * lora_* input slots hidden from node UI.
//   * Initial node size 480 x 600.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "WanLoraLoader";
const EXT_NAME = "zyf.WanLoraLoader";
const LORA_ROW_PREFIX = "lora_";
const ROW_HEIGHT = 40;
const WT_STEP = 0.05;
const INITIAL_NODE_WIDTH = 480;

// ---------------------------------------------------------------------------
// State + helpers
// ---------------------------------------------------------------------------

// Store the last right-click event on the canvas so we can position the
// context menu correctly. LiteGraph's ContextMenu uses event.clientX/Y.
let _lastCanvasMouseEvent = null;

// Capture pointerdown events on the canvas so we always have the latest
// mouse event for menu positioning.
document.addEventListener("pointerdown", (e) => {
  if (e.button === 2) {
    _lastCanvasMouseEvent = e;
  }
}, true);

let _loraTreeCache = null;
let _loraTreeTime = 0;
const CACHE_TTL_MS = 60_000;

async function fetchLoraTree(force = false) {
  const now = Date.now();
  if (!force && _loraTreeCache && now - _loraTreeTime < CACHE_TTL_MS) {
    return _loraTreeCache;
  }
  try {
    const res = await api.fetchApi("/zyf_wan_lora/lora-tree");
    const data = await res.json();
    _loraTreeCache = (data && data.tree) || [];
  } catch (err) {
    console.warn("[zyf-WAN-Lora-Loader] failed to fetch lora tree:", err);
    _loraTreeCache = [];
  }
  _loraTreeTime = now;
  return _loraTreeCache;
}

async function findPair(loraFilename, side = "auto") {
  if (!loraFilename) return {};
  try {
    const res = await api.fetchApi(
      "/zyf_wan_lora/find-pair?lora=" + encodeURIComponent(loraFilename) +
      "&side=" + encodeURIComponent(side)
    );
    return await res.json();
  } catch (err) {
    console.warn("[zyf-WAN-Lora-Loader] find-pair failed:", err);
    return {};
  }
}

async function fetchTxt(loraFilename) {
  if (!loraFilename) return { found: false };
  try {
    const res = await api.fetchApi(
      "/zyf_wan_lora/find-txt?lora=" + encodeURIComponent(loraFilename)
    );
    return await res.json();
  } catch (err) {
    console.warn("[zyf-WAN-Lora-Loader] find-txt failed:", err);
    return { found: false };
  }
}

async function saveTxt(txtPath, content, encoding) {
  try {
    const res = await api.fetchApi("/zyf_wan_lora/save-txt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: txtPath, content, encoding: encoding || "utf-8" }),
    });
    return await res.json();
  } catch (err) {
    console.warn("[zyf-WAN-Lora-Loader] save-txt failed:", err);
    return { success: false, error: String(err) };
  }
}

async function openLoraFolder(loraFilename) {
  if (!loraFilename) return { success: false, error: "No LoRA selected" };
  try {
    const res = await api.fetchApi(
      "/zyf_wan_lora/open-folder?lora=" + encodeURIComponent(loraFilename)
    );
    return await res.json();
  } catch (err) {
    console.warn("[zyf-WAN-Lora-Loader] open-folder failed:", err);
    return { success: false, error: String(err) };
  }
}

function makeEmptyRow(index) {
  return {
    index,
    on: true,
    lora: null,
    lora_high: null,
    lora_low: null,
    strength_high: 1.0,
    strength_low: 1.0,
    // Track whether the user explicitly cleared a side via the context
    // menu, so `rebuildRows` can respect that choice on refresh / new row
    // instead of silently re-pairing it.
    cleared_high: false,
    cleared_low: false,
  };
}

function roundWt(v) {
  return Math.round(v * 100) / 100;
}

function pointInBounds(pos, bounds) {
  if (!bounds) return false;
  const xStart = bounds[0];
  const xEnd = xStart + (bounds.length > 2 ? bounds[2] : bounds[1]);
  const clickedX = pos[0] >= xStart && pos[0] <= xEnd;
  if (bounds.length === 2) return clickedX;
  return clickedX && pos[1] >= bounds[1] && pos[1] <= bounds[1] + bounds[3];
}

function basename(path) {
  if (!path) return "";
  const i = path.lastIndexOf("/");
  const j = path.lastIndexOf("\\");
  const sep = i >= 0 && j >= 0 ? Math.max(i, j) : Math.max(i, j);
  return sep >= 0 ? path.slice(sep + 1) : path;
}

/**
 * Return "parentFolder/filename" from a relative path, omitting the full tree.
 * E.g. "WAN/wan2.2/26增强走路姿势/Wan2.2-i2v_Normal-H.safetensors"
 *   -> "26增强走路姿势/Wan2.2-i2v_Normal-H.safetensors"
 */
function shortPath(path) {
  if (!path) return "";
  // Normalize to forward slashes.
  const p = path.replace(/\\/g, "/");
  const parts = p.split("/");
  if (parts.length <= 1) return p;
  return parts.slice(-2).join("/");
}

/**
 * Build a LiteGraph ContextMenu from the folder tree.
 * Folders open sub-menus; files call `onPick(path)`.
 */
function buildTreeMenu(tree, onPick) {
  const items = [];
  for (const node of tree) {
    if (node.type === "folder") {
      items.push({
        content: "📁 " + node.name,
        has_submenu: true,
        submenu: {
          options: buildTreeMenu(node.children || [], onPick),
        },
      });
    } else {
      items.push({
        content: node.name,
        value: node.path,
        callback: () => onPick(node.path),
      });
    }
  }
  return items;
}

// ---------------------------------------------------------------------------
// Base widget
// ---------------------------------------------------------------------------

class ZyfBaseWidget {
  constructor(name) {
    this.name = name;
    this.type = "custom";
    this.options = {};
    this.y = 0;
    this.last_y = 0;
    this.hitAreas = {};
    this._activeAreas = [];
  }

  serializeValue(_node, _index) {
    return this.value;
  }

  computeSize(width) {
    return [width, LiteGraph.NODE_WIDGET_HEIGHT || 20];
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown") {
      this._activeAreas = [];
      let anyHandled = false;
      for (const part of Object.values(this.hitAreas)) {
        if (!part || !part.bounds) continue;
        if (pointInBounds(pos, part.bounds)) {
          this._activeAreas.push(part);
          if (part.onMove) part._moveActive = true;
          if (part.onDown) {
            const r = part.onDown.call(this, event, pos, node, part);
            if (r) anyHandled = true;
          }
        }
      }
      return anyHandled;
    }

    if (event.type === "pointermove") {
      let anyHandled = false;
      for (const part of Object.values(this.hitAreas)) {
        if (part && part.onMove && part._moveActive) {
          const r = part.onMove.call(this, event, pos, node, part);
          if (r) anyHandled = true;
        }
      }
      return anyHandled;
    }

    if (event.type === "pointerup") {
      let anyHandled = false;
      for (const part of Object.values(this.hitAreas)) {
        if (!part || !part.bounds) continue;
        if (pointInBounds(pos, part.bounds)) {
          if (part.onUp) {
            const r = part.onUp.call(this, event, pos, node, part);
            if (r) anyHandled = true;
          }
          if (part.onClick && this._activeAreas.includes(part)) {
            const r = part.onClick.call(this, event, pos, node, part);
            if (r) anyHandled = true;
          }
        }
        if (part) part._moveActive = false;
      }
      this._activeAreas = [];
      return anyHandled;
    }

    return false;
  }
}

// ---------------------------------------------------------------------------
// LoRA row widget
// ---------------------------------------------------------------------------

class WanLoraRowWidget extends ZyfBaseWidget {
  constructor(name) {
    super(name);
    this.value = makeEmptyRow(parseInt(name.split("_")[1], 10) || 1);
    this.options = { serialize: true };
    this.h = ROW_HEIGHT;
    this._loraTree = [];
    this._pairInfo = {};
    this._wtDragMoved = false;
    this.hitAreas = {
      toggle:  { bounds: [0, 0], onDown: this.onToggleDown },
      loraHighSel: { bounds: [0, 0], onClick: this.onLoraHighSelectClick },
      loraLowSel:  { bounds: [0, 0], onClick: this.onLoraLowSelectClick },
      wtHighDec: { bounds: [0, 0], onClick: this.onWtHighDec },
      wtHighInc: { bounds: [0, 0], onClick: this.onWtHighInc },
      wtHighVal: { bounds: [0, 0], onDown: this.onWtHighDown, onMove: this.onWtHighMove, onUp: this.onWtHighUp, onClick: this.onWtHighClick },
      wtLowDec:  { bounds: [0, 0], onClick: this.onWtLowDec },
      wtLowInc:  { bounds: [0, 0], onClick: this.onWtLowInc },
      wtLowVal:  { bounds: [0, 0], onDown: this.onWtLowDown, onMove: this.onWtLowMove, onUp: this.onWtLowUp, onClick: this.onWtLowClick },
    };
  }

  setLoraTree(tree) {
    this._loraTree = tree || [];
  }

  computeSize(width) {
    return [width, this.h];
  }

  onToggleDown(_event, _pos, node) {
    this.value.on = !this.value.on;
    node.setDirtyCanvas(true, true);
    return true;
  }

  async onLoraHighSelectClick(_event, _pos, node) {
    const tree = await fetchLoraTree();
    this.setLoraTree(tree);
    const items = [
      { content: "— clear high —", value: null, callback: () => {
        this.value.lora_high = null;
        this.value.lora = null;
        // Mark as explicitly cleared so rebuildRows won't re-pair it.
        this.value.cleared_high = true;
        node.setDirtyCanvas(true, true);
      }},
      null,
    ];
    const treeItems = buildTreeMenu(this._loraTree, async (path) => {
      // Assign selected file to high noise; the other sibling becomes low noise.
      this.value.lora_high = path;
      this.value.lora = path;
      // Active high selection: drop the cleared flag.
      this.value.cleared_high = false;
      this.value.cleared_low = false;
      const info = await findPair(path, "auto");
      this._pairInfo = info;
      const siblings = info.siblings || [];
      const other = siblings.find(f => f !== path);
      this.value.lora_low = other || null;
      node.setDirtyCanvas(true, true);
    });
    items.push(...treeItems);
    new LiteGraph.ContextMenu(items, { event: _event, title: "Select High-Noise LoRA" });
    return true;
  }

  async onLoraLowSelectClick(_event, _pos, node) {
    const tree = await fetchLoraTree();
    this.setLoraTree(tree);
    const items = [
      { content: "— clear low —", value: null, callback: () => {
        this.value.lora_low = null;
        // Mark as explicitly cleared so rebuildRows won't re-pair it.
        this.value.cleared_low = true;
        node.setDirtyCanvas(true, true);
      }},
      null,
    ];
    const treeItems = buildTreeMenu(this._loraTree, async (path) => {
      // Assign selected file to low noise; the other sibling becomes high noise.
      this.value.lora_low = path;
      // Active low selection: drop the cleared flag.
      this.value.cleared_low = false;
      this.value.cleared_high = false;
      const info = await findPair(path, "auto");
      this._pairInfo = info;
      const siblings = info.siblings || [];
      const other = siblings.find(f => f !== path);
      this.value.lora_high = other || null;
      node.setDirtyCanvas(true, true);
    });
    items.push(...treeItems);
    new LiteGraph.ContextMenu(items, { event: _event, title: "Select Low-Noise LoRA" });
    return true;
  }

  onWtHighDec(_event, _pos, node) {
    this.value.strength_high = roundWt((this.value.strength_high ?? 1) - WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtHighInc(_event, _pos, node) {
    this.value.strength_high = roundWt((this.value.strength_high ?? 1) + WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtHighClick(_event, _pos, node) {
    if (this._wtDragMoved) return true;
    const canvas = app.canvas;
    if (canvas && canvas.prompt) {
      canvas.prompt("Value", String(this.value.strength_high ?? 1), (v) => {
        const n = parseFloat(v);
        if (!isNaN(n)) {
          this.value.strength_high = roundWt(n);
          node.setDirtyCanvas(true, true);
        }
      }, _event);
    }
    return true;
  }
  onWtHighDown(_event, _pos, _node) {
    this._wtDragMoved = false;
    this._wtStartX = _event.clientX || _pos[0];
    this._wtStartVal = this.value.strength_high ?? 1;
    return true;
  }
  onWtHighMove(_event, _pos, node) {
    const clientX = _event.clientX != null ? _event.clientX : _pos[0];
    const dx = clientX - this._wtStartX;
    if (Math.abs(dx) > 2) this._wtDragMoved = true;
    this.value.strength_high = roundWt(this._wtStartVal + dx * WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtHighUp(_event, _pos, node) {
    node.setDirtyCanvas(true, true);
    return true;
  }

  onWtLowDec(_event, _pos, node) {
    this.value.strength_low = roundWt((this.value.strength_low ?? 1) - WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtLowInc(_event, _pos, node) {
    this.value.strength_low = roundWt((this.value.strength_low ?? 1) + WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtLowClick(_event, _pos, node) {
    if (this._wtDragMoved) return true;
    const canvas = app.canvas;
    if (canvas && canvas.prompt) {
      canvas.prompt("Value", String(this.value.strength_low ?? 1), (v) => {
        const n = parseFloat(v);
        if (!isNaN(n)) {
          this.value.strength_low = roundWt(n);
          node.setDirtyCanvas(true, true);
        }
      }, _event);
    }
    return true;
  }
  onWtLowDown(_event, _pos, _node) {
    this._wtDragMoved = false;
    this._wtStartX = _event.clientX || _pos[0];
    this._wtStartVal = this.value.strength_low ?? 1;
    return true;
  }
  onWtLowMove(_event, _pos, node) {
    const clientX = _event.clientX != null ? _event.clientX : _pos[0];
    const dx = clientX - this._wtStartX;
    if (Math.abs(dx) > 2) this._wtDragMoved = true;
    this.value.strength_low = roundWt(this._wtStartVal + dx * WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtLowUp(_event, _pos, node) {
    node.setDirtyCanvas(true, true);
    return true;
  }

  // Right-click context menu
  onContextMenu(event, _pos, node) {
    const items = [];

    // Delete row option
    items.push({
      content: " Delete this row",
      callback: () => {
        const idx = node.widgets.indexOf(this);
        if (idx >= 0) {
          node.widgets.splice(idx, 1);
          // Reindex remaining rows
          const rows = collectRowWidgets(node);
          rows.forEach((r, i) => {
            r.value.index = i + 1;
            r.name = `${LORA_ROW_PREFIX}${i + 1}`;
          });
          node.size[1] = Math.max(node.size[1], node.computeSize()[1]);
          node.setDirtyCanvas(true, true);
        }
      }
    });

    // Open LoRA folder option (uses whichever side is selected)
    {
      const loraForFolder = this.value.lora_high || this.value.lora_low;
      items.push({
        content: "📁 Open LoRA Folder",
        disabled: !loraForFolder,
        callback: () => {
          if (loraForFolder) openLoraFolder(loraForFolder);
        },
      });
    }

    // View TXT file option (only if LoRA is selected)
    if (this.value.lora) {
      items.push({
        content: "📄 View TXT",
        callback: () => this.openTxtViewer(node)
      });
    } else {
      items.push({
        content: " View TXT (disabled)",
        disabled: true
      });
    }

    new LiteGraph.ContextMenu(items, {
      event: event,
      title: "Row Options"
    });
    return true;
  }

  async openTxtViewer(node) {
    const loraFile = this.value.lora_high || this.value.lora_low;
    if (!loraFile) return;

    const result = await fetchTxt(loraFile);
    if (!result.found) {
      console.warn("[zyf-WAN-Lora-Loader] No TXT file found for:", this.value.lora);
      return;
    }

    const txtPath = result.path;
    const txtEncoding = result.encoding || "utf-8";

    // Create modal dialog
    const modal = document.createElement("div");
    modal.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      justify-content: center;
      align-items: flex-start;
      padding-top: 5vh;
      z-index: 10000;
    `;

    const dialog = document.createElement("div");
    dialog.style.cssText = `
      background: #1e1e1e;
      border: 1px solid #3a3f4a;
      border-radius: 8px;
      padding: 20px;
      width: 85%;
      max-width: 900px;
      height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    `;

    // Header row: title + buttons
    const header = document.createElement("div");
    header.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      padding-bottom: 10px;
      border-bottom: 1px solid #3a3f4a;
    `;

    const title = document.createElement("div");
    title.style.cssText = `
      color: #d4d4d4;
      font-size: 14px;
      font-weight: bold;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-right: 10px;
    `;
    title.textContent = `TXT: ${txtPath}`;

    const btnGroup = document.createElement("div");
    btnGroup.style.cssText = `
      display: flex;
      gap: 8px;
      flex-shrink: 0;
    `;

    const saveBtn = document.createElement("button");
    saveBtn.style.cssText = `
      background: #2b6cb0;
      color: white;
      border: none;
      padding: 5px 16px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    `;
    saveBtn.textContent = " Save";

    const closeBtn = document.createElement("button");
    closeBtn.style.cssText = `
      background: #a04040;
      color: white;
      border: none;
      padding: 5px 15px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    `;
    closeBtn.textContent = "Close";

    btnGroup.appendChild(saveBtn);
    btnGroup.appendChild(closeBtn);
    header.appendChild(title);
    header.appendChild(btnGroup);

    // Status bar (for save feedback)
    const statusBar = document.createElement("div");
    statusBar.style.cssText = `
      color: #98c379;
      font-size: 12px;
      margin-bottom: 8px;
      min-height: 18px;
      display: none;
    `;

    // Editable textarea
    const textarea = document.createElement("textarea");
    textarea.style.cssText = `
      flex: 1;
      background: #2a2f38;
      color: #d4d4d4;
      border: 1px solid #3a3f4a;
      border-radius: 4px;
      padding: 10px;
      font-family: monospace;
      font-size: 12px;
      resize: none;
      outline: none;
      white-space: pre-wrap;
      word-wrap: break-word;
    `;
    textarea.value = result.content;
    textarea.readOnly = false;

    // Track changes to show unsaved indicator
    let hasChanges = false;
    textarea.addEventListener("input", () => {
      hasChanges = true;
      statusBar.style.display = "block";
      statusBar.style.color = "#e5c07b";
      statusBar.textContent = "● Unsaved changes";
    });

    // Save handler
    saveBtn.onclick = async () => {
      saveBtn.disabled = true;
      statusBar.style.display = "block";
      statusBar.style.color = "#e5c07b";
      statusBar.textContent = "Saving...";

      const res = await saveTxt(txtPath, textarea.value, txtEncoding);

      if (res.success) {
        hasChanges = false;
        statusBar.style.color = "#98c379";
        statusBar.textContent = "Saved successfully";
        setTimeout(() => { statusBar.style.display = "none"; }, 2000);
      } else {
        statusBar.style.color = "#e06c75";
        statusBar.textContent = "Save failed: " + (res.error || "Unknown error");
      }
      saveBtn.disabled = false;
    };

    // Close handler – warn if unsaved
    closeBtn.onclick = () => {
      if (hasChanges) {
        if (confirm("You have unsaved changes. Close anyway?")) {
          document.body.removeChild(modal);
        }
      } else {
        document.body.removeChild(modal);
      }
    };

    dialog.appendChild(header);
    dialog.appendChild(statusBar);
    dialog.appendChild(textarea);
    modal.appendChild(dialog);

    modal.onclick = (e) => {
      if (e.target === modal) {
        if (hasChanges) {
          if (confirm("You have unsaved changes. Close anyway?")) {
            document.body.removeChild(modal);
          }
        } else {
          document.body.removeChild(modal);
        }
      }
    };

    document.body.appendChild(modal);
    textarea.focus();
  }

  draw(ctx, node, width, y, height) {
    const margin = 6;
    const innerMargin = 4;
    const accent = LiteGraph.WIDGET_OUTLINE_COLOR || "#3a3f4a";
    const bg = LiteGraph.WIDGET_BGCOLOR || "#1c1f25";
    const text = LiteGraph.WIDGET_TEXT_COLOR || "#d4d4d4";
    const muted = "#7f8794";
    const blue = "#5b8def";
    const green = "#98c379";
    const redOn = "#e06c75";
    const purple = "#c678dd";

    // Background row.
    ctx.fillStyle = this.value.index % 2 === 0 ? "#262b34" : "#1f232b";
    ctx.fillRect(0, y, width, height);
    ctx.strokeStyle = "#2a2f38";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y + height - 0.5);
    ctx.lineTo(width, y + height - 0.5);
    ctx.stroke();

    let posX = margin;

    // --- Toggle -----------------------------------------------------
    const tw = 28;
    const th = 20;
    const trackX = posX;
    const trackY = y + (height - th) / 2;
    ctx.fillStyle = this.value.on ? green : redOn;
    ctx.globalAlpha = this.value.on ? 1.0 : 0.4;
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(trackX, trackY, tw, th, th / 2);
    ctx.fill();
    ctx.stroke();
    const knobR = th * 0.35;
    const knobX = this.value.on ? trackX + tw - th / 2 : trackX + th / 2;
    const knobY = trackY + th / 2;
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(knobX, knobY, knobR, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1.0;
    this.hitAreas.toggle.bounds = [trackX, trackY, tw, th];
    posX += tw + innerMargin;

    // --- LoRA selector area: two stacked boxes --------------------
    const selAreaW = width - posX - margin - 76 - innerMargin;
    const boxH = 17;
    const gap = 0;  // tightly adjacent
    const boxHighY = y + 2;
    const boxLowY = boxHighY + boxH + gap;

    const highName = this.value.lora_high || "(no high-noise pair)";
    const lowName = this.value.lora_low || "(no low-noise pair)";
    const hasHigh = !!this.value.lora_high;
    const hasLow = !!this.value.lora_low;

    const maxTextW = selAreaW - 20;

    // --- High noise box (purple-ish background) --------------------
    const highBoxBg = "#261a38";
    const highBoxBorder = "#7b529e";
    ctx.fillStyle = highBoxBg;
    ctx.strokeStyle = highBoxBorder;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(posX, boxHighY, selAreaW, boxH, 3);
    ctx.fill();
    ctx.stroke();
    this.hitAreas.loraHighSel.bounds = [posX, boxHighY, selAreaW, boxH];

    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = "10px monospace";
    ctx.fillStyle = this.value.on ? (hasHigh ? "#c678dd" : muted) : muted;
    ctx.fillText(truncateText(ctx, "H: " + shortPath(highName), maxTextW), posX + 6, boxHighY + boxH / 2);

    // High box dropdown arrow
    ctx.fillStyle = muted;
    ctx.font = "8px sans-serif";
    ctx.fillText("▼", posX + selAreaW - 14, boxHighY + boxH / 2);

    // --- Low noise box (blue-ish background) -----------------------
    const lowBoxBg = "#1a2740";
    const lowBoxBorder = "#3b5f9e";
    ctx.fillStyle = lowBoxBg;
    ctx.strokeStyle = lowBoxBorder;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(posX, boxLowY, selAreaW, boxH, 3);
    ctx.fill();
    ctx.stroke();
    this.hitAreas.loraLowSel.bounds = [posX, boxLowY, selAreaW, boxH];

    ctx.font = "10px monospace";
    ctx.fillStyle = this.value.on ? (hasLow ? "#5b8def" : muted) : muted;
    ctx.fillText(truncateText(ctx, "L: " + shortPath(lowName), maxTextW), posX + 6, boxLowY + boxH / 2);

    // Low box dropdown arrow
    ctx.fillStyle = muted;
    ctx.font = "8px sans-serif";
    ctx.fillText("▼", posX + selAreaW - 14, boxLowY + boxH / 2);

    posX += selAreaW + innerMargin;

    // --- Weight area: aligned with LoRA boxes --------------------
    const wtW = 76;
    const wtH = 15;
    const btnW = 17;

    // High noise weight - aligned with high noise box
    const wtHighY = boxHighY + (boxH - wtH) / 2;
    drawWeightControl(ctx, this.value.on, this.value.strength_high, null,
      posX, wtHighY, wtW, wtH, btnW, accent, bg, text, blue, muted);
    this.hitAreas.wtHighDec.bounds = [posX, wtHighY, btnW, wtH];
    this.hitAreas.wtHighVal.bounds = [posX + btnW, wtHighY, wtW - btnW * 2, wtH];
    this.hitAreas.wtHighInc.bounds = [posX + wtW - btnW, wtHighY, btnW, wtH];

    // Low noise weight - aligned with low noise box
    const wtLowY = boxLowY + (boxH - wtH) / 2;
    drawWeightControl(ctx, this.value.on, this.value.strength_low, null,
      posX, wtLowY, wtW, wtH, btnW, accent, bg, text, blue, muted);
    this.hitAreas.wtLowDec.bounds = [posX, wtLowY, btnW, wtH];
    this.hitAreas.wtLowVal.bounds = [posX + btnW, wtLowY, wtW - btnW * 2, wtH];
    this.hitAreas.wtLowInc.bounds = [posX + wtW - btnW, wtLowY, btnW, wtH];
  }

  serializeValue(_node, _index) {
    return {
      index: this.value.index,
      on: this.value.on !== false,
      lora: this.value.lora_high || this.value.lora,
      lora_high: this.value.lora_high,
      lora_low: this.value.lora_low,
      strength_high: this.value.strength_high ?? 1.0,
      strength_low: this.value.strength_low ?? 1.0,
      // Persist the user-intent "cleared" flags so they survive
      // page reloads / workflow saves.
      cleared_high: this.value.cleared_high === true,
      cleared_low: this.value.cleared_low === true,
    };
  }
}

// ---------------------------------------------------------------------------
// Weight control drawing helper
// ---------------------------------------------------------------------------

function drawWeightControl(ctx, on, value, label, x, y, w, h, btnW, accent, bg, text, blue, muted) {
  ctx.fillStyle = bg;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 4);
  ctx.fill();
  ctx.stroke();

  const midY = y + h / 2;

  ctx.fillStyle = on ? "#2a2f38" : "#1e2128";
  ctx.beginPath();
  ctx.roundRect(x, y + 1, btnW, h - 2, 3);
  ctx.fill();
  ctx.strokeStyle = accent;
  ctx.stroke();

  ctx.fillStyle = on ? "#2a2f38" : "#1e2128";
  ctx.beginPath();
  ctx.roundRect(x + w - btnW, y + 1, btnW, h - 2, 3);
  ctx.fill();
  ctx.strokeStyle = accent;
  ctx.stroke();

  ctx.fillStyle = on ? blue : muted;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "bold 12px sans-serif";
  ctx.fillText("−", x + btnW / 2, midY);
  ctx.fillText("+", x + w - btnW / 2, midY);

  ctx.fillStyle = on ? text : muted;
  ctx.font = "11px monospace";
  ctx.fillText(Number(value || 0).toFixed(2), x + w / 2, midY);

  // Draw label only if provided (e.g. "H" or "L").
  if (label) {
    ctx.fillStyle = muted;
    ctx.font = "8px monospace";
    ctx.fillText(label, x + w / 2, y + h * 0.2);
  }
}

// ---------------------------------------------------------------------------
// Button + title widgets
// ---------------------------------------------------------------------------

class ZyfButtonWidget extends ZyfBaseWidget {
  constructor(name, label, color, onClick) {
    super(name);
    this.label = label;
    this.color = color;
    this._onClick = onClick;
    this.value = label;
    this.options = { serialize: false };
    this.isMouseDownedAndOver = false;
    this.hitAreas = {
      button: { bounds: [0, 0], onDown: this._onDown, onUp: this._onUp, onClick: this._onClickHandler },
    };
  }

  computeSize(width) {
    return [width, 30];
  }

  _onDown(_event, _pos, _node) {
    this.isMouseDownedAndOver = true;
    return true;
  }

  _onUp(_event, _pos, _node) {
    this.isMouseDownedAndOver = false;
    return true;
  }

  _onClickHandler(event, pos, node) {
    return this._onClick(event, pos, node);
  }

  draw(ctx, _node, width, y, height) {
    const margin = 6;
    const radius = 4;
    ctx.fillStyle = this.isMouseDownedAndOver ? this._darken(this.color) : this.color;
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR || "#3a3f4a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(margin, y + 3, width - margin * 2, height - 6, radius);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.font = "11px sans-serif";
    ctx.fillText(this.label, width / 2, y + height / 2);
    this.hitAreas.button.bounds = [margin, y + 3, width - margin * 2, height - 6];
  }

  _darken(hex) {
    const m = hex.match(/^#?([0-9a-f]{6})$/i);
    if (!m) return hex;
    const n = parseInt(m[1], 16);
    const r = Math.round(((n >> 16) & 255) * 0.8);
    const g = Math.round(((n >> 8) & 255) * 0.8);
    const b = Math.round((n & 255) * 0.8);
    return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
  }
}

class ZyfTitleWidget extends ZyfBaseWidget {
  constructor(name, text) {
    super(name);
    this.text = text;
    this.value = text;
    this.options = { serialize: false };
  }

  computeSize(width) {
    return [width, 22];
  }

  draw(ctx, _node, width, y, height) {
    ctx.fillStyle = "#1f232b";
    ctx.fillRect(0, y, width, height);
    ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR || "#d4d4d4";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = "bold 12px sans-serif";
    ctx.fillText(this.text, 8, y + height / 2);
  }
}

// ---------------------------------------------------------------------------
// Small drawing utilities
// ---------------------------------------------------------------------------

function truncateText(ctx, text, maxW) {
  if (!text) return "";
  if (ctx.measureText(text).width <= maxW) return text;
  let cut = text.length;
  while (cut > 4 && ctx.measureText(text.slice(0, cut) + "…").width > maxW) {
    cut -= 2;
  }
  return text.slice(0, cut) + "…";
}

// ---------------------------------------------------------------------------
// Node wiring
// ---------------------------------------------------------------------------

function addCustomWidget(node, widget) {
  if (typeof node.addCustomWidget === "function") {
    return node.addCustomWidget(widget);
  }
  node.widgets = node.widgets || [];
  node.widgets.push(widget);
  return widget;
}

function moveWidgetTo(node, widget, index) {
  node.widgets = node.widgets || [];
  const cur = node.widgets.indexOf(widget);
  if (cur >= 0) node.widgets.splice(cur, 1);
  node.widgets.splice(index, 0, widget);
}

function collectRowWidgets(node) {
  return (node.widgets || []).filter((w) => w instanceof WanLoraRowWidget);
}

function addRowWidget(node, row) {
  const widget = new WanLoraRowWidget(`${LORA_ROW_PREFIX}${row.index}`);
  widget.value = { ...makeEmptyRow(row.index), ...row };
  // No input slot needed – ComfyUI serializes widget values directly into
  // the prompt via serializeValue(), and the backend receives them as
  // keyword arguments (lora_1, lora_2, ...).
  addCustomWidget(node, widget);
  const btnIdx = node.widgets.findIndex(
    (w) => w && w.name === "_zyf_btn_add_lora"
  );
  if (btnIdx >= 0) {
    moveWidgetTo(node, widget, btnIdx);
  }
  return widget;
}

function ensureHeader(node) {
  node.widgets = node.widgets || [];
  if (node._zyfHeaderDone) return;
  node._zyfHeaderDone = true;

  addCustomWidget(node, new ZyfButtonWidget(
    "_zyf_btn_add_lora", "+ Add LoRA", "#2b6cb0",
    (_event, _pos, n) => {
      const existing = collectRowWidgets(n);
      const nextIdx = existing.length
        ? Math.max(...existing.map((r) => r.value.index)) + 1
        : 1;
      const newRow = addRowWidget(n, makeEmptyRow(nextIdx));
      // Grow height by one row, preserve width.
      n.size[1] = Math.max(n.size[1], n.computeSize()[1]);
      n.setDirtyCanvas(true, true);
      rebuildRows(n);
      // Immediately open the LoRA selector for the new row.
      if (newRow && newRow.onLoraHighSelectClick) {
        newRow.onLoraHighSelectClick(_event, _pos, n);
      }
      return true;
    }
  ));
}

function ensureRowsAndButtons(node) {
  ensureHeader(node);
  if (collectRowWidgets(node).length === 0) {
    addRowWidget(node, makeEmptyRow(1));
  }
}

async function rebuildRows(node) {
  const tree = await fetchLoraTree();
  for (const r of collectRowWidgets(node)) {
    r.setLoraTree(tree);
    if (r.value.lora_high) {
      // High is set → verify it exists and re-pair low to it.
      const info = await findPair(r.value.lora_high, "high");
      r._pairInfo = info;
      r.value.lora_high = info.lora_high || null;
      r.value.lora_low = info.lora_low || null;
      // Active selections on both sides: drop any stale cleared flags.
      r.value.cleared_high = false;
      r.value.cleared_low = false;
    } else if (r.value.lora_low) {
      // Only low is set. Look up the auto-pair, but respect the user's
      // choice to keep high cleared.
      const info = await findPair(r.value.lora_low, "low");
      r._pairInfo = info;
      r.value.lora_low = info.lora_low || null;
      if (!r.value.cleared_high) {
        r.value.lora_high = info.lora_high || null;
      } else {
        // Keep high cleared per user intent.
        r.value.lora_high = null;
      }
      r.value.cleared_low = false;
    }
  }
  node.setDirtyCanvas(true, true);
}

// ---------------------------------------------------------------------------
// Extension registration
// ---------------------------------------------------------------------------

/**
 * Show a context menu for a LoRA row widget.
 * Called from getSlotMenuOptions when a right-click lands on a LoRA row.
 */
function showRowContextMenu(targetWidget, node) {
  const index = node.widgets.indexOf(targetWidget);
  const rows = collectRowWidgets(node);
  const canMoveUp = index > 0 && rows.indexOf(node.widgets[index - 1]) >= 0;
  const canMoveDown = index < node.widgets.length - 1 && rows.indexOf(node.widgets[index + 1]) >= 0;

  const menuItems = [
    {
      content: "📄 View TXT",
      disabled: !targetWidget.value.lora_high && !targetWidget.value.lora_low,
      callback: () => {
        if (targetWidget.value.lora_high || targetWidget.value.lora_low) {
          targetWidget.openTxtViewer(node);
        }
      },
    },
    {
      content: "📁 Open LoRA Folder",
      disabled: !targetWidget.value.lora_high && !targetWidget.value.lora_low,
      callback: () => {
        const f = targetWidget.value.lora_high || targetWidget.value.lora_low;
        if (f) openLoraFolder(f);
      },
    },
    null,
    {
      content: "⬆️ Move Up",
      disabled: !canMoveUp,
      callback: () => {
        moveWidgetTo(node, targetWidget, index - 1);
        // Reindex all rows to match the new widget order.
        const remaining = collectRowWidgets(node);
        remaining.forEach((r, i) => {
          r.value.index = i + 1;
          r.name = `${LORA_ROW_PREFIX}${i + 1}`;
        });
        rebuildRows(node);
        node.setDirtyCanvas(true, true);
      },
    },
    {
      content: "⬇️ Move Down",
      disabled: !canMoveDown,
      callback: () => {
        moveWidgetTo(node, targetWidget, index + 1);
        // Reindex all rows to match the new widget order.
        const remaining = collectRowWidgets(node);
        remaining.forEach((r, i) => {
          r.value.index = i + 1;
          r.name = `${LORA_ROW_PREFIX}${i + 1}`;
        });
        rebuildRows(node);
        node.setDirtyCanvas(true, true);
      },
    },
    null,
    {
      content: "🗑️ Remove",
      callback: () => {
        node.widgets.splice(index, 1);
        const remaining = collectRowWidgets(node);
        remaining.forEach((r, i) => {
          r.value.index = i + 1;
          r.name = `${LORA_ROW_PREFIX}${i + 1}`;
        });
        node.size[1] = Math.max(node.size[1], node.computeSize()[1]);
        node.setDirtyCanvas(true, true);
      },
    },
  ];

  new LiteGraph.ContextMenu(menuItems, {
    title: "LoRA Row",
    event: _lastCanvasMouseEvent,
  });
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== NODE_TYPE) return;

    nodeType.title = "Wan 2.2 LoRA Loader (zyf)";

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
      const node = this;
      node.serialize_widgets = true;
      ensureRowsAndButtons(node);
      rebuildRows(node).catch((err) =>
        console.error("[zyf-WAN-Lora-Loader] rebuild failed:", err)
      );
      node.size[0] = INITIAL_NODE_WIDTH;
      node.size[1] = node.computeSize()[1];
      node.setDirtyCanvas(true, true);
      return r;
    };

    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const node = this;
      node.serialize_widgets = true;

      node.widgets = (node.widgets || []).filter(
        (w) => !(w instanceof ZyfButtonWidget || w instanceof ZyfTitleWidget || w instanceof WanLoraRowWidget)
      );
      node._zyfHeaderDone = false;

      const r = origConfigure ? origConfigure.call(this, info) : undefined;

      const wv = (info && info.widgets_values) || [];
      const restored = [];
      for (const v of wv) {
        if (
          v && typeof v === "object" && typeof v.index === "number" &&
          ("lora" in v || "lora_high" in v || "lora_low" in v || "strength_high" in v)
        ) {
          restored.push({ ...makeEmptyRow(v.index), ...v });
        }
      }

      ensureHeader(node);
      if (restored.length) {
        for (const row of restored.sort((a, b) => a.index - b.index)) {
          addRowWidget(node, row);
        }
      } else {
        addRowWidget(node, makeEmptyRow(1));
      }

      node.size[0] = Math.max(node.size[0] || INITIAL_NODE_WIDTH, INITIAL_NODE_WIDTH);
      node.size[1] = Math.max(node.size[1] || 0, node.computeSize()[1]);
      node.setDirtyCanvas(true, true);
      rebuildRows(node);

      return r;
    };

    // Right-click context menu for LoRA rows.
    // LiteGraph handles right-click via pointerdown, NOT contextmenu.
    // We follow rgthree's approach: override getSlotInPosition to detect
    // when the click is on a LoRA row widget, then override
    // getSlotMenuOptions to show our custom menu and suppress the default.
    const origGetSlotInPosition = nodeType.prototype.getSlotInPosition;
    nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
      // Check our custom row widgets first.
      for (const w of (this.widgets || [])) {
        if (w instanceof WanLoraRowWidget && w.last_y != null) {
          const wTop = this.pos[1] + w.last_y;
          const wH = w.computeSize(this.size[0])[1];
          if (canvasX >= this.pos[0] && canvasX <= this.pos[0] + this.size[0] &&
              canvasY >= wTop && canvasY < wTop + wH) {
            return { widget: w, output: { type: "ZYF_LORA_ROW" } };
          }
        }
      }
      // Fall back to default behavior.
      return origGetSlotInPosition ? origGetSlotInPosition.call(this, canvasX, canvasY) : null;
    };

    const origGetSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
    nodeType.prototype.getSlotMenuOptions = function (slot) {
      if (slot && slot.widget instanceof WanLoraRowWidget) {
        showRowContextMenu(slot.widget, this);
        return undefined; // Suppress the default LiteGraph menu.
      }
      return origGetSlotMenuOptions ? origGetSlotMenuOptions.call(this, slot) : null;
    };
  },
});

// =========================================================================
// Single LoRA Loader (zyf)
// =========================================================================
// A simplified version of the Wan 2.2 loader: single MODEL+CLIP in/out,
// no high/low pairing, no auto-select logic. Each row picks one LoRA
// with a single strength value.

const SINGLE_NODE_TYPE = "SingleLoraLoader";
const SINGLE_EXT_NAME = "zyf.SingleLoraLoader";
const SINGLE_ROW_HEIGHT = 30;

function makeEmptySingleRow(index) {
  return {
    index,
    on: true,
    lora: null,
    strength: 1.0,
  };
}

// ---------------------------------------------------------------------------
// SingleLoraRowWidget – one LoRA selector + one weight per row
// ---------------------------------------------------------------------------

class SingleLoraRowWidget extends ZyfBaseWidget {
  constructor(name) {
    super(name);
    this.value = makeEmptySingleRow(parseInt(name.split("_")[1], 10) || 1);
    this.options = { serialize: true };
    this.h = SINGLE_ROW_HEIGHT;
    this._loraTree = [];
    this._wtDragMoved = false;
    this.hitAreas = {
      toggle:  { bounds: [0, 0], onDown: this.onToggleDown },
      loraSel: { bounds: [0, 0], onClick: this.onLoraSelectClick },
      wtDec:   { bounds: [0, 0], onClick: this.onWtDec },
      wtInc:   { bounds: [0, 0], onClick: this.onWtInc },
      wtVal:   { bounds: [0, 0], onDown: this.onWtDown, onMove: this.onWtMove, onUp: this.onWtUp, onClick: this.onWtClick },
    };
  }

  setLoraTree(tree) {
    this._loraTree = tree || [];
  }

  computeSize(width) {
    return [width, this.h];
  }

  onToggleDown(_event, _pos, node) {
    this.value.on = !this.value.on;
    node.setDirtyCanvas(true, true);
    return true;
  }

  async onLoraSelectClick(_event, _pos, node) {
    const tree = await fetchLoraTree();
    this.setLoraTree(tree);
    const items = [
      { content: "— clear —", value: null, callback: () => {
        this.value.lora = null;
        node.setDirtyCanvas(true, true);
      }},
      null,
    ];
    const treeItems = buildTreeMenu(this._loraTree, (path) => {
      this.value.lora = path;
      node.setDirtyCanvas(true, true);
    });
    items.push(...treeItems);
    new LiteGraph.ContextMenu(items, { event: _event, title: "Select LoRA" });
    return true;
  }

  onWtDec(_event, _pos, node) {
    this.value.strength = roundWt((this.value.strength ?? 1) - WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtInc(_event, _pos, node) {
    this.value.strength = roundWt((this.value.strength ?? 1) + WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtClick(_event, _pos, node) {
    if (this._wtDragMoved) return true;
    const canvas = app.canvas;
    if (canvas && canvas.prompt) {
      canvas.prompt("Value", String(this.value.strength ?? 1), (v) => {
        const n = parseFloat(v);
        if (!isNaN(n)) {
          this.value.strength = roundWt(n);
          node.setDirtyCanvas(true, true);
        }
      }, _event);
    }
    return true;
  }
  onWtDown(_event, _pos, _node) {
    this._wtDragMoved = false;
    this._wtStartX = _event.clientX || _pos[0];
    this._wtStartVal = this.value.strength ?? 1;
    return true;
  }
  onWtMove(_event, _pos, node) {
    const clientX = _event.clientX != null ? _event.clientX : _pos[0];
    const dx = clientX - this._wtStartX;
    if (Math.abs(dx) > 2) this._wtDragMoved = true;
    this.value.strength = roundWt(this._wtStartVal + dx * WT_STEP);
    node.setDirtyCanvas(true, true);
    return true;
  }
  onWtUp(_event, _pos, node) {
    node.setDirtyCanvas(true, true);
    return true;
  }

  // Right-click context menu
  onContextMenu(event, _pos, node) {
    const items = [];

    items.push({
      content: " Delete this row",
      callback: () => {
        const idx = node.widgets.indexOf(this);
        if (idx >= 0) {
          node.widgets.splice(idx, 1);
          const rows = collectSingleRowWidgets(node);
          rows.forEach((r, i) => {
            r.value.index = i + 1;
            r.name = `${LORA_ROW_PREFIX}${i + 1}`;
          });
          node.size[1] = Math.max(node.size[1], node.computeSize()[1]);
          node.setDirtyCanvas(true, true);
        }
      }
    });

    // Open LoRA folder option
    items.push({
      content: "📁 Open LoRA Folder",
      disabled: !this.value.lora,
      callback: () => {
        if (this.value.lora) openLoraFolder(this.value.lora);
      },
    });

    // View TXT file option
    if (this.value.lora) {
      items.push({
        content: "📄 View TXT",
        callback: () => this.openTxtViewer(node)
      });
    } else {
      items.push({
        content: " View TXT (disabled)",
        disabled: true
      });
    }

    new LiteGraph.ContextMenu(items, {
      event: event,
      title: "Row Options"
    });
    return true;
  }

  async openTxtViewer(node) {
    const loraFile = this.value.lora;
    if (!loraFile) return;

    const result = await fetchTxt(loraFile);
    if (!result.found) {
      console.warn("[zyf-SingleLoraLoader] No TXT file found for:", this.value.lora);
      return;
    }

    const txtPath = result.path;
    const txtEncoding = result.encoding || "utf-8";

    const modal = document.createElement("div");
    modal.style.cssText = `
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: rgba(0, 0, 0, 0.7);
      display: flex;
      justify-content: center;
      align-items: flex-start;
      padding-top: 5vh;
      z-index: 10000;
    `;

    const dialog = document.createElement("div");
    dialog.style.cssText = `
      background: #1e1e1e;
      border: 1px solid #3a3f4a;
      border-radius: 8px;
      padding: 20px;
      width: 85%;
      max-width: 900px;
      height: 85vh;
      display: flex;
      flex-direction: column;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    `;

    const header = document.createElement("div");
    header.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
      padding-bottom: 10px;
      border-bottom: 1px solid #3a3f4a;
    `;

    const title = document.createElement("div");
    title.style.cssText = `
      color: #d4d4d4;
      font-size: 14px;
      font-weight: bold;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      margin-right: 10px;
    `;
    title.textContent = `TXT: ${txtPath}`;

    const btnGroup = document.createElement("div");
    btnGroup.style.cssText = `
      display: flex;
      gap: 8px;
      flex-shrink: 0;
    `;

    const saveBtn = document.createElement("button");
    saveBtn.style.cssText = `
      background: #2b6cb0;
      color: white;
      border: none;
      padding: 5px 16px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    `;
    saveBtn.textContent = " Save";

    const closeBtn = document.createElement("button");
    closeBtn.style.cssText = `
      background: #a04040;
      color: white;
      border: none;
      padding: 5px 15px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
    `;
    closeBtn.textContent = "Close";

    btnGroup.appendChild(saveBtn);
    btnGroup.appendChild(closeBtn);
    header.appendChild(title);
    header.appendChild(btnGroup);

    const statusBar = document.createElement("div");
    statusBar.style.cssText = `
      color: #98c379;
      font-size: 12px;
      margin-bottom: 8px;
      min-height: 18px;
      display: none;
    `;

    const textarea = document.createElement("textarea");
    textarea.style.cssText = `
      flex: 1;
      background: #2a2f38;
      color: #d4d4d4;
      border: 1px solid #3a3f4a;
      border-radius: 4px;
      padding: 10px;
      font-family: monospace;
      font-size: 12px;
      resize: none;
      outline: none;
      white-space: pre-wrap;
      word-wrap: break-word;
    `;
    textarea.value = result.content;
    textarea.readOnly = false;

    let hasChanges = false;
    textarea.addEventListener("input", () => {
      hasChanges = true;
      statusBar.style.display = "block";
      statusBar.style.color = "#e5c07b";
      statusBar.textContent = "● Unsaved changes";
    });

    saveBtn.onclick = async () => {
      saveBtn.disabled = true;
      statusBar.style.display = "block";
      statusBar.style.color = "#e5c07b";
      statusBar.textContent = "Saving...";

      const res = await saveTxt(txtPath, textarea.value, txtEncoding);

      if (res.success) {
        hasChanges = false;
        statusBar.style.color = "#98c379";
        statusBar.textContent = "Saved successfully";
        setTimeout(() => { statusBar.style.display = "none"; }, 2000);
      } else {
        statusBar.style.color = "#e06c75";
        statusBar.textContent = "Save failed: " + (res.error || "Unknown error");
      }
      saveBtn.disabled = false;
    };

    closeBtn.onclick = () => {
      if (hasChanges) {
        if (confirm("You have unsaved changes. Close anyway?")) {
          document.body.removeChild(modal);
        }
      } else {
        document.body.removeChild(modal);
      }
    };

    dialog.appendChild(header);
    dialog.appendChild(statusBar);
    dialog.appendChild(textarea);
    modal.appendChild(dialog);

    modal.onclick = (e) => {
      if (e.target === modal) {
        if (hasChanges) {
          if (confirm("You have unsaved changes. Close anyway?")) {
            document.body.removeChild(modal);
          }
        } else {
          document.body.removeChild(modal);
        }
      }
    };

    document.body.appendChild(modal);
    textarea.focus();
  }

  draw(ctx, node, width, y, height) {
    const margin = 6;
    const innerMargin = 4;
    const accent = LiteGraph.WIDGET_OUTLINE_COLOR || "#3a3f4a";
    const bg = LiteGraph.WIDGET_BGCOLOR || "#1c1f25";
    const text = LiteGraph.WIDGET_TEXT_COLOR || "#d4d4d4";
    const muted = "#7f8794";
    const blue = "#5b8def";
    const green = "#98c379";
    const redOn = "#e06c75";

    // Background row.
    ctx.fillStyle = this.value.index % 2 === 0 ? "#262b34" : "#1f232b";
    ctx.fillRect(0, y, width, height);
    ctx.strokeStyle = "#2a2f38";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, y + height - 0.5);
    ctx.lineTo(width, y + height - 0.5);
    ctx.stroke();

    let posX = margin;

    // --- Toggle -----------------------------------------------------
    const tw = 28;
    const th = 20;
    const trackX = posX;
    const trackY = y + (height - th) / 2;
    ctx.fillStyle = this.value.on ? green : redOn;
    ctx.globalAlpha = this.value.on ? 1.0 : 0.4;
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(trackX, trackY, tw, th, th / 2);
    ctx.fill();
    ctx.stroke();
    const knobR = th * 0.35;
    const knobX = this.value.on ? trackX + tw - th / 2 : trackX + th / 2;
    const knobY = trackY + th / 2;
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(knobX, knobY, knobR, 0, Math.PI * 2);
    ctx.fill();
    ctx.globalAlpha = 1.0;
    this.hitAreas.toggle.bounds = [trackX, trackY, tw, th];
    posX += tw + innerMargin;

    // --- LoRA selector box ------------------------------------------
    const selAreaW = width - posX - margin - 76 - innerMargin;
    const boxH = height - 6;
    const boxY = y + (height - boxH) / 2;

    const loraName = this.value.lora || "(no LoRA selected)";
    const hasLora = !!this.value.lora;
    const maxTextW = selAreaW - 20;

    const boxBg = "#261a38";
    const boxBorder = "#7b529e";
    ctx.fillStyle = boxBg;
    ctx.strokeStyle = boxBorder;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(posX, boxY, selAreaW, boxH, 3);
    ctx.fill();
    ctx.stroke();
    this.hitAreas.loraSel.bounds = [posX, boxY, selAreaW, boxH];

    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = "10px monospace";
    ctx.fillStyle = this.value.on ? (hasLora ? "#c678dd" : muted) : muted;
    ctx.fillText(truncateText(ctx, shortPath(loraName), maxTextW), posX + 6, boxY + boxH / 2);

    // Dropdown arrow
    ctx.fillStyle = muted;
    ctx.font = "8px sans-serif";
    ctx.fillText("▼", posX + selAreaW - 14, boxY + boxH / 2);

    posX += selAreaW + innerMargin;

    // --- Weight control ---------------------------------------------
    const wtW = 76;
    const wtH = height - 8;
    const wtY = y + (height - wtH) / 2;
    const btnW = 17;

    drawWeightControl(ctx, this.value.on, this.value.strength, null,
      posX, wtY, wtW, wtH, btnW, accent, bg, text, blue, muted);
    this.hitAreas.wtDec.bounds = [posX, wtY, btnW, wtH];
    this.hitAreas.wtVal.bounds = [posX + btnW, wtY, wtW - btnW * 2, wtH];
    this.hitAreas.wtInc.bounds = [posX + wtW - btnW, wtY, btnW, wtH];
  }

  serializeValue(_node, _index) {
    return {
      index: this.value.index,
      on: this.value.on !== false,
      lora: this.value.lora,
      strength: this.value.strength ?? 1.0,
    };
  }
}

// ---------------------------------------------------------------------------
// Single loader helpers
// ---------------------------------------------------------------------------

function collectSingleRowWidgets(node) {
  return (node.widgets || []).filter((w) => w instanceof SingleLoraRowWidget);
}

function addSingleRowWidget(node, row) {
  const widget = new SingleLoraRowWidget(`${LORA_ROW_PREFIX}${row.index}`);
  widget.value = { ...makeEmptySingleRow(row.index), ...row };
  addCustomWidget(node, widget);
  const btnIdx = node.widgets.findIndex(
    (w) => w && w.name === "_zyf_btn_add_lora"
  );
  if (btnIdx >= 0) {
    moveWidgetTo(node, widget, btnIdx);
  }
  return widget;
}

function ensureSingleHeader(node) {
  node.widgets = node.widgets || [];
  if (node._zyfSingleHeaderDone) return;
  node._zyfSingleHeaderDone = true;

  addCustomWidget(node, new ZyfButtonWidget(
    "_zyf_btn_add_lora", "+ Add LoRA", "#2b6cb0",
    (_event, _pos, n) => {
      const existing = collectSingleRowWidgets(n);
      const nextIdx = existing.length
        ? Math.max(...existing.map((r) => r.value.index)) + 1
        : 1;
      const newRow = addSingleRowWidget(n, makeEmptySingleRow(nextIdx));
      n.size[1] = Math.max(n.size[1], n.computeSize()[1]);
      n.setDirtyCanvas(true, true);
      rebuildSingleRows(n);
      if (newRow && newRow.onLoraSelectClick) {
        newRow.onLoraSelectClick(_event, _pos, n);
      }
      return true;
    }
  ));
}

function ensureSingleRowsAndButtons(node) {
  ensureSingleHeader(node);
  if (collectSingleRowWidgets(node).length === 0) {
    addSingleRowWidget(node, makeEmptySingleRow(1));
  }
}

async function rebuildSingleRows(node) {
  const tree = await fetchLoraTree();
  for (const r of collectSingleRowWidgets(node)) {
    r.setLoraTree(tree);
  }
  node.setDirtyCanvas(true, true);
}

function showSingleRowContextMenu(targetWidget, node) {
  const index = node.widgets.indexOf(targetWidget);
  const rows = collectSingleRowWidgets(node);
  const canMoveUp = index > 0 && rows.indexOf(node.widgets[index - 1]) >= 0;
  const canMoveDown = index < node.widgets.length - 1 && rows.indexOf(node.widgets[index + 1]) >= 0;

  const menuItems = [
    {
      content: "📄 View TXT",
      disabled: !targetWidget.value.lora,
      callback: () => {
        if (targetWidget.value.lora) {
          targetWidget.openTxtViewer(node);
        }
      },
    },
    {
      content: "📁 Open LoRA Folder",
      disabled: !targetWidget.value.lora,
      callback: () => {
        if (targetWidget.value.lora) {
          openLoraFolder(targetWidget.value.lora);
        }
      },
    },
    null,
    {
      content: "⬆️ Move Up",
      disabled: !canMoveUp,
      callback: () => {
        moveWidgetTo(node, targetWidget, index - 1);
        const remaining = collectSingleRowWidgets(node);
        remaining.forEach((r, i) => {
          r.value.index = i + 1;
          r.name = `${LORA_ROW_PREFIX}${i + 1}`;
        });
        rebuildSingleRows(node);
        node.setDirtyCanvas(true, true);
      },
    },
    {
      content: "⬇️ Move Down",
      disabled: !canMoveDown,
      callback: () => {
        moveWidgetTo(node, targetWidget, index + 1);
        const remaining = collectSingleRowWidgets(node);
        remaining.forEach((r, i) => {
          r.value.index = i + 1;
          r.name = `${LORA_ROW_PREFIX}${i + 1}`;
        });
        rebuildSingleRows(node);
        node.setDirtyCanvas(true, true);
      },
    },
    null,
    {
      content: "🗑️ Remove",
      callback: () => {
        node.widgets.splice(index, 1);
        const remaining = collectSingleRowWidgets(node);
        remaining.forEach((r, i) => {
          r.value.index = i + 1;
          r.name = `${LORA_ROW_PREFIX}${i + 1}`;
        });
        node.size[1] = Math.max(node.size[1], node.computeSize()[1]);
        node.setDirtyCanvas(true, true);
      },
    },
  ];

  new LiteGraph.ContextMenu(menuItems, {
    title: "LoRA Row",
    event: _lastCanvasMouseEvent,
  });
}

// ---------------------------------------------------------------------------
// Single LoRA Loader extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: SINGLE_EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData, _app) {
    if (nodeData.name !== SINGLE_NODE_TYPE) return;

    nodeType.title = "Simple LoRA Loader (zyf)";

    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = origOnNodeCreated ? origOnNodeCreated.apply(this, arguments) : undefined;
      const node = this;
      node.serialize_widgets = true;
      ensureSingleRowsAndButtons(node);
      rebuildSingleRows(node).catch((err) =>
        console.error("[zyf-SingleLoraLoader] rebuild failed:", err)
      );
      node.size[0] = INITIAL_NODE_WIDTH;
      node.size[1] = node.computeSize()[1];
      node.setDirtyCanvas(true, true);
      return r;
    };

    const origConfigure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      const node = this;
      node.serialize_widgets = true;

      node.widgets = (node.widgets || []).filter(
        (w) => !(w instanceof ZyfButtonWidget || w instanceof ZyfTitleWidget || w instanceof SingleLoraRowWidget)
      );
      node._zyfSingleHeaderDone = false;

      const r = origConfigure ? origConfigure.call(this, info) : undefined;

      const wv = (info && info.widgets_values) || [];
      const restored = [];
      for (const v of wv) {
        if (
          v && typeof v === "object" && typeof v.index === "number" &&
          ("lora" in v || "strength" in v)
        ) {
          restored.push({ ...makeEmptySingleRow(v.index), ...v });
        }
      }

      ensureSingleHeader(node);
      if (restored.length) {
        for (const row of restored.sort((a, b) => a.index - b.index)) {
          addSingleRowWidget(node, row);
        }
      } else {
        addSingleRowWidget(node, makeEmptySingleRow(1));
      }

      node.size[0] = Math.max(node.size[0] || INITIAL_NODE_WIDTH, INITIAL_NODE_WIDTH);
      node.size[1] = Math.max(node.size[1] || 0, node.computeSize()[1]);
      node.setDirtyCanvas(true, true);
      rebuildSingleRows(node);

      return r;
    };

    // Right-click context menu for single LoRA rows.
    const origGetSlotInPosition = nodeType.prototype.getSlotInPosition;
    nodeType.prototype.getSlotInPosition = function (canvasX, canvasY) {
      for (const w of (this.widgets || [])) {
        if (w instanceof SingleLoraRowWidget && w.last_y != null) {
          const wTop = this.pos[1] + w.last_y;
          const wH = w.computeSize(this.size[0])[1];
          if (canvasX >= this.pos[0] && canvasX <= this.pos[0] + this.size[0] &&
              canvasY >= wTop && canvasY < wTop + wH) {
            return { widget: w, output: { type: "ZYF_SINGLE_LORA_ROW" } };
          }
        }
      }
      return origGetSlotInPosition ? origGetSlotInPosition.call(this, canvasX, canvasY) : null;
    };

    const origGetSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
    nodeType.prototype.getSlotMenuOptions = function (slot) {
      if (slot && slot.widget instanceof SingleLoraRowWidget) {
        showSingleRowContextMenu(slot.widget, this);
        return undefined;
      }
      return origGetSlotMenuOptions ? origGetSlotMenuOptions.call(this, slot) : null;
    };
  },
});
