/* Palimpsest frontend. Plain script, no build step: served by the backend
   from the same origin, so every call below is a relative /api URL.

   The contract rule holds here too: this file never recomputes a number the
   backend measured. Load factor, held flags, answers and timings are shown
   exactly as they arrive. */
(function () {
  "use strict";

  var CLASSES = ["THEORY", "FORMULA", "PLOT_SKETCH", "DIAGRAM", "TABLE"];
  var COLOURS = { THEORY: "#2a78d6", FORMULA: "#eb6834", PLOT_SKETCH: "#1baf7a", DIAGRAM: "#8a5cd6", TABLE: "#c4407a" };
  var LABELS = { THEORY: "Theory", FORMULA: "Formula", PLOT_SKETCH: "Plot", DIAGRAM: "Diagram", TABLE: "Table" };
  var BACKBONE_NAMES = { squeezenet1_1: "SqueezeNet 1.1", mobilenet_v2: "MobileNetV2", resnet18: "ResNet-18", palimpsest_net: "PalimpsestNet" };

  var $ = function (id) { return document.getElementById(id); };
  var main = $("main");
  var state = { view: "notebook", noteId: null, pageId: null, models: [], health: null };

  // ------------------------------------------------------------------ api

  function api(path, options) {
    return fetch("/api" + path, options).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (body) {
          var detail = body && body.detail;
          throw new Error(typeof detail === "string" ? detail : r.status + " " + r.statusText);
        });
      }
      return r.json();
    });
  }
  function postFile(path, file, extra) {
    var form = new FormData();
    form.append("file", file);
    Object.keys(extra || {}).forEach(function (k) { form.append(k, extra[k]); });
    return api(path, { method: "POST", body: form });
  }
  function sendJSON(path, method, body) {
    return api(path, { method: method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  }

  // ------------------------------------------------------------------ dom helpers

  function el(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.slice(0, 2) === "on") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    });
    (children || []).forEach(function (c) { if (c) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c); });
    return node;
  }
  function svgEl(tag, attrs, text) {
    var node = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function fmt(x, digits) { return x === null || x === undefined ? "—" : Number(x).toFixed(digits); }
  function sig(x) { return Math.abs(x) >= 1e4 || (Math.abs(x) < 1e-3 && x !== 0) ? x.toExponential(3) : String(+x.toPrecision(4)); }
  function loadImage(src) {
    return new Promise(function (resolve, reject) {
      var img = new Image();
      img.onload = function () { resolve(img); };
      img.onerror = function () { reject(new Error("could not load " + src)); };
      img.src = src;
    });
  }

  // ------------------------------------------------------------------ status & models

  function refreshHealth() {
    return api("/health").then(function (h) { drawHealth(h); })
      .catch(function () {
        $("status").innerHTML = "";
        $("status").appendChild(el("div", {}, [el("span", { class: "dot off" }), "Backend not reachable"]));
      });
  }

  // The privacy line is generated from the reader, not written into the
  // markup, keyed by `where` so it cannot drift from whichever reader is
  // actually selected.
  var PRIVACY = {
    local: "Capture, recognition and plotting run on this laptop. No page leaves the vault folder.",
    none:  "Regions are classified on this laptop and held unread. Nothing is transcribed and nothing is sent."
  };

  function drawHealth(h) {
    state.health = h;
    var on = h.available;
    var box = $("status");
    box.innerHTML = "";

    box.appendChild(el("div", {}, [el("span", { class: "dot" + (on ? "" : " off") }),
      on ? "Reader " + h.model + " ready" : "Reader " + h.model + " offline"]));

    var pick = el("select", { "aria-label": "Which model reads the page",
                              style: "margin:6px 0 2px" });
    (h.choices || []).forEach(function (c) {
      pick.appendChild(el("option", { value: c.name, selected: c.name === h.name,
                                      text: c.label + " \u00b7 " + c.where }));
    });
    pick.addEventListener("change", function () {
      pick.disabled = true;
      api("/reader", { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify({ name: pick.value }) })
        .then(drawHealth)
        .catch(function (e) {
          // Put the selection back where it was rather than leaving the
          // dropdown showing a reader the server did not accept.
          drawHealth(state.health);
          $("status").appendChild(el("div", { class: "modelnote", text: e.message }));
        });
    });
    box.appendChild(pick);

    var chosen = (h.choices || []).filter(function (c) { return c.name === h.name; })[0];
    if (chosen) box.appendChild(el("div", { class: "modelnote", text: chosen.detail }));
    // `why` is the actionable half: selected but keyless is a different and
    // more fixable state than selected but unreachable.
    if (!on && h.why) {
      box.appendChild(el("div", { class: "modelnote", style: "color:var(--warn)", text: h.why }));
    }

    box.appendChild(el("div", { style: "margin-top:6px", text: "Classifier on " + h.device.toUpperCase() }));
    box.appendChild(el("div", { style: "margin-top:8px", text: PRIVACY[h.where] || PRIVACY.local }));
  }

  // Every origin gets its own words. This was a ternary on "real", so a
  // third origin fell into the else and two different checkpoints both
  // read "SqueezeNet 1.1 · synthetic" in the dropdown with no way to tell
  // them apart. A lookup fails loudly on an unknown origin instead of
  // quietly colliding with an existing label.
  var ORIGIN_NAMES = {
    synthetic: "synthetic pages",
    real: "your labelled pages",
    taught: "taught by the reader",
  };
  function modelLabel(m) {
    return (BACKBONE_NAMES[m.backbone] || m.backbone) + " · "
      + (ORIGIN_NAMES[m.origin] || m.origin);
  }

  function refreshModels() {
    return api("/models").then(function (models) {
      state.models = models;
      var select = $("model");
      select.innerHTML = "";
      models.forEach(function (m) {
        select.appendChild(el("option", { value: m.key, selected: m.active, text: modelLabel(m) }));
      });
      describeActive();
    });
  }
  function describeActive() {
    var m = state.models.find(function (x) { return x.active; });
    if (!m) { $("modelnote").textContent = "no trained checkpoint found"; return; }
    var parts = [];
    if (m.latency_ms) parts.push(fmt(m.latency_ms, 2) + " ms/crop");
    if (m.ece !== null && m.ece !== undefined) parts.push("ECE " + fmt(m.ece, 4));
    if (m.real_after) parts.push("your pages F1 " + fmt(m.real_before && m.real_before.macro_f1, 2) + " → " + fmt(m.real_after.macro_f1, 2));
    else if (m.synthetic_macro_f1) parts.push("synthetic F1 " + fmt(m.synthetic_macro_f1, 3));
    $("modelnote").textContent = parts.join(" · ");
  }
  $("model").addEventListener("change", function (e) {
    sendJSON("/models", "POST", { key: e.target.value }).then(refreshModels).then(refreshHealth);
  });

  // ------------------------------------------------------------------ views

  function setView(view) {
    if (state.view === "write" && view !== "write") leaveWrite();
    state.view = view;
    if (history.replaceState) history.replaceState(null, "", view === "notebook" ? "#" : "#" + view);
    ["write", "notebook", "label", "fix"].forEach(function (v) {
      $("tab-" + v).setAttribute("aria-selected", view === v);
      $("side-" + v).hidden = view !== v;
    });
    if (view !== "label") main.dataset.view = "";  // so the labeller rebuilds when we come back
    if (view === "notebook") {
      refreshNotes();
      state.noteId ? openNote(state.noteId) : showNotebookEmpty();
    } else if (view === "write") {
      renderWrite();
    } else if (view === "fix") {
      renderFix();
    } else {
      refreshPages().then(function (pages) { renderLabeler(); if (state.pageId) openPage(state.pageId); else if (pages.length) openPage(pages[0].id); });
    }
  }
  $("tab-write").addEventListener("click", function () { setView("write"); });
  $("tab-notebook").addEventListener("click", function () { setView("notebook"); });
  $("tab-label").addEventListener("click", function () { setView("label"); });
  $("tab-fix").addEventListener("click", function () { setView("fix"); });

  // ================================================================== NOTEBOOK

  $("capture").addEventListener("click", function () { $("file-compile").click(); });
  $("file-compile").addEventListener("change", function (e) {
    if (e.target.files[0]) compileFile(e.target.files[0]);
    e.target.value = "";
  });

  function refreshNotes() {
    return api("/notes").then(function (notes) {
      var list = $("notes");
      list.innerHTML = "";
      if (!notes.length) list.appendChild(el("p", { class: "empty-side", text: "Nothing compiled yet." }));
      notes.forEach(function (n) {
        list.appendChild(el("button", { class: "nb", "aria-current": n.note_id === state.noteId, onclick: function () { openNote(n.note_id); } }, [
          el("span", { text: n.title }),
          el("span", { class: "n" + (n.held ? " warn" : ""), text: n.held ? n.held + " held" : String(n.regions) }),
        ]));
      });
      return notes;
    });
  }

  function showNotebookEmpty() {
    main.className = "doc";
    main.innerHTML = "";
    var drop = el("div", { class: "drop" }, [
      el("h2", { text: "Compile a page" }),
      el("p", { text: "Photograph a page of rough notes, or drop the image here. It is read, split into regions, classified, and typeset on this machine." }),
      el("button", { class: "btn solid", onclick: function () { $("file-compile").click(); }, text: "Choose a photo" }),
    ]);
    ["dragenter", "dragover"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.add("over"); }); });
    ["dragleave", "drop"].forEach(function (t) { drop.addEventListener(t, function (e) { e.preventDefault(); drop.classList.remove("over"); }); });
    drop.addEventListener("drop", function (e) { var f = e.dataTransfer.files[0]; if (f) compileFile(f); });
    main.appendChild(el("div", { class: "state" }, [drop]));
  }

  function showBusy(title, text) {
    main.className = "doc";
    main.innerHTML = "";
    main.appendChild(el("div", { class: "state", role: "status" }, [el("div", { class: "spinner" }), el("h2", { text: title }), el("p", { text: text })]));
  }
  function showError(title, err, retry) {
    main.className = "doc";
    main.innerHTML = "";
    main.appendChild(el("div", { class: "state error", role: "alert" }, [
      el("h2", { text: title }), el("p", { text: err.message || String(err) }),
      retry ? el("button", { class: "btn", onclick: retry, text: "Try again" }) : null,
    ]));
  }

  function compileFile(file) {
    showBusy("Reading the page", "Preprocessing, finding regions and classifying them takes well under a second. Reading each region with the local model takes a few seconds per region.");
    postFile("/compile", file, { title: file.name.replace(/\.[^.]+$/, "") })
      .then(function (note) { state.noteId = note.note_id; refreshNotes(); return renderNote(note); })
      .catch(function (err) { showError("The page did not compile", err, function () { compileFile(file); }); });
  }

  function openNote(id) {
    state.noteId = id;
    refreshNotes();
    api("/notes/" + encodeURIComponent(id)).then(renderNote)
      .catch(function (err) { showError("Could not open this note", err); });
  }

  function renderNote(note, target) {
    target = target || main;
    var pageSrc = "/api/notes/" + encodeURIComponent(note.note_id) + "/page.png";
    return loadImage(pageSrc).catch(function () { return null; }).then(function (img) {
      if (target === main) main.className = "doc";
      target.innerHTML = "";
      var held = note.blocks.filter(function (b) { return b.held; }).length;
      var typeset = note.blocks.length - held;
      var b = note.budget;
      var model = state.models.find(function (m) { return m.key === note.backbone; });

      target.appendChild(el("header", { class: "dochead" }, [el("div", { class: "hd" }, [
        el("p", { class: "kicker", text: "Compiled note · " + note.note_id.slice(0, 15) }),
        el("h1", { text: note.title }),
        el("p", { class: "standfirst", text: "Compiled in " + (b.service_ms / 1000).toFixed(2) + " seconds. " +
          (note.blocks.length ? note.blocks.length + " regions were found, " + typeset + " typeset" +
            (held ? ", and " + held + (held === 1 ? " is" : " are") + " waiting on you." : ".") : "No regions were found on this page.") }),
        el("p", { class: "byline" }, [
          el("span", {}, ["Regions ", el("b", { text: String(note.blocks.length) })]),
          el("span", {}, ["Mean confidence ", el("b", { text: fmt(note.mean_confidence, 2) })]),
          el("span", {}, ["Load factor ", el("b", { class: b.over_budget ? "warn" : "", text: fmt(b.load_factor, 2) })]),
          el("span", {}, ["Bottleneck ", el("b", { text: b.dominant_stage })]),
          el("span", {}, ["Classifier ", el("b", { text: model ? modelLabel(model) : note.backbone })]),
          el("span", {}, ["Reader ", el("b", { text: state.health ? state.health.model + ", " + state.health.where : "local" })]),
        ]),
      ])]));

      if (!note.blocks.length) {
        target.appendChild(el("div", { class: "row" }, [el("div", { class: "body" }, [el("div", { class: "refused", text: "Segmentation found no ink regions. A photo with more contrast, or the page filling more of the frame, usually fixes this." })])]));
      }

      var firstProse = true;
      note.blocks.forEach(function (block, i) {
        var row = el("div", { class: "row", "data-block": block.id, style: "animation-delay:" + Math.min(i * 50, 400) + "ms" });
        row.appendChild(facsimile(block, img, note));
        var body = el("div", { class: "body" });
        if (block.held) body.appendChild(reviewPanel(block));
        else body.appendChild(renderContent(block, firstProse));
        if (!block.held && block.region_class === "THEORY") firstProse = false;
        body.appendChild(correctionBar(note, block));
        row.appendChild(body);
        target.appendChild(row);
      });

      var prov = el("div", { class: "prov" }, [el("span", { text: "Provenance" })]);
      note.blocks.forEach(function (bl) {
        prov.appendChild(el("span", { class: "pill", "data-low": bl.held ? "1" : null }, [el("i"), bl.region_class + " " + fmt(bl.confidence, 2)]));
      });
      prov.appendChild(el("span", { style: "margin-left:auto", text:
        "preprocess " + fmt(b.preprocess_ms, 0) + " · classify " + fmt(b.infer_ms, 0) + " · read " + fmt(b.route_ms, 0) + " ms of " + fmt(b.budget_ms, 0) }));
      target.appendChild(el("div", { class: "row" }, [prov]));

      var extra = note.model_notes.slice();
      var symbols = Object.keys(note.symbols || {});
      if (symbols.length) extra.unshift("symbol table: " + symbols.map(function (k) { return k + " = " + sig(note.symbols[k]); }).join(", "));
      if (extra.length) target.appendChild(el("div", { class: "row" }, [el("div", { class: "notes" }, extra.map(function (t) { return el("div", { text: t }); }))]));
    });
  }

  function facsimile(block, img, note) {
    var low = block.held;
    var btn = el("button", { class: "marg", "data-low": low ? "1" : null,
      "aria-label": "Open the source page. " + block.region_class + " region, confidence " + fmt(block.confidence, 2) + ".",
      onclick: function () { openSource(note, block.id); } });
    var fac = el("span", { class: "fac" });
    if (img) {
      var bx = block.box, pad = 6;
      var sx = Math.max(0, bx.x - pad), sy = Math.max(0, bx.y - pad);
      var sw = Math.min(img.naturalWidth - sx, bx.w + 2 * pad), sh = Math.min(img.naturalHeight - sy, bx.h + 2 * pad);
      var w = 208, h = Math.max(40, Math.min(240, Math.round(w * sh / sw)));
      var c = el("canvas", { width: w, height: h });
      var ctx = c.getContext("2d");
      ctx.fillStyle = "#fffdf7"; ctx.fillRect(0, 0, w, h);
      var scale = Math.min(w / sw, h / sh);
      ctx.drawImage(img, sx, sy, sw, sh, (w - sw * scale) / 2, (h - sh * scale) / 2, sw * scale, sh * scale);
      fac.appendChild(c);
    }
    btn.appendChild(fac);
    btn.appendChild(el("span", { class: "cf" }, [block.region_class, el("br"), el("em", { text: fmt(block.confidence, 2) + " / " + fmt(block.threshold, 2) })]));
    return btn;
  }

  // ------------------------------------------------------------------ fix
  // One region at a time, not a list. The list version put fifty-three rows
  // on a page and asked the reader to decide fifty-three times while looking
  // at all of them, which is the slowest possible shape for a queue of
  // independent yes/no judgements: every row competes for attention with
  // every other, and none of them is big enough to actually read.
  //
  // A card is the right shape because the task is genuinely one-at-a-time.
  // The crop gets the whole width, the judgement is two keys, and the
  // correction form only appears once the answer is "no" — so confirming,
  // which is most of them, costs a single keystroke.
  //
  // Ordered most-confident-first, which is the opposite of the obvious. A
  // held region is already handled; nothing was written into the note. The
  // expensive mistakes are the ones written in as fact at 1.00.

  var fixFilter = "all";
  var fix = { queue: [], at: 0, done: 0, report: {} };

  function renderFix() {
    var main = $("main");
    main.textContent = "";
    main.dataset.view = "fix";
    var stage = el("div", { class: "cardstage", id: "cardstage" }, [
      el("p", { class: "qempty", text: "Loading…" }),
    ]);
    main.appendChild(stage);
    api("/review?limit=200&only=" + fixFilter).then(function (data) {
      fix.queue = data.regions; fix.at = 0; fix.done = 0; fix.report = data.report || {};
      renderFixSidebar(data);
      showCard();
    }).catch(function (e) {
      stage.textContent = "";
      stage.appendChild(el("p", { class: "qempty", text: e.message }));
    });
  }

  function renderFixSidebar(data) {
    var stats = $("fixstats");
    if (stats) {
      stats.textContent = "";
      var rep = data.report || fix.report || {};
      [["left to check", data.pending !== undefined ? data.pending : fix.queue.length - fix.at],
       ["recorded", rep.entries || 0], ["wrong class", rep.class_errors || 0],
       ["wrong reading", rep.reading_errors || 0], ["confirmed", rep.confirmed || 0]]
        .forEach(function (pair) {
          stats.appendChild(el("div", {}, [pair[0] + " ", el("b", { text: String(pair[1]) })]));
        });
    }
    var filters = $("fixfilters");
    if (!filters) return;
    filters.textContent = "";
    [["all", "Everything"], ["confident", "Confident only (≥ 0.90)"], ["held", "Held only"]]
      .forEach(function (pair) {
        filters.appendChild(el("button", { class: "nb",
          "aria-current": fixFilter === pair[0] ? "true" : null,
          onclick: function () { fixFilter = pair[0]; renderFix(); } }, [pair[1]]));
      });
  }

  function showCard() {
    var stage = $("cardstage");
    if (!stage) return;
    stage.textContent = "";
    var r = fix.queue[fix.at];
    if (!r) {
      stage.appendChild(el("div", { class: "cardend" }, [
        el("h2", { text: fix.done ? "That is the lot." : "Nothing to check." }),
        el("p", { text: fix.done
          ? "You went through " + fix.done + ". Every correction is in the training data now."
          : "Compile a page and come back." }),
        fix.done ? el("button", { class: "btn", text: "Go round again", onclick: renderFix }) : null,
      ]));
      return;
    }
    stage.appendChild(cardFor(r));
  }

  function cardFor(r) {
    var card = el("div", { class: "card", tabindex: "-1" });
    card.appendChild(el("div", { class: "cardcount", text:
      (fix.at + 1) + " of " + fix.queue.length }));

    var holder = el("div", { class: "cardcrop" });
    card.appendChild(holder);
    loadImage("/api/notes/" + encodeURIComponent(r.note_id) + "/page.png").then(function (img) {
      var b = r.box, pad = 10;
      var sx = Math.max(0, b.x - pad), sy = Math.max(0, b.y - pad);
      var sw = Math.min(img.naturalWidth - sx, b.w + 2 * pad);
      var sh = Math.min(img.naturalHeight - sy, b.h + 2 * pad);
      var w = 760, h = Math.max(90, Math.min(340, Math.round(w * sh / sw)));
      var c = el("canvas", { width: w, height: h });
      var ctx = c.getContext("2d");
      ctx.fillStyle = "#fffdf7"; ctx.fillRect(0, 0, w, h);
      var k = Math.min(w / sw, h / sh);
      ctx.drawImage(img, sx, sy, sw, sh, (w - sw * k) / 2, (h - sh * k) / 2, sw * k, sh * k);
      holder.appendChild(c);
    }).catch(function () {
      holder.appendChild(el("p", { class: "qmeta", text: "the page image could not be loaded" }));
    });

    card.appendChild(el("div", { class: "cardguess" }, [
      el("span", { class: "cardkind" }, [LABELS[r.region_class] || r.region_class,
        el("em", { text: " " + fmt(r.confidence, 2) }), r.held ? el("b", { text: " held" }) : null]),
      r.read_as
        ? el("div", { class: "qread", text: r.read_as })
        : el("div", { class: "qread" }, [el("em", { text: "nothing was read from this region" })]),
      el("span", { class: "qmeta", text: r.title.slice(0, 40) || r.note_id }),
    ]));

    var status = el("span", { class: "fixnote" });

    function send(payload, label) {
      status.textContent = "saving…";
      api("/notes/" + encodeURIComponent(r.note_id) + "/blocks/" + encodeURIComponent(r.block_id) + "/correction",
          { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
        .then(function (res) {
          fix.done++;
          fix.report = res.report || fix.report;
          fly(card, label === "right" ? 1 : -1, function () {
            fix.at++;
            showCard();
            api("/review?limit=1&only=" + fixFilter).then(renderFixSidebar).catch(function () {});
          });
        })
        .catch(function (e) { status.textContent = e.message; });
    }

    var actions = el("div", { class: "cardacts" }, [
      el("button", { class: "swipe no", onclick: function () { openFixForm(card, r, send); } }, [
        el("span", { class: "key", text: "←" }), "Not what I wrote"]),
      el("button", { class: "swipe yes", onclick: function () {
        send({ should_be: r.read_as, correct_class: r.region_class }, "right");
      } }, [el("span", { class: "key", text: "→" }), "That’s right"]),
    ]);
    card.appendChild(actions);
    card.appendChild(el("div", { class: "cardform", id: "cardform" }));
    card.appendChild(status);

    card.fixSend = send;
    card.fixRegion = r;
    dragToSwipe(card, r, send);
    setTimeout(function () { card.focus(); }, 0);
    return card;
  }

  function openFixForm(card, r, send) {
    var slot = card.querySelector(".cardform");
    if (!slot || slot.dataset.open) return;
    slot.dataset.open = "1";
    card.querySelector(".cardacts").hidden = true;

    var text = el("textarea", { class: "fixtext", rows: "2",
      "aria-label": "What this region actually says" });
    text.value = r.read_as || "";
    var pick = el("select", { class: "fixclass", "aria-label": "What kind of region this is" });
    Object.keys(LABELS).forEach(function (k) {
      pick.appendChild(el("option", { value: k, text: LABELS[k], selected: k === r.region_class }));
    });
    slot.appendChild(el("div", { class: "fixrow" }, [
      el("label", { class: "fixlabel", text: "Should say" }), text]));
    slot.appendChild(el("div", { class: "fixrow" }, [
      el("span", { class: "fixlabel" }), mathKeys(text)]));
    slot.appendChild(el("div", { class: "fixrow" }, [
      el("label", { class: "fixlabel", text: "Kind" }), pick,
      el("button", { class: "fixsave", text: "Save and next", onclick: function () {
        send({ should_be: text.value, correct_class: pick.value }, "left");
      } }),
      el("button", { class: "fixlink", text: "cancel", onclick: function () {
        slot.textContent = ""; delete slot.dataset.open;
        card.querySelector(".cardacts").hidden = false; card.focus();
      } }),
    ]));
    text.focus();
    text.select();
  }

  // Characters a keyboard does not have and a formula needs. Typing LaTeX
  // by hand is the thing that makes a correction interface go unused, and
  // the three that actually come up on an engineering page are a subscript,
  // a superscript and a multiplication dot.
  //
  // They insert LaTeX, not Unicode lookalikes: the text goes back as the
  // block's `tex`, so `B_{0}` is a subscript everywhere downstream while
  // `B₀` would be a character the solver cannot read and KaTeX would not
  // typeset. Pressing one with a selection wraps it.
  var MATH_KEYS = [
    { label: "x₂", title: "Subscript  (Ctrl+-)", wrap: ["_{", "}"] },
    { label: "x²", title: "Superscript  (Ctrl+6)", wrap: ["^{", "}"] },
    { label: "·", title: "Multiply  (Ctrl+8)", insert: " \\cdot " },
    { label: "⁄", title: "Fraction", wrap: ["\\frac{", "}{}"] },
    { label: "√", title: "Square root", wrap: ["\\sqrt{", "}"] },
  ];

  function applyMathKey(field, key) {
    var start = field.selectionStart, end = field.selectionEnd;
    var before = field.value.slice(0, start);
    var middle = field.value.slice(start, end);
    var after = field.value.slice(end);
    var caret;
    if (key.insert) {
      field.value = before + key.insert + middle + after;
      caret = before.length + key.insert.length + middle.length;
    } else {
      field.value = before + key.wrap[0] + middle + key.wrap[1] + after;
      // Land inside the braces when nothing was selected, after the
      // inserted text when something was.
      caret = before.length + key.wrap[0].length + middle.length;
    }
    field.focus();
    field.setSelectionRange(caret, caret);
  }

  function mathKeys(field) {
    var bar = el("span", { class: "mathkeys", role: "group",
      "aria-label": "Insert a mathematical symbol" });
    MATH_KEYS.forEach(function (key) {
      bar.appendChild(el("button", { class: "mathkey", type: "button", title: key.title,
        text: key.label, onclick: function () { applyMathKey(field, key); } }));
    });
    // Keyboard equivalents, so a correction never needs the mouse.
    field.addEventListener("keydown", function (e) {
      if (!(e.ctrlKey || e.metaKey)) return;
      var hit = { "-": MATH_KEYS[0], "6": MATH_KEYS[1], "8": MATH_KEYS[2] }[e.key];
      if (hit) { applyMathKey(field, hit); e.preventDefault(); }
    });
    return bar;
  }

  // Throw the card off in the direction it was answered, then advance. The
  // motion is the receipt: it is the only confirmation that the keystroke
  // landed, so it runs even though it is longer than the usual UI budget.
  function fly(card, dir, then) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return then();
    card.style.transition = "transform 260ms var(--ease), opacity 260ms var(--ease)";
    card.style.transform = "translateX(" + (dir * 120) + "%) rotate(" + (dir * 7) + "deg)";
    card.style.opacity = "0";
    setTimeout(then, 240);
  }

  function dragToSwipe(card, r, send) {
    var startX = null, dx = 0;
    card.addEventListener("pointerdown", function (e) {
      if (e.target.closest("button, textarea, select")) return;
      startX = e.clientX; dx = 0;
      card.setPointerCapture(e.pointerId);
      card.style.transition = "none";
    });
    card.addEventListener("pointermove", function (e) {
      if (startX === null) return;
      dx = e.clientX - startX;
      card.style.transform = "translateX(" + dx + "px) rotate(" + (dx / 60) + "deg)";
      card.dataset.lean = dx > 70 ? "yes" : dx < -70 ? "no" : "";
    });
    function release() {
      if (startX === null) return;
      var moved = dx;
      startX = null; card.dataset.lean = "";
      card.style.transition = "transform 160ms var(--ease)";
      if (moved > 110) return send({ should_be: r.read_as, correct_class: r.region_class }, "right");
      if (moved < -110) { card.style.transform = ""; return openFixForm(card, r, send); }
      card.style.transform = "";
    }
    card.addEventListener("pointerup", release);
    card.addEventListener("pointercancel", release);
  }


  // The reading can be wrong while the class and the confidence are right:
  // `x^2 = 49` came back `x^2 = 4` at 1.00. Until there was somewhere to say
  // so, the only record of that was the writer noticing and shrugging. The
  // two halves are asked separately because they are fixed by different
  // work: a wrong class is a training label, a wrong reading is a test case
  // for the reader.
  function blockText(block) {
    var c = block.content || {};
    return c.tex || c.markdown || c.caption || "";
  }

  function correctionBar(note, block) {
    var bar = el("div", { class: "fixbar" });
    var done = block.correction;

    function settled(entry) {
      bar.textContent = "";
      var what = entry.kind === "confirmed" ? "Marked correct."
        : entry.kind === "class" ? "Recorded: should be " + LABELS[entry.correct_class] + "."
        : entry.kind === "reading" ? "Correction saved. This region is now a test case for the reader."
        : "Recorded: should be " + LABELS[entry.correct_class] + ", read as \u201c" + entry.should_be + "\u201d.";
      bar.appendChild(el("span", { class: "fixdone", text: what }));
      bar.appendChild(el("button", { class: "fixlink", text: "change", onclick: function () { open(); } }));
    }

    function open() {
      bar.textContent = "";
      var text = el("textarea", { class: "fixtext", rows: "2",
        "aria-label": "What this region actually says" });
      text.value = (done && done.should_be) || blockText(block);
      var pick = el("select", { class: "fixclass", "aria-label": "What kind of region this is" });
      Object.keys(LABELS).forEach(function (k) {
        pick.appendChild(el("option", { value: k, text: LABELS[k], selected: k === ((done && done.correct_class) || block.region_class) }));
      });
      var status = el("span", { class: "fixnote" });

      function send(payload) {
        status.textContent = "saving\u2026";
        api("/notes/" + encodeURIComponent(note.note_id) + "/blocks/" + encodeURIComponent(block.id) + "/correction",
            { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
          .then(function (r) { block.correction = r.correction; settled(r.correction); })
          .catch(function (e) { status.textContent = e.message; });
      }

      bar.appendChild(el("div", { class: "fixrow" }, [
        el("label", { class: "fixlabel", text: "Should say" }), text,
      ]));
      bar.appendChild(el("div", { class: "fixrow" }, [
        el("label", { class: "fixlabel", text: "Kind" }), pick,
        el("button", { class: "fixsave", text: "Save correction", onclick: function () {
          send({ should_be: text.value, correct_class: pick.value });
        } }),
        el("button", { class: "fixlink", text: "It\u2019s right", onclick: function () {
          send({ should_be: blockText(block), correct_class: block.region_class });
        } }),
        el("button", { class: "fixlink", text: "cancel", onclick: function () { shut(); } }),
        status,
      ]));
      text.focus();
    }

    function shut() {
      bar.textContent = "";
      if (done) return settled(done);
      bar.appendChild(el("button", { class: "fixlink", text: "Not what I wrote?",
        title: "Say what this region should have been", onclick: open }));
    }

    shut();
    return bar;
  }

  function reviewPanel(block) {
    return el("div", { class: "review" }, [
      el("h4", { text: "Held for review" }),
      el("span", { text: "Read as " + LABELS[block.region_class].toLowerCase() + " at " + fmt(block.confidence, 2) + ". " + (block.hold_reason || "") + ". Nothing has been written into the note." }),
    ]);
  }

  function renderContent(block, lede) {
    var c = block.content;
    if (!c) return el("div", { class: "refused", text: "This region was admitted but produced no content." });
    if (c.kind === "prose") {
      var wrap = el("div");
      var paragraphs = c.markdown.split(/\n{2,}/).map(function (p) { return p.replace(/\n/g, " ").trim(); }).filter(Boolean);
      paragraphs.forEach(function (p, i) { wrap.appendChild(el("p", { class: lede && i === 0 ? "lede" : null, text: p })); });
      return wrap;
    }
    if (c.kind === "formula") {
      var eq = el("div", { class: "eq" });
      var tex = el("span");
      try { window.katex.render(c.tex, tex, { displayMode: false, throwOnError: false }); }
      catch (e) { tex.textContent = c.tex; tex.className = "mono"; }
      eq.appendChild(tex);
      if (c.answer !== null && c.answer !== undefined) eq.appendChild(el("span", { class: "answer", text: "= " + sig(c.answer) }));
      var meta = [];
      var bound = Object.keys(c.bound || {});
      if (bound.length) meta.push("with " + bound.map(function (k) { return k + " = " + sig(c.bound[k]); }).join(", "));
      if (c.free && c.free.length) meta.push("free: " + c.free.join(", "));
      return el("div", {}, [eq, el("p", { class: "eqsrc" }, [el("code", { text: c.tex }), el("span", { text: meta.join(" · ") || (c.expr ? "typeset and solved" : "typeset only") })])]);
    }
    if (c.kind === "table") return markdownTable(c.markdown);
    if (c.kind === "figure") return c.refused ? el("div", { class: "refused", text: "No figure drawn: " + c.refused + "." }) : figure(c);
    return el("div", { class: "refused", text: "Unknown content kind " + c.kind });
  }

  function markdownTable(md) {
    var lines = md.split("\n").map(function (l) { return l.trim(); }).filter(function (l) { return l.charAt(0) === "|"; });
    var cells = function (l) { return l.replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map(function (s) { return s.trim().replace(/\\\|/g, "|"); }); };
    var rows = lines.filter(function (l) { return !/^\|[\s:|-]+\|$/.test(l); }).map(cells);
    var table = el("table");
    if (!rows.length) return el("div", { class: "refused", text: "Empty table." });
    table.appendChild(el("thead", {}, [el("tr", {}, rows[0].map(function (h) { return el("th", { text: h }); }))]));
    table.appendChild(el("tbody", {}, rows.slice(1).map(function (r) { return el("tr", {}, r.map(function (d) { return el("td", { text: d }); })); })));
    return table;
  }

  function figure(c) {
    var W = 536, H = 228, L = 44, R = 16, T = 14, B = 30;
    var xs = c.series_x, ys = c.series_y;
    var finite = ys.filter(function (y) { return y !== null && isFinite(y); });
    var y0 = c.y_clip ? c.y_clip[0] : Math.min.apply(null, finite), y1 = c.y_clip ? c.y_clip[1] : Math.max.apply(null, finite);
    if (y0 === y1) { y0 -= 1; y1 += 1; }
    var x0 = xs[0], x1 = xs[xs.length - 1];
    var px = function (x) { return L + (x - x0) / (x1 - x0) * (W - L - R); };
    var py = function (y) { return T + (y1 - Math.min(Math.max(y, y0), y1)) / (y1 - y0) * (H - T - B); };
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Plot of " + (c.source_expr || "the formula") + " against " + c.free_symbol });
    var grid = svgEl("g", { stroke: "#e4ded0", "stroke-width": 1 });
    for (var g = 0; g <= 4; g++) grid.appendChild(svgEl("path", { d: "M" + L + " " + (T + g * (H - T - B) / 4) + "h" + (W - L - R) }));
    svg.appendChild(grid);
    svg.appendChild(svgEl("path", { d: "M" + L + " " + T + "V" + (H - B) + "H" + (W - R), stroke: "#b5ad9c", fill: "none" }));
    var ticks = svgEl("g", { "font-family": "IBM Plex Mono", "font-size": 9, fill: "#746b59" });
    ticks.appendChild(svgEl("text", { x: L - 6, y: T + 3, "text-anchor": "end" }, sig(y1)));
    ticks.appendChild(svgEl("text", { x: L - 6, y: H - B + 3, "text-anchor": "end" }, sig(y0)));
    ticks.appendChild(svgEl("text", { x: L, y: H - B + 16 }, sig(x0)));
    ticks.appendChild(svgEl("text", { x: W - R, y: H - B + 16, "text-anchor": "end" }, sig(x1) + " " + c.free_symbol));
    svg.appendChild(ticks);
    // A None in the series is a pole or a gap: start a new subpath rather than drawing through it.
    var d = "", pen = false, last = null;
    xs.forEach(function (x, i) {
      var y = ys[i];
      if (y === null || !isFinite(y)) { pen = false; return; }
      d += (pen ? "L" : "M") + px(x).toFixed(1) + " " + py(y).toFixed(1);
      pen = true; last = [px(x), py(y)];
    });
    svg.appendChild(svgEl("path", { d: d, fill: "none", stroke: "var(--series-1)", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    if (last) {
      svg.appendChild(svgEl("circle", { cx: last[0], cy: last[1], r: 3, fill: "var(--series-1)", stroke: "#fffdf7", "stroke-width": 1.8 }));
      svg.appendChild(svgEl("text", { x: Math.min(last[0], W - R) - 6, y: Math.max(T + 10, last[1] - 8), "text-anchor": "end", "font-family": "IBM Plex Sans", "font-size": 10.5, "font-weight": 500, fill: "#2a78d6" }, c.source_expr || "f"));
    }
    return el("figure", {}, [svg, el("figcaption", {}, [
      el("span", { text: c.caption || "" }),
      el("span", { class: "mono", text: (c.range_source ? "range: " + c.range_source : "") + (c.y_clip ? " · y clipped" : "") }),
    ])]);
  }

  function openSource(note, focusId) {
    var W = note.source_width, H = note.source_height;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "The source page with every detected region outlined" });
    svg.appendChild(svgEl("image", { href: "/api/notes/" + encodeURIComponent(note.note_id) + "/page.png", x: 0, y: 0, width: W, height: H }));
    var fs = Math.max(12, W / 70);
    note.blocks.forEach(function (b) {
      var colour = b.held ? "#b5733a" : "#7e704f";
      svg.appendChild(svgEl("rect", { x: b.box.x, y: b.box.y, width: b.box.w, height: b.box.h, fill: b.id === focusId ? "rgba(126,112,79,.08)" : "none",
        stroke: colour, "stroke-width": b.id === focusId ? 3 : 1.5, "stroke-dasharray": b.held ? "8 5" : "none" }));
      svg.appendChild(svgEl("text", { x: b.box.x + 2, y: Math.max(fs, b.box.y - 5), "font-family": "IBM Plex Mono", "font-size": fs, fill: colour }, b.region_class + " " + fmt(b.confidence, 2)));
    });
    $("full-body").innerHTML = "";
    $("full-body").appendChild(svg);
    $("full-meta").textContent = W + " × " + H + " · " + note.blocks.length + " regions";
    $("full").showModal();
  }
  $("full-close").addEventListener("click", function () { $("full").close(); });
  $("full").addEventListener("click", function (e) { if (e.target === $("full")) $("full").close(); });

  // ================================================================== LABEL & TRAIN

  var ed = { page: null, img: null, regions: [], sel: -1, cls: "THEORY", drag: null, dirty: false, canvas: null };

  $("upload").addEventListener("click", function () { $("file-pages").click(); });
  $("file-pages").addEventListener("change", function (e) {
    var files = Array.prototype.slice.call(e.target.files);
    e.target.value = "";
    if (files.length) uploadPages(files);
  });

  function refreshPages() {
    return api("/pages").then(function (pages) {
      var list = $("pages");
      list.innerHTML = "";
      var done = pages.filter(function (p) { return p.labelled; }).length;
      $("pagecount").textContent = pages.length ? done + "/" + pages.length + " labelled" : "";
      if (!pages.length) list.appendChild(el("p", { class: "empty-side", text: "No pages yet. Upload photos of your notes to start labelling." }));
      pages.forEach(function (p) {
        list.appendChild(el("button", { class: "nb", "aria-current": p.id === state.pageId, onclick: function () { guardUnsaved(function () { openPage(p.id); }); } }, [
          el("span", { text: p.source_name || p.id }),
          el("span", { class: "n" + (p.labelled ? " done" : ""), text: (p.labelled ? "✓ " : "") + p.regions + " · " + p.split }),
        ]));
      });
      state.pages = pages;
      return pages;
    });
  }

  function guardUnsaved(next) {
    if (!ed.dirty || window.confirm("This page has unsaved labels. Discard them?")) { ed.dirty = false; next(); }
  }

  function uploadPages(files) {
    renderLabeler();
    var i = 0, first = null;
    var step = function () {
      if (i >= files.length) { refreshPages().then(function () { if (first) openPage(first); }); refreshStats(); return; }
      $("hint").textContent = "Uploading and proposing boxes for page " + (i + 1) + " of " + files.length + "…";
      postFile("/pages", files[i]).then(function (rec) { if (!first) first = rec.id; i++; step(); })
        .catch(function (err) { $("hint").textContent = files[i].name + ": " + err.message; i++; step(); });
    };
    step();
  }

  function renderLabeler() {
    if (main.dataset.view === "label") return;
    main.dataset.view = "label";
    main.className = "labeler";
    main.innerHTML = "";
    var chips = CLASSES.map(function (c, i) {
      return el("button", { class: "chip", "data-cls": c, "aria-pressed": c === ed.cls, onclick: function () { setClass(c); } },
        [el("i", { style: "background:" + COLOURS[c] }), LABELS[c], el("kbd", { text: String(i + 1) })]);
    });
    var left = el("section", {}, [
      el("div", { class: "toolbar" }, chips.concat([el("span", { class: "spacer" }),
        el("button", { class: "btn", id: "btn-compile-page", onclick: compileCurrentPage, text: "Compile with these boxes" }),
        el("button", { class: "btn solid", id: "btn-save", onclick: function () { saveLabels(true); }, text: "Save & next" })])),
      el("div", { class: "stage", id: "stage" }, [el("div", { class: "state", style: "margin:80px auto" }, [el("p", { text: "Upload pages, or pick one from the list." })])]),
      el("p", { class: "hint", id: "hint", text: "Drag on empty paper to draw a box · click to select · drag to move · drag a corner to resize · 1–5 set class · Delete removes · Ctrl+S saves · Enter saves and moves on" }),
    ]);
    var right = el("aside", {}, [
      el("div", { class: "panel" }, [el("h3", { text: "Your labelled data" }), el("div", { id: "stats" })]),
      el("div", { class: "panel" }, [
        el("h3", { text: "Train on your pages" }),
        el("p", { text: "Fine-tunes a classifier from its synthetic weights on every labelled page. Scored before and after on your own held-out pages." }),
        el("div", { class: "row2" }, [el("select", { id: "train-backbone", "aria-label": "Backbone to fine-tune" },
          ["squeezenet1_1", "mobilenet_v2", "resnet18", "palimpsest_net"].map(function (b) { return el("option", { value: b, text: BACKBONE_NAMES[b] }); })),
          el("input", { id: "train-epochs", type: "number", min: 1, max: 50, value: 8, "aria-label": "Epochs" })]),
        el("button", { class: "btn solid", id: "train-go", onclick: startTraining, text: "Train" }),
        el("div", { class: "meter" }, [el("span", { id: "train-meter" })]),
        el("div", { class: "result", id: "train-result" }),
      ]),
    ]);
    main.appendChild(left);
    main.appendChild(right);
    refreshStats();
    pollTraining();
  }

  function refreshStats() {
    return api("/pages/stats").then(function (s) {
      var box = $("stats");
      if (!box) return;
      box.innerHTML = "";
      box.appendChild(el("p", { text: s.labelled_pages + " pages labelled (" + s.pages_by_split.train + " train · " + s.pages_by_split.val + " val · " + s.pages_by_split.test + " test), " + s.regions + " regions." }));
      var t = el("table", { class: "counts" }, [el("thead", {}, [el("tr", {}, ["Class", "Train", "Val", "Test"].map(function (h) { return el("th", { text: h }); }))])]);
      var tb = el("tbody");
      CLASSES.forEach(function (c) {
        var row = s.per_class[c];
        tb.appendChild(el("tr", {}, [el("td", { text: LABELS[c] }),
          el("td", { class: row.train < 10 ? "low" : null, text: String(row.train) }), el("td", { text: String(row.val) }), el("td", { text: String(row.test) })]));
      });
      t.appendChild(tb);
      box.appendChild(t);
    });
  }

  function openPage(id) {
    renderLabeler();
    state.pageId = id;
    refreshPages();
    return api("/pages/" + encodeURIComponent(id)).then(function (page) {
      return loadImage("/api/pages/" + encodeURIComponent(id) + "/image.png").then(function (img) {
        ed.page = page; ed.img = img; ed.sel = -1; ed.dirty = false;
        ed.regions = page.regions.map(function (r) { return { box: r.box.slice(), cls: r["class"], proposed: !page.labelled, conf: r.confidence }; });
        var canvas = el("canvas", { width: img.naturalWidth, height: img.naturalHeight, tabindex: 0, "aria-label": "Page " + (page.source_name || id) + ". Draw boxes around regions." });
        $("stage").innerHTML = "";
        $("stage").appendChild(canvas);
        ed.canvas = canvas;
        bindCanvas(canvas);
        draw();
        $("hint").textContent = (page.labelled ? "Labelled page" : "Unlabelled: dashed boxes are the machine's proposals, correct them and save") + " · split " + page.split + " · " + page.width + "×" + page.height;
      });
    }).catch(function (err) { $("hint").textContent = err.message; });
  }

  function setClass(c) {
    ed.cls = c;
    document.querySelectorAll(".chip").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.cls === c); });
    if (ed.sel >= 0) { ed.regions[ed.sel].cls = c; ed.regions[ed.sel].proposed = false; ed.dirty = true; draw(); }
  }

  function toImage(e) {
    var r = ed.canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * ed.canvas.width / r.width, (e.clientY - r.top) * ed.canvas.height / r.height];
  }
  function handleSize() { return 10 * ed.canvas.width / ed.canvas.getBoundingClientRect().width; }
  function hitCorner(p) {
    if (ed.sel < 0) return -1;
    var b = ed.regions[ed.sel].box, s = handleSize();
    var corners = [[b[0], b[1]], [b[0] + b[2], b[1]], [b[0], b[1] + b[3]], [b[0] + b[2], b[1] + b[3]]];
    for (var i = 0; i < 4; i++) if (Math.abs(p[0] - corners[i][0]) < s && Math.abs(p[1] - corners[i][1]) < s) return i;
    return -1;
  }
  function hitBox(p) {
    // Smallest containing box wins, so a region inside another stays reachable.
    var best = -1, area = Infinity;
    ed.regions.forEach(function (r, i) {
      var b = r.box;
      if (p[0] >= b[0] && p[0] <= b[0] + b[2] && p[1] >= b[1] && p[1] <= b[1] + b[3] && b[2] * b[3] < area) { best = i; area = b[2] * b[3]; }
    });
    return best;
  }

  function bindCanvas(c) {
    c.addEventListener("pointerdown", function (e) {
      c.setPointerCapture(e.pointerId);
      c.focus();
      var p = toImage(e), corner = hitCorner(p);
      if (corner >= 0) { ed.drag = { mode: "resize", corner: corner, start: p, orig: ed.regions[ed.sel].box.slice() }; return; }
      var hit = hitBox(p);
      if (hit >= 0) {
        ed.sel = hit;
        setClassChips(ed.regions[hit].cls);
        ed.drag = { mode: "move", start: p, orig: ed.regions[hit].box.slice() };
      } else {
        ed.regions.push({ box: [p[0], p[1], 0, 0], cls: ed.cls, proposed: false });
        ed.sel = ed.regions.length - 1;
        ed.drag = { mode: "create", start: p };
      }
      draw();
    });
    c.addEventListener("pointermove", function (e) {
      if (!ed.drag) { c.style.cursor = hitCorner(toImage(e)) >= 0 ? "nwse-resize" : hitBox(toImage(e)) >= 0 ? "move" : "crosshair"; return; }
      var p = toImage(e), d = ed.drag, r = ed.regions[ed.sel];
      if (d.mode === "create") {
        r.box = [Math.min(d.start[0], p[0]), Math.min(d.start[1], p[1]), Math.abs(p[0] - d.start[0]), Math.abs(p[1] - d.start[1])];
      } else if (d.mode === "move") {
        r.box = [d.orig[0] + p[0] - d.start[0], d.orig[1] + p[1] - d.start[1], d.orig[2], d.orig[3]];
      } else {
        var x0 = d.orig[0], y0 = d.orig[1], x1 = x0 + d.orig[2], y1 = y0 + d.orig[3];
        if (d.corner === 0 || d.corner === 2) x0 = p[0]; else x1 = p[0];
        if (d.corner === 0 || d.corner === 1) y0 = p[1]; else y1 = p[1];
        r.box = [Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0)];
      }
      r.proposed = false;
      ed.dirty = true;
      draw();
    });
    var end = function () {
      if (!ed.drag) return;
      var r = ed.regions[ed.sel];
      if (ed.drag.mode === "create" && (r.box[2] < 8 || r.box[3] < 8)) { ed.regions.splice(ed.sel, 1); ed.sel = -1; }
      else if (ed.drag.mode === "move" && r.box[0] === ed.drag.orig[0] && r.box[1] === ed.drag.orig[1]) { /* a click, not a move */ }
      ed.drag = null;
      draw();
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
  }
  function setClassChips(c) {
    ed.cls = c;
    document.querySelectorAll(".chip").forEach(function (b) { b.setAttribute("aria-pressed", b.dataset.cls === c); });
  }

  function draw() {
    var c = ed.canvas, ctx = c.getContext("2d");
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(ed.img, 0, 0);
    var unit = c.width / c.getBoundingClientRect().width;
    ed.regions.forEach(function (r, i) {
      var b = r.box, col = COLOURS[r.cls], selected = i === ed.sel;
      ctx.save();
      ctx.lineWidth = (selected ? 3 : 2) * unit;
      ctx.strokeStyle = col;
      ctx.setLineDash(r.proposed ? [7 * unit, 5 * unit] : []);
      ctx.fillStyle = selected ? col + "22" : col + "0d";
      ctx.fillRect(b[0], b[1], b[2], b[3]);
      ctx.strokeRect(b[0], b[1], b[2], b[3]);
      ctx.setLineDash([]);
      var label = LABELS[r.cls] + (r.proposed && r.conf ? " " + r.conf.toFixed(2) + "?" : "");
      ctx.font = "500 " + 12 * unit + "px IBM Plex Sans, sans-serif";
      var tw = ctx.measureText(label).width + 10 * unit, th = 17 * unit;
      var ty = b[1] - th >= 0 ? b[1] - th : b[1];
      ctx.fillStyle = col;
      ctx.fillRect(b[0], ty, tw, th);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, b[0] + 5 * unit, ty + 12.5 * unit);
      if (selected) {
        var s = 5 * unit;
        [[b[0], b[1]], [b[0] + b[2], b[1]], [b[0], b[1] + b[3]], [b[0] + b[2], b[1] + b[3]]].forEach(function (q) {
          ctx.fillStyle = "#fff"; ctx.fillRect(q[0] - s, q[1] - s, 2 * s, 2 * s);
          ctx.strokeStyle = col; ctx.lineWidth = 1.5 * unit; ctx.strokeRect(q[0] - s, q[1] - s, 2 * s, 2 * s);
        });
      }
      ctx.restore();
    });
  }

  function saveLabels(next) {
    if (!ed.page) return Promise.resolve();
    var body = { regions: ed.regions.map(function (r) { return { box: r.box.map(Math.round), region_class: r.cls }; }) };
    return sendJSON("/pages/" + encodeURIComponent(ed.page.id) + "/labels", "PUT", body).then(function (rec) {
      ed.dirty = false;
      $("hint").textContent = "Saved " + rec.regions.length + " regions.";
      refreshStats();
      return refreshPages().then(function (pages) {
        if (!next) return openPage(ed.page.id);
        var idx = pages.findIndex(function (p) { return p.id === ed.page.id; });
        var after = pages.slice(idx + 1).concat(pages.slice(0, idx)).find(function (p) { return !p.labelled; });
        if (after) return openPage(after.id);
        $("hint").textContent = "Saved. Every page is labelled, so Train is ready when you are.";
        return openPage(ed.page.id);
      });
    }).catch(function (err) { $("hint").textContent = "Not saved: " + err.message; });
  }

  function compileCurrentPage() {
    if (!ed.page) return;
    var go = function () {
      setView("notebook");
      showBusy("Compiling your boxes", "Classifying and reading exactly the regions you drew.");
      api("/pages/" + encodeURIComponent(ed.page.id) + "/compile", { method: "POST" })
        .then(function (note) { state.noteId = note.note_id; refreshNotes(); main.dataset.view = ""; return renderNote(note); })
        .catch(function (err) { showError("The page did not compile", err); });
    };
    if (ed.dirty) saveLabels(false).then(go); else go();
  }

  // Arrow keys are the whole point of the card: a queue of yes/no judgements
  // is only fast if answering costs one keystroke and never needs the mouse.
  document.addEventListener("keydown", function (e) {
    if (state.view !== "fix") return;
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") {
      // Inside the correction form, Escape backs out and Ctrl+Enter saves.
      if (e.key === "Escape") { renderFixCancel(); e.preventDefault(); }
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
        var save = document.querySelector(".cardform .fixsave");
        if (save) { save.click(); e.preventDefault(); }
      }
      return;
    }
    var card = document.querySelector(".card");
    if (!card || !card.fixSend) return;
    if (e.key === "ArrowRight") {
      card.fixSend({ should_be: card.fixRegion.read_as,
                     correct_class: card.fixRegion.region_class }, "right");
      e.preventDefault();
    } else if (e.key === "ArrowLeft") {
      openFixForm(card, card.fixRegion, card.fixSend);
      e.preventDefault();
    }
  });

  function renderFixCancel() {
    var slot = document.querySelector(".cardform");
    var card = document.querySelector(".card");
    if (!slot || !card) return;
    slot.textContent = ""; delete slot.dataset.open;
    var acts = card.querySelector(".cardacts");
    if (acts) acts.hidden = false;
    card.focus();
  }

  document.addEventListener("keydown", function (e) {
    if (state.view !== "label" || !ed.page) return;
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    var n = parseInt(e.key, 10);
    if (n >= 1 && n <= 5) { setClass(CLASSES[n - 1]); e.preventDefault(); }
    else if ((e.key === "Delete" || e.key === "Backspace") && ed.sel >= 0) { ed.regions.splice(ed.sel, 1); ed.sel = -1; ed.dirty = true; draw(); e.preventDefault(); }
    else if (e.key === "Escape") { ed.sel = -1; draw(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { saveLabels(false); e.preventDefault(); }
    else if (e.key === "Enter") { saveLabels(true); e.preventDefault(); }
  });
  window.addEventListener("resize", function () { if (ed.canvas && state.view === "label") draw(); });

  // ------------------------------------------------------------------ training

  var pollTimer = null;
  function startTraining() {
    var body = { backbone: $("train-backbone").value, epochs: parseInt($("train-epochs").value, 10) || 8 };
    $("train-go").disabled = true;
    sendJSON("/train", "POST", body).then(function () { pollTraining(); })
      .catch(function (err) { $("train-result").innerHTML = ""; $("train-result").appendChild(el("span", { class: "err", text: err.message })); $("train-go").disabled = false; });
  }
  function pollTraining() {
    clearTimeout(pollTimer);
    api("/train/status").then(function (s) {
      var out = $("train-result");
      if (!out) return;
      out.innerHTML = "";
      var meter = $("train-meter");
      var frac = s.state === "done" ? 1 : s.epochs ? (s.epoch || 0) / s.epochs : 0;
      meter.style.setProperty("--sx", frac);
      $("train-go").disabled = !!s.running;
      var line = function (k, v) { out.appendChild(el("div", {}, [k + " ", el("b", { text: v })])); };
      if (s.state === "idle") line("Status", "not trained yet");
      if (s.state === "building") line("Status", "building the dataset");
      if (s.state === "training") {
        line("Epoch", (s.epoch || 0) + " of " + s.epochs);
        if (s.val_macro_f1 !== undefined) line("Val macro-F1", fmt(s.val_macro_f1, 3));
      }
      if (s.before) line("Before, your test pages", "F1 " + fmt(s.before.macro_f1, 3) + " on " + s.before.samples + " regions");
      if (s.state === "done") {
        if (s.after) line("After fine-tuning", "F1 " + fmt(s.after.macro_f1, 3));
        line("Took", fmt(s.train_seconds, 0) + " s");
        refreshModels();
      }
      if (s.state === "failed") out.appendChild(el("div", { class: "err", text: "Failed: " + (s.error || "unknown error") }));
      if (s.running) pollTimer = setTimeout(pollTraining, 2000);
    }).catch(function () {});
  }


  // ================================================================== WRITE
  // The workbench: ink on the left, the compiled note on the right. The page
  // is a real canvas at a fixed 1600 x 1200, which is the pipeline's working
  // size, so a drawn page is never resampled and the boxes that come back sit
  // exactly on the strokes. Strokes are kept as point lists, not pixels, so
  // undo is "drop the last stroke and redraw", and they survive a reload.

  // 1600x1500 rather than 4:3, so the sheet reaches the bottom of the pane
  // instead of leaving a band of dead surface under it. Kept in step with
  // the `aspect-ratio` on `.sheet`.
  var PAD_W = 1600, PAD_H = 1500, PAPER = "#ffffff", INK = "#2b2620";
  var WIDTHS = { fine: 2.6, medium: 4, bold: 6 };
  var pad = { strokes: [], live: true, tool: "pen", width: "medium", note: null, busy: false, timer: null, dirty: false, hover: null, rev: 0, compiledRev: -1 };
  try { pad.strokes = JSON.parse(localStorage.getItem("palimpsest.pad") || "[]"); } catch (e) { pad.strokes = []; }
  function savePad() { try { localStorage.setItem("palimpsest.pad", JSON.stringify(pad.strokes)); } catch (e) { /* storage full or blocked: drawing still works */ } }

  $("newpage").addEventListener("click", function () {
    if (pad.strokes.length && !window.confirm("Start a new page? The current drawing is cleared.")) return;
    pad.strokes = []; pad.note = null; pad.rev++; savePad();
    if (state.view === "write") renderWrite(); else setView("write");
  });

  function leaveWrite() { clearTimeout(pad.timer); }

  function renderWrite() {
    main.className = "";
    main.innerHTML = "";
    var tool = function (id, label, pressed, onclick, title) {
      return el("button", { class: "tool", id: id, "aria-pressed": pressed === undefined ? null : String(pressed), onclick: onclick, title: title || null, text: label });
    };
    var widthSel = el("select", { id: "pad-width", "aria-label": "Pen width", style: "width:auto;height:30px", onchange: function (e) { pad.width = e.target.value; } },
      ["fine", "medium", "bold"].map(function (w) { return el("option", { value: w, selected: w === pad.width, text: w[0].toUpperCase() + w.slice(1) }); }));
    var bar = el("div", { class: "inkbar" }, [
      tool("pad-pen", "Pen", pad.tool === "pen", function () { setTool("pen"); }, "P"),
      tool("pad-eraser", "Eraser", pad.tool === "eraser", function () { setTool("eraser"); }, "E"),
      widthSel,
      el("span", { class: "sep" }),
      tool("pad-undo", "Undo", undefined, undo, "Ctrl+Z"),
      tool("pad-clear", "Clear", undefined, function () { if (!pad.strokes.length || window.confirm("Clear the page?")) { pad.strokes = []; pad.note = null; changed(); } }),
      el("span", { class: "spacer" }),
      el("label", {}, [el("input", { type: "checkbox", id: "pad-live", checked: pad.live, onchange: function (e) { pad.live = e.target.checked; } }), "Live"]),
      tool("pad-train", "Add to training pages", undefined, addToTraining, "Send this page to Label, where you box and label its regions"),
      el("button", { class: "btn solid", id: "pad-compile", onclick: function () { compilePad(true); }, text: "Compile" }),
    ]);
    var canvas = el("canvas", { id: "pad", width: PAD_W, height: PAD_H, tabindex: 0, "aria-label": "Writing page. Draw with the mouse or a pen." });
    var overlay = el("canvas", { id: "padov", width: PAD_W, height: PAD_H, "aria-hidden": "true" });
    var left = el("section", { class: "pane-ink" }, [bar,
      el("div", { class: "paper" }, [el("div", { class: "sheet" }, [canvas, overlay])]),
      el("div", { class: "inkstatus", id: "pad-status" })]);
    var splitter = el("div", { class: "splitter", role: "separator", "aria-orientation": "vertical", "aria-label": "Resize panes", tabindex: 0 });
    var right = el("section", { class: "pane-note doc", id: "pad-note" });
    var bench = el("div", { class: "bench", id: "bench" }, [left, splitter, right]);
    main.appendChild(bench);

    pad.canvas = canvas; pad.overlay = overlay;
    bindPad(canvas);
    bindSplitter(splitter, bench);
    redraw();
    if (pad.note && pad.compiledRev === pad.rev) showPadNote(pad.note); else padEmpty();
    padStatus();
  }

  function padEmpty() {
    var right = $("pad-note");
    if (!right) return;
    right.innerHTML = "";
    right.appendChild(el("div", { class: "state" }, [
      el("h2", { text: pad.strokes.length ? "Ready to compile" : "Write on the page" }),
      el("p", { text: pad.strokes.length ? "Press Compile, or pause with Live on." : "Prose, a formula like rho = lambda / mu, a quick sketch of a graph, a table. Leave a little space between regions so they are found as separate blocks." }),
    ]));
  }

  function setTool(t) {
    pad.tool = t;
    if ($("pad-pen")) { $("pad-pen").setAttribute("aria-pressed", t === "pen"); $("pad-eraser").setAttribute("aria-pressed", t === "eraser"); }
    if (pad.canvas) pad.canvas.classList.toggle("erasing", t === "eraser");
  }

  function padPoint(e) {
    var r = pad.canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * PAD_W / r.width, (e.clientY - r.top) * PAD_H / r.height, e.timeStamp];
  }

  // Width tapers with speed, as a nib does: slow strokes are heavier. A pen
  // that reports pressure uses that instead; a mouse reports a constant 0.5.
  function segmentWidth(stroke, a, b, pressure) {
    var base = WIDTHS[stroke.w] || WIDTHS.medium;
    if (stroke.eraser) return base * 6;
    if (pressure !== null) return base * (0.5 + pressure);
    var dt = Math.max(1, b[2] - a[2]);
    var v = Math.hypot(b[0] - a[0], b[1] - a[1]) / dt;   // canvas px per ms
    return base * Math.max(0.55, Math.min(1.35, 1.35 - v * 0.18));
  }

  // The eraser removes pixels rather than painting white ones. It used to
  // paint PAPER, which worked only because the bitmap was an opaque white
  // page; now that the paper and its dot grid are the CSS background, a
  // white stroke would scrub a solid patch across the grid and leave it
  // there. `destination-out` takes the ink back out and lets the page show.
  function inkMode(ctx, eraser) {
    ctx.globalCompositeOperation = eraser ? "destination-out" : "source-over";
    return eraser ? "#000" : INK;   // any opaque colour erases; alpha is what counts
  }

  function drawSegment(ctx, stroke, i) {
    var a = stroke.p[i - 1], b = stroke.p[i];
    ctx.strokeStyle = inkMode(ctx, stroke.eraser);
    ctx.lineWidth = stroke.ws[i];
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
    ctx.globalCompositeOperation = "source-over";
  }

  function redraw() {
    if (!pad.canvas) return;
    var ctx = pad.canvas.getContext("2d");
    // Cleared, not filled: the paper and its dot grid are the CSS background
    // of this element, and a white fill here would cover them. It also keeps
    // the bitmap to ink alone, which is what `padBlob` has to send.
    ctx.clearRect(0, 0, PAD_W, PAD_H);
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    pad.strokes.forEach(function (s) {
      if (s.p.length === 1) { ctx.fillStyle = inkMode(ctx, s.eraser); ctx.beginPath(); ctx.arc(s.p[0][0], s.p[0][1], (WIDTHS[s.w] || 4) / 2, 0, 7); ctx.fill(); ctx.globalCompositeOperation = "source-over"; }
      for (var i = 1; i < s.p.length; i++) drawSegment(ctx, s, i);
    });
    drawOverlay();
  }

  function bindPad(c) {
    var current = null;
    c.addEventListener("pointerdown", function (e) {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      c.setPointerCapture(e.pointerId);
      clearTimeout(pad.timer);
      current = { p: [padPoint(e)], ws: [0], w: pad.width, eraser: pad.tool === "eraser" };
      pad.strokes.push(current);
      var ctx = c.getContext("2d");
      ctx.fillStyle = inkMode(ctx, current.eraser);
      ctx.beginPath(); ctx.arc(current.p[0][0], current.p[0][1], (WIDTHS[current.w] || 4) / 2, 0, 7); ctx.fill();
      ctx.globalCompositeOperation = "source-over";
    });
    c.addEventListener("pointermove", function (e) {
      if (!current) { hoverRegion(padPoint(e)); return; }
      var events = e.getCoalescedEvents ? e.getCoalescedEvents() : [e];
      var ctx = c.getContext("2d");
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      events.forEach(function (ev) {
        var p = padPoint(ev), last = current.p[current.p.length - 1];
        if (Math.hypot(p[0] - last[0], p[1] - last[1]) < 1.2) return;
        var pressure = ev.pointerType === "pen" && ev.pressure > 0 ? ev.pressure : null;
        var w = segmentWidth(current, last, p, pressure);
        var prev = current.ws[current.ws.length - 1] || w;
        current.p.push(p);
        current.ws.push(prev ? prev * 0.6 + w * 0.4 : w);   // smoothed, so width never jumps
        drawSegment(ctx, current, current.p.length - 1);
      });
    });
    var end = function () {
      if (!current) return;
      // Store integers: a page is a few thousand points, and localStorage is small.
      current.p = current.p.map(function (q) { return [Math.round(q[0]), Math.round(q[1]), 0]; });
      current.ws = current.ws.map(function (w) { return Math.round(w * 10) / 10; });
      current = null;
      changed();
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("pointerleave", function () { if (!current) setHover(null); });
  }

  function changed() {
    pad.rev++;
    savePad();
    redraw();
    padStatus();
    clearTimeout(pad.timer);
    if (pad.live && pad.strokes.length) pad.timer = setTimeout(function () { compilePad(false); }, 2200);
    if (!pad.strokes.length) { pad.note = null; padEmpty(); }
  }

  function undo() {
    if (!pad.strokes.length) return;
    pad.strokes.pop();
    changed();
  }

  function padBlob() {
    // Composite the ink onto white before sending. The pad's own bitmap is
    // transparent so the CSS grid can show through it, and a transparent PNG
    // decoded with IMREAD_COLOR comes out black: the page would arrive as
    // solid ink. The grid is not drawn here and never reaches the compiler,
    // which is the point of keeping it in CSS.
    var flat = document.createElement("canvas");
    flat.width = PAD_W; flat.height = PAD_H;
    var ctx = flat.getContext("2d");
    ctx.fillStyle = PAPER;
    ctx.fillRect(0, 0, PAD_W, PAD_H);
    ctx.drawImage(pad.canvas, 0, 0);
    return new Promise(function (resolve) { flat.toBlob(resolve, "image/png"); });
  }

  function padStatus(extra) {
    var box = $("pad-status");
    if (!box) return;
    box.innerHTML = "";
    var item = function (label, value, cls) { box.appendChild(el("span", {}, [label + " ", el("b", { class: cls || null, text: value })])); };
    item("strokes", String(pad.strokes.filter(function (s) { return !s.eraser; }).length));
    if (pad.busy) box.appendChild(el("span", {}, [el("span", { class: "busy-dot" }), " compiling"]));
    var n = pad.note;
    if (n) {
      item("regions", String(n.blocks.length));
      item("compiled in", (n.budget.service_ms / 1000).toFixed(2) + " s");
      item("load factor", fmt(n.budget.load_factor, 2), n.budget.over_budget ? "warn" : "");
      if (pad.compiledRev !== pad.rev && !pad.busy) item("", "page changed since", "warn");
    }
    if (extra) box.appendChild(el("span", { text: extra }));
  }

  function compilePad(force) {
    if (!pad.canvas || !pad.strokes.length) return;
    if (pad.busy) { pad.again = true; return; }
    if (!force && pad.compiledRev === pad.rev) return;
    pad.busy = true;
    var rev = pad.rev;
    padStatus();
    if (!pad.note) {
      var right = $("pad-note");
      right.innerHTML = "";
      right.appendChild(el("div", { class: "state", role: "status" }, [el("div", { class: "spinner" }), el("h2", { text: "Reading the page" }), el("p", { text: "Finding regions, classifying them, and reading each one with the local model." })]));
    }
    padBlob().then(function (blob) {
      return postFile("/compile", new File([blob], "drawn-page.png", { type: "image/png" }), { title: "Drawn page", deskew: "false" });
    }).then(function (note) {
      pad.note = note;
      pad.compiledRev = rev;
      refreshNotes();
      if (state.view === "write") showPadNote(note);
    }).catch(function (err) {
      var right = $("pad-note");
      if (right && !pad.note) { right.innerHTML = ""; right.appendChild(el("div", { class: "state error", role: "alert" }, [el("h2", { text: "The page did not compile" }), el("p", { text: err.message })])); }
      padStatus("error: " + err.message);
    }).then(function () {
      pad.busy = false;
      padStatus();
      // Strokes added while this compile ran: run once more for the latest page.
      if (pad.again) { pad.again = false; if (pad.compiledRev !== pad.rev) compilePad(false); }
    });
  }

  function showPadNote(note) {
    var right = $("pad-note");
    if (!right) return;
    renderNote(note, right).then(function () {
      right.querySelectorAll(".row[data-block]").forEach(function (row) {
        row.addEventListener("mouseenter", function () { setHover(row.dataset.block, true); });
        row.addEventListener("mouseleave", function () { setHover(null, true); });
      });
      drawOverlay();
      padStatus();
    });
  }

  // Provenance as a live link, per the style lock: a block lights its region
  // on the page, and a region lights its block.
  function setHover(id, fromNote) {
    if (pad.hover === id) return;
    pad.hover = id;
    drawOverlay();
    var right = $("pad-note");
    if (!right) return;
    right.querySelectorAll(".row.lit").forEach(function (r) { r.classList.remove("lit"); });
    if (id) {
      var row = right.querySelector('.row[data-block="' + id + '"]');
      if (row) {
        row.classList.add("lit");
        if (!fromNote) {
          var rr = row.getBoundingClientRect(), pr = right.getBoundingClientRect();
          if (rr.top < pr.top || rr.bottom > pr.bottom) row.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
        }
      }
    }
  }

  function hoverRegion(p) {
    if (!pad.note || pad.compiledRev !== pad.rev) return setHover(null);
    var hit = null, area = Infinity;
    pad.note.blocks.forEach(function (b) {
      var x = b.box;
      if (p[0] >= x.x && p[0] <= x.x + x.w && p[1] >= x.y && p[1] <= x.y + x.h && x.w * x.h < area) { hit = b.id; area = x.w * x.h; }
    });
    setHover(hit);
  }

  function drawOverlay() {
    if (!pad.overlay) return;
    var ctx = pad.overlay.getContext("2d");
    ctx.clearRect(0, 0, PAD_W, PAD_H);
    if (!pad.note || pad.compiledRev !== pad.rev) return;   // stale boxes would point at the wrong ink
    pad.note.blocks.forEach(function (b) {
      var x = b.box, on = b.id === pad.hover, col = b.held ? "#b5733a" : "#7e704f";
      ctx.save();
      ctx.strokeStyle = col;
      ctx.globalAlpha = on ? 1 : 0.45;
      ctx.lineWidth = on ? 3 : 1.5;
      ctx.setLineDash(b.held ? [10, 6] : [6, 6]);
      ctx.strokeRect(x.x - 6, x.y - 6, x.w + 12, x.h + 12);
      if (on) { ctx.fillStyle = "rgba(126,112,79,.07)"; ctx.fillRect(x.x - 6, x.y - 6, x.w + 12, x.h + 12); }
      ctx.setLineDash([]);
      ctx.font = "500 20px IBM Plex Mono, monospace";
      ctx.fillStyle = col;
      ctx.fillText(b.region_class + " " + b.confidence.toFixed(2), x.x - 4, Math.max(20, x.y - 12));
      ctx.restore();
    });
  }

  function bindSplitter(s, bench) {
    var set = function (pct) { bench.style.setProperty("--left", Math.max(30, Math.min(75, pct)) + "%"); };
    try { var saved = localStorage.getItem("palimpsest.split"); if (saved) set(+saved); } catch (e) {}
    s.addEventListener("pointerdown", function (e) {
      s.setPointerCapture(e.pointerId);
      var move = function (ev) {
        var r = bench.getBoundingClientRect();
        var pct = (ev.clientX - r.left) / r.width * 100;
        set(pct);
        try { localStorage.setItem("palimpsest.split", String(Math.max(30, Math.min(75, pct)))); } catch (e2) {}
      };
      var up = function () { s.removeEventListener("pointermove", move); s.removeEventListener("pointerup", up); };
      s.addEventListener("pointermove", move);
      s.addEventListener("pointerup", up);
    });
    s.addEventListener("keydown", function (e) {
      var cur = parseFloat(getComputedStyle(bench).getPropertyValue("--left")) || 56;
      if (e.key === "ArrowLeft") { set(cur - 3); e.preventDefault(); }
      if (e.key === "ArrowRight") { set(cur + 3); e.preventDefault(); }
    });
  }

  function addToTraining() {
    if (!pad.strokes.length) return;
    var btn = $("pad-train");
    btn.disabled = true;
    padBlob().then(function (blob) {
      return postFile("/pages", new File([blob], "drawn-" + new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-") + ".png", { type: "image/png" }));
    }).then(function (rec) {
      state.pageId = rec.id;
      padStatus("added to Label as a " + rec.split + " page");
    }).catch(function (err) { padStatus("not added: " + err.message); })
      .then(function () { btn.disabled = false; });
  }

  document.addEventListener("keydown", function (e) {
    if (state.view !== "write") return;
    var tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { undo(); e.preventDefault(); }
    else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { compilePad(true); e.preventDefault(); }
    else if (!e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "p") setTool("pen");
    else if (!e.ctrlKey && !e.metaKey && e.key.toLowerCase() === "e") setTool("eraser");
  });

  // ------------------------------------------------------------------ boot

  refreshHealth();
  refreshModels().then(function () {
    if (["#label", "#write", "#fix"].indexOf(location.hash) >= 0) { setView(location.hash.slice(1)); return null; }
    setView("notebook");
    return refreshNotes();
  }).then(function (notes) {
    if (notes && notes.length) openNote(notes[0].note_id);
  }).catch(function (err) { showError("The backend is not reachable", err, function () { location.reload(); }); });
})();
