"use strict";

const metrics = {
  e2e_s: ["End-to-end · s", 1, "Includes prebuilt guest loading, sizing and proof traces, preprocessing, setup, proving, and serialization. Excludes compilation and verification."],
  prove_s: ["Prove · s", 1, "The full prove call only. Guest loading, tracing, preprocessing, and commitment setup are outside this timer."],
  peak_rss_bytes: ["Peak process memory · MiB", 1048576, "OS high-water resident memory over the entire proof process, including both verifications and teardown. Excludes child compilation. 1 MiB = 1,048,576 bytes."],
  verify_s: ["Parallel verify · s", 1, "Full verification with the topology-limited host-parallel pool. Pool setup is excluded."],
  verify_single_s: ["Single-thread verify · s", 1, "The second full verification, with one thread. Pool setup is excluded."],
  proof_bytes: ["Serialized proof · bytes", 1, "The complete serialized proof, not only the commitment backend payload."],
  process_s: ["Process · s", 1, "Parent-observed process launch through reaped exit: loading, proof, both verifications, logging, and teardown. Compilation is excluded."],
  setup_s: ["Commitment setup · s", 1, "Commitment-scheme setup only; a component of end-to-end time."],
  trace_rows: ["Expanded trace rows", 1, "Unpadded expanded proof-trace rows, not source guest instructions. The input-size target is a separate dimension."],
  padded_trace_rows: ["Padded trace rows", 1, "The actual power-of-two proof-trace size. The input-size target is a separate dimension."],
};
const $ = (id) => document.getElementById(id);
const cache = new Map();
let index = {runs: []};
let tier = "main";
let limit = 30;
let generation = 0;
let visible = [];
let points = [];
let chartRows = [];

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function canonical(value) {
  if (Array.isArray(value)) return value.map(canonical);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
  }
  return value;
}

const identity = (value) => JSON.stringify(canonical(value));
const number = (value) => new Intl.NumberFormat("en", {maximumFractionDigits: 3}).format(value);
const short = (value) => typeof value === "string" ? value.slice(0, 12) : "unknown";

function group(record) {
  if (record.source.type === "manual") return "preview";
  return record.kind === "history" && ["main", "nightly"].includes(record.tier) ? record.tier : "other";
}

function rawLink(record) {
  return typeof record.raw === "string" && /^runs\/(github-[0-9a-f]{40}-[1-9][0-9]*-[1-9][0-9]*|preview-[0-9a-f]{64})\.json$/.test(record.raw) ? record.raw : null;
}

function buildSettings(build) {
  if (!build) return null;
  return Object.fromEntries([
    "toolchain", "cargo_version", "target_cpu", "target_triple", "target_features", "profile",
    "rustflags", "harness_files", "host",
  ].map((key) => [key, build[key] ?? null]).concat([["runner_sha256", build.runner?.sha256 ?? null]]));
}

function references(record, data) {
  return data.cases.flatMap((testCase) => {
    const legs = new Set([...data.builds.map((build) => build.leg), ...testCase.samples.map((sample) => sample.leg)]);
    if (legs.size === 0) legs.add("head");
    return [...legs].map((leg) => {
      const build = data.builds.find((item) => item.leg === leg);
      const definition = {
        tier: data.tier, kind: data.kind, workload: testCase.workload, pcs: testCase.pcs,
        backend: testCase.backend, physical_cores: testCase.physical_cores,
        scale_log2: testCase.scale_log2, input: testCase.input,
        metric_contract: data.suite.metric_contract, leg,
      };
      const environment = {host: data.host, cpu_ids: testCase.cpu_ids, build_settings: buildSettings(build)};
      return {record, data, testCase, leg, build, definition, environment, key: identity(definition), regime: identity(environment)};
    });
  });
}

function choices(select, entries, emptyLabel) {
  const previous = select.value;
  select.replaceChildren();
  for (const [key, label] of entries) {
    const option = element("option", label);
    option.value = key;
    select.append(option);
  }
  if (entries.length === 0) select.append(element("option", emptyLabel));
  select.disabled = entries.length === 0;
  if (entries.some(([key]) => key === previous)) select.value = previous;
}

function pointLabel(point) {
  const d = point.definition;
  return `${d.workload} / ${d.pcs} / ${d.backend} / ${d.physical_cores} physical cores / 2^${d.scale_log2} input / ${short(d.input.sha256)} / ${d.leg} (${d.tier}, ${d.kind})`;
}

function renderSelectors() {
  const unique = new Map(points.map((point) => [point.key, pointLabel(point)]));
  choices($("point-select"), [...unique].sort((a, b) => a[1].localeCompare(b[1], "en", {numeric: true})), "No benchmark cases in these runs");
  renderRegimes();
}

function renderRegimes() {
  const matching = points.filter((point) => point.key === $("point-select").value);
  const unique = new Map(matching.map((point) => [point.regime, point]));
  choices($("regime-select"), [...unique].map(([key, point], i) => [key,
    `Configuration ${i + 1}: ${point.data.host.id} / ${point.data.host.cpu_model} / ${point.build?.profile ?? "unknown profile"} / ${point.build?.target_cpu ?? "unknown target"} / ${point.build?.rustflags ?? "unknown flags"}`,
  ]), "No host/build configurations");
  render();
}

function summarize(record, reference) {
  const loaded = cache.get(record.key);
  const base = {record, reference, value: null, minimum: null, maximum: null, count: 0, expected: null};
  if (record.publication_status !== "ready" || loaded?.error) {
    return {...base, status: loaded?.error ? "missing" : record.publication_status, reason: loaded?.error || record.reason};
  }
  if (!reference) {
    return {...base, status: loaded?.data.status !== "complete" ? loaded?.data.status : "not comparable",
      reason: loaded?.data.reason || "This run has no case for the selected input and configuration."};
  }
  const {testCase, data, leg} = reference;
  const samples = testCase.samples.filter((sample) => sample.phase === "measure" && sample.leg === leg);
  base.expected = data.suite.repetitions;
  base.count = samples.filter((sample) => sample.status === "complete" && sample.exit_code === 0).length;
  if (testCase.status !== "complete") return {...base, status: testCase.status, reason: testCase.reason};
  if (reference.build?.status !== "complete" || !reference.build.binaries?.[testCase.pcs]) {
    return {...base, status: "incomplete", reason: "This point has no completed build and binary provenance."};
  }
  if (samples.length !== base.expected || base.count !== base.expected) {
    return {...base, status: "incomplete", reason: "Not every expected measured repetition completed successfully."};
  }
  const metric = $("metric-select").value;
  const values = samples.map((sample) => sample.metrics[metric]);
  if (values.some((value) => typeof value !== "number" || !Number.isFinite(value) || value < 0)) {
    return {...base, status: "missing", reason: "At least one measured repetition is missing this metric."};
  }
  values.sort((a, b) => a - b);
  const middle = Math.floor(values.length / 2);
  const median = values.length % 2 ? values[middle] : values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
  const unit = metrics[metric][1];
  return {...base, status: "complete", reason: null, value: median / unit, minimum: values[0] / unit, maximum: values.at(-1) / unit};
}

function appendMeta(parent, text) {
  parent.append(element("span", text, "meta"));
}

function renderRows(rows) {
  $("history-rows").replaceChildren();
  for (const row of rows) {
    const {record, reference} = row;
    const tr = element("tr");
    const commit = element("td");
    const rev = record.source.commit;
    if (typeof rev === "string" && /^[0-9a-f]{40}$/.test(rev)) {
      const link = element("a", short(rev), "mono");
      link.href = `https://github.com/a16z/jolt/commit/${rev}`;
      commit.append(link);
    } else commit.append(element("span", "Unknown revision"));
    appendMeta(commit, `Run started ${record.started_at.replace("T", " ").replace("+00:00", "Z")} (UTC)`);
    if (record.source.type === "github") {
      appendMeta(commit, `Run ${record.source.run_id}, attempt ${record.source.run_attempt}`);
      appendMeta(commit, `Workflow: ${record.source.conclusion}`);
      if (record.source.push_range) appendMeta(commit, `Main commit ${record.source.commit_index + 1} of ${record.source.push_range.commits} in the recorded range`);
    } else appendMeta(commit, `Manual preview: ${record.source.label}`);
    appendMeta(commit, `Run status: ${record.status}`);
    const status = element("td");
    const badge = element("span", row.status, "status");
    badge.dataset.status = row.status;
    status.append(badge);
    if (row.reason) appendMeta(status, row.reason);
    const value = element("td", row.value === null ? "—" : number(row.value), "number");
    if (row.value !== null) appendMeta(value, `${number(row.minimum)}–${number(row.maximum)}`);
    const count = element("td", row.expected === null ? "—" : `${row.count} / ${row.expected}`, "number");
    if (reference) appendMeta(count, `${reference.testCase.samples.filter((sample) => sample.phase === "warmup" && sample.leg === reference.leg).length} warmups in raw data`);
    const provenance = element("td");
    const path = rawLink(record);
    if (path) {
      const download = element("a", record.publication_status === "rejected" ? "Rejected raw JSON" : "Raw JSON");
      download.href = path;
      download.download = path.split("/").at(-1);
      provenance.append(download);
    } else appendMeta(provenance, "No raw JSON received");
    if (typeof record.push_index === "string" && /^runs\/github-[0-9a-f]{40}-[1-9][0-9]*-[1-9][0-9]*\.index\.json$/.test(record.push_index)) {
      const download = element("a", "Push index");
      download.href = record.push_index;
      download.download = record.push_index.split("/").at(-1);
      const line = element("div");
      line.append(download);
      provenance.append(line);
      appendMeta(provenance, "Ancestry checked; push start is producer-recorded, not independently verified.");
    }
    if (record.source.type === "github" && Number.isSafeInteger(record.source.run_id) && Number.isSafeInteger(record.source.run_attempt)) {
      const link = element("a", "Source run");
      link.href = `https://github.com/a16z/jolt/actions/runs/${record.source.run_id}/attempts/${record.source.run_attempt}`;
      const line = element("div");
      line.append(link);
      provenance.append(line);
    }
    appendMeta(provenance, `Schema ${record.benchmark_schema ?? "unknown"} / ${record.tier} / ${record.kind}`);
    if (reference?.build?.binaries?.[reference.testCase.pcs]) {
      appendMeta(provenance, `Binary ${short(reference.build.binaries[reference.testCase.pcs].sha256)}`);
    }
    const details = element("details");
    details.append(element("summary", "Build and source"), element("pre", JSON.stringify({
      source: record.source, publisher_revision: record.publisher_revision,
      raw_sha256: record.sha256, measured_revision: reference?.testCase.samples.find((sample) => sample.leg === reference.leg)?.revision ?? null,
      build: reference?.build ?? null, suite: reference?.data.suite ?? null,
    }, null, 2)));
    provenance.append(details);
    tr.append(commit, status, value, count, provenance);
    $("history-rows").append(tr);
  }
}


function svgElement(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderChart() {
  const chart = $("chart");
  chart.replaceChildren();
  const rows = [...chartRows].reverse();
  const values = rows.flatMap((row) => row.value === null ? [] : [row.value]);
  if (!values.length) {
    chart.append(element("p", "No complete measured values for this selection. Run and failure records remain in the table.", "chart-empty"));
    return;
  }
  const width = Math.max(260, chart.clientWidth);
  const height = 280;
  const left = 64;
  const right = width - 18;
  const top = 22;
  const bottom = 204;
  const maximum = Math.max(...values) || 1;
  const svg = svgElement("svg", {viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": `${metrics[$("metric-select").value][0]} by run; missing values break the line`});
  for (let i = 0; i <= 4; i++) {
    const y = top + (bottom - top) * i / 4;
    const tick = maximum * ((4 - i) / 4);
    const label = new Intl.NumberFormat("en", {notation: "compact", maximumSignificantDigits: 3}).format(tick);
    svg.append(svgElement("line", {x1: left, x2: right, y1: y, y2: y, class: "chart-grid"}));
    svg.append(svgElement("text", {x: left - 8, y: y + 4, "text-anchor": "end", class: "chart-label"}, label));
  }
  const xAt = (i) => rows.length === 1 ? (left + right) / 2 : left + (right - left) * i / (rows.length - 1);
  let segment = [];
  const flush = () => {
    if (segment.length > 1) svg.append(svgElement("polyline", {points: segment.join(" "), class: "chart-line"}));
    segment = [];
  };
  rows.forEach((row, i) => {
    if (row.value === null) flush();
    else segment.push(`${xAt(i)},${bottom - (row.value / maximum) * (bottom - top)}`);
  });
  flush();
  rows.forEach((row, i) => {
    const x = xAt(i);
    let mark;
    if (row.value === null) {
      mark = svgElement("path", {d: `M${x - 4},238 L${x + 4},246 M${x - 4},246 L${x + 4},238`, class: "chart-gap"});
    } else {
      const y = bottom - (row.value / maximum) * (bottom - top);
      mark = svgElement("circle", {cx: x, cy: y, r: 4.5, class: "chart-point"});
    }
    mark.append(svgElement("title", {}, `${short(row.record.source.commit)}: ${row.value === null ? row.status : number(row.value)}`));
    svg.append(mark);
  });
  svg.append(svgElement("text", {x: left - 8, y: 246, "text-anchor": "end", class: "chart-label"}, "No value"));
  svg.append(svgElement("text", {x: left, y: 273, class: "chart-label"}, rows[0].record.started_at.slice(0, 10)));
  if (rows.length > 1) svg.append(svgElement("text", {x: right, y: 273, "text-anchor": "end", class: "chart-label"}, rows.at(-1).record.started_at.slice(0, 10)));
  chart.append(svg);
}

function render() {
  const metric = metrics[$("metric-select").value];
  $("metric-boundary").textContent = metric[2];
  $("value-heading").textContent = metric[0];
  const matching = points.filter((point) => point.key === $("point-select").value && point.regime === $("regime-select").value);
  const byRun = new Map(matching.map((point) => [point.record.key, point]));
  chartRows = visible.map((record) => summarize(record, byRun.get(record.key)));
  const selected = matching[0];
  $("configuration-value").textContent = selected ? JSON.stringify({
    point: selected.definition, ...selected.environment,
  }, null, 2) : "No case-level configuration is available for these runs.";
  $("point-count").textContent = `${chartRows.filter((row) => row.value !== null).length} measured values / ${chartRows.length} run records`;
  $("empty-state").hidden = visible.length !== 0;
  renderRows(chartRows);
  renderChart();
}

async function fetchRecord(record) {
  if (cache.has(record.key) || record.publication_status !== "ready") return;
  try {
    const path = rawLink(record);
    if (!path) throw new Error("The history index has no safe raw JSON path.");
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Raw JSON could not be read (HTTP ${response.status}).`);
    const data = await response.json();
    if (data.schema_version !== 1 || !Array.isArray(data.cases) || !Array.isArray(data.builds)) {
      throw new Error("Unsupported or malformed raw benchmark JSON.");
    }
    cache.set(record.key, {data});
  } catch (error) {
    cache.set(record.key, {error: error.message});
  }
}

async function loadTier() {
  const request = ++generation;
  const all = index.runs.filter((record) => group(record) === tier);
  const batch = all.slice(0, limit);
  $("load-status").textContent = "Loading run records…";
  visible = [];
  points = [];
  chartRows = [];
  $("history-rows").replaceChildren();
  $("chart").replaceChildren(element("p", "Loading measurements…", "chart-empty"));
  $("point-count").textContent = "";
  $("window-label").textContent = "";
  $("configuration-value").textContent = "Loading configuration…";
  $("empty-state").hidden = true;
  for (const id of ["point-select", "regime-select", "metric-select"]) $(id).disabled = true;
  $("load-more").disabled = true;
  for (const button of document.querySelectorAll("[data-tier]")) {
    button.setAttribute("aria-pressed", String(button.dataset.tier === tier));
  }
  $("tier-description").textContent = {
    main: "Standard suite for main commits. Each source run and rerun attempt is retained separately.",
    nightly: "Wider nightly suite on main. Larger input sizes and repetitions are never pooled with standard runs.",
    other: "Unclassified or non-history runs. Their original tier and purpose remain visible; these are not main or nightly trends.",
    preview: "Manual data with no authenticated GitHub publication provenance. Original tiers and control purposes are preserved.",
  }[tier];
  $("notice").hidden = tier !== "preview";
  $("notice").textContent = tier === "preview" ? "Manual preview only. A pilot subset does not establish full-suite coverage or a commit-to-commit trend." : "";
  await Promise.all(batch.map(fetchRecord));
  if (request !== generation) return;
  visible = batch;
  points = batch.flatMap((record) => {
    const loaded = cache.get(record.key);
    return loaded?.data ? references(record, loaded.data) : [];
  });
  renderSelectors();
  $("metric-select").disabled = false;
  $("load-status").textContent = `${all.length} archived run records in this view. All statuses and attempts are retained.`;
  $("window-label").textContent = `Showing the newest ${batch.length} of ${all.length} run records. Dates use UTC.`;
  $("load-more").hidden = batch.length >= all.length;
  $("load-more").disabled = false;
}

$("point-select").addEventListener("change", renderRegimes);
$("regime-select").addEventListener("change", render);
$("metric-select").addEventListener("change", render);
$("load-more").addEventListener("click", () => { limit += 30; loadTier(); });
for (const button of document.querySelectorAll("[data-tier]")) {
  button.addEventListener("click", () => { tier = button.dataset.tier; limit = 30; loadTier(); });
}
window.addEventListener("resize", renderChart);

async function start() {
  try {
    const response = await fetch("history.json", {cache: "no-store"});
    if (!response.ok) throw new Error(`History index could not be read (HTTP ${response.status}).`);
    index = await response.json();
    if (index.format_version !== 1 || !Array.isArray(index.runs)) throw new Error("Unsupported history index.");
    for (const optional of ["other", "preview"]) {
      document.querySelector(`[data-tier="${optional}"]`).hidden = !index.runs.some((record) => group(record) === optional);
    }
    if (index.runs.length && index.runs.every((record) => group(record) === "preview")) tier = "preview";
    await loadTier();
  } catch (error) {
    $("load-status").textContent = "History is unavailable; no measurements have been inferred.";
    $("notice").hidden = false;
    $("notice").textContent = error.message;
  }
}

start();
