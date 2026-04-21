#!/usr/bin/env python3
"""Build a LaTeX perf report from core + modular aggregated traces."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "perf" / "report_tools"


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash ")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde ")
        .replace("^", r"\textasciicircum ")
    )


def classify_core(name: str) -> str:
    """Map a core span name to a coarse stage label."""
    n = name
    if n.startswith("prove_stage") or n == "prove":
        return "stage-root"
    if "DoryCommitment" in n or "multi_pair" in n or "G1::msm" in n or "msm_" in n:
        if "tier1" in n or "tier2" in n or "combine" in n or "setup_prover" in n:
            return "commitment"
        if "prove" in n or "create_evaluation_proof" in n or "verify" in n:
            return "opening_proof"
        if "compute" in n and "message" in n:
            return "opening_proof"
        return "pcs_misc"
    if n.startswith("Booleanity"):
        return "booleanity_sc"
    if n.startswith("InstructionRa"):
        return "instruction_ra_sc"
    if "InstructionReadRaf" in n or "BytecodeReadRaf" in n:
        return "read_raf_sc"
    if n.startswith("RamValCheck") or n.startswith("RamRa"):
        return "ram_sc"
    if "Registers" in n:
        return "registers_sc"
    if n.startswith("Outer") or n.startswith("Shift") or n.startswith("InstructionInput"):
        return "r1cs_sc"
    if "Polynomial" in n or "bind" in n or "EqPoly" in n or "Unipoly" in n or "UniPoly" in n:
        return "poly_ops"
    if "Gruen" in n:
        return "gruen"
    if "Witness" in n or "generate_and_commit" in n:
        return "witness_gen"
    if "Preprocess" in n:
        return "preprocess"
    if "Program::" in n or "trace" == n or n.startswith("Program"):
        return "host"
    return "other"


def classify_modular(name: str) -> str:
    n = name
    if n.startswith("Op::"):
        n = n[4:]
    if n in ("InstanceBind", "Bind"):
        return "op.bind"
    if n in ("InstanceReduce", "Evaluate"):
        return "op.reduce"
    if "Reduce" in n and "Segmented" in n:
        return "op.segmented_reduce"
    if n in ("Commit", "DoryScheme::commit"):
        return "op.commit"
    if "Materialize" in n:
        return "op.materialize"
    if "EqTable" in n or "EqProject" in n:
        return "op.eq_table"
    if "Gruen" in n or n.startswith("gruen"):
        return "op.gruen"
    if "Squeeze" in n or "AbsorbRound" in n or "Transcript" in n:
        return "op.transcript"
    if n.startswith("Batch") or "Checkpoint" in n:
        return "op.batch"
    if "Read" in n or "Raf" in n:
        return "sumcheck.read_raf"
    if "reduce_dense" in n or "reduce_tensor" in n:
        return "cpu.reduce"
    if "interpolate" in n:
        return "cpu.interpolate"
    if "DoryScheme" in n or "multi_pair" in n or "G1::msm" in n or "msm_" in n:
        return "pcs.dory"
    if "pm::" in n or "mb::" in n or n.startswith("r1cs::"):
        return "infra.module_boundary"
    if "Polynomial" in n or "bind" in n:
        return "poly_ops"
    if "Preprocess" in n:
        return "preprocess"
    if "Program::" in n or "trace" == n:
        return "host"
    if "ComputePower" in n or "AliasEval" in n or "CaptureScalar" in n:
        return "misc.compute"
    return "other"


def load(path):
    with open(path) as f:
        return json.load(f)


def _get(aggr, name):
    for r in aggr["spans"]:
        if r["name"] == name:
            return r
    return {"total_ms": 0, "self_ms": 0, "count": 0, "mean_total_us": 0}


def _tree_rows(aggr, root, max_depth=2, top_k=6, visited=None, depth=0):
    """Return list of LaTeX-formatted tabular rows for the subtree."""
    if visited is None:
        visited = set()
    if root in visited or depth > max_depth:
        return []
    visited = visited | {root}
    spans = {r["name"]: r for r in aggr["spans"]}
    pc = aggr.get("parent_children", {})

    rows = []
    if depth == 0:
        s = spans.get(root, {"total_ms": 0, "self_ms": 0, "count": 0})
        nm = latex_escape(root)
        rows.append(
            f"\\texttt{{{nm}}} & {fmt_ms(s['total_ms'])} & {fmt_ms(s['self_ms'])} & {s['count']} \\\\"
        )
    children = pc.get(root, {})
    children_sorted = sorted(
        children.items(),
        key=lambda x: -spans.get(x[0], {"total_ms": 0})["total_ms"],
    )[:top_k]
    for c, cnt in children_sorted:
        s = spans.get(c, {"total_ms": 0, "self_ms": 0, "count": 0})
        nm = latex_escape(c)
        prefix = "\\quad " * (depth + 1) + "\\textcolor{gray}{$\\hookrightarrow$} "
        rows.append(
            f"{prefix}\\texttt{{{nm}}} & {fmt_ms(s['total_ms'])} & {fmt_ms(s['self_ms'])} & {cnt} \\\\"
        )
        rows.extend(_tree_rows(aggr, c, max_depth, top_k, visited, depth + 1))
    return rows


def stage_rollup(spans, classifier):
    """Sum self_ms by stage, keep top spans per stage."""
    rollup = {}
    for r in spans:
        s = classifier(r["name"])
        if s not in rollup:
            rollup[s] = {"self_ms": 0.0, "total_ms": 0.0, "count": 0, "members": []}
        rollup[s]["self_ms"] += r["self_ms"]
        rollup[s]["total_ms"] += r["total_ms"]
        rollup[s]["count"] += r["count"]
        rollup[s]["members"].append(r)
    for k in rollup:
        rollup[k]["members"].sort(key=lambda x: -x["self_ms"])
    return rollup


def fmt_ms(v):
    if v >= 10000:
        return f"{v/1000:,.1f}\\,s"
    if v >= 100:
        return f"{v:,.0f}\\,ms"
    if v >= 1:
        return f"{v:,.1f}\\,ms"
    return f"{v*1000:,.0f}\\,\\textmu s"


def fmt_count(v):
    if v >= 1_000_000:
        return f"{v/1e6:.1f}M"
    if v >= 1000:
        return f"{v/1000:.1f}k"
    return str(v)


def make_tex(core, modular, core_top, mod_top, summary):
    core_stage = stage_rollup(core["spans"], classify_core)
    mod_stage = stage_rollup(modular["spans"], classify_modular)

    tex = []
    tex.append(r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{xcolor}
\usepackage{colortbl}
\usepackage[table]{xcolor}
\usepackage{siunitx}
\usepackage{tcolorbox}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[T1]{fontenc}
\usepackage{inconsolata}
\definecolor{slowred}{HTML}{B00020}
\definecolor{okgreen}{HTML}{2F7D32}
\definecolor{warn}{HTML}{B57900}
\definecolor{rowalt}{HTML}{F2F2F2}
\sisetup{group-separator={,}}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{5pt}
\title{\vspace{-1.6cm}\textbf{Jolt Perf Report}\\\large Modular stack vs.\ jolt-core, sha2-chain \texttt{log\_t=16}}
\author{}
\date{2026-04-20}
\begin{document}
\maketitle
""")

    # === Headline ===
    ratio = summary["modular_prove_ms"] / summary["core_prove_ms"]
    tex.append(r"\section*{Headline}")
    tex.append(r"\begin{tcolorbox}[colback=gray!6,colframe=gray!40,sharp corners]")
    tex.append(r"\par\medskip\noindent\begin{tabular}{l r r r}")
    tex.append(r"\textbf{Metric} & \textbf{Core} & \textbf{Modular} & \textbf{Ratio} \\ \midrule")
    for label, c, m in [
        ("prove\\_ms", summary["core_prove_ms"], summary["modular_prove_ms"]),
        ("verify\\_ms", summary["core_verify_ms"], summary["modular_verify_ms"]),
        ("peak RSS (MB)", summary["core_rss_mb"], summary["modular_rss_mb"]),
        ("proof bytes", summary["core_bytes"], summary["modular_bytes"]),
    ]:
        r_ = m / c if c else 0
        tex.append(
            f"{label} & {c:,.1f} & {m:,.1f} & {r_:.2f}x \\\\"
        )
    tex.append(r"\end{tabular}\par\medskip")
    tex.append(r"\end{tcolorbox}")
    tex.append(
        rf"\noindent Overall modular is \textbf{{{ratio:.2f}\texttimes{{}}}} slower than core on \texttt{{prove}}. "
        f"Both stacks produce valid proofs over the same workload "
        f"(sha2-chain, \\texttt{{--num-iters 16 --log-t 16}}, $\\sim 2^{{16}} = 65{{,}}536$ cycles).\\\\"
    )
    tex.append(r"\noindent\emph{Trace self-time is wall-time $\times$ parallelism; treat as CPU-time breakdown, not wall-time budget.}")
    tex.append("")

    # === Per-stage rollup side-by-side ===
    tex.append(r"\section{Per-stage CPU breakdown}")
    tex.append(r"""Spans are bucketed by coarse stage labels. Self\_ms is the time spent in the span excluding its instrumented children. The sum across stages $\neq$ wall time: on parallel sections, self\_ms accumulates across worker threads.""")

    # Core table
    tex.append(r"\subsection{Core stack}")
    tex.append(r"\par\medskip\noindent\begin{tabular}{l r r r l}")
    tex.append(r"\textbf{Stage} & \textbf{self\_ms} & \textbf{total\_ms} & \textbf{span calls} & \textbf{top span} \\ \midrule")
    core_rows = sorted(core_stage.items(), key=lambda x: -x[1]["self_ms"])
    for stage, data in core_rows:
        top = data["members"][0]["name"] if data["members"] else "-"
        top = latex_escape(top)
        if len(top) > 46:
            top = top[:43] + "..."
        tex.append(
            f"{latex_escape(stage)} & {fmt_ms(data['self_ms'])} & {fmt_ms(data['total_ms'])} & {fmt_count(data['count'])} & \\texttt{{{top}}} \\\\"
        )
    tex.append(r"\end{tabular}\par\medskip")

    # Modular table
    tex.append(r"\subsection{Modular stack}")
    tex.append(r"\par\medskip\noindent\begin{tabular}{l r r r l}")
    tex.append(r"\textbf{Stage} & \textbf{self\_ms} & \textbf{total\_ms} & \textbf{span calls} & \textbf{top span} \\ \midrule")
    mod_rows = sorted(mod_stage.items(), key=lambda x: -x[1]["self_ms"])
    for stage, data in mod_rows:
        top = data["members"][0]["name"] if data["members"] else "-"
        top = latex_escape(top)
        if len(top) > 46:
            top = top[:43] + "..."
        tex.append(
            f"{latex_escape(stage)} & {fmt_ms(data['self_ms'])} & {fmt_ms(data['total_ms'])} & {fmt_count(data['count'])} & \\texttt{{{top}}} \\\\"
        )
    tex.append(r"\end{tabular}\par\medskip")

    # === Top-N spans core ===
    tex.append(r"\section{Top spans — core}")
    tex.append(r"\begin{longtable}{r l r r r r}")
    tex.append(r"\toprule \textbf{\#} & \textbf{span} & \textbf{self\_ms} & \textbf{total\_ms} & \textbf{count} & \textbf{mean\_us} \\ \midrule \endhead")
    for i, r in enumerate(core["spans"][:core_top], 1):
        name = latex_escape(r["name"])
        if len(name) > 60:
            name = name[:57] + "..."
        tex.append(
            f"{i} & \\texttt{{{name}}} & {r['self_ms']:,.1f} & {r['total_ms']:,.1f} & {fmt_count(r['count'])} & {r['mean_total_us']:,.1f} \\\\"
        )
    tex.append(r"\bottomrule \end{longtable}")

    # === Top-N spans modular ===
    tex.append(r"\section{Top spans — modular}")
    tex.append(r"\begin{longtable}{r l r r r r}")
    tex.append(r"\toprule \textbf{\#} & \textbf{span} & \textbf{self\_ms} & \textbf{total\_ms} & \textbf{count} & \textbf{mean\_us} \\ \midrule \endhead")
    for i, r in enumerate(modular["spans"][:mod_top], 1):
        name = latex_escape(r["name"])
        if len(name) > 60:
            name = name[:57] + "..."
        tex.append(
            f"{i} & \\texttt{{{name}}} & {r['self_ms']:,.1f} & {r['total_ms']:,.1f} & {fmt_count(r['count'])} & {r['mean_total_us']:,.1f} \\\\"
        )
    tex.append(r"\bottomrule \end{longtable}")

    # === Side-by-side: spans that exist in both ===
    core_names = {r["name"]: r for r in core["spans"]}
    mod_names = {r["name"]: r for r in modular["spans"]}
    common = []
    for name, c in core_names.items():
        if name in mod_names and (c["self_ms"] >= 1 or mod_names[name]["self_ms"] >= 1):
            m = mod_names[name]
            common.append((name, c, m))
    common.sort(key=lambda x: -(x[1]["self_ms"] + x[2]["self_ms"]))

    tex.append(r"\section{Spans present in both stacks}")
    tex.append(r"These spans have the same name in both the core and modular traces — direct apples-to-apples comparison. Ratio = modular / core self\_ms.")
    tex.append(r"\begin{longtable}{l r r r r r}")
    tex.append(r"\toprule \textbf{span} & \textbf{core self\_ms} & \textbf{core count} & \textbf{mod self\_ms} & \textbf{mod count} & \textbf{ratio} \\ \midrule \endhead")
    for name, c, m in common[:35]:
        ratio_self = m["self_ms"] / c["self_ms"] if c["self_ms"] > 0 else float("inf")
        color = ""
        if ratio_self > 5:
            color = r"\rowcolor{slowred!12}"
        elif ratio_self > 2:
            color = r"\rowcolor{warn!18}"
        nm = latex_escape(name)
        if len(nm) > 55:
            nm = nm[:52] + "..."
        tex.append(
            f"{color} \\texttt{{{nm}}} & {c['self_ms']:,.1f} & {fmt_count(c['count'])} & {m['self_ms']:,.1f} & {fmt_count(m['count'])} & {ratio_self:.2f}x \\\\"
        )
    tex.append(r"\bottomrule \end{longtable}")

    # === Modular-only (infra overhead) ===
    only_mod = [
        r for r in modular["spans"]
        if r["name"] not in core_names and r["self_ms"] >= 50
    ]
    only_mod.sort(key=lambda x: -x["self_ms"])
    tex.append(r"\section{Modular-only spans (infrastructure tax)}")
    tex.append(r"Spans that appear only in the modular trace and contribute $\geq 50$\,ms of self-time. These are candidates for structural overhead the modular stack pays that the core stack does not.")
    tex.append(r"\begin{longtable}{r l r r r r}")
    tex.append(r"\toprule \textbf{\#} & \textbf{span} & \textbf{self\_ms} & \textbf{total\_ms} & \textbf{count} & \textbf{mean\_us} \\ \midrule \endhead")
    for i, r in enumerate(only_mod[:30], 1):
        nm = latex_escape(r["name"])
        if len(nm) > 55:
            nm = nm[:52] + "..."
        tex.append(
            f"{i} & \\texttt{{{nm}}} & {r['self_ms']:,.1f} & {r['total_ms']:,.1f} & {fmt_count(r['count'])} & {r['mean_total_us']:,.1f} \\\\"
        )
    tex.append(r"\bottomrule \end{longtable}")

    # === Core-only (things modular may not emit spans for) ===
    only_core = [
        r for r in core["spans"]
        if r["name"] not in mod_names and r["self_ms"] >= 10
    ]
    only_core.sort(key=lambda x: -x["self_ms"])
    tex.append(r"\section{Core-only spans (differences in what modular instruments)}")
    tex.append(r"Spans that only appear in the core trace. Either modular does equivalent work but under a different span name (re-buckets into \texttt{Op::*}), or the work is simply absent on the modular path.")
    tex.append(r"\begin{longtable}{r l r r r r}")
    tex.append(r"\toprule \textbf{\#} & \textbf{span} & \textbf{self\_ms} & \textbf{total\_ms} & \textbf{count} & \textbf{mean\_us} \\ \midrule \endhead")
    for i, r in enumerate(only_core[:30], 1):
        nm = latex_escape(r["name"])
        if len(nm) > 55:
            nm = nm[:52] + "..."
        tex.append(
            f"{i} & \\texttt{{{nm}}} & {r['self_ms']:,.1f} & {r['total_ms']:,.1f} & {fmt_count(r['count'])} & {r['mean_total_us']:,.1f} \\\\"
        )
    tex.append(r"\bottomrule \end{longtable}")

    # === Call count gaps (skipped when empty — modular uses Op::* names) ===
    count_gap = []
    for name, c, m in common:
        if c["count"] > 0 and m["count"] / c["count"] >= 2.0:
            count_gap.append((name, c, m, m["count"] / c["count"]))
    count_gap.sort(key=lambda x: -x[3])
    if count_gap:
        tex.append(r"\section{Call-count explosion --- same span, many more calls}")
        tex.append(r"Spans present in both stacks where the modular call count is $\geq 2\times$ the core count.")
        tex.append(r"\par\medskip\noindent\begin{longtable}{l r r r r}")
        tex.append(r"\toprule \textbf{span} & \textbf{core count} & \textbf{mod count} & \textbf{count ratio} & \textbf{time ratio} \\ \midrule \endhead")
        for name, c, m, cr in count_gap[:25]:
            tr = m["self_ms"] / c["self_ms"] if c["self_ms"] > 0 else float("inf")
            nm = latex_escape(name)
            if len(nm) > 55:
                nm = nm[:52] + "..."
            tex.append(
                f"\\texttt{{{nm}}} & {fmt_count(c['count'])} & {fmt_count(m['count'])} & {cr:.1f}x & {tr:.2f}x \\\\"
            )
        tex.append(r"\bottomrule \end{longtable}\par\medskip")

    # === Call-tree hot paths ===
    tex.append(r"\section{Hot-path call tree}")
    tex.append(r"Top sub-spans under the dominant parents. Columns show the span's own total\_ms and self\_ms, and how many times it was called from that parent.")

    tex.append(r"\subsection{Core --- \texttt{prove} subtree}")
    tex.append(r"\par\medskip\noindent\begin{longtable}{l r r r}")
    tex.append(r"\toprule \textbf{node} & \textbf{total\_ms} & \textbf{self\_ms} & \textbf{calls} \\ \midrule \endhead")
    for row in _tree_rows(core, "prove", max_depth=2, top_k=5):
        tex.append(row)
    tex.append(r"\bottomrule \end{longtable}\par\medskip")

    tex.append(r"\subsection{Modular --- \texttt{modular\_prove} subtree}")
    tex.append(r"\par\medskip\noindent\begin{longtable}{l r r r}")
    tex.append(r"\toprule \textbf{node} & \textbf{total\_ms} & \textbf{self\_ms} & \textbf{calls} \\ \midrule \endhead")
    for row in _tree_rows(modular, "modular_prove", max_depth=3, top_k=6):
        tex.append(row)
    tex.append(r"\bottomrule \end{longtable}\par\medskip")

    # === Modular stage pie ===
    tex.append(r"\section{Modular prove --- time by top-level op}")
    tex.append(r"Children of the single \texttt{modular\_prove} root span, ordered by total\_ms. This is the most trustworthy wall-time rollup on the modular side because it's explicitly scoped.")
    tex.append(r"\par\medskip\noindent\begin{longtable}{l r r r}")
    tex.append(r"\toprule \textbf{op} & \textbf{total\_ms} & \textbf{\% of prove} & \textbf{calls} \\ \midrule \endhead")
    mp_children = modular.get("parent_children", {}).get("modular_prove", {})
    m_spans = {r["name"]: r for r in modular["spans"]}
    total_prove = m_spans.get("modular_prove", {}).get("total_ms", summary["modular_prove_ms"])
    for c, cnt in sorted(mp_children.items(), key=lambda x: -m_spans.get(x[0], {"total_ms": 0})["total_ms"]):
        s = m_spans.get(c, {"total_ms": 0, "count": 0})
        pct = 100.0 * s["total_ms"] / total_prove if total_prove else 0
        nm = latex_escape(c)
        tex.append(f"\\texttt{{{nm}}} & {fmt_ms(s['total_ms'])} & {pct:.1f}\\% & {cnt} \\\\")
    tex.append(r"\bottomrule \end{longtable}\par\medskip")

    # === stage subtree - deeper ===
    tex.append(r"\subsection{Modular --- \texttt{stage} subtree (8 stage invocations)}")
    tex.append(r"\par\medskip\noindent\begin{longtable}{l r r r}")
    tex.append(r"\toprule \textbf{node} & \textbf{total\_ms} & \textbf{self\_ms} & \textbf{calls} \\ \midrule \endhead")
    for row in _tree_rows(modular, "stage", max_depth=3, top_k=6):
        tex.append(row)
    tex.append(r"\bottomrule \end{longtable}\par\medskip")

    # === mb::* Materialize breakdown ===
    tex.append(r"\subsection{Modular --- \texttt{Materialize} / \texttt{mb::*} family}")
    tex.append(r"The materialize-on-demand path for derived polynomials. \texttt{mb::EqProject} dominates because it runs every sumcheck round.")
    tex.append(r"\par\medskip\noindent\begin{longtable}{l r r r}")
    tex.append(r"\toprule \textbf{node} & \textbf{total\_ms} & \textbf{self\_ms} & \textbf{calls} \\ \midrule \endhead")
    for row in _tree_rows(modular, "Materialize", max_depth=2, top_k=10):
        tex.append(row)
    tex.append(r"\bottomrule \end{longtable}\par\medskip")

    # === pm::Derived breakdown ===
    tex.append(r"\subsection{Modular --- \texttt{pm::Derived} (lazy witness poly materialization)}")
    tex.append(r"Every row is a polynomial that the modular stack derives on-demand during prove. Core materializes these once in \texttt{generate\_and\_commit\_witness\_polynomials} (\textasciitilde 819\,ms total).")
    tex.append(r"\par\medskip\noindent\begin{longtable}{l r r}")
    tex.append(r"\toprule \textbf{derived poly} & \textbf{total\_ms} & \textbf{calls} \\ \midrule \endhead")
    d_children = modular.get("parent_children", {}).get("pm::Derived", {})
    for c, cnt in sorted(d_children.items(), key=lambda x: -m_spans.get(x[0], {"total_ms": 0})["total_ms"]):
        s = m_spans.get(c, {"total_ms": 0})
        nm = latex_escape(c)
        tex.append(f"\\texttt{{{nm}}} & {fmt_ms(s['total_ms'])} & {cnt} \\\\")
    tex.append(r"\bottomrule \end{longtable}\par\medskip")

    # === Interpolate / reduce_dense / gruen dominance ===
    tex.append(r"\section{The three heaviest modular-only kernels}")
    tex.append(r"These three spans together account for the majority of the modular overhead. Each is part of the CPU backend's evaluation path for sumcheck kernels.")
    heavy = [
        "reduce_dense",
        "interpolate_inplace",
        "CpuBackend::gruen_segmented_reduce",
        "CpuBackend::segmented_reduce",
        "CpuBackend::eq_project",
        "CpuBackend::transpose_from_host",
        "CpuBackend::eq_table",
    ]
    tex.append(r"\par\medskip\noindent\begin{tabular}{l r r r r}")
    tex.append(r"\textbf{kernel} & \textbf{self\_ms} & \textbf{count} & \textbf{mean\_us} & \textbf{parent} \\ \midrule")
    for h in heavy:
        s = m_spans.get(h)
        if not s:
            continue
        # Find dominant parent
        parents = []
        for p, cs in modular.get("parent_children", {}).items():
            if h in cs:
                parents.append((p, cs[h]))
        parents.sort(key=lambda x: -x[1])
        parent = parents[0][0] if parents else "ROOT"
        nm = latex_escape(h)
        p = latex_escape(parent)
        tex.append(
            f"\\texttt{{{nm}}} & {s['self_ms']:,.1f} & {fmt_count(s['count'])} & {s['mean_total_us']:,.1f} & \\texttt{{{p}}} \\\\"
        )
    tex.append(r"\end{tabular}\par\medskip")

    # === Observations ===
    tex.append(r"\section{Structural observations}")
    ins_seg = m_spans.get("InstanceSegmentedReduce", {"total_ms": 0})["total_ms"]
    ins_bind = m_spans.get("InstanceBind", {"total_ms": 0})["total_ms"]
    ins_reduce = m_spans.get("InstanceReduce", {"total_ms": 0})["total_ms"]
    mat = m_spans.get("Materialize", {"total_ms": 0})["total_ms"]
    mat_unless = m_spans.get("MaterializeUnlessFresh", {"total_ms": 0})["total_ms"]
    open_ms = m_spans.get("Open", {"total_ms": 0})["total_ms"]
    commit_ms = m_spans.get("Commit", {"total_ms": 0})["total_ms"]
    derived_ram = sum(
        m_spans.get(k, {"total_ms": 0})["total_ms"]
        for k in ["derived::ram_val", "derived::ram_ra_indicator", "derived::ram_combined_ra"]
    )

    tex.append(r"\begin{enumerate}")
    tex.append(
        rf"\item \textbf{{Sumcheck hot path: {ins_seg + ins_bind + ins_reduce:,.0f}\,ms (\textasciitilde{100*(ins_seg+ins_bind+ins_reduce)/total_prove:.0f}\% of modular prove).}} "
        r"Modular's \texttt{InstanceSegmentedReduce + InstanceBind + InstanceReduce} trio replaces what core does inside compiled \texttt{prove\_stageN} closures. The combined overhead vs. core's same-shape sumchecks is \textasciitilde 20\texttimes{}. "
        r"Every round, modular dispatches through \texttt{CpuBackend} kernels that core inlines directly. See the call-count explosion table."
    )
    tex.append(
        rf"\item \textbf{{Derived-polynomial materialization: pm::Derived = {_get(modular, 'pm::Derived')['total_ms']:,.0f}\,ms.}} "
        rf"RAM witness polys alone ($\texttt{{ram\_val}} + \texttt{{ram\_ra\_indicator}} + \texttt{{ram\_combined\_ra}}$) cost {derived_ram:,.0f}\,ms. "
        r"Core materializes equivalent polynomials once up-front in \texttt{generate\_and\_commit\_witness\_polynomials} (\textasciitilde 819\,ms). Modular pays this per-use, not per-poly. "
        r"If these polys are re-derived on each \texttt{Bind} call, caching or bulk-derivation is a big target."
    )
    tex.append(
        rf"\item \textbf{{Materialize/EqProject subsystem: {mat + mat_unless:,.0f}\,ms.}} "
        rf"The \texttt{{mb::EqProject + CpuBackend::eq\_project}} path alone is {_get(modular, 'CpuBackend::eq_project')['self_ms']:,.0f}\,ms self. "
        r"Core's equivalent \texttt{EqPolynomial::evals\_parallel} is 9.0\,ms --- a \textasciitilde 1000\texttimes{} gap. Likely re-computing eq tables every round rather than binding incrementally."
    )
    tex.append(
        rf"\item \textbf{{PCS commit+open is roughly parity.}} "
        rf"Modular \texttt{{Commit}} = {commit_ms:,.0f}\,ms vs. core's committed polynomial generation = 819\,ms. "
        rf"Modular \texttt{{Open}} ({open_ms:,.0f}\,ms) is already comparable to core \texttt{{create\_evaluation\_proof}} (706\,ms). "
        r"Dory is not the modular bottleneck; sumcheck and witness-derivation are."
    )
    tex.append(
        rf"\item \textbf{{Verify delta is suspicious: core 83.8\,ms vs modular 0.8\,ms.}} "
        r"Modular's \texttt{jolt\_verifier::verify} is two orders of magnitude faster, suggesting the Dory PCS opening is \emph{not} being verified on the modular path. "
        r"Confirm whether the native modular verify is pipelined against the Dory opening proof before declaring modular verify a win."
    )
    tex.append(
        rf"\item \textbf{{Memory: 3.3\texttimes.}} "
        r"Modular peak RSS 5.16\,GB vs core 1.55\,GB. Likely sources: lazily-derived polys held in CpuBackend buffers + per-op output buffers not recycled + multiple copies of eq tables."
    )
    tex.append(
        rf"\item \textbf{{Per-round dispatch: \texttt{{interpolate\_inplace}} runs {_get(modular, 'interpolate_inplace')['count']:,} times in modular vs. not at all as a distinct span in core}} "
        r"(core inlines the interpolation inside its sumcheck compute\_message). "
        r"Batching \texttt{interpolate\_inplace} across kernels in the same round is the prototypical amortization target."
    )
    tex.append(r"\end{enumerate}")

    tex.append(r"\section{Reproduction}")
    tex.append(r"""\begin{verbatim}
cargo run --release -p jolt-bench -- \
  --program sha2-chain --num-iters 16 --log-t 16 \
  --iters 1 --warmup 0 --stack core \
  --trace-chrome report_core --json perf/report-core-fresh.json

cargo run --release -p jolt-bench -- \
  --program sha2-chain --num-iters 16 --log-t 16 \
  --iters 1 --warmup 0 --stack modular \
  --trace-chrome report_modular --json perf/report-modular-fresh.json

python3 perf/report_tools/analyze_trace.py \
  benchmark-runs/perfetto_traces/report_core.json \
  perf/report_tools/core_spans.json
python3 perf/report_tools/analyze_trace.py \
  benchmark-runs/perfetto_traces/report_modular.json \
  perf/report_tools/modular_spans.json

python3 perf/report_tools/build_report.py
(cd perf/report_tools && pdflatex -interaction=nonstopmode perf_report.tex)
\end{verbatim}""")

    tex.append(r"\end{document}")
    return "\n".join(tex)


def main():
    core_aggr = load(TOOLS / "core_spans.json")
    mod_aggr = load(TOOLS / "modular_spans.json")
    summary = load(TOOLS / "summary.json")
    out_tex = TOOLS / "perf_report.tex"
    tex = make_tex(core_aggr, mod_aggr, core_top=40, mod_top=40, summary=summary)
    out_tex.write_text(tex)
    print(f"wrote {out_tex}")


if __name__ == "__main__":
    main()
