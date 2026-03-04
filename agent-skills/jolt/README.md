# Jolt AI Coding Skill

Teaches AI coding agents how to wrap Rust functions in Jolt zero-knowledge proofs.

## Install

```bash
npx skills add a16z/jolt
```

Works with Claude Code, Cursor, Codex, and [18+ other agents](https://vercel.com/docs/agent-resources/skills).

### Fallback (Claude Code only)

```bash
curl -sfL jolt.rs/skill | bash
```

## What it does

When you say "make this Jolt provable" (or similar), the agent will:

1. Identify a pure Rust function to prove
2. Adapt the signature for the RISC-V guest (no floats, no I/O)
3. Scaffold a Jolt project (`jolt new`)
4. Write guest and host code
5. Build and run the proof
