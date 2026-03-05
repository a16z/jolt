const { danger, warn, fail, message } = require("danger");

// ---------------------------------------------------------------------------
// PR size guard
// ---------------------------------------------------------------------------
const linesChanged = danger.github.pr.additions + danger.github.pr.deletions;

if (linesChanged > 1000) {
  fail(
    `This PR changes **${linesChanged} lines**. ` +
      "Please break it into smaller PRs (target < 500 lines)."
  );
} else if (linesChanged > 500) {
  warn(
    `This PR changes **${linesChanged} lines**. ` +
      "Consider breaking it into smaller PRs for easier review."
  );
}

// ---------------------------------------------------------------------------
// New dependencies added
// ---------------------------------------------------------------------------
const modifiedCargos = danger.git.modified_files.filter(
  (f) => f.endsWith("Cargo.toml") && !f.includes("examples/")
);

if (modifiedCargos.length > 0) {
  warn(
    "This PR modifies `Cargo.toml` files: " +
      modifiedCargos.map((f) => `\`${f}\``).join(", ") +
      ". If new dependencies were added, please justify them in the PR description."
  );
}

// ---------------------------------------------------------------------------
// Cargo.lock changes
// ---------------------------------------------------------------------------
if (danger.git.modified_files.includes("Cargo.lock")) {
  message(
    "`Cargo.lock` was updated. Verify the dependency resolution changes are intentional."
  );
}

// ---------------------------------------------------------------------------
// Security-critical paths
// ---------------------------------------------------------------------------
const securityCriticalPaths = [
  "jolt-core/src/poly/commitment/",
  "jolt-core/src/zkvm/spartan/",
  "jolt-core/src/subprotocols/",
  "jolt-core/src/zkvm/r1cs/",
  "crates/jolt-sumcheck/",
  "crates/jolt-spartan/",
  "crates/jolt-dory/",
];

const touchedCritical = danger.git.modified_files.filter((f) =>
  securityCriticalPaths.some((p) => f.startsWith(p))
);

if (touchedCritical.length > 0) {
  warn(
    "This PR touches **security-critical** paths:\n" +
      touchedCritical.map((f) => `- \`${f}\``).join("\n") +
      "\n\nThese modules affect proof soundness. Please ensure thorough review."
  );
}

// ---------------------------------------------------------------------------
// PR description quality
// ---------------------------------------------------------------------------
const body = danger.github.pr.body || "";

if (body.length < 50) {
  fail(
    "PR description is too short. Please include a summary, testing strategy, " +
      "and security considerations."
  );
}

// ---------------------------------------------------------------------------
// Crate README freshness
// ---------------------------------------------------------------------------
const crateNames = [
  "jolt-field",
  "jolt-poly",
  "jolt-sumcheck",
  "jolt-openings",
  "jolt-spartan",
  "jolt-instructions",
  "jolt-transcript",
  "jolt-dory",
  "jolt-zkvm",
];

const allChanged = [
  ...danger.git.modified_files,
  ...danger.git.created_files,
  ...danger.git.deleted_files,
];

const cratesWithSrcChanges = crateNames.filter((name) =>
  allChanged.some(
    (f) =>
      f.startsWith(`crates/${name}/src/`) || f === `crates/${name}/Cargo.toml`
  )
);

const cratesWithReadmeUpdates = crateNames.filter((name) =>
  allChanged.includes(`crates/${name}/README.md`)
);

const staleReadmes = cratesWithSrcChanges.filter(
  (name) => !cratesWithReadmeUpdates.includes(name)
);

if (staleReadmes.length > 0) {
  warn(
    "These crates had source changes but their `README.md` was not updated:\n" +
      staleReadmes.map((n) => `- \`crates/${n}/README.md\``).join("\n") +
      "\n\nIf the public API changed, please update the README."
  );
}

// ---------------------------------------------------------------------------
// CI / workflow changes
// ---------------------------------------------------------------------------
const workflowChanges = danger.git.modified_files.filter((f) =>
  f.startsWith(".github/")
);

if (workflowChanges.length > 0) {
  warn(
    "This PR modifies CI/GitHub configuration:\n" +
      workflowChanges.map((f) => `- \`${f}\``).join("\n") +
      "\n\nPlease verify workflow changes are secure (no secret exfiltration, " +
      "no `pull_request_target` with checkout of PR head, etc.)."
  );
}
