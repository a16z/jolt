---
name: new-spec
description: Create a new spec file from the template
arguments: "<feature-name>"
---

Create a new spec file in the `specs/` directory.

## Steps

1. Validate the argument: must be lowercase alphanumeric with dashes (e.g. `streaming-prover`). Reject otherwise.
2. Get the GitHub username: run `gh api user --jq .login`.
3. Get today's date in `YYYY-MM-DD` format.
4. Create `specs/<feature-name>.md` by copying the template from `specs/TEMPLATE.md` with these substitutions:
   - `[Feature Name]` → the feature name (keep dashes, this is a working title the author will edit)
   - `Author(s)` value → `@<github-username>`
   - `Created` value → today's date
   - `Status` stays `proposed`
   - `PR` stays empty
5. Print the path and remind the author to fill in the spec and open a PR.

Do NOT modify `TEMPLATE.md` itself. Do NOT add a date prefix to the filename — the PR number will be prepended by a GitHub Action after the PR is opened.
