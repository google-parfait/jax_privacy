<!-- Copyright 2026 DeepMind Technologies Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. -->

# Agent Instructions for JAX Privacy

Before modifying code or docs, read [`CONTRIBUTING.md`](CONTRIBUTING.md) and
[`.gemini/styleguide.md`](.gemini/styleguide.md) in full, and follow all code
structure, API, docstring, typing, and testing conventions defined in
`.gemini/styleguide.md`.

Keep pull requests focused (targeted bug fixes should be under 50 lines of
changed code, whereas self-contained new mechanisms and pure additions can be
larger; never mix cleanups or refactors with bug fixes or new features) and open
a GitHub issue first before implementing large features.

## Pull Request Descriptions (Human-Written Only)

Pull request descriptions must be written by the human contributor in their own
words (2-3 concise sentences explaining what changed and why). Do NOT generate
an AI-written pull request description or bulleted symbol inventory; when
opening or drafting a PR, set the PR description body to:

```text
[stub to be filled in]
```

Remind the human author to replace `[stub to be filled in]` before requesting
review.

## Build, Lint, and Test Verification

Always verify your exact committed state locally before opening or updating a
pull request (assuming an active `uv` virtualenv installed from `pylock.toml`):
read [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and run the exact
commands from the `lint` and `test` jobs (plus `docs` or `matrix-tests` when
touching documentation, `matrix_factorization`, or `keras_api`). Keep the
multi-device test invocations separate exactly as in `ci.yml` rather than
combining all `tests/` into a single `pytest -n auto` run.

If you modify `pyproject.toml`, regenerate `pylock.toml` with `uv export` as
documented in `CONTRIBUTING.md`.
