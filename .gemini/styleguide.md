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

<!-- Single source of truth for JAX Privacy code review and coding conventions.
     Loaded automatically by Gemini Code Assist on GitHub PRs, and referenced
     from AGENTS.md for local coding agents. -->

# JAX Privacy Code Review Style Guide

Do NOT post review comments on pull requests from `test_*` or `cl/*` branches
(automated internal syncs), and do not spend review comments on mechanical
formatting, whitespace, or lint issues that CI (`pyink`, `flake8`, `pylint`,
`pyrefly`) already enforces. Keep review comments high-signal, concise, and
focused on bugs, maintainability, and the repository conventions below.

## 0. Human-Written Pull Request Descriptions

*   **Flag `[stub to be filled in]` or AI-written PR descriptions**: Pull
    request descriptions must be brief (2-3 sentences) and written by the human
    contributor in their own words. Flag the PR description if it still contains
    `[stub to be filled in]` or exhibits clear signs of AI-generated prose (for
    example, exhaustive file-by-file or symbol-by-symbol bullet lists, generic
    `Summary / Key Changes / Test Plan` boilerplate, em-dashes, or line-by-line
    narration of the diff), and ask the author to replace it with a short
    human-written summary.

## 1. Change Structure & API Boundaries

*   **Targeted fixes vs. new functionality**: Targeted bug fixes should be
    minimal and surgical (under 50 changed lines) without unrelated refactors or
    cosmetic churn. New mechanisms and pure additions can be larger when
    self-contained (in a dedicated helper function or module) and
    wired in via a small, targeted update to an existing function so changes are
    easy to maintain and reason about. Flag PRs that mix unrelated cleanups with
    functional changes, or refactors that add more lines than they remove.
*   **Minimal public API & validation**: Keep the public API surface small; do
    not expose `_`-prefixed modules or internal state (`self._field`) in public
    signatures or docstrings. Prefer pure functions over stateless wrapper
    classes, `@property` over parameterless `as_*()` or `get_*()` methods,
    explicit config fields over inferring booleans from `arg is not None`, and
    lightweight `__init__` methods. Recommend reusing or extending
    `jax_privacy/_validate.py` (`from jax_privacy import _validate`, such as
    `_validate.positive`, `_validate.non_negative`, `_validate.in_range`,
    `_validate.ndim`) where applicable instead of ad-hoc inline `ValueError`
    checks.

## 2. Conciseness & Formatting

*   **Diff hygiene & dead code**: Match the surrounding file's formatting and
    variable names strictly. Flag unrelated cosmetic edits, stale imports, dead
    helpers, or default arguments matching the callee's built-in default.
*   **Compact expressions & inlining**: Keep short calls, assignments,
    `ValueError` raises, and `logging` statements on one line (within 80
    characters) when readable. Default to inlining attribute accesses
    (`self.max_batch_size` rather than 1-2 letter aliases like `bs` or `n`),
    introducing temporaries only to avoid multi-line wraps or heavy repetition.
*   **No multi-line generator wraps & clean returns**: When a generator inside
    `any(...)`, `all(...)`, or `sum(...)` would wrap across 3+ lines, extract a
    1-line list comprehension into a named variable above it. Split complex
    constructor or return expressions into named temporaries before a 1-line
    `return`.
*   **Minimal branching**: Prefer default-then-override (`x = default` followed
    by `if cond: x = override`) over `if / else` initialization, `for` over
    bounded `while`, `if not items:` over `len(items) == 0`, and omit
    special-case branches when the general code path handles empty inputs.

## 3. Comments, Docstrings & Typing

*   **Inline comment brevity**: Default to zero inline `#` comments; flag blocks
    with more comment lines than code lines or comments that restate the code.
    Keep necessary `#` comments to 1-2 lines maximum without backticks, and put
    algorithmic implementation details in `#` comments rather than public
    docstrings.
*   **Docstrings & docs**: Keep module docstrings short (~10-15 lines, public
    API only) and generic (no caller-specific or dataset-specific notes). Public
    APIs require Google-style rST docstrings (double backticks `` ``code`` ``
    for code literals, `:math:` for math, single-line `Args:` / `Attributes:`
    entries, omitting boilerplate `Raises: ValueError` when obvious). File-local
    `_`-prefixed helpers, ABC subclass overrides, and small self-explanatory
    classes must use 1-line docstrings only (no `Args:` or `Returns:`).
    Standalone `.md` docs in `docs/` use MyST Markdown (`$math$`,
    ```` ```{note} ```` directives, no `[TOC]` or `> [!NOTE]`).
*   **Imports & types**: Import modules rather than individual symbols (`from
    jax_privacy import _validate`, top-of-file aliases in tests). Use `from
    __future__ import annotations`, `from absl import logging` (not stdlib
    `logging`), `collections.abc`, `X | None`, full `jax.typing.ArrayLike` or
    `np.typing.ArrayLike` (no type aliases), and `@jax.jit(...)` decorator
    factories. Resolve ergonomic input unions at public API boundaries and keep
    private helpers on canonical types; short `_`-prefixed helpers may omit
    type annotations when obvious to fit `def` on one line. Never add
    `# pyrefly: ignore` or `# type: ignore` in new code.

## 4. Testing

*   **Thorough, fast unit tests**: Every bug fix needs a regression test. Prefer
    parameterizing existing tests (`@parameterized`, using `defaults | kwargs`
    on one line) over duplicating test methods, and use seeded randomness
    (`jax.random.key` or `np.random.default_rng`) or reference-implementation
    checks on small shapes so CI stays fast.

## 5. [If applicable] Differential Privacy Correctness

*   **Config-time validation & sensitivity**: Only when a PR modifies core DP
    mechanisms, accounting, or batching, verify that configuration parameters
    are validated at initialization time (never via data-dependent runtime
    branches or error messages on sensitive records) and that microbatching,
    gradient accumulation, sharding, and subsampling preserve the claimed
    `DpEvent` sensitivity and amplification guarantees.
