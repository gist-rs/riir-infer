#!/usr/bin/env python3
"""fence_gate.py — riir-infer's import fence (the public-substrate fence).

The repo is upstream of every engine by design, so it carries exactly two
fences, both mechanized here:

  F1  PATH-DEP FENCE — every `path = "..."` dependency in any tracked
      Cargo.toml must stay INSIDE this repo or resolve under the one
      sanctioned public sibling prefix (`../katgpt-rs`). Any other
      `../riir-*` or absolute escape is a finding.

  F2  IMPORT FENCE — tracked `*.rs` must not reference any workspace
      crate (`riir_engine`, `riir_gpu`, `riir_games*`, `riir_router`,
      `riir_chain`, `riir_neuron_db`, ...) outside comments and string
      literals. The scan masks `//` line comments, `/* */` block
      comments and double-quoted strings before matching, because the
      module headers legitimately SAY "NO cognition" in prose — a raw
      grep overcounts (the comment-only-ref lesson), and a fence that
      cries wolf is a fence nobody runs.

Findings are pinned by MEMBERSHIP in scripts/fence_expected.txt
(reason per row, reds in BOTH directions): a finding with no pin fails,
and a pin whose finding no longer exists fails. The expected file is
deliberately EMPTY today — a row reading "not written yet" would be a
backlog wearing a pin.

Two floors (a ceiling is green over whatever the instrument can see):
  MIN_RS_FILES  — the rust walk size; below it the walk is blind (exit 2)
  MIN_TOMLS     — the manifest walk size; same rule

Exit codes: 0 pass · 1 findings / pin mismatch / self-test failure ·
2 blindness (walk below floor, or git unavailable).

--self-test plants a known violation in a throwaway fixture tree and
requires the detector to fire. It runs as part of every invocation, and
the fixture path never spawns a subprocess — the first cut did, and
each fixture run spawned another fixture run (a process bomb on the
invoking shell).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MIN_RS_FILES = 20
MIN_TOMLS = 1

# The one sanctioned sibling prefix (public upstream). Everything else
# that escapes the repo is a finding.
ALLOWED_SIBLING_PREFIX = "../katgpt-rs"

# Foreign workspace crates are forbidden in code positions — the repo is
# upstream of ALL of them. This repo's OWN crates (riir_infer_core,
# riir_infer_gpu) are exempt: self-references are the point of the repo.
FOREIGN_RIIR = r"riir_(?!infer_core\b|infer_gpu\b)[a-z0-9_]+"
RIIR_TOKEN = re.compile(r"\b" + FOREIGN_RIIR)
USE_RIIR = re.compile(r"\buse\s+" + FOREIGN_RIIR)
EXTERN_CRATE_RIIR = re.compile(r"\bextern\s+crate\s+" + FOREIGN_RIIR)
PATH_TOKEN = re.compile(r"\b" + FOREIGN_RIIR + r"\s*::")

PATH_DEP = re.compile(r'path\s*=\s*"([^"]+)"')


def sh(args: list[str], cwd: Path) -> str:
    out = subprocess.run(
        args, cwd=cwd, capture_output=True,
        encoding="utf-8", errors="backslashreplace",
    )
    if out.returncode != 0:
        raise RuntimeError(f"git failed ({out.returncode}): {out.stderr[:200]}")
    return out.stdout


def tracked(root: Path, pattern: str) -> list[str]:
    txt = sh(["git", "ls-files", "--cached", "--others",
              "--exclude-standard", "--", pattern], root)
    return sorted(l for l in txt.splitlines() if l.strip())


def mask_comments_and_strings(src: str) -> str:
    """Blank out // and /* */ comments and "..." string contents.

    Conservative by design: masking too much can only HIDE a violation
    from F2 (the pin file and review cover that direction); masking too
    little manufactures findings from prose. Char literals are not
    modeled — a `'` in a lifetime position is left alone, which cannot
    fabricate a riir_* token.
    """
    out: list[str] = []
    i = 0
    n = len(src)
    in_line = in_block = in_str = False
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if in_line:
            out.append("\n" if c == "\n" else " ")
            if c == "\n":
                in_line = False
            i += 1
        elif in_block:
            out.append("\n" if c == "\n" else " ")
            if c == "*" and nxt == "/":
                out.append(" ")
                i += 2
                in_block = False
            else:
                i += 1
        elif in_str:
            out.append("\n" if c == "\n" else " ")
            if c == "\\":
                out.append(" ")
                i += 2
            elif c == '"':
                i += 1
                in_str = False
            else:
                i += 1
        else:
            if c == "/" and nxt == "/":
                in_line = True
                out.append("  ")
                i += 2
            elif c == "/" and nxt == "*":
                in_block = True
                out.append("  ")
                i += 2
            elif c == '"':
                in_str = True
                out.append(" ")
                i += 1
            else:
                out.append(c)
                i += 1
    return "".join(out)


def scan_rs(root: Path) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    for rel in tracked(root, "*.rs"):
        text = (root / rel).read_text(encoding="utf-8",
                                      errors="backslashreplace")
        masked = mask_comments_and_strings(text)
        for ln, line in enumerate(masked.splitlines(), 1):
            if (USE_RIIR.search(line) or EXTERN_CRATE_RIIR.search(line)
                    or PATH_TOKEN.search(line)):
                findings.append((rel, ln, line.strip()[:120]))
    return findings


def path_dep_escapes(manifest_dir: Path, repo_root: Path, p: str) -> bool:
    """F1 predicate: does this `path = "..."` dep escape the repo?

    Containment, not a leading-`..` heuristic: a WORKSPACE MEMBER
    manifest legitimately reaches the repo root via `../..` (the
    riir-infer-gpu -> riir-infer-core dep), which the first cut — `not
    p.startswith("..")` — misread as an escape and would have red the
    slice-2 manifest by construction. A dep is a finding iff its
    normalized resolution leaves repo_root and it is not the one
    sanctioned public sibling prefix.
    """
    if p.startswith(ALLOWED_SIBLING_PREFIX):
        return False
    resolved = Path(os.path.normpath(os.path.join(str(manifest_dir), p)))
    return not (resolved == repo_root or repo_root in resolved.parents)


def scan_tomls(root: Path) -> list[tuple[str, int, str]]:
    findings: list[tuple[str, int, str]] = []
    for rel in tracked(root, "*.toml"):
        manifest_dir = (root / rel).parent
        text = (root / rel).read_text(encoding="utf-8",
                                      errors="backslashreplace")
        for ln, line in enumerate(text.splitlines(), 1):
            m = PATH_DEP.search(line)
            if not m:
                continue
            p = m.group(1)
            if path_dep_escapes(manifest_dir, root, p):
                findings.append((rel, ln, f"path dep escapes: {p}"))
    return findings


def load_pins(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, _, reason = line.partition("::")
        key = key.strip()
        if not reason.strip():
            raise SystemExit(f"⛔ pin row without a reason: {key!r}")
        if key in pins:
            raise SystemExit(f"⛔ duplicate pin row: {key!r}")
        pins[key] = reason.strip()
    return pins


def pin_key(rel: str, ln: int, text: str) -> str:
    # LINE-FREE: a line number drifts on every edit above it and a pin
    # file that reds on noise is one people delete. The hit text is the
    # identity; the ordinal disambiguates genuine duplicates.
    return f"{rel}::{text}"


def run_gate(root: Path) -> int:
    rs = tracked(root, "*.rs")
    tomls = tracked(root, "*.toml")
    print(f"walk: {len(rs)} tracked .rs / {len(tomls)} tracked .toml")
    if len(rs) < MIN_RS_FILES or len(tomls) < MIN_TOMLS:
        print(f"⛔ BLIND WALK — floors min_rs={MIN_RS_FILES} "
              f"min_tomls={MIN_TOMLS}; refusing a green zero (exit 2)")
        return 2

    findings = scan_tomls(root) + scan_rs(root)
    pins = load_pins(root / "scripts" / "fence_expected.txt")

    unpinned: list[tuple[str, int, str]] = []
    keyed: dict[str, int] = {}
    for rel, ln, text in findings:
        k = pin_key(rel, ln, text)
        keyed[k] = keyed.get(k, 0) + 1
        if k in pins or f"{k}#{keyed[k]}" in pins:
            continue
        unpinned.append((rel, ln, text))
    stale = [k for k in pins if k not in keyed]

    for rel, ln, text in unpinned:
        print(f"⛔ UNDEFENDED {rel}:{ln} — {text}")
    for k in stale:
        print(f"⛔ STALE PIN — {k} (finding gone; remove the row)")

    if unpinned or stale:
        print(f"✗ fence_gate FAILED — {len(unpinned)} undefended, "
              f"{len(stale)} stale pins")
        return 1
    print(f"✓ fence_gate PASSED — 0 undefended, {len(pins)} pinned")
    return 0


def run_scanners(files: list[Path], root: Path) -> tuple[list, list]:
    """Scanner halves over an explicit file list (fixture mode has no git)."""
    rs_findings: list[tuple[str, int, str]] = []
    toml_findings: list[tuple[str, int, str]] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        rel = str(f.relative_to(root))
        if f.suffix == ".rs":
            masked = mask_comments_and_strings(text)
            for ln, line in enumerate(masked.splitlines(), 1):
                if (USE_RIIR.search(line) or EXTERN_CRATE_RIIR.search(line)
                        or PATH_TOKEN.search(line)):
                    rs_findings.append((rel, ln, line.strip()[:120]))
        elif f.suffix == ".toml":
            manifest_dir = f.parent
            for ln, line in enumerate(text.splitlines(), 1):
                m = PATH_DEP.search(line)
                if not m:
                    continue
                p = m.group(1)
                if path_dep_escapes(manifest_dir, root, p):
                    toml_findings.append((rel, ln,
                                          f"path dep escapes: {p}"))
    return rs_findings, toml_findings


def selftest() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src").mkdir()
        (root / "src/lib.rs").write_text(
            'use riir_engine::quant;\nfn main() { riir_gpu::x(); }\n',
            encoding="utf-8")
        (root / "src/clean.rs").write_text(
            "// NO cognition — prose mention is fine\n"
            'fn f() -> &\'static str { "riir_chain in a string is fine" }\n',
            encoding="utf-8")
        (root / "Cargo.toml").write_text(
            '[dependencies]\n'
            'katgpt-core = { path = "../katgpt-rs/crates/katgpt-core" }\n'
            'bad = { path = "../riir-ai/crates/riir-engine" }\n',
            encoding="utf-8")
        (root / "src/self_ref.rs").write_text(
            'use riir_infer_core::quant;\n'
            'fn f() { riir_infer_gpu::k(); }\n',
            encoding="utf-8")
        # F1 containment: a workspace MEMBER manifest legitimately
        # reaches the repo root via `../..` (the riir-infer-gpu ->
        # riir-infer-core dep) — INSIDE, never a finding; one level
        # further escapes and must fire.
        (root / "crates/member").mkdir(parents=True)
        (root / "crates/member/Cargo.toml").write_text(
            '[dependencies]\n'
            'core = { path = "../.." }\n'
            'bad = { path = "../../../riir-engine" }\n',
            encoding="utf-8")
        rs_f, toml_f = run_scanners(
            [root / "src/lib.rs", root / "src/clean.rs",
             root / "src/self_ref.rs", root / "Cargo.toml",
             root / "crates/member/Cargo.toml"],
            root)
        fired_rs = len(rs_f) >= 2
        fired_toml = len(toml_f) >= 2
        escaped_only = all(not f[2].endswith(': ../..')
                            for f in toml_f)
        clean_masked = mask_comments_and_strings(
            (root / "src/clean.rs").read_text(encoding="utf-8"))
        prose_safe = not RIIR_TOKEN.search(clean_masked)
        # Normalize separators: on Windows the scanner yields backslash
        # relatives and the literal below is forward-slash (the gate was
        # authored + validated on POSIX; this comparison made every Windows
        # selftest run red with the scanner working correctly).
        self_ref_safe = all(str(rel).replace("\\", "/") == "src/lib.rs"
                            for rel, _, _ in rs_f)
        if not (fired_rs and fired_toml and prose_safe and self_ref_safe
                and escaped_only):
            ok = False
            print(f"⛔ selftest: rs={rs_f} toml={toml_f} "
                  f"prose_safe={prose_safe} self_ref_safe={self_ref_safe} "
                  f"escaped_only={escaped_only}")
    print("✓ selftest: 4 violations fired, prose masking + self-ref "
          "exemption + member-dep containment held"
          if ok else "✗ selftest FAILED")
    return 0 if ok else 1


def main() -> int:
    args = sys.argv[1:]
    if "--self-test-only" in args:
        return selftest()
    rc = selftest()
    if rc != 0:
        return rc
    return run_gate(REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
