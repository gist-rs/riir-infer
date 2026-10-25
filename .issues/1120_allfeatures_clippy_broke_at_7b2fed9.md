# Issue 1120 — all-features clippy broke at `7b2fed9`: spike_census_t3_kv_ab_ternary + kv_evict

**Status:** OPEN — pre-existing breakage found by the Issue-1118 G3 posture sweep; unclaimed; owner of the surface is the `7b2fed9` lane (the 4090 zerocopy fix).

**Claimed:** — none

## The finding

`cargo clippy --all-features --all-targets --keep-going` fails with 5 errors at `ddc7105` (2026-10-25), **none** of them in the issue-1118 tokenizer change:

- `src/bin/spike_census_t3_kv_ab_ternary.rs:142` — E0277: `C cannot be sent between threads safely` (a `forward_qwen_deltanet_ternary_qkv` call whose generic cache handle lost its `Send` bound coverage under `7b2fed9`'s SPLIT-handle split).
- `src/deltanet/kv_evict.rs:710` — E0061: a function takes 11 arguments, 10 supplied (the same fix added a parameter; this caller was not updated).

## Reproduce

```sh
cargo clippy --all-features --all-targets --keep-going 2>&1 | grep -E "^error" -A6
```

Confirmed PRE-EXISTING: a clean worktree at `8a4f379` (which includes `7b2fed9`, before the issue-1118 work) shows the identical 5 errors — the zerocopy lane landed without an all-features sweep (its own record says it self-documents on the default posture).

## Scope

Fix the two callers at the all-features posture (Send bound or handle split propagation + the missing argument), then re-run the full all-features/all-targets sweep to zero. The issue-1118 lane's G3 claim is scoped to default + the tokenizer-serving postures, which are green; this issue un-blocks a repo-wide all-features green claim.
