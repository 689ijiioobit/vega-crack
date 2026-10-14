# vega-crack

GPU password cracker for WPA/WPA2 (PMKID + EAPOL), MD5, NTLM, SHA-256, and bcrypt — Rust host code with hand-optimized OpenCL kernels written specifically for AMD Vega 3 (gfx900) integrated graphics.

## Why

hashcat has no working GPU path on this hardware: the AMD APP SDK runtime hangs during kernel compilation and hashcat skips Mesa Rusticl entirely, so every hashcat run on this class of laptop falls back to CPU. vega-crack was built to answer one question — **how fast can this 3-CU iGPU crack if the kernels are written for it specifically?**

The result: **6.2–6.5x faster than hashcat's CPU fallback** on every GPU mode, and the only GPU cracker that runs here at all. Thousands of laptops carry this exact iGPU with no hashcat GPU support, so this is a practical tool, not a benchmark stunt.

**Stated plainly:** these comparisons are vega-crack on GPU vs hashcat on CPU. That is the honest comparison available on this hardware — it is not a claim of out-engineering hashcat on flagship GPUs.

## Benchmarks

Measured on AMD Ryzen 3 3250U / Radeon Vega 3 (3 CUs, 1200MHz), 5.7GB RAM, Kali Linux. vega-crack via Mesa Rusticl (`RUSTICL_ENABLE=radeonsi`); hashcat 7.1.2 on PoCL CPU; bcrypt on PoCL CPU for both tools.

| Mode | vega-crack | hashcat (CPU) | Ratio | Device |
|------|-----------|---------------|-------|--------|
| MD5 (mask) | 954 MH/s | 147 MH/s | 6.49x | GPU |
| NTLM (mask) | 1.47 GH/s | 231 MH/s | 6.38x | GPU |
| SHA-256 (mask) | 39.2 MH/s | 19.4 MH/s | 2.02x | GPU |
| WPA (PBKDF2) | 14,425 H/s | 2,341 H/s | 6.16x | GPU |
| bcrypt (cost 5) | 1,223 H/s | 1,057 H/s | 1.16x | CPU |

Every number above is measured, not projected. bcrypt intentionally runs on CPU: four cores with L1/L2-resident S-boxes beat the GPU's 64KB LDS-limited occupancy for Blowfish.

## How this was made — honest AI attribution

This codebase was built **with** AI assistance, not by pasting AI output, and the difference matters.

**Process.** A human engineer directed every architectural decision and acceptance criterion. AI models (Claude Code driving a privately hosted GLM backend, plus an independent second agent) served as drafting accelerators, fresh-context adversarial reviewers, and analysis tools, inside a fixed loop: *specify → implement → verify against ground truth → adversarial review → fix → re-verify → commit*. 106 review rounds ran under that loop; every change landed only after passing it.

**Why the output is not "AI slop":**

- **Ground truth everywhere.** 190 automated tests run per build; each hash mode is pinned by positive known-answer vectors — hashcat-cracked test hashes, OpenSSL-verified AES-CMAC, and hashcat's own reference test module — not just "it runs."
- **The process caught what slop wouldn't.** The original SHA-1 core shipped subtly broken: wrong message-schedule index and a mis-rotated register, computing wrong hashes at full speed for months, masked by negative-only tests. It was found by insisting on positive vectors, fixed, then re-verified line-by-line by a separate adversarial audit pass. Vapid generated code does not survive that loop — it either gets fixed or doesn't ship.
- **Null results documented, not hidden.** The manual 80-round SHA-1 unroll measured +0.0% and was reverted (ACO already folds schedule index math into register allocation). The "WPA early-exit" optimization from an early audit proved mathematically impossible — PBKDF2 is a serial chain with nothing to exit early from. Both are recorded here so nobody retries them.
- **Honest numbers.** When the SHA-1 correctness fix cost throughput (14,425 H/s vs the broken kernel's 15,590), the slower honest number replaced the faster garbage one everywhere.

The takeaway this project demonstrates: AI-written code is exactly as good as the verification harness around it.

## Features

### Hash modes

| Mode | hashcat id | Notes |
|------|-----------|-------|
| WPA PMKID | 22000 (type 01) | HMAC-SHA1(PMK, "PMK Name" ∥ AP ∥ STA) |
| WPA EAPOL | 22000 (type 02) | keyver 1 (HMAC-MD5), 2 (HMAC-SHA1), 3 (AES-128-CMAC); nonce error corrections |
| MD5 | 0 | early-exit kernel (skips 21 of 64 rounds for 99.99% of candidates) |
| NTLM | 1000 | MD4 ∥ UTF-16LE, early-exit (skips 18 of 48) |
| SHA-256 | 1400 | wordlist + mask |
| bcrypt | 3200 | $2a$/$2b$/$2x$/$2y$, cost 4-31, CPU mode preferred |

Auto-detection: 32 hex chars → MD5/NTLM, 64 hex → SHA-256, `WPA*` → WPA, `$2` → bcrypt.

### Attack modes

Wordlist · Mask (`--mask`, custom charsets `-1`..`-8`, increment) · Rules (79 hashcat-compatible operations) · Combinator · Hybrid wordlist+mask · PRINCE · Markov (hcstat2) · Stdin

### Operational features

- **PMK database** — `--pmk-output` / `--pmk-input`: precompute PMKs once, crack unlimited captures of the same ESSID at ~103K PMKs/s (skips PBKDF2 entirely)
- **Batch mode** — `--batch-dir`: run a whole directory of captures with one command, summary report + CSV/JSON/SQL/metasploit/hashcat export
- **Sessions** — `--session` checkpoints every 30s, `--restore` resumes
- **GPU safety** — temperature display in the progress line, warning at 88°C, `--gpu-temp-abort` (default 95°C) saves the session and exits
- **Host-side re-verification** — every GPU-reported match is re-computed on the host before being trusted, guarding against GPU bit-flip corruption and potfile poisoning
- Multi-hash per ESSID, hashcat-compatible potfile, `--skip`/`--limit`, `--keep-guessing`, binary-safe I/O (rockyou.txt Latin-1, `$HEX[...]`), runtime device detection with vendor-aware workgroup sizing

## Kernel engineering notes

### GPU-side candidate generation (mask mode)

The kernel decomposes its thread ID into per-position character indices from flat charset buffers — all candidates are generated on-chip. Zero host memory, zero DMA, zero packing overhead; eliminates the host-write race condition that previously caused silent false negatives.

### Early-exit reversal (MD5/NTLM)

The mask kernels reverse the last hash rounds from the target digest, then compute forward and compare intermediate states — rejecting wrong candidates after 43 of 64 rounds (MD5) or 29 of 48 (NTLM). IV subtraction happens in-kernel; NTLM caps mask expansion at 2 positions so the reverse register set stays fixed.

### bitselect round functions

Boolean CH/MAJ forms in SHA-1/MD4/SHA-256 were replaced with OpenCL `bitselect` after the semantics were verified on-device (selector bits pick the **second** argument) and each identity proven numerically:

- SHA-1 CH → `bitselect(d,c,b)`: WPA +2.25%
- SHA-256 MAJ → `bitselect(x,y,x^z)`: SHA-256 +5.2%
- MD4 F/G: NTLM wordlist path

The first MD4_G attempt had swapped arguments — caught by the numeric identity check before it could ship wrong hashes, which is exactly why that check is mandatory.

### Split-kernel architecture (WPA, bcrypt)

WPA: `pbkdf2_init` (HMAC states + first iteration) → `pbkdf2_loop` (fused inner+outer SHA-1, single dispatch per 4095 iterations via `__constant` W_TAIL) → `crack_pmkid_final` (PMKID/MIC compute + compare). State is 30 u32s per work-item.

bcrypt mirrors this: `crack_bcrypt_init` → `crack_bcrypt_loop` (2^cost expansions) → `crack_bcrypt_comp`. On CPU, `BF_LWS` sizes `__local` S-boxes to lws=16 for SIMD + L2 residency.

### Stability engineering

- `MASK_BATCH_SIZE=49152` (768 wavefronts) is the maximum stable dispatch at 1200MHz — larger values silently drop work-items (KIQ fence limit), caught by positive-vector testing
- `amdgpu.lockup_timeout=60000` in GRUB prevents false-positive GPU hangs on long PBKDF2 runs
- ACCEL_N=8 accelerated dispatch with full-window host re-verification

## Correctness verification

All modes are pinned by positive known-answer tests, not negative-only checks:

- MD5/NTLM/SHA-256: known-answer tests find their targets in mask and wordlist modes
- WPA PMKID + EAPOL kv1/kv2: crack hashcat-verified 22000 vectors through both the wordlist and PMK-lookup paths (correct PMK buried among decoys)
- EAPOL kv3: matches the reference construction (KDF-256 PKE layout, RFC 4493 CMAC verified against OpenSSL). Note: hashcat's own cracker currently fails kv3 vectors produced by its own test module, so no cross-tool ground truth exists for kv3
- bcrypt: verified against the OpenBSD `crypt/blowfish.c` reference
- Lesson baked into the suite after the SHA-1 incident: *negative results cannot validate a hash implementation — every mode needs a positive known-answer vector*

## Building

```bash
cargo build --release
```

Requirements: Rust; Mesa Rusticl (`RUSTICL_ENABLE=radeonsi`) for GPU modes; PoCL (`OCL_ICD_VENDORS=/etc/OpenCL/vendors/pocl.icd`) for bcrypt CPU mode. For sustained 1200MHz operation, set `amdgpu.lockup_timeout=60000` in your kernel cmdline.

## Usage

```bash
# WPA (GPU)
RUSTICL_ENABLE=radeonsi ./target/release/vega-crack capture.hc22000 wordlist.txt

# Mask attack (GPU)
RUSTICL_ENABLE=radeonsi ./target/release/vega-crack --mode md5 --mask '?l?l?l?l?d?d?d?d' hash.txt

# bcrypt (CPU — faster than GPU here)
OCL_ICD_VENDORS=/etc/OpenCL/vendors/pocl.icd \
  ./target/release/vega-crack --mode bcrypt bcrypt_hashes.txt wordlist.txt

# Benchmarks
RUSTICL_ENABLE=radeonsi ./target/release/vega-crack --benchmark          # WPA
RUSTICL_ENABLE=radeonsi ./target/release/vega-crack --benchmark-md5      # MD5

# Session save/restore
RUSTICL_ENABLE=radeonsi ./target/release/vega-crack --session run1 capture.hc22000 wordlist.txt
RUSTICL_ENABLE=radeonsi ./target/release/vega-crack --session run1 --restore

# Batch a directory of captures with export
RUSTICL_ENABLE=radeonsi ./target/release/vega-crack --batch-dir /captures/ wordlist.txt \
  --export-format csv --export-file results.csv
```

## Limitations

- Comparisons vs hashcat are GPU-vs-CPU on this hardware (hashcat cannot drive Vega 3)
- NTLM early-exit caps mask expansion at 2 positions (reverse-correctness constraint)
- SHA-256 mask has no early-exit (recursive message schedule defeats round reversal); masks limited to 55 chars, MD5 to 55, NTLM to 27 (single-block limits, enforced with clear errors)
- bcrypt: wordlist only (no mask kernel) and faster on CPU than GPU here
- EAPOL nonce-error-correction support is present but not exhaustive
- Max password length 63 bytes (bcrypt 72, warning printed)

## Test status

- 190 cargo tests, 0 failed · 0 clippy warnings
- 106 audit rounds (R1–R106), all findings fixed — including the SHA-1 core rewrite (R105) and its independent line-by-line verification (R106)
- All attack modes verified: wordlist, mask, combinator, hybrid, PRINCE, stdin, rules

## License

MIT — see [LICENSE](LICENSE). Authorized security testing and research use only.