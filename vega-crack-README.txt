# vega-crack

A custom GPU-accelerated password cracker for WPA/WPA2, MD5, and NTLM hashes, optimized for AMD Vega 3 (gfx900) integrated graphics. Built in Rust with OpenCL kernels, targeting the ROCm 7.2.1 OpenCL stack.

## Why vega-crack?

vega-crack was built from scratch to squeeze maximum performance from a low-power AMD Vega 3 iGPU (3 CUs, 1200MHz, 256 VGPRs). Through extensive kernel optimization — chunked message scheduling, `v_add3_u32` fusion, fused HMAC, and constant memory exploitation — it achieves **WPA cracking speeds 4% faster than hashcat** on the same hardware, despite having 10-20x fewer compute units than hashcat's typical target GPUs.

## Benchmarks

All benchmarks on AMD Vega 3 (gfx900, 3 CUs, 1200MHz) with ROCm 7.2.1 OpenCL.

| Mode | vega-crack | hashcat | Speedup |
|------|-----------|---------|---------|
| **WPA (PBKDF2-SHA1)** | **21,400 H/s** | 20,600 H/s | **+4.0%** |
| **MD5 (mask)** | 549M H/s | 1,124M H/s | 0.49x |
| **NTLM (mask)** | 651M H/s | 1,778M H/s | 0.37x |
| **PMK lookup** | 103K PMKs/s | N/A | — |

### WPA optimization stack

The +4% advantage over hashcat comes from a series of kernel optimizations:

1. **Maj bitselect** — `bitselect(b, c, b^d)` compiles to 2 GCN instructions vs 3 (+2.8%)
2. **Chunked message schedule** — double-buffered `w_cur[16]+w_next[16]` breaks circular buffer false dependencies (+1.4%)
3. **Add3 grouping** — `t = (rotate(a,5) + f + e) + (K + w)` triggers `v_add3_u32` fusion, the breakthrough (+6.3%)
4. **Fused HMAC-SHA1** — inner + outer hash in one function, shared `w_cur/w_next` arrays
5. **`__constant W_TAIL`** — padding constants in constant memory (K-cache), `STATE_UINTS` reduced 41→30
6. **Single dispatch** — `SPLIT_LOOP_COUNT=4095` eliminates 15 state save/restore cycles
7. **Redundant reload removal** — `sha1_process_block` takes `const w_in`, so `w[5..15]` persists across inner/outer hashes

### MD5/NTLM mask kernel (hashcat-style architecture)

MD5 and NTLM use a rewritten mask attack kernel matching hashcat's architecture:
- CPU pre-computes `words_buf_r` (first-position charset as u32 values)
- GPU loads one base password per thread (pre-packed as u32 words)
- Inner loop: `w[0] = w0l | words_buf_r[il_pos]` — one OR instruction per candidate
- Dynamic `il_cnt` (expands first 1-4 positions to fill first u32 word)
- No per-candidate modulo, division, or global memory charset reads
- Unrolled MD5 (64 explicit rounds) and MD4 (48 explicit rounds with bitselect G)

## Features

### Hash modes
- **WPA PMKID** (hashcat mode 22000, type 01) — PMKID = HMAC-SHA1(PMK, "PMK Name" || AP || STA)
- **WPA EAPOL** (hashcat mode 22000, type 02) — full 4-way handshake with key version 1 (HMAC-MD5) and 2 (HMAC-SHA1-AES) support
- **MD5** (hashcat mode 0) — raw MD5 hash
- **NTLM** (hashcat mode 1000) — MD4(UTF-16LE(password))

### Attack modes
- **Wordlist** — standard dictionary attack with deduplication and encoding detection
- **Mask** (`--mask`) — GPU-generated candidates from mask patterns (`?l?l?l?d?d?d?d`)
  - Custom charsets (`-1` through `-8`)
  - Increment mode (`--increment`, `--increment-min`, `--increment-max`)
  - Charset types: `?l` (lowercase), `?u` (upper), `?d` (digit), `?s` (special), `?a` (all), `?h` (hex), `?b` (bytes)
- **Rules** (`--rules`) — hashcat-compatible rule-based mutations applied CPU-side
- **Combinator** (`--combinator-left`, `--combinator-right`) — two wordlists combined
- **Hybrid** (`--hybrid-wordlist`, `--hybrid-mask`) — wordlist + mask concatenation
- **Markov** (`--markov`) — probability-ordered candidate generation
- **Stdin** (`-`) — pipe passwords from stdin

### PMK precomputation database
- **`--pmk-output <file>`** — save precomputed PMKs during cracking (password:pmk_hex format)
- **`--pmk-input <file>`** — load precomputed PMKs and skip PBKDF2 entirely
  - 103K PMKs/second (5x faster than full PBKDF2 cracking)
  - Ideal for repeated ESSID targets: precompute once, crack unlimited captures
  - PMKID-only (EAPOL support planned)

### Session management
- **`--session <name>`** — save/restore cracking progress
- Periodic checkpoints during long-running mask attacks
- Resume with `--session <name> --restore`

### Other features
- **`--benchmark`** — GPU speed test (no hash file needed)
- **`--keep-guessing`** — continue after first crack (find all matches)
- **`--potfile`** / **`--potfile-disable`** — hashcat-compatible potfile
- **`--outfile`** / **`--outfile-format`** — output file with configurable format
- **`--skip`** / **`--limit`** — skip N candidates or limit total
- **`--quiet`** / **`--verbose`** / **`--status`** — output control
- **`--hex-charset`** — interpret custom charset values as hex
- **Multi-hash** — crack multiple hashes simultaneously per ESSID group
- **Host-side re-verification** — defense against GPU bit-flip corruption
- **GPU bit-flip defense** — full batch scan via candidate index reconstruction
- **Deduplication** — automatic password dedup with configurable tracking limit

## Architecture

### Split kernel (WPA)
1. **`pbkdf2_init`** — HMAC precompute (ipad/opad), first PBKDF2 iteration
2. **`pbkdf2_loop`** — 4095 iterations of fused HMAC-SHA1 (single dispatch)
3. **`crack_pmkid_final`** — extract PMK, compute PMKID/EAPOL MIC, compare

State layout (30 uints per work-item):
- `[0..4]` ipad state, `[5..9]` opad state
- `[10..14]` U1, `[15..19]` U2
- `[20..24]` T1, `[25..29]` T2

### GPU mask kernel (MD5/NTLM)
- `crack_md5_mask` / `crack_ntlm_mask` — hashcat-style architecture
- CPU: `build_words_buf_r()` pre-computes first-position charset
- CPU: `pack_base_password()` packs positions 1..N-1 as u32 words
- GPU: inner loop does one OR instruction per candidate, inline hash computation

### PMK lookup kernel
- `crack_pmk_lookup` — takes precomputed PMKs directly, computes PMKID, compares
- Skips PBKDF2 entirely (~4000x fewer SHA1 compressions)
- PMKID-only (EAPOL support planned)

## Requirements

- AMD GPU with ROCm OpenCL support (gfx900/gfx906+)
- ROCm 7.2.1+ (amdocl64 ICD)
- Rust (cargo)
- Environment variables for Vega 3:
  ```
  HSA_XNACK=1
  HSA_OVERRIDE_GFX_VERSION=9.0.0
  HSA_ENABLE_INTERRUPT=0
  HSA_ENABLE_SDMA=0
  ```

## Building

```bash
cd vega-crack
cargo build --release
```

## Usage

### WPA PMKID cracking
```bash
./target/release/vega-crack capture.hc22000 wordlist.txt
```

### WPA with PMK output (for later reuse)
```bash
./target/release/vega-crack --pmk-output pmks.txt capture.hc22000 wordlist.txt
```

### WPA with PMK lookup (skip PBKDF2)
```bash
./target/release/vega-crack --pmk-input pmks.txt capture.hc22000
```

### Mask attack (MD5)
```bash
./target/release/vega-crack --mode md5 --mask '?l?l?l?l?d?d?d?d' hash.txt
```

### Mask attack (NTLM)
```bash
./target/release/vega-crack --mode ntlm --mask '?l?l?l?l?d?d?d?d' hash.txt
```

### Benchmark
```bash
./target/release/vega-crack --benchmark wpa
```

### Session save/restore
```bash
./target/release/vega-crack --session myattack capture.hc22000 wordlist.txt
# ... interrupted ...
./target/release/vega-crack --session myattack --restore
```

## Limitations

- MD5/NTLM gap to hashcat: 2-2.5x (limited by 3 CUs and VECT_SIZE=1 on GCN)
- PMK lookup: PMKID only (EAPOL support planned)
- WPA EAPOL: nonce error corrections supported but not exhaustive
- Markov mode: single-threaded CPU generation (GPU dispatch is parallel)
- Maximum password length: 63 bytes (802.11 spec)
- No OpenCL binary caching (kernel recompiles on each run, ~3s)

## License

MIT