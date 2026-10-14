# vega-crack vs hashcat — Feature Comparison & Gap Analysis

Based on reading the full hashcat source code (hashcat-src/, cloned Sep 2026).

## What vega-crack HAS (working, tested, reviewed)

### Attack Modes
- ✅ **Straight (wordlist)** — -a 0 equivalent, wordlist reading, stdin via "-"
- ✅ **Mask/bruteforce** — -a 3 equivalent, `--mask`, charsets ?l?u?d?s?h?H?a?b, custom ?1-?4
- ✅ **Increment** — `--increment`, `--increment-min`, `--increment-max`

### Hash Support
- ✅ **PMKID** (WPA*01*, mode 22000 type 01)
- ✅ **EAPOL keyver 2** (HMAC-SHA1, WPA*02*)
- ✅ **16800 deprecated format** (PMKID*MAC_AP*MAC_STA*ESSID)
- ✅ **Host-side verification** (verify.rs — recompute PMKID/EAPOL on CPU before declaring success)
- ✅ **GPU false-positive batch re-verification**
- ✅ **Nonce error corrections** (auto-detect from message_pair, `--nonce-error-corrections` override)
- ✅ **EAPOL keyver 1** (HMAC-MD5/WPA1) and **keyver 3** (AES-128-CMAC/WPA2)

### CLI
- ✅ `--mask`, `-1/-2/-3/-4`, `--increment`, `--increment-min/max`
- ✅ `--quiet`, `--verbose`, `--batch-size`
- ✅ Exit codes: 0=found, 1=not found, 2=GPU corruption

### Crypto (Kernel — vega_wpa.cl)
- ✅ SHA1, HMAC-SHA1, PBKDF2 (4096 rounds)
- ✅ PMKID computation (HMAC-SHA1 of "PMK Name" + AP + STA)
- ✅ PRF-512 (4 HMAC-SHA1 calls, 64-byte PTK)
- ✅ EAPOL MIC computation (HMAC-SHA1 of EAPOL frame)
- ✅ sha1_continue_multi (arbitrary-length EAPOL frames)

## What vega-crack LACKS (from hashcat source comparison)

### HIGH PRIORITY — Core cracking features

1. **~~Rule-based attack~~** ✅ DONE — 70 ops, 13 audit rounds (R1-R13)

2. **~~Multi-hash cracking~~** ✅ DONE — salt reuse via ESSID grouping

3. **~~Nonce error corrections~~** ✅ DONE — auto-detect from message_pair

### MEDIUM PRIORITY — Usability features

4. **Outfile** (`-o` / `--outfile`)
   - Write cracked hashes to file in configurable format
   - `--outfile-format`: 1=hash, 2=plain, 3=hash:plain, etc.
   - `--outfile-json`: JSON format
   - `--outfile-autohex-disable`: control $HEX[] encoding
   - hashcat src/outfile.c (840 lines)

5. **Session/restore** (`--session` / `--restore`)
   - Save cracking position to resume later
   - `--restore-file-path`, `--restore-disable`
   - hashcat src/restore.c (858 lines)

6. **Potfile** (`--potfile-path` / `--potfile-disable` / `--show` / `--left`)
   - Track cracked hashes, skip already-cracked
   - `--show`: display cracked, `--left`: display uncracked
   - Custom potfile format for WPA: `PMK*essid:password`
   - hashcat src/potfile.c (1399 lines)

7. **Status display** (`--status` / `--status-json` / `--status-timer`)
   - Real-time status screen: speed, ETA, progress, device temps
   - Machine-readable format for automation
   - hashcat src/status.c (2905 lines)

8. **Combinator attack** (`-a 1`)
   - Two wordlists, concatenate candidates (word1 + word2)
   - hashcat src/combinator.c (157 lines)

9. **Hybrid attack** (`-a 6` / `-a 7`)
   - -a 6: wordlist + mask (word + ?d?d?d)
   - -a 7: mask + wordlist (?d?d?d + word)
   - hashcat src/straight.c, slow_candidates.c

### LOW PRIORITY — Advanced features

10. **Markov chains** (`--markov-hcstat2` / `--markov-disable`)
    - Probabilistic candidate ordering based on character transition frequencies
    - hashcat hashcat.hcstat2 file, src/mpsp.c

11. **Benchmark** (`-b` / `--benchmark-all`)
    - Measure hash rate for a given mode
    - hashcat src/benchmark.c

12. **Keep guessing** (`--keep-guessing`)
    - Continue after finding a hash (find all matches)
    - hashcat src/hashes.c

13. **Wordlist dedup**
    - Skip duplicate entries in wordlist
    - hashcat src/wordlist.c

14. **Loopback** (`--loopback`)
    - Add cracked passwords back as candidates
    - hashcat src/loopback.c

15. **Skip/limit** (`-s` / `-l`)
    - Skip N words, limit to N words
    - hashcat src/user_options.c

16. **EAPOL keyver 1** (HMAC-MD5) and **keyver 3** (AES-128-CMAC)
    - hashcat supports all 3 key versions via separate aux kernels:
      aux1=keyver1 (MD5), aux2=keyver2 (SHA1), aux3=keyver3 (AES-CMAC)
    - We reject keyver 1 and 3 with error message
    - Would need MD5 and AES-128-CMAC in the kernel

17. **Custom charset ?5-?8**
    - hashcat supports 8 custom charsets, we only support 4

18. **hex-charset** (`--hex-charset`)
    - Specify charsets in hex

19. **Increment-inverse** (`--increment-inverse`)
    - Increment from right-to-left instead of left-to-right

20. **Brain server/client**
    - Distributed cracking coordination
    - hashcat src/brain.c

### Observations from hashcat source

1. **Hashcat's WPA kernel** (m22000-pure.cl, 1152 lines) uses a split init/loop/comp design:
   - `m22000_init`: PBKDF2 ipad/opad precomputation
   - `m22000_loop`: PBKDF2 iteration loop (runs on GPU, 4096 rounds)
   - `m22000_aux1-4`: Final comparison (aux1=PMKID, aux2=EAPOL-v1, aux3=EAPOL-v2, aux4=EAPOL-v3)
   - We do it all in one kernel (crack_pmkid) — simpler but less flexible for multi-hash

2. **Hashcat's PKE precomputation**: The module precomputes PKE (Pairwise Key Expansion)
   data on the host, including nonce comparison and byte swapping. Our kernel does
   this on GPU. Hashcat's approach saves GPU cycles for multi-hash scenarios.

3. **Hashcat uses SIMD vectorization** (u32x types) — we use scalar u32. On Vega 3
   this may not matter much (no hardware SIMD), but hashcat gets 4x throughput
   on SIMD-capable GPUs.

4. **Hashcat's module system** (module_XXXXX.c) — each hash mode is a dynamically
   loaded plugin. We have a single binary. For WPA-only this is fine.

5. **Hashcat's autotune** (src/autotune.c) — automatically finds optimal kernel-accel
   and kernel-loops. We use fixed batch_size (user-configurable).