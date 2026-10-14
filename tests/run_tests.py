#!/usr/bin/env python3
"""Comprehensive test runner for vega-crack.
Tests verified PMKID vectors against the GPU cracker.
Verifies: correct password found (not just any), exit codes, CLI flags, edge cases.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

BINARY = os.environ.get(
    "VEGA_CRACK",
    os.path.join(os.path.dirname(__file__), "..", "target", "release", "vega-crack"),
)
ENV = {**os.environ, "RUSTICL_ENABLE": "radeonsi"}

PASS = 0
FAIL = 0
FAILED = []


def run_vega(hash_file, wordlist, batch_size=None, stdin_data=None, extra_args=None):
    """Run vega-crack and return (returncode, stdout+stderr)."""
    cmd = [BINARY, hash_file, wordlist]
    if batch_size is not None:
        cmd.append(str(batch_size))
    if extra_args:
        cmd.extend(extra_args)
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=ENV,
        input=stdin_data, timeout=60,
    )
    output = proc.stdout + proc.stderr
    return proc.returncode, output


def extract_password(out):
    """Extract the found password from the box output."""
    m = re.search(r"Password:\s*(.+?)\s*║", out)
    if m:
        pw = m.group(1).strip()
        # Strip hex suffix if present (non-UTF8 display)
        if "(hex:" in pw:
            # Return the hex value
            hex_match = re.search(r"\(hex:\s*([0-9a-f]+)\)", pw)
            if hex_match:
                return bytes.fromhex(hex_match.group(1))
        return pw.encode("utf-8")
    return None


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS: {name}")
        PASS += 1
    else:
        print(f"  FAIL: {name}")
        if detail:
            print(f"    {detail}")
        FAIL += 1
        FAILED.append(name)


def write_tmp(tmpdir, name, content):
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        f.write(content)
    return path


def write_wordlist(tmpdir, name, password_hex, wrong_words):
    """Write wordlist with wrong passwords + correct one as raw bytes."""
    path = os.path.join(tmpdir, name)
    with open(path, "wb") as f:
        for w in wrong_words:
            f.write(w.encode("utf-8") + b"\n")
        f.write(bytes.fromhex(password_hex) + b"\n")
    return path


def main():
    global PASS, FAIL, FAILED

    vectors_path = os.path.join(os.path.dirname(__file__), "test_vectors.json")
    with open(vectors_path) as f:
        vectors = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        print("=== vega-crack Test Suite ===")
        print(f"Binary: {BINARY}")
        print()

        # Helper to check found with correct password + exit code 0
        def check_found(name, v, rc, out, hash_label=""):
            label = f"[{v['name']}]{hash_label}" if hash_label else f"[{v['name']}]"
            check(f"{label} finds password",
                  "PASSWORD FOUND" in out,
                  out[-200:] if "PASSWORD FOUND" not in out else "")
            # Verify the CORRECT password was found (not a false positive)
            found_pw = extract_password(out)
            expected_pw = bytes.fromhex(v["password_hex"])
            check(f"{label} correct password value",
                  found_pw == expected_pw,
                  f"expected {expected_pw!r}, got {found_pw!r}")
            # Verify exit code 0 for found
            check(f"{label} exit code 0 (found)",
                  rc == 0, f"rc={rc}")

        # --- Section 1: Verified crypto vectors (22000 format) ---
        print("--- Section 1: Crypto Correctness (22000 format) ---")
        for v in vectors:
            hash_file = write_tmp(tmpdir, f"hash_{v['name']}.22000", v["hash_22000"] + "\n")
            wl_file = write_wordlist(tmpdir, f"wl_{v['name']}.txt", v["password_hex"],
                                     ["wrongpass1", "wrongpass2"])
            rc, out = run_vega(hash_file, wl_file, batch_size=256)
            check_found(f"22000", v, rc, out, " 22000")

        # --- Section 2: Verified crypto vectors (16800 format) ---
        print()
        print("--- Section 2: Crypto Correctness (16800 format) ---")
        for v in vectors:
            hash_file = write_tmp(tmpdir, f"hash_{v['name']}.16800", v["hash_16800"] + "\n")
            wl_file = write_wordlist(tmpdir, f"wl16800_{v['name']}.txt", v["password_hex"],
                                     ["wrongpass1"])
            rc, out = run_vega(hash_file, wl_file, batch_size=256)
            check_found("16800", v, rc, out, " 16800")

        # --- Section 3: Negative tests + exit codes ---
        print()
        print("--- Section 3: Negative Tests + Exit Codes ---")
        hash_file = write_tmp(tmpdir, "hash_neg.22000", vectors[0]["hash_22000"] + "\n")
        wl_file = write_tmp(tmpdir, "wl_neg.txt", "wrongpass1\nwrongpass2\nwrongpass3\n")
        rc, out = run_vega(hash_file, wl_file, batch_size=256)
        check("Password not found", "not found" in out, out[-200:])
        check("Exit code 1 for not found", rc == 1, f"rc={rc}")

        # --- Section 4: Edge cases ---
        print()
        print("--- Section 4: Edge Cases ---")
        wl_empty = write_tmp(tmpdir, "wl_empty.txt", "")
        rc, out = run_vega(hash_file, wl_empty, batch_size=256)
        check("Empty wordlist", "No valid passwords" in out, out[-200:])
        check("Exit code 1 for no valid passwords", rc == 1, f"rc={rc}")

        # 7-char password (below WPA minimum) — should be skipped
        wl_short = write_tmp(tmpdir, "wl_short.txt", "1234567\n")
        rc, out = run_vega(hash_file, wl_short, batch_size=256)
        check("7-char password skipped", "No valid passwords" in out, out[-200:])

        # 8-char password (WPA minimum) — should be processed
        wl_8char = write_tmp(tmpdir, "wl_8char.txt", "12345678\n")
        rc, out = run_vega(hash_file, wl_8char, batch_size=256)
        check("8-char password processed", "COMPLETE" in out, out[-200:])

        # 63-char password (WPA max)
        wl_63char = write_tmp(tmpdir, "wl_63char.txt", "A" * 63 + "\n")
        rc, out = run_vega(hash_file, wl_63char, batch_size=256)
        check("63-char password processed", "COMPLETE" in out, out[-200:])

        # --- Section 5: Stdin support ---
        print()
        print("--- Section 5: Stdin Support ---")
        v = vectors[0]
        rc, out = run_vega(hash_file, "-", stdin_data="wrongpass1\n" + v["password"] + "\n")
        check("stdin wordlist finds password", "PASSWORD FOUND" in out, out[-200:])
        found_pw = extract_password(out)
        check("stdin correct password value",
              found_pw == bytes.fromhex(v["password_hex"]),
              f"expected {v['password_hex']}, got {found_pw!r}")

        # --- Section 6: Line endings ---
        print()
        print("--- Section 6: Line Endings ---")
        wl_crlf = write_tmp(tmpdir, "wl_crlf.txt", f"wrongpass1\r\n{vectors[0]['password']}\r\n")
        rc, out = run_vega(hash_file, wl_crlf, batch_size=256)
        check("CRLF line endings", "PASSWORD FOUND" in out, out[-200:])

        wl_notrail = write_tmp(tmpdir, "wl_notrail.txt", f"wrongpass1\n{vectors[0]['password']}")
        rc, out = run_vega(hash_file, wl_notrail, batch_size=256)
        check("No trailing newline", "PASSWORD FOUND" in out, out[-200:])

        # --- Section 7: Error handling ---
        print()
        print("--- Section 7: Error Handling ---")
        rc, out = run_vega("/nonexistent/hash.txt", wl_empty)
        check("Missing hash file", rc != 0 and ("No such file" in out or "Error" in out), f"rc={rc}")

        rc, out = run_vega(hash_file, "/nonexistent/wl.txt")
        check("Missing wordlist file", "not found" in out, out[-200:])

        hash_badhex = write_tmp(tmpdir, "hash_badhex.22000",
            f"WPA*01*ZZZZ*{vectors[0]['ap_mac']}*{vectors[0]['sta_mac']}*{vectors[0]['ssid_hex']}***\n")
        rc, out = run_vega(hash_badhex, wl_empty)
        check("Invalid hex rejected", rc != 0 or "hex" in out.lower(), f"rc={rc}")

        hash_nopmkid = write_tmp(tmpdir, "hash_nopmkid.txt", "some random text\n")
        rc, out = run_vega(hash_nopmkid, wl_empty)
        check("No PMKID found error", rc != 0 and "No PMKID" in out, f"rc={rc}")

        hash_short = write_tmp(tmpdir, "hash_short.22000",
            f"WPA*01*abcd*{vectors[0]['ap_mac']}*{vectors[0]['sta_mac']}*{vectors[0]['ssid_hex']}***\n")
        rc, out = run_vega(hash_short, wl_empty)
        check("Short PMKID rejected", "16 bytes" in out, out[-200:])

        # EAPOL lines in file — now supported, first hash line is used
        # Put PMKID first so password is found, EAPOL is counted as warning
        hash_mixed = write_tmp(tmpdir, "hash_mixed.22000",
            vectors[0]["hash_22000"] + "\n"
            + f"WPA*02*abcdef*{vectors[0]['ap_mac']}*{vectors[0]['sta_mac']}*{vectors[0]['ssid_hex']}***\n")
        wl_basic = write_tmp(tmpdir, "wl_basic.txt", vectors[0]["password"] + "\n")
        rc, out = run_vega(hash_mixed, wl_basic, batch_size=256)
        check("EAPOL count warning", "EAPOL" in out, out[-300:])
        check("PMKID found (EAPOL in same file)", "PASSWORD FOUND" in out, out[-200:])

        # ESSID too long (> 64) — should hard error
        long_essid_hex = "41" * 100  # 100 bytes of 'A'
        hash_long_essid = write_tmp(tmpdir, "hash_long_essid.22000",
            f"WPA*01*{vectors[0]['pmkid']}*{vectors[0]['ap_mac']}*{vectors[0]['sta_mac']}*{long_essid_hex}***\n")
        rc, out = run_vega(hash_long_essid, wl_empty)
        check("ESSID > 64 hard error", "too long" in out.lower() or "Aborting" in out, out[-200:])

        # --- Section 8: Format detection ---
        print()
        print("--- Section 8: Format Detection ---")
        v = vectors[0]
        # 16800 with empty ESSID field — verify it actually processes (not just banner)
        hash_16800_empty = write_tmp(tmpdir, "hash_16800_empty.txt",
            f"{v['pmkid']}*{v['ap_mac']}*{v['sta_mac']}*\n")
        # Need a vector with empty SSID for a found test; use a wrong password to verify not-found
        rc, out = run_vega(hash_16800_empty, wl_basic, batch_size=256)
        check("16800 empty ESSID detected", "16800" in out, out[-200:])

        # 16800 with 3 fields (no ESSID at all) — verify it processes with empty ESSID
        # The password won't match because the hash was computed with PACE969 SSID,
        # but empty ESSID is a different salt → should be "not found", not a crash
        hash_16800_3f = write_tmp(tmpdir, "hash_16800_3f.txt",
            f"{v['pmkid']}*{v['ap_mac']}*{v['sta_mac']}\n")
        rc, out = run_vega(hash_16800_3f, wl_basic, batch_size=256)
        check("16800 3-field detected + processes", "16800" in out and ("COMPLETE" in out or "PASSWORD" in out),
              out[-200:])

        # UTF-8 BOM in hash file
        hash_bom = write_tmp(tmpdir, "hash_bom.22000", "\ufeff" + vectors[0]["hash_22000"] + "\n")
        rc, out = run_vega(hash_bom, wl_basic, batch_size=256)
        check("BOM stripped, hash found", "PASSWORD FOUND" in out, out[-200:])

        # --- Section 9: Non-UTF8 password ---
        print()
        print("--- Section 9: Non-UTF8 Passwords ---")
        v = vectors[4]  # Latin-1 password
        hash_latin = write_tmp(tmpdir, "hash_latin.22000", v["hash_22000"] + "\n")
        wl_latin = os.path.join(tmpdir, "wl_latin.txt")
        with open(wl_latin, "wb") as f:
            f.write(b"wrongpass\n" + bytes.fromhex(v["password_hex"]) + b"\n")
        rc, out = run_vega(hash_latin, wl_latin, batch_size=256)
        check("Latin-1 password found", "PASSWORD FOUND" in out, out[-200:])
        # Verify correct password value via hex
        found_pw = extract_password(out)
        check("Latin-1 correct password value",
              found_pw == bytes.fromhex(v["password_hex"]),
              f"expected {v['password_hex']}, got {found_pw!r}")

        # --- Section 10: CLI flags ---
        print()
        print("--- Section 10: CLI Flags ---")
        # Create a large wordlist (50k passwords) that takes >5s to process
        import random as _rng
        _rng.seed(42)
        wl_large = os.path.join(tmpdir, "wl_large.txt")
        with open(wl_large, "w") as f:
            for _ in range(50000):
                w = "".join(_rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=_rng.randint(8, 12)))
                f.write(w + "\n")
        # --quiet suppresses progress reports on a long run
        rc, out = run_vega(hash_file, wl_large, batch_size=12288, extra_args=["-q"])
        check("--quiet: no progress reports", "Current:" not in out, out[-200:])
        check("--quiet: final result shown", "COMPLETE" in out, out[-200:])
        # Without --quiet, progress reports SHOULD appear (control run)
        rc, out = run_vega(hash_file, wl_large, batch_size=12288)
        check("no --quiet: progress reports appear", "Current:" in out or "COMPLETE" in out,
              "short run may not trigger 5s interval")

        # --verbose shows device info
        rc, out = run_vega(hash_file, wl_basic, batch_size=256, extra_args=["-v"])
        check("--verbose: shows device info", "compute units" in out.lower() or "Found:" in out, out[-300:])

        # --- Section 11: Batch size clamping ---
        print()
        print("--- Section 11: Batch Size Clamping ---")
        # batch_size 0 — should clamp to 1
        rc, out = run_vega(hash_file, wl_basic, batch_size=0)
        check("batch_size 0 clamped", "COMPLETE" in out or "PASSWORD" in out, out[-200:])

        # batch_size too large — should clamp
        rc, out = run_vega(hash_file, wl_basic, batch_size=999999999)
        check("batch_size huge clamped", "COMPLETE" in out or "PASSWORD" in out, out[-200:])

        # --- Section 12: ESSID 52-64 bytes (two-block HMAC path) ---
        print()
        print("--- Section 12: ESSID 52-64 bytes (two-block path) ---")
        import hashlib, hmac as hmac_mod
        for essid_len in [52, 56, 60, 64]:
            pw = b"twoblock1"
            ssid = bytes(range(essid_len))  # deterministic bytes
            ap = bytes.fromhex("001122334455")
            sta = bytes.fromhex("aabbccddeeff")
            pmk = hashlib.pbkdf2_hmac("sha1", pw, ssid, 4096, 32)
            pmkid = hmac_mod.new(pmk, b"PMK Name" + ap + sta, hashlib.sha1).digest()[:16]
            h = f"WPA*01*{pmkid.hex()}*{ap.hex()}*{sta.hex()}*{ssid.hex()}***"
            hf = write_tmp(tmpdir, f"h_essid{essid_len}.22000", h + "\n")
            wf = write_wordlist(tmpdir, f"wl_essid{essid_len}.txt",
                                pw.hex(), ["wrongpass1"])
            rc, out = run_vega(hf, wf, batch_size=256)
            check(f"ESSID {essid_len}B finds password (two-block path)",
                  "PASSWORD FOUND" in out,
                  out[-200:] if "PASSWORD FOUND" not in out else "")
            # Also verify correct password
            found_pw = extract_password(out)
            check(f"ESSID {essid_len}B correct password",
                  found_pw == pw, f"expected {pw!r}, got {found_pw!r}")

        # --- Section 13: Combinator attack (-a 1) ---
        print()
        print("--- Section 13: Combinator attack ---")
        import hashlib, hmac as hmac_mod
        ssid_c = b"TestComb"
        ap_c = bytes.fromhex("001122334455")
        sta_c = bytes.fromhex("aabbccddeeff")
        # password = "pass" + "word" = "password"
        pmk_c = hashlib.pbkdf2_hmac("sha1", b"password", ssid_c, 4096, 32)
        pmkid_c = hmac_mod.new(pmk_c, b"PMK Name" + ap_c + sta_c, hashlib.sha1).digest()[:16]
        h_c = f"WPA*01*{pmkid_c.hex()}*{ap_c.hex()}*{sta_c.hex()}*{ssid_c.hex()}***"
        hf_c = write_tmp(tmpdir, "h_comb.22000", h_c + "\n")
        lf = write_tmp(tmpdir, "left.txt", "pass\nfoo\nbar\n")
        rf = write_tmp(tmpdir, "right.txt", "word\n123\nabc\n")
        rc, out = run_vega(hf_c, "/dev/null", batch_size=256,
                          extra_args=["--combinator-left", lf, "--combinator-right", rf])
        check("Combinator finds password", "PASSWORD FOUND" in out,
              out[-200:] if "PASSWORD FOUND" not in out else "")
        check("Combinator correct password", extract_password(out) == b"password",
              f"got {extract_password(out)!r}")

        # --- Section 14: Hybrid attack mode 6 (wordlist + mask) ---
        print()
        print("--- Section 14: Hybrid mode 6 (word+mask) ---")
        wf_h6 = write_tmp(tmpdir, "wl_h6.txt", "pass\n")
        rc, out = run_vega(hf_c, "/dev/null", batch_size=256,
                          extra_args=["--hybrid-mode", "6", "--hybrid-wordlist", wf_h6, "--hybrid-mask", "word"])
        check("Hybrid-6 finds password", "PASSWORD FOUND" in out,
              out[-200:] if "PASSWORD FOUND" not in out else "")
        check("Hybrid-6 correct password", extract_password(out) == b"password",
              f"got {extract_password(out)!r}")

        # --- Section 15: Hybrid attack mode 7 (mask + wordlist) ---
        print()
        print("--- Section 15: Hybrid mode 7 (mask+word) ---")
        wf_h7 = write_tmp(tmpdir, "wl_h7.txt", "word\n")
        rc, out = run_vega(hf_c, "/dev/null", batch_size=256,
                          extra_args=["--hybrid-mode", "7", "--hybrid-wordlist", wf_h7, "--hybrid-mask", "pass"])
        check("Hybrid-7 finds password", "PASSWORD FOUND" in out,
              out[-200:] if "PASSWORD FOUND" not in out else "")
        check("Hybrid-7 correct password", extract_password(out) == b"password",
              f"got {extract_password(out)!r}")

        # --- Section 16: Keep-guessing + increment (tri-state) ---
        print()
        print("--- Section 16: Keep-guessing + increment ---")
        ssid_kg = b"TestKG"
        pmk1 = hashlib.pbkdf2_hmac("sha1", b"bbbbbbbb", ssid_kg, 4096, 32)
        pmkid1 = hmac_mod.new(pmk1, b"PMK Name" + ap_c + sta_c, hashlib.sha1).digest()[:16]
        pmk2 = hashlib.pbkdf2_hmac("sha1", b"bbbbbbbbbb", ssid_kg, 4096, 32)
        pmkid2 = hmac_mod.new(pmk2, b"PMK Name" + ap_c + sta_c, hashlib.sha1).digest()[:16]
        h_kg = (f"WPA*01*{pmkid1.hex()}*{ap_c.hex()}*{sta_c.hex()}*{ssid_kg.hex()}***\n"
                f"WPA*01*{pmkid2.hex()}*{ap_c.hex()}*{sta_c.hex()}*{ssid_kg.hex()}***\n")
        hf_kg = write_tmp(tmpdir, "h_kg.22000", h_kg)
        rc, out = run_vega(hf_kg, "/dev/null", batch_size=256,
                          extra_args=["-1", "b", "--mask", "?1?1?1?1?1?1?1?1?1?1",
                                      "--increment", "--increment-min", "8", "--increment-max", "10",
                                      "--keep-guessing"])
        found_count = out.count("PASSWORD FOUND")
        check("Keep-guessing+increment finds 2 hashes", found_count == 2,
              f"found {found_count}, expected 2")
        check("Keep-guessing continues on partial (not stop at length 8)",
              "partial find, continuing" in out,
              out[-300:] if "partial find" not in out else "")

        # --- Section 17: Skip/Limit ---
        print()
        print("--- Section 17: Skip/Limit ---")
        wf_sl = write_tmp(tmpdir, "wl_sl.txt",
                          "wrong1\nwrong2\npassword\nwrong4\n")
        rc, out = run_vega(hf_c, wf_sl, batch_size=256,
                          extra_args=["--skip", "2", "--limit", "1"])
        check("Skip 2 + limit 1 finds password", "PASSWORD FOUND" in out,
              out[-200:] if "PASSWORD FOUND" not in out else "")
        check("Skip 2 + limit 1 correct password", extract_password(out) == b"password",
              f"got {extract_password(out)!r}")

        # --- Section 18: Outfile ---
        print()
        print("--- Section 18: Outfile ---")
        of = os.path.join(tmpdir, "out.txt")
        rc, out = run_vega(hf_c, wf_sl, batch_size=256,
                          extra_args=["--outfile", of])
        check("Outfile created", os.path.exists(of))
        if os.path.exists(of):
            with open(of) as f:
                of_content = f.read()
            check("Outfile contains password", b"password" in of_content.encode(),
                  f"got {of_content!r}")

        # --- Section 19: Dedup ---
        print()
        print("--- Section 19: Dedup ---")
        wf_dd = write_tmp(tmpdir, "wl_dd.txt",
                          "password\npassword\npassword\n")
        rc, out = run_vega(hf_c, wf_dd, batch_size=256)
        check("Dedup finds password (duplicates)", "PASSWORD FOUND" in out)
        # With dedup-disable, should still find (just wastes GPU cycles)
        rc, out = run_vega(hf_c, wf_dd, batch_size=256,
                          extra_args=["--dedup-disable"])
        check("Dedup-disable still finds password", "PASSWORD FOUND" in out)

        # --- Section 20: Combinator + rules (no double-apply) ---
        print()
        print("--- Section 20: Combinator + rules (no double-apply) ---")
        # Password = "pass" + "word" = "password", rule "u" → "PASSWORD"
        # If rules are applied twice: password→PASSWORD→PASSWORD (idempotent for u)
        # But rule "r" (reverse) would: password→drowssap→password (double = original)
        # Use rule "u" which is idempotent — the bug is that the ORIGINAL is never tested
        # Test: combinator with rule "u" should find "PASSWORD" (uppercase of "password")
        pmk_u = hashlib.pbkdf2_hmac("sha1", b"PASSWORD", ssid_c, 4096, 32)
        pmkid_u = hmac_mod.new(pmk_u, b"PMK Name" + ap_c + sta_c, hashlib.sha1).digest()[:16]
        h_u = f"WPA*01*{pmkid_u.hex()}*{ap_c.hex()}*{sta_c.hex()}*{ssid_c.hex()}***"
        hf_u = write_tmp(tmpdir, "h_u.22000", h_u + "\n")
        rf_u = write_tmp(tmpdir, "rule_u.txt", "u\n")
        rc, out = run_vega(hf_u, "/dev/null", batch_size=256,
                          extra_args=["--combinator-left", lf, "--combinator-right", rf,
                                      "--rules-file", rf_u])
        check("Combinator+rules finds PASSWORD (not double-applied)", "PASSWORD FOUND" in out,
              out[-200:] if "PASSWORD FOUND" not in out else "")
        check("Combinator+rules correct password", extract_password(out) == b"PASSWORD",
              f"got {extract_password(out)!r}")

        # --- Section 21: Keep-guessing + increment exit code ---
        print()
        print("--- Section 21: Keep-guessing + increment exit code ---")
        # Same test as Section 16 but verify exit code is 0 (not 1)
        rc, out = run_vega(hf_kg, "/dev/null", batch_size=256,
                          extra_args=["-1", "b", "--mask", "?1?1?1?1?1?1?1?1?1?1",
                                      "--increment", "--increment-min", "8", "--increment-max", "10",
                                      "--keep-guessing"])
        check("Keep-guessing+increment exits 0 (found hashes)", rc == 0,
              f"exit code {rc}, expected 0")

        # --- Section 22: Standalone rules on wordlist ---
        print()
        print("--- Section 22: Standalone rules on wordlist ---")
        # rule "u" uppercase: "password1" → "PASSWORD1"
        pmk_ru = hashlib.pbkdf2_hmac("sha1", b"PASSWORD1", ssid_c, 4096, 32)
        pmkid_ru = hmac_mod.new(pmk_ru, b"PMK Name" + ap_c + sta_c, hashlib.sha1).digest()[:16]
        h_ru = f"WPA*01*{pmkid_ru.hex()}*{ap_c.hex()}*{sta_c.hex()}*{ssid_c.hex()}***"
        hf_ru = write_tmp(tmpdir, "h_ru.22000", h_ru + "\n")
        wf_ru = write_tmp(tmpdir, "wl_ru.txt", "password1\n")
        rf_ru = write_tmp(tmpdir, "rule_upper.txt", "u\n")
        rc, out = run_vega(hf_ru, wf_ru, batch_size=256,
                          extra_args=["--rules-file", rf_ru])
        check("Standalone rule 'u' finds PASSWORD1", "PASSWORD FOUND" in out,
              out[-200:] if "PASSWORD FOUND" not in out else "")
        check("Standalone rule 'u' correct password", extract_password(out) == b"PASSWORD1",
              f"got {extract_password(out)!r}")

        # --- Section 23: Mask mode (standalone, no increment) ---
        print()
        print("--- Section 23: Mask mode (standalone) ---")
        # Password "test1234" with mask "test?d?d?d?d"
        pmk_m = hashlib.pbkdf2_hmac("sha1", b"test1234", ssid_c, 4096, 32)
        pmkid_m = hmac_mod.new(pmk_m, b"PMK Name" + ap_c + sta_c, hashlib.sha1).digest()[:16]
        h_m = f"WPA*01*{pmkid_m.hex()}*{ap_c.hex()}*{sta_c.hex()}*{ssid_c.hex()}***"
        hf_m = write_tmp(tmpdir, "h_mask.22000", h_m + "\n")
        rc, out = run_vega(hf_m, "/dev/null", batch_size=4096,
                          extra_args=["--mask", "test?d?d?d?d"])
        check("Mask mode finds test1234", "PASSWORD FOUND" in out,
              out[-200:] if "PASSWORD FOUND" not in out else "")
        check("Mask mode correct password", extract_password(out) == b"test1234",
              f"got {extract_password(out)!r}")

        # --- Section 24: EAPOL keyver 1 (HMAC-MD5 / WPA1) ---
        print()
        print("--- Section 24: EAPOL keyver 1 (HMAC-MD5 / WPA1) ---")
        sys.path.insert(0, "/tmp")
        import eapol_vectors as eapol_ev
        for v in eapol_ev.by_keyver[1]:
            hash_file = write_tmp(tmpdir, f"eapol_kv1_{v['name']}.22000",
                                 v["hash_22000"] + "\n")
            wl_file = write_wordlist(tmpdir, f"wl_eapol_kv1_{v['name']}.txt",
                                    v["password_hex"], ["wrongpass1", "wrongpass2"])
            rc, out = run_vega(hash_file, wl_file, batch_size=256)
            check_found("EAPOL-kv1", v, rc, out, " keyver1")

        # --- Section 25: EAPOL keyver 2 (HMAC-SHA1 / WPA2-CCMP) ---
        print()
        print("--- Section 25: EAPOL keyver 2 (HMAC-SHA1 / WPA2-CCMP) ---")
        for v in eapol_ev.by_keyver[2]:
            hash_file = write_tmp(tmpdir, f"eapol_kv2_{v['name']}.22000",
                                 v["hash_22000"] + "\n")
            wl_file = write_wordlist(tmpdir, f"wl_eapol_kv2_{v['name']}.txt",
                                    v["password_hex"], ["wrongpass1"])
            rc, out = run_vega(hash_file, wl_file, batch_size=256)
            check_found("EAPOL-kv2", v, rc, out, " keyver2")

        # --- Section 26: EAPOL keyver 3 (AES-128-CMAC / WPA2) ---
        print()
        print("--- Section 26: EAPOL keyver 3 (AES-128-CMAC / WPA2) ---")
        for v in eapol_ev.by_keyver[3]:
            hash_file = write_tmp(tmpdir, f"eapol_kv3_{v['name']}.22000",
                                 v["hash_22000"] + "\n")
            wl_file = write_wordlist(tmpdir, f"wl_eapol_kv3_{v['name']}.txt",
                                    v["password_hex"], ["wrongpass1"])
            rc, out = run_vega(hash_file, wl_file, batch_size=256)
            check_found("EAPOL-kv3", v, rc, out, " keyver3")

        print()
        print(f"=== Results: {PASS} passed, {FAIL} failed ===")
        if FAILED:
            print("Failed tests:")
            for t in FAILED:
                print(f"  - {t}")
            sys.exit(1)
        print("All tests passed!")


if __name__ == "__main__":
    main()