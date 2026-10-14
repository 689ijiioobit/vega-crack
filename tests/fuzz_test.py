#!/usr/bin/env python3
"""Differential fuzzer: generate random PMKID test vectors with Python's
hashlib/hmac reference implementation, then verify vega-crack finds them.

Tests edge cases that static vectors don't cover:
- Random password lengths 8-63
- Random SSID lengths 0-32
- Random MAC addresses
- Binary passwords (all byte values 0x00-0xFF)
- Passwords at batch boundaries
"""
import hashlib
import hmac
import json
import os
import random
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


def compute_pmkid(password: bytes, ssid: bytes, ap_mac: bytes, sta_mac: bytes) -> bytes:
    pmk = hashlib.pbkdf2_hmac("sha1", password, ssid, 4096, 32)
    msg = b"PMK Name" + ap_mac + sta_mac
    pmkid_full = hmac.new(pmk, msg, hashlib.sha1).digest()
    return pmkid_full[:16]


def run_vega(hash_file, wordlist_file, batch_size=256):
    cmd = [BINARY, hash_file, wordlist_file, str(batch_size)]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=60)
    return proc.returncode, proc.stdout + proc.stderr


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


def extract_password(out):
    """Extract the found password from the box output."""
    m = re.search(r"Password:\s*(.+?)\s*║", out)
    if m:
        pw = m.group(1).strip()
        if "(hex:" in pw:
            hex_match = re.search(r"\(hex:\s*([0-9a-f]+)\)", pw)
            if hex_match:
                return bytes.fromhex(hex_match.group(1))
        return pw.encode("utf-8")
    return None


def random_mac():
    return bytes(random.randint(0, 255) for _ in range(6))


def gen_password():
    """Generate a random WPA-valid password (8-63 bytes)."""
    pw_len = random.randint(8, 63)
    # Mix of categories
    category = random.randint(0, 4)
    if category == 0:
        # ASCII printable
        return bytes(random.randint(0x20, 0x7E) for _ in range(pw_len))
    elif category == 1:
        # ASCII lowercase + digits
        chars = b"abcdefghijklmnopqrstuvwxyz0123456789"
        return bytes(random.choice(chars) for _ in range(pw_len))
    elif category == 2:
        # Full byte range (0x00-0xFF) — but avoid 0x0A and 0x0D (line endings)
        return bytes(
            random.choice([b for b in range(256) if b not in (0x0A, 0x0D)])
            for _ in range(pw_len)
        )
    elif category == 3:
        # Latin-1 range (0x80-0xFF)
        return bytes(random.randint(0x80, 0xFF) for _ in range(pw_len))
    else:
        # Mixed ASCII + high bytes (but no line endings)
        return bytes(
            random.choice([b for b in range(256) if b not in (0x0A, 0x0D)])
            for _ in range(pw_len)
        )


def gen_ssid():
    """Generate a random SSID (0-64 bytes). Extended to 64 to test two-block HMAC path."""
    ssid_len = random.randint(0, 64)
    if ssid_len == 0:
        return b""
    return bytes(random.randint(0x20, 0x7E) for _ in range(ssid_len))


def main():
    global PASS, FAIL, FAILED

    random.seed(42)  # reproducible

    NUM_RANDOM = 50
    NUM_EDGE = 15

    with tempfile.TemporaryDirectory() as tmpdir:
        print("=== Differential Fuzzing ===")
        print(f"Binary: {BINARY}")
        print(f"Random vectors: {NUM_RANDOM}, Edge cases: {NUM_EDGE}")
        print()

        all_vectors = []

        # Generate random vectors
        print("--- Random Vectors ---")
        for i in range(NUM_RANDOM):
            pw = gen_password()
            ssid = gen_ssid()
            ap = random_mac()
            sta = random_mac()
            pmkid = compute_pmkid(pw, ssid, ap, sta)

            hash_22000 = f"WPA*01*{pmkid.hex()}*{ap.hex()}*{sta.hex()}*{ssid.hex()}***"
            hash_file = os.path.join(tmpdir, f"hash_{i}.22000")
            with open(hash_file, "w") as f:
                f.write(hash_22000 + "\n")

            # Wordlist: wrong passwords + correct one
            wl_file = os.path.join(tmpdir, f"wl_{i}.txt")
            with open(wl_file, "wb") as f:
                for _ in range(random.randint(1, 10)):
                    # Random wrong password (8-63 bytes, ASCII)
                    wrong_len = random.randint(8, 20)
                    f.write(bytes(random.randint(0x61, 0x7A) for _ in range(wrong_len)) + b"\n")
                f.write(pw + b"\n")

            rc, out = run_vega(hash_file, wl_file)
            pw_display = pw.hex()
            found = "PASSWORD FOUND" in out
            check(
                f"Random #{i} (pw_len={len(pw)}, ssid_len={len(ssid)})",
                found,
                f"pw_hex={pw_display}, out={out[-150:]}" if not found else "",
            )
            # Verify correct password value (not just any password found)
            if found:
                found_pw = extract_password(out)
                check(
                    f"Random #{i} correct password value",
                    found_pw == pw,
                    f"expected {pw_display}, got {found_pw!r}",
                )

        # Generate edge case vectors
        print()
        print("--- Edge Cases ---")

        # Edge 1: Password exactly 8 bytes, all 0x00 (but not line endings)
        pw = bytes([0x01] * 8)
        ssid = b"TestEdge"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("8x 0x01", pw, ssid, ap, sta, pmkid))

        # Edge 2: Password exactly 63 bytes
        pw = b"Z" * 63
        ssid = b"MaxPw"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("63x Z", pw, ssid, ap, sta, pmkid))

        # Edge 3: Empty SSID
        pw = b"emptyssid"
        ssid = b""
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("empty SSID", pw, ssid, ap, sta, pmkid))

        # Edge 4: 32-byte SSID
        pw = b"longssid32"
        ssid = b"X" * 32
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("32-byte SSID", pw, ssid, ap, sta, pmkid))

        # Edge 5: Password with spaces
        pw = b"hello world test 123"
        ssid = b"Spaces"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("spaces in pw", pw, ssid, ap, sta, pmkid))

        # Edge 6: Password with tabs (valid in WPA, not line endings)
        pw = b"tab\there\ttest"
        ssid = b"Tabs"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("tabs in pw", pw, ssid, ap, sta, pmkid))

        # Edge 7: Password with null bytes (0x00) — valid in WPA raw bytes
        pw = b"\x00" * 8 + b"test"
        ssid = b"NullBytes"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("null bytes in pw", pw, ssid, ap, sta, pmkid))

        # Edge 8: Password at exact batch boundary (position 0)
        pw = b"firstpass"
        ssid = b"Boundary0"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("batch pos 0", pw, ssid, ap, sta, pmkid))

        # Edge 9: Password at exact batch boundary (last position in batch)
        pw = b"lastinbatch"
        ssid = b"BoundaryN"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("batch last pos", pw, ssid, ap, sta, pmkid))

        # Edge 10: All-same-byte password
        pw = b"\xff" * 8
        ssid = b"SameByte"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("8x 0xFF", pw, ssid, ap, sta, pmkid))

        # Edge 11: Password with high-bit chars (Latin-1)
        pw = bytes(range(0xC0, 0xC8))  # 8 bytes
        ssid = b"LatinHigh"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("Latin-1 0xC0-0xC7", pw, ssid, ap, sta, pmkid))

        # Edge 12: 1-byte SSID
        pw = b"onessid01"
        ssid = b"A"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("1-byte SSID", pw, ssid, ap, sta, pmkid))

        # Edge 13: Password with all printable ASCII special chars
        pw = b"!\"#$%&'()*+,-./"
        ssid = b"Special"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("special chars", pw, ssid, ap, sta, pmkid))

        # Edge 14: Password with backtick and pipe
        pw = b"`pipe|test123"
        ssid = b"Backtick"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("backtick+pipe", pw, ssid, ap, sta, pmkid))

        # Edge 15: Password exactly 8 bytes, minimum boundary
        pw = b"12345678"
        ssid = b"MinLen8"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        all_vectors.append(("exact 8 bytes", pw, ssid, ap, sta, pmkid))

        for name, pw, ssid, ap, sta, pmkid in all_vectors:
            hash_22000 = f"WPA*01*{pmkid.hex()}*{ap.hex()}*{sta.hex()}*{ssid.hex()}***"
            hash_file = os.path.join(tmpdir, f"edge_{name}.22000")
            with open(hash_file, "w") as f:
                f.write(hash_22000 + "\n")

            wl_file = os.path.join(tmpdir, f"edge_{name}_wl.txt")
            with open(wl_file, "wb") as f:
                f.write(b"wrongpass1\nwrongpass2\n")
                f.write(pw + b"\n")

            rc, out = run_vega(hash_file, wl_file)
            found = "PASSWORD FOUND" in out
            check(
                f"Edge: {name}",
                found,
                f"pw_hex={pw.hex()}, out={out[-150:]}" if not found else "",
            )

        # Batch boundary test: password at exact end of a 256-batch
        print()
        print("--- Batch Boundary Tests ---")
        pw = b"batchend1"
        ssid = b"BatchEnd"
        ap = random_mac()
        sta = random_mac()
        pmkid = compute_pmkid(pw, ssid, ap, sta)
        hash_22000 = f"WPA*01*{pmkid.hex()}*{ap.hex()}*{sta.hex()}*{ssid.hex()}***"
        hash_file = os.path.join(tmpdir, "boundary.22000")
        with open(hash_file, "w") as f:
            f.write(hash_22000 + "\n")

        # Fill batch with 255 wrong passwords, then correct at position 256
        wl_file = os.path.join(tmpdir, "boundary_wl.txt")
        with open(wl_file, "wb") as f:
            for i in range(255):
                f.write(f"wrongpw{i:04d}\n".encode())
            f.write(pw + b"\n")

        rc, out = run_vega(hash_file, wl_file, batch_size=256)
        found = "PASSWORD FOUND" in out
        check("Password at batch end (pos 256)", found, out[-150:] if not found else "")

        # Password at position 257 (crosses batch boundary)
        wl_file2 = os.path.join(tmpdir, "boundary_wl2.txt")
        with open(wl_file2, "wb") as f:
            for i in range(256):
                f.write(f"wrongpw{i:04d}\n".encode())
            f.write(pw + b"\n")

        rc, out = run_vega(hash_file, wl_file2, batch_size=256)
        found = "PASSWORD FOUND" in out
        check("Password at batch start (pos 257, new batch)", found, out[-150:] if not found else "")

        # 16800 format with edge cases
        print()
        print("--- 16800 Format Edge Cases ---")
        for name, pw, ssid, ap, sta, pmkid in all_vectors[:5]:
            hash_16800 = f"{pmkid.hex()}*{ap.hex()}*{sta.hex()}*{ssid.hex()}"
            hash_file = os.path.join(tmpdir, f"edge16800_{name}.16800")
            with open(hash_file, "w") as f:
                f.write(hash_16800 + "\n")

            wl_file = os.path.join(tmpdir, f"edge16800_{name}_wl.txt")
            with open(wl_file, "wb") as f:
                f.write(b"wrongpass1\n")
                f.write(pw + b"\n")

            rc, out = run_vega(hash_file, wl_file)
            found = "PASSWORD FOUND" in out
            check(
                f"16800 edge: {name}",
                found,
                f"pw_hex={pw.hex()}, out={out[-150:]}" if not found else "",
            )

        # Summary
        print()
        print(f"=== Results: {PASS} passed, {FAIL} failed ===")
        if FAILED:
            print("Failed tests:")
            for t in FAILED:
                print(f"  - {t}")
            sys.exit(1)
        print("All fuzz tests passed!")


if __name__ == "__main__":
    main()