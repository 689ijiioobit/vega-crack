#!/usr/bin/env python3
"""Generate verified PMKID test vectors for vega-crack testing.
Computes PMKID = HMAC-SHA1-128(PMK, "PMK Name" || AP_MAC || STA_MAC)
where PMK = PBKDF2-HMAC-SHA1(password, SSID, 4096, 32)
"""
import os
import hashlib
import hmac
import struct

def compute_pmkid(password: bytes, ssid: bytes, ap_mac: bytes, sta_mac: bytes) -> bytes:
    pmk = hashlib.pbkdf2_hmac('sha1', password, ssid, 4096, 32)
    msg = b"PMK Name" + ap_mac + sta_mac
    pmkid_full = hmac.new(pmk, msg, hashlib.sha1).digest()
    return pmkid_full[:16]  # PMKID is first 16 bytes

def to_hex(data: bytes) -> str:
    return data.hex()

# Test vectors
vectors = []

# Vector 1: Simple ASCII password and SSID
pw1 = b"password123"
ssid1 = b"PACE969"
ap1 = bytes.fromhex("4604ba734d4e")
sta1 = bytes.fromhex("89acf0e761f4")
pmkid1 = compute_pmkid(pw1, ssid1, ap1, sta1)
vectors.append(("password123 + PACE969", pw1, ssid1, ap1, sta1, pmkid1))

# Vector 2: hashcat! example
pw2 = b"hashcat!"
ssid2 = b"hashcat!"
ap2 = bytes.fromhex("4604ba734d4e")
sta2 = bytes.fromhex("89acf0e761f4")
pmkid2 = compute_pmkid(pw2, ssid2, ap2, sta2)
vectors.append(("hashcat! + hashcat!", pw2, ssid2, ap2, sta2, pmkid2))

# Vector 3: 8-char minimum password
pw3 = b"12345678"
ssid3 = b"TestNet"
ap3 = bytes.fromhex("001122334455")
sta3 = bytes.fromhex("aabbccddeeff")
pmkid3 = compute_pmkid(pw3, ssid3, ap3, sta3)
vectors.append(("12345678 + TestNet", pw3, ssid3, ap3, sta3, pmkid3))

# Vector 4: 63-char max password
pw4 = b"A" * 63
ssid4 = b"MaxLenTest"
ap4 = bytes.fromhex("001122334455")
sta4 = bytes.fromhex("aabbccddeeff")
pmkid4 = compute_pmkid(pw4, ssid4, ap4, sta4)
vectors.append(("63 A's + MaxLenTest", pw4, ssid4, ap4, sta4, pmkid4))

# Vector 5: Non-ASCII password (Latin-1)
pw5 = b"\xe9\xe8\xea\xeb\xec\xed\xee\xef"  # 8 Latin-1 chars
ssid5 = b"Latin1"
ap5 = bytes.fromhex("001122334455")
sta5 = bytes.fromhex("aabbccddeeff")
pmkid5 = compute_pmkid(pw5, ssid5, ap5, sta5)
vectors.append(("Latin-1 chars + Latin1", pw5, ssid5, ap5, sta5, pmkid5))

# Vector 6: Empty SSID
pw6 = b"testpass1"
ssid6 = b""
ap6 = bytes.fromhex("001122334455")
sta6 = bytes.fromhex("aabbccddeeff")
pmkid6 = compute_pmkid(pw6, ssid6, ap6, sta6)
vectors.append(("testpass1 + empty SSID", pw6, ssid6, ap6, sta6, pmkid6))

# Vector 7: 32-byte max SSID
pw7 = b"longssid1"
ssid7 = b"X" * 32
ap7 = bytes.fromhex("001122334455")
sta7 = bytes.fromhex("aabbccddeeff")
pmkid7 = compute_pmkid(pw7, ssid7, ap7, sta7)
vectors.append(("longssid1 + 32-byte SSID", pw7, ssid7, ap7, sta7, pmkid7))

# Vector 8: Password with special chars
pw8 = b"P@ssw0rd!#$%"
ssid8 = b"Special"
ap8 = bytes.fromhex("001122334455")
sta8 = bytes.fromhex("aabbccddeeff")
pmkid8 = compute_pmkid(pw8, ssid8, ap8, sta8)
vectors.append(("P@ssw0rd!#$% + Special", pw8, ssid8, ap8, sta8, pmkid8))

import json
results = []
for name, pw, ssid, ap, sta, pmkid in vectors:
    entry = {
        "name": name,
        "password": pw.decode('latin-1'),
        "password_hex": pw.hex(),
        "ssid": ssid.decode('latin-1'),
        "ssid_hex": ssid.hex(),
        "ap_mac": ap.hex(),
        "sta_mac": sta.hex(),
        "pmkid": pmkid.hex(),
        # 22000 format: WPA*01*PMKID*AP*STA*ESSID***
        "hash_22000": f"WPA*01*{pmkid.hex()}*{ap.hex()}*{sta.hex()}*{ssid.hex()}***",
        # 16800 format: PMKID*AP*STA*ESSID
        "hash_16800": f"{pmkid.hex()}*{ap.hex()}*{sta.hex()}*{ssid.hex()}",
    }
    results.append(entry)
    print(f"[{name}]")
    print(f"  Password: {entry['password']} (hex: {entry['password_hex']})")
    print(f"  SSID: {entry['ssid']} (hex: {entry['ssid_hex']})")
    print(f"  AP MAC: {entry['ap_mac']}")
    print(f"  STA MAC: {entry['sta_mac']}")
    print(f"  PMKID: {entry['pmkid']}")
    print(f"  22000: {entry['hash_22000']}")
    print(f"  16800: {entry['hash_16800']}")
    print()

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_vectors.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"Written {len(results)} test vectors to tests/test_vectors.json")