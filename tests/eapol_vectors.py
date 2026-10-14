#!/usr/bin/env python3
"""Generate EAPOL test vectors for vega-crack testing.

Uses the CORRECT PRF-512 with null separator per IEEE 802.11-2020 §12.7.1.3:
  R = Label || 0x00 || Min(AA,SPA) || Max(AA,SPA) || Min(ANonce,SNonce) || Max(ANonce,SNonce)

This module provides:
  by_keyver[1] = list of keyver 1 (HMAC-MD5) test vectors
  by_keyver[2] = list of keyver 2 (HMAC-SHA1) test vectors
  by_keyver[3] = list of keyver 3 (AES-128-CMAC) test vectors

Each vector has: name, password, password_hex, ssid, ssid_hex, ap_mac, sta_mac,
anonce, snonce, eapol, eapol_len, mic, pmk, kck, hash_22000
"""
import hashlib
import hmac
import json
import os
import struct


def pbkdf2_sha1(password: bytes, ssid: bytes, iterations: int = 4096, out_len: int = 32) -> bytes:
    return hashlib.pbkdf2_hmac('sha1', password, ssid, iterations, out_len)


def prf_512(pmk: bytes, mac_min: bytes, mac_max: bytes,
            nonce_min: bytes, nonce_max: bytes) -> bytes:
    """PRF-512 with null separator per IEEE 802.11-2020 §12.7.1.3."""
    label = b"Pairwise key expansion\x00"  # 23 bytes with null separator
    B = mac_min + mac_max + nonce_min + nonce_max  # 76 bytes
    ptk = b""
    for i in range(4):
        h = hmac.new(pmk, label + B + bytes([i]), hashlib.sha1).digest()
        ptk += h
    return ptk[:64]


def compute_mic_keyver1(kck: bytes, eapol_zeroed: bytes) -> bytes:
    """HMAC-MD5 (WPA1/TKIP)."""
    return hmac.new(kck, eapol_zeroed, hashlib.md5).digest()[:16]


def kdf_256(pmk: bytes, mac_min: bytes, mac_max: bytes,
            nonce_min: bytes, nonce_max: bytes) -> bytes:
    """KDF-256 for keyver 3 — HMAC-SHA-256, matches hashcat.
    PKE = [0x01, 0x00] + "Pairwise key expansion" + B + [0x80, 0x01]
    KCK = HMAC-SHA-256(PMK, PKE)[0:16]
    """
    pke = bytes([0x01, 0x00]) + b"Pairwise key expansion" + \
          mac_min + mac_max + nonce_min + nonce_max + bytes([0x80, 0x01])
    return hmac.new(pmk, pke, hashlib.sha256).digest()[:16]


def compute_mic_keyver2(kck: bytes, eapol_zeroed: bytes) -> bytes:
    """HMAC-SHA1 (WPA2/CCMP)."""
    return hmac.new(kck, eapol_zeroed, hashlib.sha1).digest()[:16]


def compute_mic_keyver3(kck: bytes, eapol_zeroed: bytes) -> bytes:
    """AES-128-CMAC (WPA2). RFC 4493. Verified against RFC 4493 test vectors."""
    sbox = [
        0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
        0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
        0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
        0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
        0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
        0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
        0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
        0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
        0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
        0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
        0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
        0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
        0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
        0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
        0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
        0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
    ]
    rcon = [0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36]

    def _gmul2(a):
        return ((a << 1) ^ 0x1b) & 0xff if a & 0x80 else (a << 1) & 0xff
    def _gmul3(a):
        return _gmul2(a) ^ a

    def _aes128_key_expansion(key):
        ek = bytearray(176)
        ek[:16] = key
        for i in range(10):
            w_temp = bytearray(ek[i*16+12:i*16+16])
            w_temp = w_temp[1:] + w_temp[:1]
            for j in range(4): w_temp[j] = sbox[w_temp[j]]
            w_temp[0] ^= rcon[i]
            for j in range(4):
                base = (i+1)*16 + j*4
                if j == 0:
                    for k in range(4): ek[base+k] = ek[i*16+j*4+k] ^ w_temp[k]
                else:
                    for k in range(4): ek[base+k] = ek[i*16+j*4+k] ^ ek[base-4+k]
        return bytes(ek)

    def _aes128_encrypt(block, ek):
        state = bytearray(block)
        for i in range(16): state[i] ^= ek[i]
        for rnd in range(1, 10):
            for i in range(16): state[i] = sbox[state[i]]
            temp = bytearray(state)
            for row in range(4):
                for col in range(4): state[col*4+row] = temp[((col+row)%4)*4+row]
            for col in range(4):
                s0,s1,s2,s3 = state[col*4],state[col*4+1],state[col*4+2],state[col*4+3]
                state[col*4]   = _gmul2(s0)^_gmul3(s1)^s2^s3
                state[col*4+1] = s0^_gmul2(s1)^_gmul3(s2)^s3
                state[col*4+2] = s0^s1^_gmul2(s2)^_gmul3(s3)
                state[col*4+3] = _gmul3(s0)^s1^s2^_gmul2(s3)
            for i in range(16): state[i] ^= ek[rnd*16+i]
        for i in range(16): state[i] = sbox[state[i]]
        temp = bytearray(state)
        for row in range(4):
            for col in range(4): state[col*4+row] = temp[((col+row)%4)*4+row]
        for i in range(16): state[i] ^= ek[160+i]
        return bytes(state)

    def _left_shift_1(block):
        result = bytearray(16)
        carry = 0
        for i in range(15, -1, -1):
            result[i] = ((block[i] << 1) | carry) & 0xff
            carry = block[i] >> 7
        return bytes(result), carry

    ek = _aes128_key_expansion(kck)
    L = _aes128_encrypt(b'\x00'*16, ek)
    K1, c1 = _left_shift_1(L)
    if c1:
        K1 = bytearray(K1); K1[15] ^= 0x87; K1 = bytes(K1)
    K2, c2 = _left_shift_1(K1)
    if c2:
        K2 = bytearray(K2); K2[15] ^= 0x87; K2 = bytes(K2)

    msg = eapol_zeroed
    n = (len(msg) + 15) // 16 if len(msg) > 0 else 1
    X = b'\x00'*16
    for i in range(n - 1):
        block = msg[i*16:(i+1)*16]
        Y = bytes(a ^ b for a, b in zip(X, block))
        X = _aes128_encrypt(Y, ek)

    last_block = bytearray(16)
    last_start = (n - 1) * 16
    last_len = len(msg) - last_start
    for i in range(last_len):
        last_block[i] = msg[last_start + i]

    if len(msg) > 0 and len(msg) % 16 == 0:
        last_block = bytes(a ^ b for a, b in zip(last_block, K1))
    else:
        pad_byte = last_len & 0xf
        last_block[pad_byte] ^= 0x80
        last_block = bytes(a ^ b for a, b in zip(last_block, K2))

    Y = bytes(a ^ b for a, b in zip(X, last_block))
    return _aes128_encrypt(Y, ek)[:16]


def make_eapol_frame(keyver, snonce, mic_placeholder=b'\x00'*16):
    """Build a minimal 99-byte EAPOL-Key frame for testing."""
    # EAPOL-Key frame structure (99 bytes):
    # 0: Version (1)
    # 1: Type (3 = EAPOL-Key)
    # 2-3: Length (0x005f = 95)
    # 4: Descriptor Type (2 = RSN)
    # 5-6: Key Information (keyver in bits 0-2)
    # 7-8: Key Length
    # 9-16: Replay Counter (8 bytes)
    # 17-48: Key Nonce (SNonce, 32 bytes)
    # 49-64: Key IV (16 bytes, zeros)
    # 65-72: Key RSC (8 bytes, zeros)
    # 73-80: Key ID (8 bytes, zeros)
    # 81-96: Key MIC (16 bytes, placeholder)
    # 97-98: Key Data Length (2 bytes, zeros)

    frame = bytearray(99)
    frame[0] = 0x01  # Version
    frame[1] = 0x03  # Type = EAPOL-Key
    frame[2] = 0x00  # Length high
    frame[3] = 0x5f  # Length low (95)
    frame[4] = 0x02  # Descriptor Type = RSN

    # Key Information: keyver in bits 0-2, bit 3 = MIC, bit 7 = ACK
    key_info = keyver | (1 << 3) | (1 << 8)  # keyver + MIC flag + ACK
    frame[5] = (key_info >> 8) & 0xff
    frame[6] = key_info & 0xff

    frame[7] = 0x00  # Key Length high
    frame[8] = 0x10  # Key Length low (16)

    # Replay Counter
    frame[9] = 0x01

    # SNonce at offset 17
    frame[17:49] = snonce

    # MIC at offset 81
    frame[81:97] = mic_placeholder

    return bytes(frame)


def make_vector(name, password, ssid, ap_mac, sta_mac, anonce, snonce, keyver):
    """Generate a complete EAPOL test vector."""
    pmk = pbkdf2_sha1(password, ssid)

    eapol = make_eapol_frame(keyver, snonce)

    # Zero MIC for computation
    eapol_zeroed = bytearray(eapol)
    eapol_zeroed[81:97] = b'\x00' * 16
    eapol_zeroed = bytes(eapol_zeroed)

    mac_min = min(ap_mac, sta_mac)
    mac_max = max(ap_mac, sta_mac)
    nonce_min = min(anonce, snonce)
    nonce_max = max(anonce, snonce)

    if keyver == 3:
        # keyver 3: KDF-256 (HMAC-SHA-256) — matches hashcat
        kck = kdf_256(pmk, mac_min, mac_max, nonce_min, nonce_max)
    else:
        # keyver 1/2: PRF-512 (HMAC-SHA-1) with null separator
        ptk = prf_512(pmk, mac_min, mac_max, nonce_min, nonce_max)
        kck = ptk[:16]

    if keyver == 1:
        mic = compute_mic_keyver1(kck, eapol_zeroed)
    elif keyver == 2:
        mic = compute_mic_keyver2(kck, eapol_zeroed)
    elif keyver == 3:
        mic = compute_mic_keyver3(kck, eapol_zeroed)
    else:
        raise ValueError(f"Unsupported keyver {keyver}")

    # Build the 22000 hash line (MIC is in the hash, EAPOL has MIC zeroed)
    hash_22000 = (
        f"WPA*02*{mic.hex()}*{ap_mac.hex()}*{sta_mac.hex()}*"
        f"{ssid.hex()}*{anonce.hex()}*{eapol_zeroed.hex()}*a2"
    )

    return {
        "name": name,
        "password": password.decode('latin-1'),
        "password_hex": password.hex(),
        "ssid": ssid.decode('latin-1'),
        "ssid_hex": ssid.hex(),
        "ap_mac": ap_mac.hex(),
        "sta_mac": sta_mac.hex(),
        "anonce": anonce.hex(),
        "snonce": snonce.hex(),
        "eapol": eapol.hex(),
        "eapol_len": len(eapol),
        "mic": mic.hex(),
        "pmk": pmk.hex(),
        "kck": kck.hex(),
        "keyver": keyver,
        "hash_22000": hash_22000,
    }


# === Generate all test vectors ===

_common_ap = bytes.fromhex("4604ba734d4e")
_common_sta = bytes.fromhex("89acf0e761f4")
_common_anonce = bytes.fromhex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
_common_snonce = bytes.fromhex("202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f")

_all_vectors = []

# Keyver 2 (HMAC-SHA1 / WPA2-CCMP) — 3 vectors
for name, pw, ssid, ap, sta in [
    ("password123 + PACE969", b"password123", b"PACE969", _common_ap, _common_sta),
    ("hashcat! + hashcat!", b"hashcat!", b"hashcat!", _common_ap, _common_sta),
    ("12345678 + TestNet", b"12345678", b"TestNet",
     bytes.fromhex("001122334455"), bytes.fromhex("aabbccddeeff")),
]:
    v = make_vector(name, pw, ssid, ap, sta, _common_anonce, _common_snonce, 2)
    _all_vectors.append(v)

# Keyver 1 (HMAC-MD5 / WPA1-TKIP) — 2 vectors
for name, pw, ssid, ap, sta in [
    ("password123 + PACE969 (kv1)", b"password123", b"PACE969", _common_ap, _common_sta),
    ("12345678 + TestNet (kv1)", b"12345678", b"TestNet",
     bytes.fromhex("001122334455"), bytes.fromhex("aabbccddeeff")),
]:
    v = make_vector(name, pw, ssid, ap, sta, _common_anonce, _common_snonce, 1)
    _all_vectors.append(v)

# Keyver 3 (AES-128-CMAC / WPA2) — 2 vectors
for name, pw, ssid, ap, sta in [
    ("password123 + PACE969 (kv3)", b"password123", b"PACE969", _common_ap, _common_sta),
    ("12345678 + TestNet (kv3)", b"12345678", b"TestNet",
     bytes.fromhex("001122334455"), bytes.fromhex("aabbccddeeff")),
]:
    v = make_vector(name, pw, ssid, ap, sta, _common_anonce, _common_snonce, 3)
    _all_vectors.append(v)

# Organize by keyver for the test runner
by_keyver = {1: [], 2: [], 3: []}
for v in _all_vectors:
    by_keyver[v["keyver"]].append(v)


if __name__ == "__main__":
    # Save to JSON
    out_path = os.path.join(os.path.dirname(__file__), "eapol_vectors.json")
    with open(out_path, "w") as f:
        json.dump(_all_vectors, f, indent=2)
    print(f"Written {len(_all_vectors)} EAPOL test vectors to {out_path}")
    for v in _all_vectors:
        print(f"  [{v['keyver']}] {v['name']}: MIC={v['mic']}")