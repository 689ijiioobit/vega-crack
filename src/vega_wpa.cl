// vega_wpa.cl — OpenCL kernel for WPA PMKID cracking
// Self-contained: PBKDF2-HMAC-SHA1 + PMKID computation
// No external includes needed

// === SHA1 Constants ===
// Constant block tail for PBKDF2 SHA1: w[5..15] = padding + length
// 64-byte HMAC key block + 20-byte data = 84 bytes = 0x80000000 at byte 20,
// length in bits = 672 at word 15.
__constant uint W_TAIL[11] = {
    0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 672
};

// OPT-2: Use OpenCL rotate() built-in — ROCm LLVM maps this to a single
// hardware rotation instruction. amd_bitalign was tested and is SLOWER
// (13,378 H/s vs 16,156 H/s) because LLVM already optimizes rotate() better.
#define SHA1_ROTL(x, n) rotate((x), (uint)(n))

// SHA1 state: 5 uint words
typedef struct {
    uint h[5];
} sha1_state_t;

// Initialize SHA1 state
static inline void sha1_init(sha1_state_t *s) {
    s->h[0] = 0x67452301;
    s->h[1] = 0xEFCDAB89;
    s->h[2] = 0x98BADCFE;
    s->h[3] = 0x10325476;
    s->h[4] = 0xC3D2E1F0;
}

// Process one 64-byte block (16 big-endian uints in w_in)
// Process one 64-byte block (16 big-endian uints in w_in).
// R105 CRITICAL FIX: the previous fully-unrolled version was WRONG —
// (a) message schedule used w_cur[14] where SHA-1 needs w[(i+2)&15]
//     (w[i-14] mis-transcribed as an absolute index), and
// (b) the round register-cycle rotated the wrong variable (rotate(e,5)
//     after a=t reassigned the accumulator into a).
// Verified broken on-device: sha1("abc") != A9993E36... Every SHA-1 path
// (WPA PMKID, EAPOL MIC, PBKDF2) computed wrong hashes. Replaced with the
// same rolling-window pattern used by the (correct) sha256_process_block.
static inline void sha1_process_block(sha1_state_t *s, const uint *w_in) {
    uint w[16];
    for (int i = 0; i < 16; i++) w[i] = w_in[i];

    uint a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3], e = s->h[4];
    // OPT: bitselect forms of CH/MAJ (verified on-device: OpenCL bitselect(a,b,c)
    // = c-bits pick b). CH = bitselect(d, c, b) compiles to a single V_BFI vs 4
    // VALU for the boolean form; MAJ = bitselect(b, c, b^d) = XOR+BFI (2 VALU)
    // vs 5. Measured: 14,106 -> 14,425 H/s (+2.25%) on WPA benchmark.
    // (A manual full unroll with named W registers measured +0.0% — ACO already
    // folds the &15 index arithmetic — and was reverted.)
    for (int i = 0; i < 20; i++) {
        if (i >= 16) {
            w[i & 15] = rotate(w[(i + 13) & 15] ^ w[(i + 8) & 15] ^ w[(i + 2) & 15] ^ w[i & 15], 1u);
        }
        uint t = rotate(a, 5u) + bitselect(d, c, b) + e + 1518500249u + w[i & 15];
        e = d; d = c; c = rotate(b, 30u); b = a; a = t;
    }
    for (int i = 20; i < 40; i++) {
        w[i & 15] = rotate(w[(i + 13) & 15] ^ w[(i + 8) & 15] ^ w[(i + 2) & 15] ^ w[i & 15], 1u);
        uint t = rotate(a, 5u) + (b ^ c ^ d) + e + 1859775393u + w[i & 15];
        e = d; d = c; c = rotate(b, 30u); b = a; a = t;
    }
    for (int i = 40; i < 60; i++) {
        w[i & 15] = rotate(w[(i + 13) & 15] ^ w[(i + 8) & 15] ^ w[(i + 2) & 15] ^ w[i & 15], 1u);
        uint t = rotate(a, 5u) + bitselect(b, c, b ^ d) + e + 2400959708u + w[i & 15];
        e = d; d = c; c = rotate(b, 30u); b = a; a = t;
    }
    for (int i = 60; i < 80; i++) {
        w[i & 15] = rotate(w[(i + 13) & 15] ^ w[(i + 8) & 15] ^ w[(i + 2) & 15] ^ w[i & 15], 1u);
        uint t = rotate(a, 5u) + (b ^ c ^ d) + e + 3395469782u + w[i & 15];
        e = d; d = c; c = rotate(b, 30u); b = a; a = t;
    }
    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d; s->h[4] += e;
}

// Fused HMAC-SHA1: inner + outer hash in one function.
// Shares w_cur/w_next arrays between inner and outer, saving 32 VGPRs.
// Input: ipad[5], opad[5], msg[5]. Output: out[5].
// Fused HMAC-SHA1: inner + outer hash in one function.
// R105 CRITICAL FIX: previous inlined-unrolled body had the broken SHA-1
// schedule/round pattern (see sha1_process_block). Rewritten on top of the
// corrected sha1_process_block. Input: ipad[5], opad[5], msg[5] (20 bytes).
// Output: out[5]. Block layout: msg words 0-4 + W_TAIL padding (0x80 at byte
// 20, 672-bit length) — single 64-byte block per hash.
static inline void hmac_sha1_run(const uint *ipad, const uint *opad,
                                  const uint *msg, uint *out) {
    uint w[16];
    w[0] = msg[0]; w[1] = msg[1]; w[2] = msg[2]; w[3] = msg[3]; w[4] = msg[4];
    for (int i = 0; i < 11; i++) w[5 + i] = W_TAIL[i];

    sha1_state_t s;
    for (int i = 0; i < 5; i++) s.h[i] = ipad[i];
    sha1_process_block(&s, w);          // inner: ipad-state + msg block

    for (int i = 0; i < 5; i++) w[i] = s.h[i];  // inner digest = outer message
    for (int i = 0; i < 5; i++) s.h[i] = opad[i];
    sha1_process_block(&s, w);          // outer: opad-state + inner-digest block

    for (int i = 0; i < 5; i++) out[i] = s.h[i];
}

// One SHA-1 block from a saved state, updating the state in place.
// R105 CRITICAL FIX: same broken unrolled body as sha1_process_block —
// rewritten on the corrected rolling-window block function.




// Pack bytes into 16 big-endian uints (zero-padded)
// Defense-in-depth: clamp len to 64 to prevent OOB write past block[15]
static inline void bytes_to_block(const uchar *data, uint len, uint *block) {
    if (len > 64) len = 64;
    for (int i = 0; i < 16; i++) block[i] = 0;
    for (uint i = 0; i < len; i++) {
        block[i >> 2] |= ((uint)data[i]) << (24 - (i & 3) * 8);
    }
}

// Continue SHA1 from a saved state with a single-block message (len <= 55)
// The saved state already has 64 bytes processed (ipad/opad).
// Defense-in-depth: clamp len to 55 — at len=56 the 0x80 padding byte
// lands in block[14] which is then overwritten by the length field,
// silently corrupting the hash.
static inline void sha1_continue_single(const uint *saved_h, const uchar *msg, uint len, uint *out_h) {
    if (len > 55) len = 55;
    sha1_state_t s;
    for (int i = 0; i < 5; i++) s.h[i] = saved_h[i];
    uint block[16];
    bytes_to_block(msg, len, block);
    uint pad_word = len >> 2;
    uint pad_byte = len & 3;
    block[pad_word] |= ((uint)0x80) << (24 - pad_byte * 8);
    ulong bits = ((ulong)64 + len) * 8;
    block[14] = (uint)(bits >> 32);
    block[15] = (uint)bits;
    sha1_process_block(&s, block);
    for (int i = 0; i < 5; i++) out_h[i] = s.h[i];
}

// Continue SHA1 from a saved state with a message of arbitrary length.
// Processes full 64-byte blocks, then handles the final block with padding.
// Used for EAPOL frames up to 256 bytes (total HMAC inner = 64 + 256 = 320 bytes = 5 blocks).
static inline void sha1_continue_multi(const uint *saved_h, const uchar *msg, uint len, uint *out_h) {
    sha1_state_t s;
    for (int i = 0; i < 5; i++) s.h[i] = saved_h[i];
    ulong bits = ((ulong)64 + len) * 8;  // 64 = precomputed ipad block

    // Process full 64-byte blocks
    uint offset = 0;
    while (offset + 64 <= len) {
        uint block[16];
        bytes_to_block(msg + offset, 64, block);
        sha1_process_block(&s, block);
        offset += 64;
    }

    // Final block: remaining bytes + 0x80 padding + length field
    uint rem = len - offset;
    if (rem > 55) {
        // Remaining doesn't fit padding+length in one block — need two blocks
        // Block 1: remaining bytes + 0x80 (no length)
        uint block1[16];
        bytes_to_block(msg + offset, rem, block1);
        uint pw = rem >> 2;
        uint pb = rem & 3;
        block1[pw] |= ((uint)0x80) << (24 - pb * 8);
        sha1_process_block(&s, block1);
        // Block 2: zeros + 8-byte length
        uint block2[16];
        for (int i = 0; i < 14; i++) block2[i] = 0;
        block2[14] = (uint)(bits >> 32);
        block2[15] = (uint)bits;
        sha1_process_block(&s, block2);
    } else {
        // Remaining + 0x80 + zeros + length fits in one block
        uint block[16];
        bytes_to_block(msg + offset, rem, block);
        uint pw = rem >> 2;
        uint pb = rem & 3;
        block[pw] |= ((uint)0x80) << (24 - pb * 8);
        block[14] = (uint)(bits >> 32);
        block[15] = (uint)bits;
        sha1_process_block(&s, block);
    }

    for (int i = 0; i < 5; i++) out_h[i] = s.h[i];
}

// Precompute HMAC ipad/opad states for a key
static inline void hmac_precompute(const uchar *key, uint key_len,
                            uint *ipad_state, uint *opad_state) {
    uint k[16];
    for (int i = 0; i < 16; i++) k[i] = 0;

    if (key_len <= 64) {
        for (uint i = 0; i < key_len; i++) {
            k[i >> 2] |= ((uint)key[i]) << (24 - (i & 3) * 8);
        }
    } else {
        // Key > 64 bytes: hash it first, then use the 20-byte digest as the key.
        // Process ALL full 64-byte blocks, then a one- or two-block final.
        // (Previous code only processed the first 64 bytes + clamped remainder,
        //  which was incorrect for keys > 128 bytes and produced wrong hashes
        //  for keys 120-128 bytes. Now matches sha1_continue_multi pattern.)
        uint khash[5];
        sha1_state_t s;
        sha1_init(&s);
        uint offset = 0;
        while (offset + 64 <= key_len) {
            uint blk[16];
            bytes_to_block(key + offset, 64, blk);
            sha1_process_block(&s, blk);
            offset += 64;
        }
        uint rem = key_len - offset;
        ulong bits = (ulong)key_len * 8;
        if (rem > 55) {
            // Two-block final: remaining bytes + 0x80 in block1, zeros + length in block2
            uint block1[16];
            bytes_to_block(key + offset, rem, block1);
            uint pw = rem >> 2; uint pb = rem & 3;
            block1[pw] |= ((uint)0x80) << (24 - pb * 8);
            sha1_process_block(&s, block1);
            uint block2[16];
            for (int i = 0; i < 14; i++) block2[i] = 0;
            block2[14] = (uint)(bits >> 32);
            block2[15] = (uint)bits;
            sha1_process_block(&s, block2);
        } else {
            uint block[16];
            bytes_to_block(key + offset, rem, block);
            uint pw = rem >> 2; uint pb = rem & 3;
            block[pw] |= ((uint)0x80) << (24 - pb * 8);
            block[14] = (uint)(bits >> 32);
            block[15] = (uint)bits;
            sha1_process_block(&s, block);
        }
        for (int i = 0; i < 5; i++) khash[i] = s.h[i];
        for (int i = 0; i < 5; i++) k[i] = khash[i];
    }

    // XOR each byte with ipad (0x36) / opad (0x5c)
    // Since k[] is big-endian uints, XOR the whole uint with the repeated constant
    uint ipad_block[16], opad_block[16];
    for (int i = 0; i < 16; i++) {
        ipad_block[i] = k[i] ^ 0x36363636;
        opad_block[i] = k[i] ^ 0x5c5c5c5c;
    }

    sha1_state_t s;
    sha1_init(&s);
    sha1_process_block(&s, ipad_block);
    for (int i = 0; i < 5; i++) ipad_state[i] = s.h[i];

    sha1_init(&s);
    sha1_process_block(&s, opad_block);
    for (int i = 0; i < 5; i++) opad_state[i] = s.h[i];
}

// HMAC-SHA1 using precomputed ipad/opad states (handles msg of any length)
static inline void hmac_sha1_precomputed(const uint *ipad_state, const uint *opad_state,
                                   const uchar *msg, uint msg_len, uint *mac) {
    uint inner_h[5];
    if (msg_len <= 55) {
        sha1_continue_single(ipad_state, msg, msg_len, inner_h);
    } else {
        sha1_continue_multi(ipad_state, msg, msg_len, inner_h);
    }

    // Outer hash: feed inner_h (5 uints = 20 bytes) directly as uint block words.
    // This avoids the uint→bytes→uint roundtrip that bytes_to_block would do.
    // inner_h is big-endian SHA1 output, same as block word ordering — no swap needed.
    sha1_state_t outer;
    for (int i = 0; i < 5; i++) outer.h[i] = opad_state[i];
    uint block[16];
    for (int i = 0; i < 5; i++) block[i] = inner_h[i];
    // Padding: 20 bytes of data → 0x80 at byte 20 (word 5, byte 0)
    block[5] = 0x80000000;
    for (int i = 6; i < 14; i++) block[i] = 0;
    // Length: 64 (ipad block) + 20 (inner hash) = 84 bytes = 672 bits
    block[14] = 0;
    block[15] = 672;
    sha1_process_block(&outer, block);
    for (int i = 0; i < 5; i++) mac[i] = outer.h[i];
}

// === AES-128 Constants (for EAPOL keyver 3 / AES-128-CMAC) ===
// All AES operations use big-endian byte ordering, same as the SHA1 path.
// References: FIPS-197 (AES), RFC 4493 (AES-CMAC).

// AES S-box (256 entries)
__constant uchar AES_SBOX[256] = {
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// AES TE0 table: combines SubBytes + ShiftRows + MixColumns into one lookup per byte.
// TE0[x] = [2*S[x], S[x], S[x], 3*S[x]] packed as a big-endian u32.
// TE1/TE2/TE3 are byte-rotations of TE0 for the ShiftRows offsets.
// Generated from the S-box and GF(2^8) multiplication constants.
__constant uint AES_TE0[256] = {
    0xc66363a5, 0xf87c7c84, 0xee777799, 0xf67b7b8d, 0xfff2f20d, 0xd66b6bbd, 0xde6f6fb1, 0x91c5c554,
    0x60303050, 0x02010103, 0xce6767a9, 0x562b2b7d, 0xe7fefe19, 0xb5d7d762, 0x4dababe6, 0xec76769a,
    0x8fcaca45, 0x1f82829d, 0x89c9c940, 0xfa7d7d87, 0xeffafa15, 0xb25959eb, 0x8e4747c9, 0xfbf0f00b,
    0x41adadec, 0xb3d4d467, 0x5fa2a2fd, 0x45afafea, 0x239c9cbf, 0x53a4a4f7, 0xe4727296, 0x9bc0c05b,
    0x75b7b7c2, 0xe1fdfd1c, 0x3d9393ae, 0x4c26266a, 0x6c36365a, 0x7e3f3f41, 0xf5f7f702, 0x83cccc4f,
    0x6834345c, 0x51a5a5f4, 0xd1e5e534, 0xf9f1f108, 0xe2717193, 0xabd8d873, 0x62313153, 0x2a15153f,
    0x0804040c, 0x95c7c752, 0x46232365, 0x9dc3c35e, 0x30181828, 0x379696a1, 0x0a05050f, 0x2f9a9ab5,
    0x0e070709, 0x24121236, 0x1b80809b, 0xdfe2e23d, 0xcdebeb26, 0x4e272769, 0x7fb2b2cd, 0xea75759f,
    0x1209091b, 0x1d83839e, 0x582c2c74, 0x341a1a2e, 0x361b1b2d, 0xdc6e6eb2, 0xb45a5aee, 0x5ba0a0fb,
    0xa45252f6, 0x763b3b4d, 0xb7d6d661, 0x7db3b3ce, 0x5229297b, 0xdde3e33e, 0x5e2f2f71, 0x13848497,
    0xa65353f5, 0xb9d1d168, 0x00000000, 0xc1eded2c, 0x40202060, 0xe3fcfc1f, 0x79b1b1c8, 0xb65b5bed,
    0xd46a6abe, 0x8dcbcb46, 0x67bebed9, 0x7239394b, 0x944a4ade, 0x984c4cd4, 0xb05858e8, 0x85cfcf4a,
    0xbbd0d06b, 0xc5efef2a, 0x4faaaae5, 0xedfbfb16, 0x864343c5, 0x9a4d4dd7, 0x66333355, 0x11858594,
    0x8a4545cf, 0xe9f9f910, 0x04020206, 0xfe7f7f81, 0xa05050f0, 0x783c3c44, 0x259f9fba, 0x4ba8a8e3,
    0xa25151f3, 0x5da3a3fe, 0x804040c0, 0x058f8f8a, 0x3f9292ad, 0x219d9dbc, 0x70383848, 0xf1f5f504,
    0x63bcbcdf, 0x77b6b6c1, 0xafdada75, 0x42212163, 0x20101030, 0xe5ffff1a, 0xfdf3f30e, 0xbfd2d26d,
    0x81cdcd4c, 0x180c0c14, 0x26131335, 0xc3ecec2f, 0xbe5f5fe1, 0x359797a2, 0x884444cc, 0x2e171739,
    0x93c4c457, 0x55a7a7f2, 0xfc7e7e82, 0x7a3d3d47, 0xc86464ac, 0xba5d5de7, 0x3219192b, 0xe6737395,
    0xc06060a0, 0x19818198, 0x9e4f4fd1, 0xa3dcdc7f, 0x44222266, 0x542a2a7e, 0x3b9090ab, 0x0b888883,
    0x8c4646ca, 0xc7eeee29, 0x6bb8b8d3, 0x2814143c, 0xa7dede79, 0xbc5e5ee2, 0x160b0b1d, 0xaddbdb76,
    0xdbe0e03b, 0x64323256, 0x743a3a4e, 0x140a0a1e, 0x924949db, 0x0c06060a, 0x4824246c, 0xb85c5ce4,
    0x9fc2c25d, 0xbdd3d36e, 0x43acacef, 0xc46262a6, 0x399191a8, 0x319595a4, 0xd3e4e437, 0xf279798b,
    0xd5e7e732, 0x8bc8c843, 0x6e373759, 0xda6d6db7, 0x018d8d8c, 0xb1d5d564, 0x9c4e4ed2, 0x49a9a9e0,
    0xd86c6cb4, 0xac5656fa, 0xf3f4f407, 0xcfeaea25, 0xca6565af, 0xf47a7a8e, 0x47aeaee9, 0x10080818,
    0x6fbabad5, 0xf0787888, 0x4a25256f, 0x5c2e2e72, 0x381c1c24, 0x57a6a6f1, 0x73b4b4c7, 0x97c6c651,
    0xcbe8e823, 0xa1dddd7c, 0xe874749c, 0x3e1f1f21, 0x964b4bdd, 0x61bdbddc, 0x0d8b8b86, 0x0f8a8a85,
    0xe0707090, 0x7c3e3e42, 0x71b5b5c4, 0xcc6666aa, 0x904848d8, 0x06030305, 0xf7f6f601, 0x1c0e0e12,
    0xc26161a3, 0x6a35355f, 0xae5757f9, 0x69b9b9d0, 0x17868691, 0x99c1c158, 0x3a1d1d27, 0x279e9eb9,
    0xd9e1e138, 0xebf8f813, 0x2b9898b3, 0x22111133, 0xd26969bb, 0xa9d9d970, 0x078e8e89, 0x339494a7,
    0x2d9b9bb6, 0x3c1e1e22, 0x15878792, 0xc9e9e920, 0x87cece49, 0xaa5555ff, 0x50282878, 0xa5dfdf7a,
    0x038c8c8f, 0x59a1a1f8, 0x09898980, 0x1a0d0d17, 0x65bfbfda, 0xd7e6e631, 0x844242c6, 0xd06868b8,
    0x824141c3, 0x299999b0, 0x5a2d2d77, 0x1e0f0f11, 0x7bb0b0cb, 0xa85454fc, 0x6dbbbbd6, 0x2c16163a
};

// AES round constants for key expansion (rcon[0..9] for rounds 1..10)
__constant uint AES_RCON[10] = {
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000,
    0x1b000000, 0x36000000
};

// SubWord: apply S-box to each byte of a u32 word
static inline uint aes_sub_word(uint w) {
    return ((uint)AES_SBOX[(w >> 24) & 0xFF] << 24) |
           ((uint)AES_SBOX[(w >> 16) & 0xFF] << 16) |
           ((uint)AES_SBOX[(w >>  8) & 0xFF] <<  8) |
           ((uint)AES_SBOX[ w        & 0xFF]);
}

// RotWord: rotate u32 left by 8 bits (cyclic byte rotation)
static inline uint aes_rot_word(uint w) {
    return (w << 8) | (w >> 24);
}

// AES-128 key expansion: 4 key words → 44 round-key words
// key[0..3] are big-endian u32 words (k[0] = most significant).
static inline void aes128_key_expansion(const uint *key, uint *ks) {
    for (int i = 0; i < 4; i++) ks[i] = key[i];
    for (int r = 0; r < 10; r++) {
        uint t = aes_sub_word(aes_rot_word(ks[4*r + 3])) ^ AES_RCON[r];
        ks[4*r + 4] = ks[4*r]     ^ t;
        ks[4*r + 5] = ks[4*r + 1] ^ ks[4*r + 4];
        ks[4*r + 6] = ks[4*r + 2] ^ ks[4*r + 5];
        ks[4*r + 7] = ks[4*r + 3] ^ ks[4*r + 6];
    }
}

// AES-128 encrypt one 16-byte block.
// in[0..3] and out[0..3] are big-endian u32 words (column-major state).
// ks[0..43] is the expanded key from aes128_key_expansion.
// Uses the standard T-table approach with 4 rotated views of TE0:
//   TE1 = ROTR(TE0, 8), TE2 = ROTR(TE0, 16), TE3 = ROTR(TE0, 24)
// These are computed on-the-fly to avoid storing 4 large tables.
static inline void aes128_encrypt_block(const uint *in, const uint *ks, uint *out) {
    uint s0 = in[0] ^ ks[0];
    uint s1 = in[1] ^ ks[1];
    uint s2 = in[2] ^ ks[2];
    uint s3 = in[3] ^ ks[3];

    for (int r = 1; r < 10; r++) {
        // SubBytes + ShiftRows + MixColumns + AddRoundKey (combined via TE tables)
        // TE0[x] = [2*S[x], S[x], S[x], 3*S[x]]
        // TE1[x] = ROTR(TE0[x], 8)  = [3*S[x], 2*S[x], S[x], S[x]]
        // TE2[x] = ROTR(TE0[x], 16) = [S[x], 3*S[x], 2*S[x], S[x]]
        // TE3[x] = ROTR(TE0[x], 24) = [S[x], S[x], 3*S[x], 2*S[x]]
        uint t0 = AES_TE0[(s0 >> 24) & 0xFF]
                ^ rotate(AES_TE0[(s1 >> 16) & 0xFF], (uint)24)  // TE1
                ^ rotate(AES_TE0[(s2 >>  8) & 0xFF], (uint)16)  // TE2
                ^ rotate(AES_TE0[ s3        & 0xFF], (uint) 8)  // TE3
                ^ ks[4*r];
        uint t1 = AES_TE0[(s1 >> 24) & 0xFF]
                ^ rotate(AES_TE0[(s2 >> 16) & 0xFF], (uint)24)
                ^ rotate(AES_TE0[(s3 >>  8) & 0xFF], (uint)16)
                ^ rotate(AES_TE0[ s0        & 0xFF], (uint) 8)
                ^ ks[4*r+1];
        uint t2 = AES_TE0[(s2 >> 24) & 0xFF]
                ^ rotate(AES_TE0[(s3 >> 16) & 0xFF], (uint)24)
                ^ rotate(AES_TE0[(s0 >>  8) & 0xFF], (uint)16)
                ^ rotate(AES_TE0[ s1        & 0xFF], (uint) 8)
                ^ ks[4*r+2];
        uint t3 = AES_TE0[(s3 >> 24) & 0xFF]
                ^ rotate(AES_TE0[(s0 >> 16) & 0xFF], (uint)24)
                ^ rotate(AES_TE0[(s1 >>  8) & 0xFF], (uint)16)
                ^ rotate(AES_TE0[ s2        & 0xFF], (uint) 8)
                ^ ks[4*r+3];
        s0 = t0; s1 = t1; s2 = t2; s3 = t3;
    }

    // Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    out[0] = (((uint)AES_SBOX[(s0 >> 24) & 0xFF] << 24) |
              ((uint)AES_SBOX[(s1 >> 16) & 0xFF] << 16) |
              ((uint)AES_SBOX[(s2 >>  8) & 0xFF] <<  8) |
              ((uint)AES_SBOX[ s3        & 0xFF])) ^ ks[40];
    out[1] = (((uint)AES_SBOX[(s1 >> 24) & 0xFF] << 24) |
              ((uint)AES_SBOX[(s2 >> 16) & 0xFF] << 16) |
              ((uint)AES_SBOX[(s3 >>  8) & 0xFF] <<  8) |
              ((uint)AES_SBOX[ s0        & 0xFF])) ^ ks[41];
    out[2] = (((uint)AES_SBOX[(s2 >> 24) & 0xFF] << 24) |
              ((uint)AES_SBOX[(s3 >> 16) & 0xFF] << 16) |
              ((uint)AES_SBOX[(s0 >>  8) & 0xFF] <<  8) |
              ((uint)AES_SBOX[ s1        & 0xFF])) ^ ks[42];
    out[3] = (((uint)AES_SBOX[(s3 >> 24) & 0xFF] << 24) |
              ((uint)AES_SBOX[(s0 >> 16) & 0xFF] << 16) |
              ((uint)AES_SBOX[(s1 >>  8) & 0xFF] <<  8) |
              ((uint)AES_SBOX[ s2        & 0xFF])) ^ ks[43];
}

// compute_cmac_mic: AES-128-CMAC per RFC 4493.
// Computes the 16-byte CMAC tag of eapol (MIC field already zeroed by caller).
// kck[0..3] are the 16-byte KCK as big-endian u32 words (PTK[0:16]).
// eapol is the EAPOL frame bytes; eapol_len is its length (up to 256).
// Writes 4 big-endian u32 words to mic_out (matches target comparison format).
static inline void compute_cmac_mic(const uint *kck,
                              const uchar *eapol, uint eapol_len,
                              uint *mic_out) {
    // Key expansion
    uint ks[44];
    aes128_key_expansion(kck, ks);

    // Generate subkeys K1, K2 (RFC 4493 §2.3)
    uint zero_block[4] = {0, 0, 0, 0};
    uint L[4];
    aes128_encrypt_block(zero_block, ks, L);

    // K1 = L << 1; if MSB(L) then K1[3] ^= 0x87
    uint K1[4], K2[4];
    bool msb_L = (L[0] >> 31) & 1;
    K1[3] = (L[3] << 1) & 0xfffffffe;
    K1[2] = ((L[2] << 1) | ((L[3] >> 31) & 1)) & 0xffffffff;
    K1[1] = ((L[1] << 1) | ((L[2] >> 31) & 1)) & 0xffffffff;
    K1[0] = ((L[0] << 1) | ((L[1] >> 31) & 1)) & 0xffffffff;
    if (msb_L) K1[3] ^= 0x87;

    // K2 = K1 << 1; if MSB(K1) then K2[3] ^= 0x87
    bool msb_K1 = (K1[0] >> 31) & 1;
    K2[3] = (K1[3] << 1) & 0xfffffffe;
    K2[2] = ((K1[2] << 1) | ((K1[3] >> 31) & 1)) & 0xffffffff;
    K2[1] = ((K1[1] << 1) | ((K1[2] >> 31) & 1)) & 0xffffffff;
    K2[0] = ((K1[0] << 1) | ((K1[1] >> 31) & 1)) & 0xffffffff;
    if (msb_K1) K2[3] ^= 0x87;

    // Determine number of blocks and prepare the last block
    uint n = (eapol_len + 15) / 16;
    if (n == 0) n = 1;  // Empty message → 1 padded block

    // CBC-MAC: process all blocks except the last
    uint X[4] = {0, 0, 0, 0};
    for (uint i = 0; i < n - 1; i++) {
        uint block[4];
        for (int j = 0; j < 4; j++) {
            uint off = i * 16 + j * 4;
            block[j] = ((uint)eapol[off]     << 24) |
                       ((uint)eapol[off + 1] << 16) |
                       ((uint)eapol[off + 2] <<  8) |
                       ((uint)eapol[off + 3]);
        }
        uint Y[4];
        for (int j = 0; j < 4; j++) Y[j] = X[j] ^ block[j];
        aes128_encrypt_block(Y, ks, X);
    }

    // Prepare the last block
    uint last_block[4];
    uint last_start = (n - 1) * 16;
    uint last_len = eapol_len - last_start;  // bytes in last block (1..16, or 0 for empty)

    // Zero the last block buffer
    for (int j = 0; j < 4; j++) last_block[j] = 0;

    // Copy remaining message bytes into last_block (big-endian packing)
    for (uint i = 0; i < last_len; i++) {
        last_block[i >> 2] |= ((uint)eapol[last_start + i]) << (24 - (i & 3) * 8);
    }

    // Choose subkey and apply padding if needed
    if (eapol_len > 0 && (eapol_len % 16 == 0)) {
        // Complete last block → XOR with K1
        for (int j = 0; j < 4; j++) last_block[j] ^= K1[j];
    } else {
        // Incomplete (or empty) last block → pad with 0x80 and XOR with K2
        uint pad_byte = last_len & 0xF;  // position of 0x80 (0 for empty message)
        last_block[pad_byte >> 2] |= ((uint)0x80) << (24 - (pad_byte & 3) * 8);
        for (int j = 0; j < 4; j++) last_block[j] ^= K2[j];
    }

    // Final: Y = X XOR last_block; T = AES-128(K, Y)
    uint Y[4];
    for (int j = 0; j < 4; j++) Y[j] = X[j] ^ last_block[j];
    aes128_encrypt_block(Y, ks, mic_out);
}

// === MD5 Constants (for EAPOL keyver 1 / HMAC-MD5) ===
// MD5 uses LITTLE-ENDIAN uint words — the opposite of SHA1's big-endian.
// This is critical: every byte-to-word packing and every length field
// uses little-endian ordering.

// MD5 sine table: T[i] = floor(2^32 * abs(sin(i+1)))
__constant uint MD5_T[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

// MD5 shift amounts per round
__constant uint MD5_S[64] = {
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,  5,  9, 14, 20,
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21
};

// OPT-2: Use OpenCL rotate() built-in for MD5 rotations too.
#define MD5_ROTL(x, n) rotate((x), (uint)(n))

// Byte-swap a 32-bit word (convert little-endian ↔ big-endian).
// Used to convert MD5's little-endian output to big-endian for comparison
// against targets (which the host stores as big-endian uints).
#define MD5_SWAP32(x) \
    ((((x) >> 24) & 0x000000FF) | \
     (((x) >>  8) & 0x0000FF00) | \
     (((x) <<  8) & 0x00FF0000) | \
     (((x) << 24) & 0xFF000000))

// MD5 state: 4 uint words (little-endian, unlike SHA1's 5 big-endian words)
typedef struct {
    uint h[4];
} md5_state_t;

// Initialize MD5 state
static inline void md5_init(md5_state_t *s) {
    s->h[0] = 0x67452301;
    s->h[1] = 0xefcdab89;
    s->h[2] = 0x98badcfe;
    s->h[3] = 0x10325476;
}

// Process one 64-byte block (16 little-endian uints in w_in).
// MD5 has no message schedule expansion (unlike SHA1), so the block is
// read-only — we copy to a local buffer for register allocation.
static inline void md5_process_block(md5_state_t *s, const uint *w_in) {
    uint a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3];

    uint x0=w_in[0], x1=w_in[1], x2=w_in[2], x3=w_in[3],
         x4=w_in[4], x5=w_in[5], x6=w_in[6], x7=w_in[7],
         x8=w_in[8], x9=w_in[9], x10=w_in[10], x11=w_in[11],
         x12=w_in[12], x13=w_in[13], x14=w_in[14], x15=w_in[15];

    // Round 0
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3614090360u + x0, 7u); a = tmp; }
    // Round 1
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3905402710u + x1, 12u); a = tmp; }
    // Round 2
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 606105819u + x2, 17u); a = tmp; }
    // Round 3
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3250441966u + x3, 22u); a = tmp; }
    // Round 4
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4118548399u + x4, 7u); a = tmp; }
    // Round 5
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1200080426u + x5, 12u); a = tmp; }
    // Round 6
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2821735955u + x6, 17u); a = tmp; }
    // Round 7
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4249261313u + x7, 22u); a = tmp; }
    // Round 8
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1770035416u + x8, 7u); a = tmp; }
    // Round 9
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2336552879u + x9, 12u); a = tmp; }
    // Round 10
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4294925233u + x10, 17u); a = tmp; }
    // Round 11
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2304563134u + x11, 22u); a = tmp; }
    // Round 12
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1804603682u + x12, 7u); a = tmp; }
    // Round 13
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4254626195u + x13, 12u); a = tmp; }
    // Round 14
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2792965006u + x14, 17u); a = tmp; }
    // Round 15
    { uint f_val = bitselect(d, c, b); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1236535329u + x15, 22u); a = tmp; }
    // Round 16
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4129170786u + x1, 5u); a = tmp; }
    // Round 17
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3225465664u + x6, 9u); a = tmp; }
    // Round 18
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 643717713u + x11, 14u); a = tmp; }
    // Round 19
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3921069994u + x0, 20u); a = tmp; }
    // Round 20
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3593408605u + x5, 5u); a = tmp; }
    // Round 21
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 38016083u + x10, 9u); a = tmp; }
    // Round 22
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3634488961u + x15, 14u); a = tmp; }
    // Round 23
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3889429448u + x4, 20u); a = tmp; }
    // Round 24
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 568446438u + x9, 5u); a = tmp; }
    // Round 25
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3275163606u + x14, 9u); a = tmp; }
    // Round 26
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4107603335u + x3, 14u); a = tmp; }
    // Round 27
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1163531501u + x8, 20u); a = tmp; }
    // Round 28
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2850285829u + x13, 5u); a = tmp; }
    // Round 29
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4243563512u + x2, 9u); a = tmp; }
    // Round 30
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1735328473u + x7, 14u); a = tmp; }
    // Round 31
    { uint f_val = bitselect(c, b, d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2368359562u + x12, 20u); a = tmp; }
    // Round 32
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4294588738u + x5, 4u); a = tmp; }
    // Round 33
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2272392833u + x8, 11u); a = tmp; }
    // Round 34
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1839030562u + x11, 16u); a = tmp; }
    // Round 35
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4259657740u + x14, 23u); a = tmp; }
    // Round 36
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2763975236u + x1, 4u); a = tmp; }
    // Round 37
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1272893353u + x4, 11u); a = tmp; }
    // Round 38
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4139469664u + x7, 16u); a = tmp; }
    // Round 39
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3200236656u + x10, 23u); a = tmp; }
    // Round 40
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 681279174u + x13, 4u); a = tmp; }
    // Round 41
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3936430074u + x0, 11u); a = tmp; }
    // Round 42
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3572445317u + x3, 16u); a = tmp; }
    // Round 43
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 76029189u + x6, 23u); a = tmp; }
    // Round 44
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3654602809u + x9, 4u); a = tmp; }
    // Round 45
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3873151461u + x12, 11u); a = tmp; }
    // Round 46
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 530742520u + x15, 16u); a = tmp; }
    // Round 47
    { uint f_val = (b ^ c ^ d); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3299628645u + x2, 23u); a = tmp; }
    // Round 48
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4096336452u + x0, 6u); a = tmp; }
    // Round 49
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1126891415u + x7, 10u); a = tmp; }
    // Round 50
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2878612391u + x14, 15u); a = tmp; }
    // Round 51
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4237533241u + x5, 21u); a = tmp; }
    // Round 52
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1700485571u + x12, 6u); a = tmp; }
    // Round 53
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2399980690u + x3, 10u); a = tmp; }
    // Round 54
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4293915773u + x10, 15u); a = tmp; }
    // Round 55
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2240044497u + x1, 21u); a = tmp; }
    // Round 56
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1873313359u + x8, 6u); a = tmp; }
    // Round 57
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4264355552u + x15, 10u); a = tmp; }
    // Round 58
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 2734768916u + x6, 15u); a = tmp; }
    // Round 59
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 1309151649u + x13, 21u); a = tmp; }
    // Round 60
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 4149444226u + x4, 6u); a = tmp; }
    // Round 61
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3174756917u + x11, 10u); a = tmp; }
    // Round 62
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 718787259u + x2, 15u); a = tmp; }
    // Round 63
    { uint f_val = (c ^ (b | (~d))); uint tmp = d; d = c; c = b; b = b + rotate(a + f_val + 3951481745u + x9, 21u); a = tmp; }

    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
}

// Pack bytes into 16 little-endian uints (zero-padded).
// Byte i → word i/4, byte position (i%4) → bits (i%4)*8 .. (i%4)*8+7.
// This is the LE counterpart of bytes_to_block (which is big-endian).
static inline void bytes_to_block_le(const uchar *data, uint len, uint *block) {
    if (len > 64) len = 64;
    for (int i = 0; i < 16; i++) block[i] = 0;
    for (uint i = 0; i < len; i++) {
        block[i >> 2] |= ((uint)data[i]) << ((i & 3) * 8);
    }
}

// Continue MD5 from a saved state with a single-block message (len <= 55).
// The saved state already has 64 bytes processed (ipad/opad).
// Defense-in-depth: clamp len to 55 to prevent padding corruption.
static inline void md5_continue_single(const uint *saved_h, const uchar *msg, uint len, uint *out_h) {
    if (len > 55) len = 55;
    md5_state_t s;
    for (int i = 0; i < 4; i++) s.h[i] = saved_h[i];
    uint block[16];
    bytes_to_block_le(msg, len, block);
    uint pad_word = len >> 2;
    uint pad_byte = len & 3;
    // Little-endian: 0x80 goes at byte position len → bit (len%4)*8 of word len/4
    block[pad_word] |= ((uint)0x80) << (pad_byte * 8);
    ulong bits = ((ulong)64 + len) * 8;  // 64 = precomputed ipad block
    // Little-endian length: low 32 bits in word 14, high 32 bits in word 15
    block[14] = (uint)bits;
    block[15] = (uint)(bits >> 32);
    md5_process_block(&s, block);
    for (int i = 0; i < 4; i++) out_h[i] = s.h[i];
}

// Continue MD5 from a saved state with a message of arbitrary length.
// Processes full 64-byte blocks, then handles the final block with padding.
// Used for EAPOL frames up to 256 bytes (total HMAC inner = 64 + 256 = 320 bytes).
static inline void md5_continue_multi(const uint *saved_h, const uchar *msg, uint len, uint *out_h) {
    md5_state_t s;
    for (int i = 0; i < 4; i++) s.h[i] = saved_h[i];
    ulong bits = ((ulong)64 + len) * 8;  // 64 = precomputed ipad block

    uint offset = 0;
    while (offset + 64 <= len) {
        uint block[16];
        bytes_to_block_le(msg + offset, 64, block);
        md5_process_block(&s, block);
        offset += 64;
    }

    uint rem = len - offset;
    if (rem > 55) {
        // Remaining doesn't fit padding+length in one block — need two blocks
        uint block1[16];
        bytes_to_block_le(msg + offset, rem, block1);
        uint pw = rem >> 2;
        uint pb = rem & 3;
        block1[pw] |= ((uint)0x80) << (pb * 8);
        md5_process_block(&s, block1);
        // Block 2: zeros + 8-byte length (little-endian)
        uint block2[16];
        for (int i = 0; i < 14; i++) block2[i] = 0;
        block2[14] = (uint)bits;
        block2[15] = (uint)(bits >> 32);
        md5_process_block(&s, block2);
    } else {
        uint block[16];
        bytes_to_block_le(msg + offset, rem, block);
        uint pw = rem >> 2;
        uint pb = rem & 3;
        block[pw] |= ((uint)0x80) << (pb * 8);
        block[14] = (uint)bits;
        block[15] = (uint)(bits >> 32);
        md5_process_block(&s, block);
    }

    for (int i = 0; i < 4; i++) out_h[i] = s.h[i];
}

// Precompute HMAC-MD5 ipad/opad states for a key.
// Key is packed as little-endian uints (unlike HMAC-SHA1's big-endian).
static inline void hmac_md5_precompute(const uchar *key, uint key_len,
                                 uint *ipad_state, uint *opad_state) {
    uint k[16];
    for (int i = 0; i < 16; i++) k[i] = 0;

    if (key_len <= 64) {
        for (uint i = 0; i < key_len; i++) {
            k[i >> 2] |= ((uint)key[i]) << ((i & 3) * 8);
        }
    } else {
        // Key > 64 bytes: hash it first with MD5, then use the 16-byte digest.
        // KCK is always 16 bytes so this branch is never reached for EAPOL MIC,
        // but included for defense-in-depth against future reuse.
        md5_state_t s;
        md5_init(&s);
        uint offset = 0;
        while (offset + 64 <= key_len) {
            uint block[16];
            bytes_to_block_le(key + offset, 64, block);
            md5_process_block(&s, block);
            offset += 64;
        }
        uint rem = key_len - offset;
        ulong bits = (ulong)key_len * 8;
        if (rem > 55) {
            uint block1[16];
            bytes_to_block_le(key + offset, rem, block1);
            uint pw = rem >> 2; uint pb = rem & 3;
            block1[pw] |= ((uint)0x80) << (pb * 8);
            md5_process_block(&s, block1);
            uint block2[16];
            for (int i = 0; i < 14; i++) block2[i] = 0;
            block2[14] = (uint)bits;
            block2[15] = (uint)(bits >> 32);
            md5_process_block(&s, block2);
        } else {
            uint block[16];
            bytes_to_block_le(key + offset, rem, block);
            uint pw = rem >> 2; uint pb = rem & 3;
            block[pw] |= ((uint)0x80) << (pb * 8);
            block[14] = (uint)bits;
            block[15] = (uint)(bits >> 32);
            md5_process_block(&s, block);
        }
        for (int i = 0; i < 4; i++) k[i] = s.h[i];
    }

    // XOR with ipad (0x36) / opad (0x5c). The repeated-byte constant
    // 0x36363636 / 0x5c5c5c5c works regardless of endianness.
    uint ipad_block[16], opad_block[16];
    for (int i = 0; i < 16; i++) {
        ipad_block[i] = k[i] ^ 0x36363636;
        opad_block[i] = k[i] ^ 0x5c5c5c5c;
    }

    md5_state_t s;
    md5_init(&s);
    md5_process_block(&s, ipad_block);
    for (int i = 0; i < 4; i++) ipad_state[i] = s.h[i];

    md5_init(&s);
    md5_process_block(&s, opad_block);
    for (int i = 0; i < 4; i++) opad_state[i] = s.h[i];
}

// HMAC-MD5 using precomputed ipad/opad states (handles msg of any length).
// Output: 4 uints (16 bytes) in LITTLE-ENDIAN order.
static inline void hmac_md5_precomputed(const uint *ipad_state, const uint *opad_state,
                                   const uchar *msg, uint msg_len, uint *mac) {
    uint inner_h[4];
    if (msg_len <= 55) {
        md5_continue_single(ipad_state, msg, msg_len, inner_h);
    } else {
        md5_continue_multi(ipad_state, msg, msg_len, inner_h);
    }

    // Outer hash: feed inner_h (4 uints = 16 bytes) directly as block words.
    // inner_h is little-endian MD5 output, same as block word ordering — no swap.
    md5_state_t outer;
    for (int i = 0; i < 4; i++) outer.h[i] = opad_state[i];
    uint block[16];
    for (int i = 0; i < 4; i++) block[i] = inner_h[i];
    // Padding: 16 bytes of data → 0x80 at byte 16 (word 4, byte 0 = LSB)
    // Little-endian: 0x80 at LSB of word 4 = 0x00000080
    block[4] = 0x80;
    for (int i = 5; i < 14; i++) block[i] = 0;
    // Length: 64 (ipad) + 16 (inner) = 80 bytes = 640 bits (little-endian)
    block[14] = 640;
    block[15] = 0;
    md5_process_block(&outer, block);
    for (int i = 0; i < 4; i++) mac[i] = outer.h[i];
}

// === SHA-256 (for keyver 3 KDF-256 PTK derivation) ===
// References: FIPS 180-4 (SHA-256), RFC 2104 (HMAC).
// hashcat uses HMAC-SHA-256 for keyver 3 (AES-128-CMAC) PTK derivation,
// matching modern APs that use KDF-256 instead of the older PRF-512 (SHA-1).

__constant uint SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// OPT-2: Use OpenCL rotate() built-in. rotate(x, 32-n) = right-rotate by n.
#define SHA256_ROTR(x, n) rotate((x), (uint)(32 - (n)))
// OPT-1: Ch and Maj using bitselect/algebraic simplification.
// Ch(e,f,g) = (e & f) ^ (~e & g) = bitselect(g, f, e)
#define SHA256_CH(x, y, z)  bitselect((z), (y), (x))
// Maj(a,b,c) = (a & b) ^ (a & c) ^ (b & c) = bitselect(a, b, a ^ c)
// (verified: selector a^c picks b when a≠c — majority then is b — and picks
// a when a==c. 2 VALU (XOR+BFI) vs 4 for the (x&y)^(z&(x^y)) form.)
#define SHA256_MA(x, y, z)  bitselect((x), (y), ((x) ^ (z)))
#define SHA256_EP0(x)  (SHA256_ROTR(x, 2) ^ SHA256_ROTR(x, 13) ^ SHA256_ROTR(x, 22))
#define SHA256_EP1(x)  (SHA256_ROTR(x, 6) ^ SHA256_ROTR(x, 11) ^ SHA256_ROTR(x, 25))
#define SHA256_SIG0(x) (SHA256_ROTR(x, 7) ^ SHA256_ROTR(x, 18) ^ ((x) >> 3))
#define SHA256_SIG1(x) (SHA256_ROTR(x, 17) ^ SHA256_ROTR(x, 19) ^ ((x) >> 10))

typedef struct {
    uint h[8];
} sha256_state_t;

static inline void sha256_init(sha256_state_t *s) {
    s->h[0] = 0x6a09e667; s->h[1] = 0xbb67ae85;
    s->h[2] = 0x3c6ef372; s->h[3] = 0xa54ff53a;
    s->h[4] = 0x510e527f; s->h[5] = 0x9b05688c;
    s->h[6] = 0x1f83d9ab; s->h[7] = 0x5be0cd19;
}

// Process one 64-byte block (16 big-endian uints in w_in)
// Uses a loop (not fully unrolled) to keep compiled code size manageable
// for rusticl's LLVM backend — full unrolling of 64 rounds causes compile timeouts.
static inline void sha256_process_block(sha256_state_t *s, const uint *w_in) {
    uint w[16];
    for (int i = 0; i < 16; i++) w[i] = w_in[i];

    uint a=s->h[0], b=s->h[1], c=s->h[2], d=s->h[3];
    uint e=s->h[4], f=s->h[5], g=s->h[6], h=s->h[7];

    for (int i = 0; i < 64; i++) {
        if (i >= 16) {
            w[i & 15] += SHA256_SIG0(w[(i + 1) & 15])
                       + w[(i + 9) & 15]
                       + SHA256_SIG1(w[(i + 14) & 15]);
        }
        uint t1 = h + SHA256_EP1(e) + SHA256_CH(e, f, g) + SHA256_K[i] + w[i & 15];
        uint t2 = SHA256_EP0(a) + SHA256_MA(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    s->h[0] += a; s->h[1] += b; s->h[2] += c; s->h[3] += d;
    s->h[4] += e; s->h[5] += f; s->h[6] += g; s->h[7] += h;
}

// Pack bytes into 16 big-endian uints (zero-padded)
static inline void bytes_to_block_sha256(const uchar *data, uint len, uint *block) {
    if (len > 64) len = 64;
    for (int i = 0; i < 16; i++) block[i] = 0;
    for (uint i = 0; i < len; i++)
        block[i >> 2] |= ((uint)data[i]) << (24 - (i & 3) * 8);
}

// HMAC-SHA-256 precompute: compute ipad/opad states for a key
static inline void hmac_sha256_precompute(const uchar *key, uint key_len,
                                    uint *ipad_state, uint *opad_state) {
    uint k[16];
    for (int i = 0; i < 16; i++) k[i] = 0;
    if (key_len <= 64) {
        for (uint i = 0; i < key_len; i++)
            k[i >> 2] |= ((uint)key[i]) << (24 - (i & 3) * 8);
    } else {
        // Key > 64 bytes: hash with SHA-256 first (not needed for PMK=32 bytes)
        sha256_state_t s;
        sha256_init(&s);
        uint offset = 0;
        while (offset + 64 <= key_len) {
            uint block[16];
            bytes_to_block_sha256(key + offset, 64, block);
            sha256_process_block(&s, block);
            offset += 64;
        }
        uint rem = key_len - offset;
        ulong bits = (ulong)key_len * 8;
        if (rem > 55) {
            uint block1[16];
            bytes_to_block_sha256(key + offset, rem, block1);
            uint pw = rem >> 2, pb = rem & 3;
            block1[pw] |= ((uint)0x80) << (24 - pb * 8);
            sha256_process_block(&s, block1);
            uint block2[16];
            for (int i = 0; i < 14; i++) block2[i] = 0;
            block2[14] = (uint)(bits >> 32);
            block2[15] = (uint)bits;
            sha256_process_block(&s, block2);
        } else {
            uint block[16];
            bytes_to_block_sha256(key + offset, rem, block);
            uint pw = rem >> 2, pb = rem & 3;
            block[pw] |= ((uint)0x80) << (24 - pb * 8);
            block[14] = (uint)(bits >> 32);
            block[15] = (uint)bits;
            sha256_process_block(&s, block);
        }
        for (int i = 0; i < 8; i++) k[i] = s.h[i];
    }

    uint ipad_block[16], opad_block[16];
    for (int i = 0; i < 16; i++) {
        ipad_block[i] = k[i] ^ 0x36363636;
        opad_block[i] = k[i] ^ 0x5c5c5c5c;
    }

    sha256_state_t s;
    sha256_init(&s);
    sha256_process_block(&s, ipad_block);
    for (int i = 0; i < 8; i++) ipad_state[i] = s.h[i];

    sha256_init(&s);
    sha256_process_block(&s, opad_block);
    for (int i = 0; i < 8; i++) opad_state[i] = s.h[i];
}

// HMAC-SHA-256 using precomputed ipad/opad states.
// Handles messages up to 256 bytes (sufficient for KDF-256 PKE = 102 bytes).
static inline void hmac_sha256_precomputed(const uint *ipad_state, const uint *opad_state,
                                     const uchar *msg, uint msg_len, uint *mac) {
    sha256_state_t inner;
    for (int i = 0; i < 8; i++) inner.h[i] = ipad_state[i];

    // Process full 64-byte blocks
    uint offset = 0;
    while (offset + 64 <= msg_len) {
        uint block[16];
        bytes_to_block_sha256(msg + offset, 64, block);
        sha256_process_block(&inner, block);
        offset += 64;
    }

    // Final block with padding
    uint rem = msg_len - offset;
    ulong bits = ((ulong)64 + msg_len) * 8;
    if (rem > 55) {
        // Two-block final: remaining bytes + 0x80 in block1, zeros + length in block2
        uint block1[16];
        bytes_to_block_sha256(msg + offset, rem, block1);
        uint pw = rem >> 2, pb = rem & 3;
        block1[pw] |= ((uint)0x80) << (24 - pb * 8);
        sha256_process_block(&inner, block1);
        uint block2[16];
        for (int i = 0; i < 14; i++) block2[i] = 0;
        block2[14] = (uint)(bits >> 32);
        block2[15] = (uint)bits;
        sha256_process_block(&inner, block2);
    } else {
        uint block[16];
        bytes_to_block_sha256(msg + offset, rem, block);
        uint pw = rem >> 2, pb = rem & 3;
        block[pw] |= ((uint)0x80) << (24 - pb * 8);
        block[14] = (uint)(bits >> 32);
        block[15] = (uint)bits;
        sha256_process_block(&inner, block);
    }

    uint inner_h[8];
    for (int i = 0; i < 8; i++) inner_h[i] = inner.h[i];

    // Outer hash: SHA-256(opad_state || inner_h)
    sha256_state_t outer;
    for (int i = 0; i < 8; i++) outer.h[i] = opad_state[i];
    uint oblock[16];
    for (int i = 0; i < 8; i++) oblock[i] = inner_h[i];
    oblock[8] = 0x80000000;
    for (int i = 9; i < 14; i++) oblock[i] = 0;
    oblock[14] = 0;  // (64 + 32) * 8 = 768 bits, high word = 0
    oblock[15] = 768;
    sha256_process_block(&outer, oblock);
    for (int i = 0; i < 8; i++) mac[i] = outer.h[i];
}

// KDF-256: derive KCK for keyver 3 using HMAC-SHA-256.
// hashcat's PKE format for keyver 3:
//   [0x01, 0x00] + "Pairwise key expansion" (22 bytes, no null) + B(76) + [0x80, 0x01]
//   Total = 102 bytes. The [0x80, 0x01] suffix is hashcat-specific data, not SHA-256 padding.
// KCK = HMAC-SHA-256(PMK, PKE_102_bytes)[0:16]
static inline void kdf_256(const uchar *pmk,
                     const uchar *aa, const uchar *spa,
                     const uchar *anonce, const uchar *snonce,
                     uchar *kck_out) {
    // Build B = min(AA,SPA) || max(AA,SPA) || min(ANonce,SNonce) || max(ANonce,SNonce)
    bool aa_less = false;
    for (int i = 0; i < 6; i++) {
        if (aa[i] != spa[i]) { aa_less = (aa[i] < spa[i]); break; }
    }
    const uchar *mac_min = aa_less ? aa : spa;
    const uchar *mac_max = aa_less ? spa : aa;

    bool anonce_less = false;
    for (int i = 0; i < 32; i++) {
        if (anonce[i] != snonce[i]) { anonce_less = (anonce[i] < snonce[i]); break; }
    }
    const uchar *nonce_min = anonce_less ? anonce : snonce;
    const uchar *nonce_max = anonce_less ? snonce : anonce;

    // Build PKE: [0x01, 0x00] + "Pairwise key expansion" + B + [0x80, 0x01]
    uchar pke[102];
    pke[0] = 0x01; pke[1] = 0x00;
    // "Pairwise key expansion" (22 bytes, no null)
    pke[2]='P'; pke[3]='a'; pke[4]='i'; pke[5]='r';
    pke[6]='w'; pke[7]='i'; pke[8]='s'; pke[9]='e';
    pke[10]=' '; pke[11]='k'; pke[12]='e'; pke[13]='y';
    pke[14]=' '; pke[15]='e'; pke[16]='x'; pke[17]='p';
    pke[18]='a'; pke[19]='n'; pke[20]='s'; pke[21]='i';
    pke[22]='o'; pke[23]='n';
    // B: 6+6+32+32 = 76 bytes at offset 24
    for (int i = 0; i < 6; i++) { pke[24+i] = mac_min[i]; pke[30+i] = mac_max[i]; }
    for (int i = 0; i < 32; i++) { pke[36+i] = nonce_min[i]; pke[68+i] = nonce_max[i]; }
    // Suffix
    pke[100] = 0x80; pke[101] = 0x01;

    // KCK = HMAC-SHA-256(PMK, PKE)[0:16]
    uint ipad_state[8], opad_state[8];
    hmac_sha256_precompute(pmk, 32, ipad_state, opad_state);
    uint mac[8];
    hmac_sha256_precomputed(ipad_state, opad_state, pke, 102, mac);

    // Convert 4 big-endian uints to 16 bytes
    for (int i = 0; i < 4; i++) {
        kck_out[i*4]     = (uchar)((mac[i] >> 24) & 0xFF);
        kck_out[i*4 + 1] = (uchar)((mac[i] >> 16) & 0xFF);
        kck_out[i*4 + 2] = (uchar)((mac[i] >>  8) & 0xFF);
        kck_out[i*4 + 3] = (uchar)(mac[i] & 0xFF);
    }
}

// Forward declaration — pbkdf2_sha1_precomputed is defined below.
static inline void pbkdf2_sha1_precomputed(const uint *ipad_state, const uint *opad_state,
                                     __constant const uchar *salt, uint salt_len,
                                     uint iterations, uchar *out);

// PBKDF2-HMAC-SHA1 with 4096 iterations, 32-byte output (PMK)
// Uses precomputed HMAC states and zero byte conversions in the inner loop.
// Processes both output blocks simultaneously for better ILP.
// OPT-3: salt parameter is __constant to avoid private-memory copy of ESSID.
// PBKDF2-HMAC-SHA1 using precomputed HMAC ipad/opad states.
// OPT-3+OPT-10: Caller precomputes HMAC states with a scoped password buffer,
// freeing those VGPRs before the 4096-iteration inner loop begins.
// salt is __constant (ESSID broadcast to all work-items — no private copy needed).
static inline void pbkdf2_sha1_precomputed(const uint *ipad_state, const uint *opad_state,
                                     __constant const uchar *salt, uint salt_len,
                                     uint iterations, uchar *out) {

    // Defense-in-depth: clamp salt_len to buffer capacity (msg[68] holds salt+4)
    if (salt_len > 64) salt_len = 64;

    // Precompute constant block words (never change across iterations).
    // Words 5..15: 0x80 padding at word 5, zeros for 6-13, length (672 bits) at 14-15.
    // 84 bytes total (64 ipad + 20 U_prev) × 8 = 672 bits.
    uint block_tail[11] = {
        0x80000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 672
    };

    // Compute U1 for both blocks
    uint U1[5], U2[5];
    {
        // Block 1: U1 = HMAC(password, salt || INT(1))
        uchar msg1[68];
        for (uint i = 0; i < salt_len; i++) msg1[i] = salt[i];
        msg1[salt_len]     = 0;
        msg1[salt_len + 1] = 0;
        msg1[salt_len + 2] = 0;
        msg1[salt_len + 3] = 1;
        hmac_sha1_precomputed(ipad_state, opad_state, msg1, salt_len + 4, U1);
    }
    {
        // Block 2: U1 = HMAC(password, salt || INT(2))
        uchar msg2[68];
        for (uint i = 0; i < salt_len; i++) msg2[i] = salt[i];
        msg2[salt_len]     = 0;
        msg2[salt_len + 1] = 0;
        msg2[salt_len + 2] = 0;
        msg2[salt_len + 3] = 2;
        hmac_sha1_precomputed(ipad_state, opad_state, msg2, salt_len + 4, U2);
    }

    uint T1[5], T2[5];
    for (int i = 0; i < 5; i++) { T1[i] = U1[i]; T2[i] = U2[i]; }

    // OPT-VGPR: Use a SINGLE sha1_state_t and w[16] for all 4 compressions
    // per iteration. Original merged approach used 4 state structs + 2 w arrays
    // (~52 VGPRs). This reuses 1 state + 1 w (~21 VGPRs), saving ~31 VGPRs for
    // better occupancy on Vega 3's 256-VGPR SIMD.
    uint w[16];
    sha1_state_t s;
    w[5] = block_tail[0]; w[6] = block_tail[1]; w[7] = block_tail[2];
    w[8] = block_tail[3]; w[9] = block_tail[4]; w[10] = block_tail[5];
    w[11] = block_tail[6]; w[12] = block_tail[7]; w[13] = block_tail[8];
    w[14] = block_tail[9]; w[15] = block_tail[10];

    for (uint iter = 1; iter < iterations; iter++) {
        // Block 1 inner hash
        s.h[0] = ipad_state[0]; s.h[1] = ipad_state[1]; s.h[2] = ipad_state[2];
        s.h[3] = ipad_state[3]; s.h[4] = ipad_state[4];
        w[0] = U1[0]; w[1] = U1[1]; w[2] = U1[2]; w[3] = U1[3]; w[4] = U1[4];
        sha1_process_block(&s, w);
        uint i1h0 = s.h[0], i1h1 = s.h[1], i1h2 = s.h[2], i1h3 = s.h[3], i1h4 = s.h[4];

        // Block 2 inner hash
        s.h[0] = ipad_state[0]; s.h[1] = ipad_state[1]; s.h[2] = ipad_state[2];
        s.h[3] = ipad_state[3]; s.h[4] = ipad_state[4];
        w[0] = U2[0]; w[1] = U2[1]; w[2] = U2[2]; w[3] = U2[3]; w[4] = U2[4];
        sha1_process_block(&s, w);
        uint i2h0 = s.h[0], i2h1 = s.h[1], i2h2 = s.h[2], i2h3 = s.h[3], i2h4 = s.h[4];

        // Block 1 outer hash
        s.h[0] = opad_state[0]; s.h[1] = opad_state[1]; s.h[2] = opad_state[2];
        s.h[3] = opad_state[3]; s.h[4] = opad_state[4];
        w[0] = i1h0; w[1] = i1h1; w[2] = i1h2; w[3] = i1h3; w[4] = i1h4;
        sha1_process_block(&s, w);

        U1[0] = s.h[0]; T1[0] ^= U1[0];
        U1[1] = s.h[1]; T1[1] ^= U1[1];
        U1[2] = s.h[2]; T1[2] ^= U1[2];
        U1[3] = s.h[3]; T1[3] ^= U1[3];
        U1[4] = s.h[4]; T1[4] ^= U1[4];

        // Block 2 outer hash
        s.h[0] = opad_state[0]; s.h[1] = opad_state[1]; s.h[2] = opad_state[2];
        s.h[3] = opad_state[3]; s.h[4] = opad_state[4];
        w[0] = i2h0; w[1] = i2h1; w[2] = i2h2; w[3] = i2h3; w[4] = i2h4;
        sha1_process_block(&s, w);

        U2[0] = s.h[0]; T2[0] ^= U2[0];
        U2[1] = s.h[1]; T2[1] ^= U2[1];
        U2[2] = s.h[2]; T2[2] ^= U2[2];
        U2[3] = s.h[3]; T2[3] ^= U2[3];
        U2[4] = s.h[4]; T2[4] ^= U2[4];
    }

    // Output: block 1 = T1[0..4] (20 bytes), block 2 = T2[0..1] (12 bytes)
    // Total 32 bytes (PMK)
    for (int i = 0; i < 5; i++) {
        int off = i * 4;
        out[off]     = (T1[i] >> 24) & 0xFF;
        out[off + 1] = (T1[i] >> 16) & 0xFF;
        out[off + 2] = (T1[i] >> 8) & 0xFF;
        out[off + 3] = T1[i] & 0xFF;
    }
    // Block 2: first 12 bytes (3 uints) go to out[20..31]
    for (int i = 0; i < 3; i++) {
        int off = 20 + i * 4;
        out[off]     = (T2[i] >> 24) & 0xFF;
        out[off + 1] = (T2[i] >> 16) & 0xFF;
        out[off + 2] = (T2[i] >> 8) & 0xFF;
        out[off + 3] = T2[i] & 0xFF;
    }
}

// PMKID = HMAC-SHA1(PMK, "PMK Name" || AP_MAC || STA_MAC)
// Writes only the first 4 uints (16 bytes) of the HMAC-SHA1 output to pmkid[],
// since PMKID is the first 16 bytes of the 20-byte HMAC. This saves a register
// compared to writing all 5 uints and only comparing 4.
static inline void compute_pmkid(const uchar *pmk,
                          const uchar *ap_mac,
                          const uchar *sta_mac,
                          uint *pmkid) {
    uchar msg[20];
    msg[0] = 'P'; msg[1] = 'M'; msg[2] = 'K'; msg[3] = ' ';
    msg[4] = 'N'; msg[5] = 'a'; msg[6] = 'm'; msg[7] = 'e';
    for (int i = 0; i < 6; i++) msg[8 + i] = ap_mac[i];
    for (int i = 0; i < 6; i++) msg[14 + i] = sta_mac[i];

    uint ipad_state[5], opad_state[5];
    hmac_precompute(pmk, 32, ipad_state, opad_state);
    // Full HMAC-SHA1 output is 20 bytes (5 uints), but PMKID = first 16 bytes.
    // Use a local 5-uint buffer, then copy only the 4 we need to pmkid[].
    uint mac_full[5];
    hmac_sha1_precomputed(ipad_state, opad_state, msg, 20, mac_full);
    for (int i = 0; i < 4; i++) pmkid[i] = mac_full[i];
}

// PRF-512: derive 64-byte PTK from PMK using HMAC-SHA1
// PTK = HMAC-SHA1(PMK, "Pairwise key expansion" || 0x00 || B || INT(i))[0:16] for i=0..3
// where B = min(AA,SPA) || max(AA,SPA) || min(ANonce,SNonce) || max(ANonce,SNonce)
static inline void prf_512(const uchar *pmk,
                     const uchar *aa, const uchar *spa,
                     const uchar *anonce, const uchar *snonce,
                     uchar *ptk_out) {
    // Build B = min(AA,SPA) || max(AA,SPA) || min(ANonce,SNonce) || max(ANonce,SNonce)
    uchar B[76];
    // Compare MACs (6 bytes each, big-endian)
    bool aa_less = false;
    for (int i = 0; i < 6; i++) {
        if (aa[i] != spa[i]) { aa_less = (aa[i] < spa[i]); break; }
    }
    const uchar *mac_min = aa_less ? aa : spa;
    const uchar *mac_max = aa_less ? spa : aa;
    for (int i = 0; i < 6; i++) { B[i] = mac_min[i]; B[6+i] = mac_max[i]; }

    // Compare nonces (32 bytes each)
    bool anonce_less = false;
    for (int i = 0; i < 32; i++) {
        if (anonce[i] != snonce[i]) { anonce_less = (anonce[i] < snonce[i]); break; }
    }
    const uchar *nonce_min = anonce_less ? anonce : snonce;
    const uchar *nonce_max = anonce_less ? snonce : anonce;
    for (int i = 0; i < 32; i++) { B[12+i] = nonce_min[i]; B[44+i] = nonce_max[i]; }

    // Label = "Pairwise key expansion" (22 bytes) + 0x00 null separator = 23 bytes
    // IEEE 802.11-2020 Section 12.7.1.3: R = Label || 0x00 || B
    // The 0x00 null byte after the label is mandatory — verified against
    // a real hashcat-cracked capture (hashcat! + TP-LINK_HASHCAT_TEST).
    // Total message: 23 (label+null) + 76 (B) + 1 (counter) = 100 bytes

    // Precompute HMAC states for PMK as key
    uint ipad_state[5], opad_state[5];
    hmac_precompute(pmk, 32, ipad_state, opad_state);

    // Build message: label(22) + null(1) + B(76) + counter(1) = 100 bytes
    uchar prf_msg[100];
    // "Pairwise key expansion" with 0x00 null separator
    prf_msg[0]='P'; prf_msg[1]='a'; prf_msg[2]='i'; prf_msg[3]='r';
    prf_msg[4]='w'; prf_msg[5]='i'; prf_msg[6]='s'; prf_msg[7]='e';
    prf_msg[8]=' '; prf_msg[9]='k'; prf_msg[10]='e'; prf_msg[11]='y';
    prf_msg[12]=' '; prf_msg[13]='e'; prf_msg[14]='x'; prf_msg[15]='p';
    prf_msg[16]='a'; prf_msg[17]='n'; prf_msg[18]='s'; prf_msg[19]='i';
    prf_msg[20]='o'; prf_msg[21]='n'; prf_msg[22]=0;
    for (int i = 0; i < 76; i++) prf_msg[23 + i] = B[i];
    // Counter byte at offset 99

    // 4 iterations, take 16 bytes each = 64 bytes total (PTK)
    for (uint iter = 0; iter < 4; iter++) {
        prf_msg[99] = (uchar)iter;
        uint mac[5];
        hmac_sha1_precomputed(ipad_state, opad_state, prf_msg, 100, mac);
        // Take first 16 bytes (4 uints) of 20-byte HMAC output
        for (int i = 0; i < 4; i++) {
            int off = iter * 16 + i * 4;
            ptk_out[off]   = (mac[i] >> 24) & 0xFF;
            ptk_out[off+1] = (mac[i] >> 16) & 0xFF;
            ptk_out[off+2] = (mac[i] >> 8) & 0xFF;
            ptk_out[off+3] = mac[i] & 0xFF;
        }
    }
}

// Compute EAPOL MIC: PMK → PTK → KCK → MIC
// keyver 1: MIC = HMAC-MD5(KCK, EAPOL_with_MIC_zeroed)[0:16]
// keyver 2: MIC = HMAC-SHA1(KCK, EAPOL_with_MIC_zeroed)[0:16]
// keyver 3: MIC = AES-128-CMAC(KCK, EAPOL_with_MIC_zeroed)[0:16]
// PTK derivation:
//   keyver 1/2: PRF-512 (HMAC-SHA1) per IEEE 802.11-2020 §12.7.1.3
//   keyver 3: KDF-256 (HMAC-SHA-256) — matches hashcat, for modern APs
// Writes 4 uints (16 bytes) to mic_out.
// For keyver 2 (SHA1) and 3 (CMAC), mic_out is big-endian (matches target format).
// For keyver 1 (MD5), mic_out is byte-swapped to big-endian so it matches
// the target format the host stores (big-endian uints from hex_decode).
// This function does NOT apply nonce error corrections — the caller is
// responsible for modifying anonce_local before calling this function.
static inline void compute_eapol_mic(const uchar *pmk,
                               const uchar *aa, const uchar *spa,
                               const uchar *anonce,
                               const uchar *eapol, uint eapol_len,
                               uint keyver,
                               uint *mic_out) {
    // Extract SNonce from EAPOL frame (offset 17, 32 bytes)
    uchar snonce[32];
    for (int i = 0; i < 32; i++) snonce[i] = eapol[17 + i];

    // KCK derivation depends on keyver
    uchar kck[16];
    if (keyver == 3) {
        // keyver 3: KDF-256 (HMAC-SHA-256) — matches hashcat
        kdf_256(pmk, aa, spa, anonce, snonce, kck);
    } else {
        // keyver 1/2: PRF-512 (HMAC-SHA-1) per IEEE 802.11-2020 §12.7.1.3
        uchar ptk[64];
        prf_512(pmk, aa, spa, anonce, snonce, ptk);
        for (int i = 0; i < 16; i++) kck[i] = ptk[i];
    }

    // Zero the MIC field in a local copy of EAPOL (offset 81, 16 bytes)
    // EAPOL frames can be up to 256 bytes — copy to local buffer
    uchar eapol_local[256];
    if (eapol_len > 256) eapol_len = 256;
    for (uint i = 0; i < eapol_len; i++) eapol_local[i] = eapol[i];
    // Zero MIC field (bytes 81-96)
    for (int i = 81; i < 97 && i < (int)eapol_len; i++) eapol_local[i] = 0;

    if (keyver == 1) {
        // WPA1/TKIP: MIC = HMAC-MD5(KCK, EAPOL_zeroed)[0:16]
        uint md5_ipad[4], md5_opad[4];
        hmac_md5_precompute(kck, 16, md5_ipad, md5_opad);
        uint md5_mac[4];
        hmac_md5_precomputed(md5_ipad, md5_opad, eapol_local, eapol_len, md5_mac);
        // MD5 outputs little-endian uints; targets are stored big-endian.
        // Swap each word so the comparison against targets works correctly.
        for (int i = 0; i < 4; i++) mic_out[i] = MD5_SWAP32(md5_mac[i]);
    } else if (keyver == 3) {
        // WPA2 keyver 3: MIC = AES-128-CMAC(KCK, EAPOL_zeroed)[0:16]
        // KCK is 16 bytes; pack as 4 big-endian u32 words for AES.
        uint kck_u32[4];
        for (int i = 0; i < 4; i++) {
            kck_u32[i] = ((uint)kck[i*4]     << 24) |
                         ((uint)kck[i*4 + 1] << 16) |
                         ((uint)kck[i*4 + 2] <<  8) |
                         ((uint)kck[i*4 + 3]);
        }
        compute_cmac_mic(kck_u32, eapol_local, eapol_len, mic_out);
    } else {
        // WPA2/CCMP (keyver 2): MIC = HMAC-SHA1(KCK, EAPOL_zeroed)[0:16]
        uint ipad_state[5], opad_state[5];
        hmac_precompute(kck, 16, ipad_state, opad_state);
        uint mac[5];
        hmac_sha1_precomputed(ipad_state, opad_state, eapol_local, eapol_len, mac);
        for (int i = 0; i < 4; i++) mic_out[i] = mac[i];
    }
}

// === SPLIT KERNEL: pbkdf2_init ===
// Phase 1 of split PBKDF2: precompute HMAC states and initial U1/U2/T1/T2.
// Writes state to global memory for the loop kernel to consume.
// State layout per work-item (30 uints = 120 bytes):
//   [0..4]   = ipad_state[5]
//   [5..9]   = opad_state[5]
//   [10..14] = U1[5]
//   [15..19] = U2[5]
//   [20..24] = T1[5]
//   [25..29] = T2[5]
// The loop kernel's padding/length tail is __constant W_TAIL[11] — no
// block_tail in state. State is exactly STATE_UINTS = 30 uints (120 bytes).
#define STATE_UINTS 30

__kernel void pbkdf2_init(
    __global const uchar *passwords,
    __global const uint *pw_lens,
    __constant const uchar *essid,
    __constant const uint *essid_len_ptr,
    __global uint *state,             // batch_size * STATE_UINTS
    const uint batch_size
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;

    __global const uchar *pw_global = passwords + gid * 64;
    uint pw_len = pw_lens[gid];
    if (pw_len > 63) pw_len = 63;

    __global uint *st = state + gid * STATE_UINTS;

    // Precompute HMAC ipad/opad states from password
    uint ipad_state[5], opad_state[5];
    {
        uchar pw[64];
        for (int i = 0; i < 64; i++) pw[i] = pw_global[i];
        hmac_precompute(pw, pw_len, ipad_state, opad_state);
    }

    uint essid_len = *essid_len_ptr;
    if (essid_len > 64) essid_len = 64;

    // Compute U1 = HMAC(password, salt || INT(1))
    uint U1[5];
    {
        uchar msg1[68];
        for (uint i = 0; i < essid_len; i++) msg1[i] = essid[i];
        msg1[essid_len]     = 0;
        msg1[essid_len + 1] = 0;
        msg1[essid_len + 2] = 0;
        msg1[essid_len + 3] = 1;
        hmac_sha1_precomputed(ipad_state, opad_state, msg1, essid_len + 4, U1);
    }

    // Compute U2 = HMAC(password, salt || INT(2))
    uint U2[5];
    {
        uchar msg2[68];
        for (uint i = 0; i < essid_len; i++) msg2[i] = essid[i];
        msg2[essid_len]     = 0;
        msg2[essid_len + 1] = 0;
        msg2[essid_len + 2] = 0;
        msg2[essid_len + 3] = 2;
        hmac_sha1_precomputed(ipad_state, opad_state, msg2, essid_len + 4, U2);
    }

    // T1 = U1, T2 = U2
    uint T1[5], T2[5];
    for (int i = 0; i < 5; i++) { T1[i] = U1[i]; T2[i] = U2[i]; }

    // Write state to global memory
    for (int i = 0; i < 5; i++) {
        st[i]      = ipad_state[i];
        st[5 + i]  = opad_state[i];
        st[10 + i] = U1[i];
        st[15 + i] = U2[i];
        st[20 + i] = T1[i];
        st[25 + i] = T2[i];
    }
// block_tail is now __constant W_TAIL[11]
}

// === SPLIT KERNEL: pbkdf2_loop ===
// Phase 2: run N iterations of PBKDF2, reading/writing state from global memory.
// Each call processes `loop_count` iterations. Called 4096/loop_count times.
//
// OPTIMIZATION (learned from hashcat m22000-pure.cl):
// - Process blocks SEQUENTIALLY (block 1 loop, then block 2 loop).
//   This halves live VGPRs: only U[5]+T[5]+ipad[5]+opad[5]=20 live at once
//   (was U1+U2+T1+T2+ipad+opad=30 when both blocks interleaved).
// - No intermediate inner-hash temporaries (i1h0..i1h4). Feed inner output
//   directly into w[0..4] for the outer hash. Saves 10 VGPRs.
// - In-place e update in SHA-1 round (no temp variable). Saves 1 VGPR.
//
// State layout (same as pbkdf2_init):
//   [0..4]   ipad[5]   [5..9]   opad[5]
//   [10..14] U1[5]     [15..19] U2[5]
//   [20..24] T1[5]     [25..29] T2[5]
// (Tail constants live in __constant W_TAIL[11], not in state.)
__kernel void pbkdf2_loop(
    __global uint *state,
    __constant const uchar *essid,
    __constant const uint *essid_len_ptr,
    const uint batch_size,
    const uint loop_count
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;

    __global uint *st = state + gid * STATE_UINTS;

    // Load shared state (ipad/opad — same for both blocks; tail is __constant W_TAIL)
    uint ipad_state[5], opad_state[5];
    for (int i = 0; i < 5; i++) {
        ipad_state[i] = st[i];
        opad_state[i] = st[5 + i];
    }
    // Uses W_TAIL[11] global constant.

    // === BLOCK 1: process loop_count iterations ===
    {
        uint U[5], T[5];
        for (int i = 0; i < 5; i++) {
            U[i] = st[10 + i];
            T[i] = st[20 + i];
        }

        for (uint iter = 0; iter < loop_count; iter++) {
            // Fused HMAC-SHA1: inner + outer in one function (shared w_cur/w_next)
            uint out[5];
            hmac_sha1_run(ipad_state, opad_state, U, out);
            U[0] = out[0]; T[0] ^= U[0];
            U[1] = out[1]; T[1] ^= U[1];
            U[2] = out[2]; T[2] ^= U[2];
            U[3] = out[3]; T[3] ^= U[3];
            U[4] = out[4]; T[4] ^= U[4];
        }

        // Write block 1 state back
        for (int i = 0; i < 5; i++) {
            st[10 + i] = U[i];
            st[20 + i] = T[i];
        }
    }

    // === BLOCK 2: process loop_count iterations ===
    {
        uint U[5], T[5];
        for (int i = 0; i < 5; i++) {
            U[i] = st[15 + i];
            T[i] = st[25 + i];
        }

        for (uint iter = 0; iter < loop_count; iter++) {
            // Fused HMAC-SHA1: inner + outer in one function (shared w_cur/w_next)
            uint out[5];
            hmac_sha1_run(ipad_state, opad_state, U, out);
            U[0] = out[0]; T[0] ^= U[0];
            U[1] = out[1]; T[1] ^= U[1];
            U[2] = out[2]; T[2] ^= U[2];
            U[3] = out[3]; T[3] ^= U[3];
            U[4] = out[4]; T[4] ^= U[4];
        }

        // Write block 2 state back
        for (int i = 0; i < 5; i++) {
            st[15 + i] = U[i];
            st[25 + i] = T[i];
        }
    }
}

// === SPLIT KERNEL: crack_pmkid_final ===
// Phase 3: extract PMK from final PBKDF2 state, compute PMKID/EAPOL, compare.
// This kernel has the same match-recording logic as the original crack_pmkid.
__kernel void crack_pmkid_final(
    __global const uint *state,       // batch_size * STATE_UINTS (read-only)
    __global const uchar *ap_macs,
    __global const uchar *sta_macs,
    __global const uint *targets,
    __global int *found,
    __global uint *found_idx,
    __global uint *found_hash_idx,
    __global uint *match_count,
    __global uint *match_gids,
    __global uint *match_hash_idxs,
    const uint batch_size,
    const uint num_hashes,
    __global const uint *hash_types,
    __global const uchar *anonces,
    __global const uchar *eapol_frames,
    __global const uint *eapol_lens,
    __global const uint *keyvers
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;

    __global const uint *st = state + gid * STATE_UINTS;

    // Extract PMK from T1/T2 (big-endian bytes)
    uchar pmk[32];
    for (int i = 0; i < 5; i++) {
        uint val = st[20 + i];  // T1
        int off = i * 4;
        pmk[off]     = (uchar)((val >> 24) & 0xFF);
        pmk[off + 1] = (uchar)((val >> 16) & 0xFF);
        pmk[off + 2] = (uchar)((val >>  8) & 0xFF);
        pmk[off + 3] = (uchar)(val & 0xFF);
    }
    for (int i = 0; i < 3; i++) {
        uint val = st[25 + i];  // T2
        int off = 20 + i * 4;
        pmk[off]     = (uchar)((val >> 24) & 0xFF);
        pmk[off + 1] = (uchar)((val >> 16) & 0xFF);
        pmk[off + 2] = (uchar)((val >>  8) & 0xFF);
        pmk[off + 3] = (uchar)(val & 0xFF);
    }

    // Compare PMK against all hashes in the group (same as original crack_pmkid)
    for (uint h = 0; h < num_hashes; h++) {
        uchar ap_mac_local[6];
        uchar sta_mac_local[6];
        for (int i = 0; i < 6; i++) {
            ap_mac_local[i] = ap_macs[h * 6 + i];
            sta_mac_local[i] = sta_macs[h * 6 + i];
        }

        uint ht = hash_types[h];
        if (ht >= 2) continue; // FIX F16: skip MD5/NTLM hash types

        if (ht == 1) {
            // EAPOL mode: PMK → PTK → KCK → MIC
            uint eapol_len = eapol_lens[h];
            if (eapol_len < 97 || eapol_len > 256) continue;
            uint packed = keyvers[h];
            uint keyver = packed & 0xFF;
            if (keyver != 1 && keyver != 2 && keyver != 3) continue;
            uint nec = (packed >> 8) & 0xFF;
            uint det_le = (packed >> 16) & 1;
            uint det_be = (packed >> 17) & 1;
            if (nec > 128) nec = 128;
            uchar anonce_local[32];
            uchar eapol_local[256];
            for (int i = 0; i < 32; i++) anonce_local[i] = anonces[h * 32 + i];
            for (uint i = 0; i < eapol_len; i++) eapol_local[i] = eapol_frames[h * 256 + i];

            uint bo_loops = det_le + det_be;
            if (bo_loops == 0) bo_loops = 2;
            uint anonce_tail =
                ((uint)anonce_local[28] << 24) |
                ((uint)anonce_local[29] << 16) |
                ((uint)anonce_local[30] <<  8) |
                (uint)anonce_local[31];

            __global const uint *target_h = targets + h * 4;

            for (uint nec_iter = 0; nec_iter <= nec; nec_iter++) {
                for (uint bo_pos = 0; bo_pos < bo_loops; bo_pos++) {
                    uint t = anonce_tail;
                    if (bo_loops == 1) {
                        if (det_le == 1) { t -= nec / 2; t += nec_iter; }
                        else { t = MD5_SWAP32(t); t -= nec / 2; t += nec_iter; t = MD5_SWAP32(t); }
                    } else {
                        if (bo_pos == 0) { t -= nec / 2; t += nec_iter; }
                        else { t = MD5_SWAP32(t); t -= nec / 2; t += nec_iter; t = MD5_SWAP32(t); }
                    }
                    anonce_local[28] = (uchar)((t >> 24) & 0xFF);
                    anonce_local[29] = (uchar)((t >> 16) & 0xFF);
                    anonce_local[30] = (uchar)((t >>  8) & 0xFF);
                    anonce_local[31] = (uchar)(t & 0xFF);

                    uint result[4];
                    compute_eapol_mic(pmk, ap_mac_local, sta_mac_local,
                                      anonce_local, eapol_local, eapol_len,
                                      keyver, result);

                    if (result[0] == target_h[0] && result[1] == target_h[1] &&
                        result[2] == target_h[2] && result[3] == target_h[3]) {
                        uint slot = atomic_inc(match_count);
                        if (slot < 8) { match_gids[slot] = gid; match_hash_idxs[slot] = h; }
                        atomic_cmpxchg(found, 0, 1);
                        if (slot == 0) { atomic_xchg(found_idx, gid); atomic_xchg(found_hash_idx, h); }
                    }
                }
            }
        } else {
            // PMKID mode: PMK → PMKID
            uint result[4];
            compute_pmkid(pmk, ap_mac_local, sta_mac_local, result);
            __global const uint *target_h = targets + h * 4;
            if (result[0] == target_h[0] && result[1] == target_h[1] &&
                result[2] == target_h[2] && result[3] == target_h[3]) {
                uint slot = atomic_inc(match_count);
                if (slot < 8) { match_gids[slot] = gid; match_hash_idxs[slot] = h; }
                atomic_cmpxchg(found, 0, 1);
                if (slot == 0) { atomic_xchg(found_idx, gid); atomic_xchg(found_hash_idx, h); }
            }
        }
    }
}

// Main kernel: each work-item tests one password against ALL hashes in the group.
// Supports both PMKID (type 01) and EAPOL (type 02) hash modes.
// Multi-hash: targets array contains num_hashes entries, each with its own
// MAC pair, hash type, and EAPOL data. PBKDF2 runs once per password, then
// the result is compared against all hashes.
__kernel void crack_pmkid(
    __global const uchar *passwords,
    __global const uint *pw_lens,
    __constant const uchar *essid,
    __constant const uint *essid_len_ptr,
    __global const uchar *ap_macs,        // num_hashes * 6 bytes
    __global const uchar *sta_macs,       // num_hashes * 6 bytes
    __global const uint *targets,         // num_hashes * 4 uints
    __global int *found,
    __global uint *found_idx,             // batch-local index of matching pw
    __global uint *found_hash_idx,        // which hash matched (0..num_hashes-1)
    __global uint *match_count,           // atomic counter: number of GPU matches
    __global uint *match_gids,            // up to 8 matching work-item IDs
    __global uint *match_hash_idxs,       // up to 8 matching hash indices
    const uint batch_size,
    const uint num_hashes,                // number of hashes to compare against
    __global const uint *hash_types,      // num_hashes: 0=PMKID, 1=EAPOL
    __global const uchar *anonces,        // num_hashes * 32 bytes (EAPOL only)
    __global const uchar *eapol_frames,   // num_hashes * 256 bytes (EAPOL only)
    __global const uint *eapol_lens,      // num_hashes: EAPOL frame lengths
    __global const uint *keyvers          // num_hashes: packed keyver(0-7) | nec(8-15) | det_le(16) | det_be(17)
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;
    // Note: we do NOT early-exit on found=1 here. A previous version checked
    // atomic_cmpxchg(found,0,0) and returned if another work-item had already
    // found a match. But if that match was a GPU false positive (bit flip),
    // the early exit would suppress computation of the REAL match in the same
    // batch. Host-side verification would discard the false positive and
    // advance past the batch, losing the real match → silent false negative.
    // The cost of not early-exiting is at most one batch of wasted GPU time
    // (negligible vs 4096-iteration PBKDF2). The atomic_cmpxchg at the match
    // site below ensures only the first real match records its index.

    __global const uchar *pw_global = passwords + gid * 64;
    uint pw_len = pw_lens[gid];
    // Clamp pw_len to WPA max (63) — defense against corrupted GPU buffer
    // sending an oversized length into hmac_precompute
    if (pw_len > 63) pw_len = 63;

    // OPT-10: Scope pw[64] to a block so it goes out of scope after
    // hmac_precompute, allowing the compiler to free those 16 VGPRs.
    uint ipad_state[5], opad_state[5];
    {
        uchar pw[64];
        // Copy all 64 bytes: the host zeros the tail, but unused slots from a
        // prior batch may contain stale data. Copying the full slot is safe and
        // the global memory reads are cached by the GPU L1 cache.
        for (int i = 0; i < 64; i++) pw[i] = pw_global[i];
        hmac_precompute(pw, pw_len, ipad_state, opad_state);
    }
    // pw is now out of scope — compiler can reuse those registers

    uint essid_len = *essid_len_ptr;
    if (essid_len > 64) essid_len = 64;  // Guard against malformed host input
    // OPT-3: Pass essid directly from __constant memory — no private copy.
    // The ESSID is broadcast to all work-items via __constant, so copying it
    // to private memory wastes 16 VGPRs. pbkdf2_sha1 copies salt bytes into
    // private msg[] arrays internally, so __constant access is safe here.

    uchar pmk[32];
    // OPT-3+OPT-10: Use precomputed HMAC states instead of passing password.
    // This avoids re-reading pw from global memory inside pbkdf2_sha1.
    pbkdf2_sha1_precomputed(ipad_state, opad_state, essid, essid_len, 4096, pmk);

    // Compare PMK against all hashes in the group
    for (uint h = 0; h < num_hashes; h++) {
        uchar ap_mac_local[6];
        uchar sta_mac_local[6];
        for (int i = 0; i < 6; i++) {
            ap_mac_local[i] = ap_macs[h * 6 + i];
            sta_mac_local[i] = sta_macs[h * 6 + i];
        }

        uint ht = hash_types[h];
        if (ht >= 2) continue; // FIX F16: skip MD5/NTLM hash types

        if (ht == 1) {
            // EAPOL mode: PMK → PTK → KCK → MIC
            uint eapol_len = eapol_lens[h];
            if (eapol_len < 97 || eapol_len > 256) {
                continue; // Invalid EAPOL length — skip this hash
            }
            // Unpack NEC data from keyvers: keyver(0-7) | nec(8-15) | det_le(16) | det_be(17)
            uint packed = keyvers[h];
            uint keyver = packed & 0xFF;
            if (keyver != 1 && keyver != 2 && keyver != 3) {
                continue; // Unsupported key version — skip
            }
            uint nec = (packed >> 8) & 0xFF;
            uint det_le = (packed >> 16) & 1;
            uint det_be = (packed >> 17) & 1;
            if (nec > 128) nec = 128; // Sanity cap
            uchar anonce_local[32];
            uchar eapol_local[256];
            for (int i = 0; i < 32; i++) anonce_local[i] = anonces[h * 32 + i];
            for (uint i = 0; i < eapol_len; i++) eapol_local[i] = eapol_frames[h * 256 + i];

            // Nonce error corrections (hashcat parity).
            // Captured ANonce may have a corrupted replay counter in its
            // last 4 bytes. We try modifying those bytes by ±N/2 in both
            // LE and BE byte order. When NEC=0 (common case: replaycount
            // was checked), the loop runs exactly once with no modification.
            // NEC data was unpacked from keyvers above.

            // If neither detected, try both byte orders (hashcat default)
            uint bo_loops = det_le + det_be;
            if (bo_loops == 0) bo_loops = 2;

            // Extract the last 4 bytes of ANonce as a big-endian uint
            // (ANonce is stored as raw bytes, big-endian per 802.11)
            uint anonce_tail =
                ((uint)anonce_local[28] << 24) |
                ((uint)anonce_local[29] << 16) |
                ((uint)anonce_local[30] <<  8) |
                (uint)anonce_local[31];

            __global const uint *target_h = targets + h * 4;

            for (uint nec_iter = 0; nec_iter <= nec; nec_iter++) {
                for (uint bo_pos = 0; bo_pos < bo_loops; bo_pos++) {
                    uint t = anonce_tail;

                    if (bo_loops == 1) {
                        if (det_le == 1) {
                            t -= nec / 2;
                            t += nec_iter;
                        } else { // det_be == 1
                            t = MD5_SWAP32(t);
                            t -= nec / 2;
                            t += nec_iter;
                            t = MD5_SWAP32(t);
                        }
                    } else {
                        if (bo_pos == 0) {
                            t -= nec / 2;
                            t += nec_iter;
                        } else { // bo_pos == 1
                            t = MD5_SWAP32(t);
                            t -= nec / 2;
                            t += nec_iter;
                            t = MD5_SWAP32(t);
                        }
                    }

                    // Write modified tail back into anonce_local
                    anonce_local[28] = (uchar)((t >> 24) & 0xFF);
                    anonce_local[29] = (uchar)((t >> 16) & 0xFF);
                    anonce_local[30] = (uchar)((t >>  8) & 0xFF);
                    anonce_local[31] = (uchar)(t & 0xFF);

                    uint result[4];
                    compute_eapol_mic(pmk, ap_mac_local, sta_mac_local,
                                      anonce_local, eapol_local, eapol_len,
                                      keyver, result);

                    if (result[0] == target_h[0] &&
                        result[1] == target_h[1] &&
                        result[2] == target_h[2] &&
                        result[3] == target_h[3]) {
                        uint slot = atomic_inc(match_count);
                        if (slot < 8) {
                            match_gids[slot] = gid;
                            match_hash_idxs[slot] = h;
                        }
                        atomic_cmpxchg(found, 0, 1);
                        if (slot == 0) {
                            atomic_xchg(found_idx, gid);
                            atomic_xchg(found_hash_idx, h);
                        }
                    }
                }
            }
        } else {
            // PMKID mode: PMK → PMKID
            uint result[4];
            compute_pmkid(pmk, ap_mac_local, sta_mac_local, result);

            // Compare first 4 uints (16 bytes) with target for this hash
            __global const uint *target_h = targets + h * 4;
            if (result[0] == target_h[0] &&
                result[1] == target_h[1] &&
                result[2] == target_h[2] &&
                result[3] == target_h[3]) {
                // Record match in a multi-slot array (up to 8 candidates).
                // This avoids the winner-take-all problem where a GPU false positive
                // wins atomic_cmpxchg and masks the real match. The host verifies
                // all recorded candidates — typically 1-2, not the entire batch.
                uint slot = atomic_inc(match_count);
                if (slot < 8) {
                    match_gids[slot] = gid;
                    match_hash_idxs[slot] = h;
                }
                // Also set the legacy found flag for backward compatibility
                atomic_cmpxchg(found, 0, 1);
                if (slot == 0) {
                    atomic_xchg(found_idx, gid);
                    atomic_xchg(found_hash_idx, h);
                }
            }
        }
    }
}
// === MD5 standalone hash (mode 0) ===
// Computes MD5(password) from scratch — no precomputed state.
// Handles passwords up to 63 bytes (WPA limit, but MD5 supports arbitrary length).
// MD5 uses little-endian word ordering (like our existing md5_process_block).

static inline void md5_hash_password(const uchar *pw, uint pw_len, uint *out_h) {
    md5_state_t s;
    md5_init(&s);

    // Process full 64-byte blocks
    uint offset = 0;
    while (offset + 64 <= pw_len) {
        uint block[16];
        bytes_to_block_le(pw + offset, 64, block);
        md5_process_block(&s, block);
        offset += 64;
    }

    // Final block with padding
    uint rem = pw_len - offset;
    ulong bits = (ulong)pw_len * 8;
    if (rem > 55) {
        // Two-block final: remaining + 0x80 in block1, zeros + length in block2
        uint block1[16];
        bytes_to_block_le(pw + offset, rem, block1);
        uint pw_word = rem >> 2;
        uint pw_byte = rem & 3;
        block1[pw_word] |= ((uint)0x80) << (pw_byte * 8);
        md5_process_block(&s, block1);
        uint block2[16];
        for (int i = 0; i < 14; i++) block2[i] = 0;
        block2[14] = (uint)bits;
        block2[15] = (uint)(bits >> 32);
        md5_process_block(&s, block2);
    } else {
        uint block[16];
        bytes_to_block_le(pw + offset, rem, block);
        uint pw_word = rem >> 2;
        uint pw_byte = rem & 3;
        block[pw_word] |= ((uint)0x80) << (pw_byte * 8);
        block[14] = (uint)bits;
        block[15] = (uint)(bits >> 32);
        md5_process_block(&s, block);
    }

    for (int i = 0; i < 4; i++) out_h[i] = s.h[i];
}

// MD5 cracking kernel (mode 0): each work-item tests one password.
// targets: num_hashes * 4 uints (MD5 = 16 bytes = 4 u32s, little-endian).
// MD5 digest is stored as 4 u32s in native (little-endian) order.
__kernel void crack_md5(
    __global const uchar *passwords,    // batch_size * PW_SLOT_LEN bytes
    __global const uint *pw_lens,       // batch_size password lengths
    __global const uint *targets,       // num_hashes * 4 u32s (MD5 digests)
    __global int *found,                // 0 = not found, 1 = found
    __global uint *found_idx,           // batch-local index of matching pw
    __global uint *found_hash_idx,      // which hash matched (0..num_hashes-1)
    __global uint *match_count,         // atomic counter: number of GPU matches
    __global uint *match_gids,          // up to 8 matching work-item IDs
    __global uint *match_hash_idxs,     // up to 8 matching hash indices
    const uint batch_size,
    const uint num_hashes
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;

    // Load password from global memory
    uchar pw[64];
    for (int i = 0; i < 64; i++) {
        pw[i] = passwords[gid * 64 + i];
    }
    uint pw_len = pw_lens[gid];
    // Defense-in-depth: clamp pw_len to prevent OOB read past pw[64] array.
    // The host should never set pw_len > PW_SLOT_LEN (64), but GPU memory
    // corruption (bit flips) could trigger it. Matches crack_pmkid's clamp.
    if (pw_len > 64) pw_len = 64;

    // Compute MD5(password)
    uint digest[4];
    md5_hash_password(pw, pw_len, digest);

    // Compare against all target hashes
    for (uint h = 0; h < num_hashes; h++) {
        uint target_h[4];
        target_h[0] = targets[h * 4 + 0];
        target_h[1] = targets[h * 4 + 1];
        target_h[2] = targets[h * 4 + 2];
        target_h[3] = targets[h * 4 + 3];

        if (digest[0] == target_h[0] &&
            digest[1] == target_h[1] &&
            digest[2] == target_h[2] &&
            digest[3] == target_h[3]) {
            uint slot = atomic_inc(match_count);
            if (slot < 8) {
                match_gids[slot] = gid;
                match_hash_idxs[slot] = h;
            }
            atomic_cmpxchg(found, 0, 1);
            if (slot == 0) {
                atomic_xchg(found_idx, gid);
                atomic_xchg(found_hash_idx, h);
            }
        }
    }
}

// === NTLM (hashcat mode 1000): MD4(UTF-16LE(password)) ===
// NTLM = MD4 of the UTF-16LE encoded password. MD4 is very similar to MD5
// but uses a different round function and constants. The password is first
// converted to UTF-16LE (each byte → 2 bytes with a zero high byte for ASCII).

// MD4 sine-derived constants (same as MD5 T table but only 48 entries needed
// for the 3 rounds × 16 operations). Actually MD4 uses fixed constants:
// Round 1: F(b,c,d) = (b & c) | (~b & d)
// Round 2: G(b,c,d) = (b & c) | (b & d) | (c & d)
// Round 3: H(b,c,d) = b ^ c ^ d
// MD4 uses left-rotate by varying amounts and processes the message words
// in a fixed permutation (not the additive index formula of MD5).

// MD4 state: 4 uint words (little-endian, same as MD5)
typedef struct {
    uint h[4];
} md4_state_t;

static inline void md4_init(md4_state_t *s) {
    s->h[0] = 0x67452301;
    s->h[1] = 0xEFCDAB89;
    s->h[2] = 0x98BADCFE;
    s->h[3] = 0x10325476;
}

// OPT: bitselect forms (verified on-device: OpenCL bitselect(a,b,c) = c-bits
// pick the SECOND arg). MD4_F = CH = bitselect(d, c, b) = 1 VALU (V_BFI) vs 4
// for the boolean form; MD4_G = MAJ = bitselect(b, c, b^d) = XOR+BFI (2 VALU)
// vs 5. 32 of 48 MD4 rounds use F (16) or G (16) — NTLM mask mode measured
// before/after in main.
#define MD4_F(b, c, d) (bitselect((d), (c), (b)))
#define MD4_G(b, c, d) (bitselect((b), (c), (b) ^ (d)))
#define MD4_H(b, c, d) ((b) ^ (c) ^ (d))
#define MD4_ROTL(x, n) rotate((x), (uint)(n))

static inline void md4_process_block(md4_state_t *s, const uint *w) {
    // MD4 uses the 16 message words directly (no expansion like SHA-1)
    // and operates on copies of the state.
    uint a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3];

    // Round 1: F(b,c,d), 16 ops, rotate by 3/7/11/19, index permutation
    // Permutation: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    a = MD4_ROTL(a + MD4_F(b, c, d) + w[0], 3);
    d = MD4_ROTL(d + MD4_F(a, b, c) + w[1], 7);
    c = MD4_ROTL(c + MD4_F(d, a, b) + w[2], 11);
    b = MD4_ROTL(b + MD4_F(c, d, a) + w[3], 19);
    a = MD4_ROTL(a + MD4_F(b, c, d) + w[4], 3);
    d = MD4_ROTL(d + MD4_F(a, b, c) + w[5], 7);
    c = MD4_ROTL(c + MD4_F(d, a, b) + w[6], 11);
    b = MD4_ROTL(b + MD4_F(c, d, a) + w[7], 19);
    a = MD4_ROTL(a + MD4_F(b, c, d) + w[8], 3);
    d = MD4_ROTL(d + MD4_F(a, b, c) + w[9], 7);
    c = MD4_ROTL(c + MD4_F(d, a, b) + w[10], 11);
    b = MD4_ROTL(b + MD4_F(c, d, a) + w[11], 19);
    a = MD4_ROTL(a + MD4_F(b, c, d) + w[12], 3);
    d = MD4_ROTL(d + MD4_F(a, b, c) + w[13], 7);
    c = MD4_ROTL(c + MD4_F(d, a, b) + w[14], 11);
    b = MD4_ROTL(b + MD4_F(c, d, a) + w[15], 19);

    // Round 2: G(b,c,d), 16 ops, rotate by 3/5/9/13, constant 0x5A827999
    // Permutation: [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]
    a = MD4_ROTL(a + MD4_G(b, c, d) + w[0]  + 0x5A827999u, 3);
    d = MD4_ROTL(d + MD4_G(a, b, c) + w[4]  + 0x5A827999u, 5);
    c = MD4_ROTL(c + MD4_G(d, a, b) + w[8]  + 0x5A827999u, 9);
    b = MD4_ROTL(b + MD4_G(c, d, a) + w[12] + 0x5A827999u, 13);
    a = MD4_ROTL(a + MD4_G(b, c, d) + w[1]  + 0x5A827999u, 3);
    d = MD4_ROTL(d + MD4_G(a, b, c) + w[5]  + 0x5A827999u, 5);
    c = MD4_ROTL(c + MD4_G(d, a, b) + w[9]  + 0x5A827999u, 9);
    b = MD4_ROTL(b + MD4_G(c, d, a) + w[13] + 0x5A827999u, 13);
    a = MD4_ROTL(a + MD4_G(b, c, d) + w[2]  + 0x5A827999u, 3);
    d = MD4_ROTL(d + MD4_G(a, b, c) + w[6]  + 0x5A827999u, 5);
    c = MD4_ROTL(c + MD4_G(d, a, b) + w[10] + 0x5A827999u, 9);
    b = MD4_ROTL(b + MD4_G(c, d, a) + w[14] + 0x5A827999u, 13);
    a = MD4_ROTL(a + MD4_G(b, c, d) + w[3]  + 0x5A827999u, 3);
    d = MD4_ROTL(d + MD4_G(a, b, c) + w[7]  + 0x5A827999u, 5);
    c = MD4_ROTL(c + MD4_G(d, a, b) + w[11] + 0x5A827999u, 9);
    b = MD4_ROTL(b + MD4_G(c, d, a) + w[15] + 0x5A827999u, 13);

    // Round 3: H(b,c,d), 16 ops, rotate by 3/9/11/15, constant 0x6ED9EBA1
    // Permutation: [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
    a = MD4_ROTL(a + MD4_H(b, c, d) + w[0]  + 0x6ED9EBA1u, 3);
    d = MD4_ROTL(d + MD4_H(a, b, c) + w[8]  + 0x6ED9EBA1u, 9);
    c = MD4_ROTL(c + MD4_H(d, a, b) + w[4]  + 0x6ED9EBA1u, 11);
    b = MD4_ROTL(b + MD4_H(c, d, a) + w[12] + 0x6ED9EBA1u, 15);
    a = MD4_ROTL(a + MD4_H(b, c, d) + w[2]  + 0x6ED9EBA1u, 3);
    d = MD4_ROTL(d + MD4_H(a, b, c) + w[10] + 0x6ED9EBA1u, 9);
    c = MD4_ROTL(c + MD4_H(d, a, b) + w[6]  + 0x6ED9EBA1u, 11);
    b = MD4_ROTL(b + MD4_H(c, d, a) + w[14] + 0x6ED9EBA1u, 15);
    a = MD4_ROTL(a + MD4_H(b, c, d) + w[1]  + 0x6ED9EBA1u, 3);
    d = MD4_ROTL(d + MD4_H(a, b, c) + w[9]  + 0x6ED9EBA1u, 9);
    c = MD4_ROTL(c + MD4_H(d, a, b) + w[5]  + 0x6ED9EBA1u, 11);
    b = MD4_ROTL(b + MD4_H(c, d, a) + w[13] + 0x6ED9EBA1u, 15);
    a = MD4_ROTL(a + MD4_H(b, c, d) + w[3]  + 0x6ED9EBA1u, 3);
    d = MD4_ROTL(d + MD4_H(a, b, c) + w[11] + 0x6ED9EBA1u, 9);
    c = MD4_ROTL(c + MD4_H(d, a, b) + w[7]  + 0x6ED9EBA1u, 11);
    b = MD4_ROTL(b + MD4_H(c, d, a) + w[15] + 0x6ED9EBA1u, 15);

    s->h[0] += a;
    s->h[1] += b;
    s->h[2] += c;
    s->h[3] += d;
}

// Hash a password as NTLM: convert to UTF-16LE, then MD4.
// Multi-block: handles any pw_len up to PW_SLOT_LEN (64) — full 64-byte
// UTF-16LE blocks (32 password chars each), then a one- or two-block final
// with 0x80 padding and the bit length. Mask-mode candidates are capped at
// 27 chars by the host (single-block fast path); wordlist mode accepts more.
// Optimized NTLM: pack UTF-16LE directly into MD4 message block words.
// Each pair of password bytes (b0, b1) → word = b0 | (b1 << 16) in UTF-16LE.
// This eliminates the 128-byte utf16[] intermediate buffer (32 VGPRs saved).
static inline void ntlm_hash_password(const uchar *pw, uint pw_len, uint *out_h) {
    md4_state_t s;
    md4_init(&s);

    // UTF-16LE length = pw_len * 2 bytes. Each password byte → 2 bytes.
    // Pack directly into MD4 block words: word[i] = pw[2i] | (pw[2i+1] << 16)
    // This is little-endian UTF-16LE packed into uint32.
    uint utf16_len = pw_len * 2;
    ulong bits = (ulong)utf16_len * 8;

    // Process full 64-byte blocks (32 words = 32 password bytes per block)
    uint offset = 0;
    while (offset + 32 <= pw_len) {
        uint block[16];
        for (int i = 0; i < 16; i++) {
            uint byte_idx = offset + i * 2;
            block[i] = ((uint)pw[byte_idx]) | (((uint)pw[byte_idx + 1]) << 16);
        }
        md4_process_block(&s, block);
        offset += 32;
    }

    // Final block with padding
    uint rem_words = (pw_len - offset + 1) / 2; // remaining UTF-16LE words
    if (utf16_len - offset * 2 > 55) {
        // Two-block final
        uint block1[16];
        for (int i = 0; i < 16; i++) block1[i] = 0;
        for (uint i = 0; i < rem_words && offset + i * 2 + 1 < pw_len; i++) {
            uint byte_idx = offset + i * 2;
            block1[i] = ((uint)pw[byte_idx]) | (((uint)pw[byte_idx + 1]) << 16);
        }
        // Handle odd byte
        if ((pw_len - offset) & 1) {
            block1[rem_words - 1] = ((uint)pw[offset + (rem_words - 1) * 2]);
        }
        // 0x80 padding at UTF-16LE byte position (pw_len - offset) * 2
        uint pad_byte_pos = (pw_len - offset) * 2; // position in UTF-16LE stream
        uint pad_word = pad_byte_pos / 4;
        uint pad_shift = (pad_byte_pos & 3) * 8;
        block1[pad_word] |= ((uint)0x80) << pad_shift;
        md4_process_block(&s, block1);
        uint block2[16];
        for (int i = 0; i < 14; i++) block2[i] = 0;
        block2[14] = (uint)bits;
        block2[15] = (uint)(bits >> 32);
        md4_process_block(&s, block2);
    } else {
        uint block[16];
        for (int i = 0; i < 16; i++) block[i] = 0;
        for (uint i = 0; i < rem_words && offset + i * 2 + 1 < pw_len; i++) {
            uint byte_idx = offset + i * 2;
            block[i] = ((uint)pw[byte_idx]) | (((uint)pw[byte_idx + 1]) << 16);
        }
        // Handle odd byte
        if ((pw_len - offset) & 1) {
            block[rem_words - 1] = ((uint)pw[offset + (rem_words - 1) * 2]);
        }
        // 0x80 padding
        uint pad_byte_pos = (pw_len - offset) * 2;
        uint pad_word = pad_byte_pos / 4;
        uint pad_shift = (pad_byte_pos & 3) * 8;
        block[pad_word] |= ((uint)0x80) << pad_shift;
        block[14] = (uint)bits;
        block[15] = (uint)(bits >> 32);
        md4_process_block(&s, block);
    }

    for (int i = 0; i < 4; i++) out_h[i] = s.h[i];
}

// NTLM cracking kernel (mode 1000): each work-item tests one password.
// targets: num_hashes * 4 uints (NTLM = 16 bytes = 4 u32s, little-endian).
// Same buffer layout as crack_md5.
__kernel void crack_ntlm(
    __global const uchar *passwords,    // batch_size * PW_SLOT_LEN bytes
    __global const uint *pw_lens,       // batch_size password lengths
    __global const uint *targets,       // num_hashes * 4 u32s (NTLM digests)
    __global int *found,                // 0 = not found, 1 = found
    __global uint *found_idx,           // batch-local index of matching pw
    __global uint *found_hash_idx,      // which hash matched (0..num_hashes-1)
    __global uint *match_count,         // atomic counter: number of GPU matches
    __global uint *match_gids,          // up to 8 matching work-item IDs
    __global uint *match_hash_idxs,     // up to 8 matching hash indices
    const uint batch_size,
    const uint num_hashes
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;

    // Load password from global memory
    uchar pw[64];
    for (int i = 0; i < 64; i++) {
        pw[i] = passwords[gid * 64 + i];
    }
    uint pw_len = pw_lens[gid];
    if (pw_len > 64) pw_len = 64;

    // Compute NTLM = MD4(UTF-16LE(password))
    uint digest[4];
    ntlm_hash_password(pw, pw_len, digest);

    // Compare against all target hashes
    for (uint h = 0; h < num_hashes; h++) {
        uint target_h[4];
        target_h[0] = targets[h * 4 + 0];
        target_h[1] = targets[h * 4 + 1];
        target_h[2] = targets[h * 4 + 2];
        target_h[3] = targets[h * 4 + 3];

        if (digest[0] == target_h[0] &&
            digest[1] == target_h[1] &&
            digest[2] == target_h[2] &&
            digest[3] == target_h[3]) {
            uint slot = atomic_inc(match_count);
            if (slot < 8) {
                match_gids[slot] = gid;
                match_hash_idxs[slot] = h;
            }
            atomic_cmpxchg(found, 0, 1);
            if (slot == 0) {
                atomic_xchg(found_idx, gid);
                atomic_xchg(found_hash_idx, h);
            }
        }
    }
}

// === GPU MASK KERNELS (INNER-LOOP VECTORIZED) ===
// Each thread processes MASK_LOOPS consecutive candidates, amortizing kernel
// launch overhead and global memory reads across MASK_LOOPS hash computations.
// Thread gid handles candidates [start_index + gid*MASK_LOOPS, ... +MASK_LOOPS).
// Only the fastest-changing position (pos 0) changes between inner iterations;
// slower positions are precomputed once from the base index.

#define MASK_LOOPS 8

// Precompute password from a flat candidate index using odometer decomposition.


// Bloom filter check for multi-hash fast rejection.
// Two bitmaps (s1, s2) of BLOOM_BITS bits each. Both bits must be set
// for a "maybe match" — false positive rate ~1/BLOOM_BITS².
// Returns true if the digest might match a target (bloom filter says yes).
#define BLOOM_BITS 8192u   // 256 u32s per bitmap = 8KB total
#define BLOOM_MASK (BLOOM_BITS - 1u)

static inline bool bloom_check(
    __constant const uint *bloom_s1,
    __constant const uint *bloom_s2,
    const uint dg0, const uint dg1, const uint dg2, const uint dg3,
    const uint dg4, const uint dg5, const uint dg6, const uint dg7,
    const uint num_digest_u32s
) {
    // hash1: XOR all digest u32s
    uint h1 = dg0 ^ dg1 ^ dg2 ^ dg3;
    if (num_digest_u32s > 4) h1 ^= dg4 ^ dg5 ^ dg6 ^ dg7;
    // hash2: rotate-mix — must match host's (i*7+5)%32 rotation schedule
    uint h2 = rotate(dg0, 5u) ^ rotate(dg1, 12u) ^ rotate(dg2, 19u) ^ rotate(dg3, 26u);
    if (num_digest_u32s > 4) h2 ^= rotate(dg4, 1u) ^ rotate(dg5, 8u) ^ rotate(dg6, 15u) ^ rotate(dg7, 22u);

    uint idx1 = (h1 & BLOOM_MASK) >> 5;  // bit index → u32 index
    uint bit1 = 1u << (h1 & 31u);
    uint idx2 = (h2 & BLOOM_MASK) >> 5;
    uint bit2 = 1u << (h2 & 31u);

    return (bloom_s1[idx1] & bit1) && (bloom_s2[idx2] & bit2);
}

// Record a match using the 8-slot atomic match array.
static inline void record_match(
    __global int *found,
    __global uint *found_idx,
    __global uint *found_hash_idx,
    __global uint *match_count,
    __global uint *match_gids,
    __global uint *match_hash_idxs,
    const uint gid,
    const uint hash_idx
) {
    uint slot = atomic_inc(match_count);
    if (slot < 8) {
        match_gids[slot] = gid;
        match_hash_idxs[slot] = hash_idx;
    }
    atomic_cmpxchg(found, 0, 1);
    if (slot == 0) {
        atomic_xchg(found_idx, gid);
        atomic_xchg(found_hash_idx, hash_idx);
    }
}

// GPU mask kernel for MD5 (mode 0) — inner-loop vectorized.
// Each thread processes MASK_LOOPS consecutive candidates.
// GPU mask kernel for MD5 (mode 0) — hashcat-style architecture.
// GPU-side candidate generation: ALL positions decomposed from thread index.
// No host DMA for passwords — purely compute-bound.
__kernel void crack_md5_mask(
    __global const uchar *charset, __global const uint *cs_offsets, __global const uint *cs_lengths,
    __constant uint *words_buf_r, const uint il_cnt,
    const ulong start_base_idx, const uint num_expanded_positions, const uint num_positions,
    __constant uint *targets,
    __global int *found, __global uint *found_idx, __global uint *found_hash_idx,
    __global uint *match_count, __global uint *match_gids, __global uint *match_hash_idxs,
    const uint batch_size, const uint num_hashes,
    __constant uint *bloom_s1, __constant uint *bloom_s2)
{
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;
    // Generate base password from thread index — replaces host DMA.
    // Decompose (start_base_idx + gid) into remaining position char indices.
    uint w[16];
    for (int i = 0; i < 16; i++) w[i] = 0;
    ulong base_idx = start_base_idx + (ulong)gid;
    for (uint pos = num_expanded_positions; pos < num_positions; pos++) {
        uint cs_len = cs_lengths[pos];
        uint char_idx = (uint)(base_idx % (ulong)cs_len);
        base_idx /= (ulong)cs_len;
        uchar c = charset[cs_offsets[pos] + char_idx];
        w[pos >> 2] |= (uint)c << ((pos & 3) * 8);
    }
    uint pw_len = num_positions;
    if (pw_len > 55) return; // FIX: single-block MD5 limit, multi-block not supported in mask mode
    uint w0l = w[0];
    uint pad_word = pw_len >> 2, pad_shift = (pw_len & 3) * 8, bits = pw_len * 8;
    if (pad_word == 0) w0l |= (0x80u << pad_shift);
    #define H0 0x67452301u
    #define H1 0xefcdab89u
    #define H2 0x98badcfeu
    #define H3 0x10325476u

    // Pre-compute the MD5 block outside the inner loop.
    // Only block[0] changes per il_pos (w0l | words_buf_r[il_pos]).
    // block[1-13] = w[1-13] (unchanged)
    // block[14] = bits (length in bits)
    // block[15] = 0
    // block[pad_word] |= (0x80 << pad_shift) (padding)
    uint block[16];
    for (int i = 1; i < 16; i++) block[i] = w[i];
    block[pad_word] = w[pad_word] | (0x80u << pad_shift);
    if (pad_word < 14) { block[14] = bits; block[15] = 0; } // FIX N8: don't overwrite padding

            uint kc1 = 0xE8C7B756u + block[1];
    uint kc2 = 0x242070DBu + block[2];
    uint kc3 = 0xC1BDCEEEu + block[3];
    uint kc4 = 0xF57C0FAFu + block[4];
    uint kc5 = 0x4787C62Au + block[5];
    uint kc6 = 0xA8304613u + block[6];
    uint kc7 = 0xFD469501u + block[7];
    uint kc8 = 0x698098D8u + block[8];
    uint kc9 = 0x8B44F7AFu + block[9];
    uint kc10 = 0xFFFF5BB1u + block[10];
    uint kc11 = 0x895CD7BEu + block[11];
    uint kc12 = 0x6B901122u + block[12];
    uint kc13 = 0xFD987193u + block[13];
    uint kc14 = 0xA679438Eu + block[14];
    uint kc15 = 0x49B40821u + block[15];
    uint kc16 = 0xF61E2562u + block[1];
    uint kc17 = 0xC040B340u + block[6];
    uint kc18 = 0x265E5A51u + block[11];
    uint kc20 = 0xD62F105Du + block[5];
    uint kc21 = 0x02441453u + block[10];
    uint kc22 = 0xD8A1E681u + block[15];
    uint kc23 = 0xE7D3FBC8u + block[4];
    uint kc24 = 0x21E1CDE6u + block[9];
    uint kc25 = 0xC33707D6u + block[14];
    uint kc26 = 0xF4D50D87u + block[3];
    uint kc27 = 0x455A14EDu + block[8];
    uint kc28 = 0xA9E3E905u + block[13];
    uint kc29 = 0xFCEFA3F8u + block[2];
    uint kc30 = 0x676F02D9u + block[7];
    uint kc31 = 0x8D2A4C8Au + block[12];
    uint kc32 = 0xFFFA3942u + block[5];
    uint kc33 = 0x8771F681u + block[8];
    uint kc34 = 0x6D9D6122u + block[11];
    uint kc35 = 0xFDE5380Cu + block[14];
    uint kc36 = 0xA4BEEA44u + block[1];
    uint kc37 = 0x4BDECFA9u + block[4];
    uint kc38 = 0xF6BB4B60u + block[7];
    uint kc39 = 0xBEBFBC70u + block[10];
    uint kc40 = 0x289B7EC6u + block[13];
    uint kc42 = 0xD4EF3085u + block[3];
    uint kc43 = 0x04881D05u + block[6];
    uint kc44 = 0xD9D4D039u + block[9];
    uint kc45 = 0xE6DB99E5u + block[12];
    // === SINGLE-HASH EARLY-EXIT PATH (hashcat m00000s optimization) ===
    // Pre-compute reverse MD5 from target: reverse 15 I-steps + 2 H-steps
    // Then forward 43 steps, check, skip remaining 21 steps for 99.99% of candidates
    if (num_hashes == 1 && il_cnt > 1) {
        uint search0 = targets[0] - 0x67452301u, search1 = targets[1] - 0xefcdab89u, search2 = targets[2] - 0x98badcfeu, search3 = targets[3] - 0x10325476u;

        // Reverse 15 I-round steps (steps 64->50, none use w[0])
        // MD5_STEP_REV_I: a -= b; a = rotate(a, 32-s); a -= I(b,c,d); a -= x; a -= K
        #define MD5_I_R(a,b,c,d,x,K,s) { a -= b; a = rotate(a, (32u - s)); a -= (c ^ (b | (~d))); a -= x; a -= K; }
        uint a_rev = search0, b_rev = search1, c_rev = search2, d_rev = search3;
        MD5_I_R(b_rev, c_rev, d_rev, a_rev, block[9],  0xEB86D391u, 21u);
        MD5_I_R(c_rev, d_rev, a_rev, b_rev, block[2],  0x2AD7D2BBu, 15u);
        MD5_I_R(d_rev, a_rev, b_rev, c_rev, block[11], 0xBD3AF235u, 10u);
        MD5_I_R(a_rev, b_rev, c_rev, d_rev, block[4],  0xF7537E82u, 6u);
        MD5_I_R(b_rev, c_rev, d_rev, a_rev, block[13], 0x4E0811A1u, 21u);
        MD5_I_R(c_rev, d_rev, a_rev, b_rev, block[6],  0xA3014314u, 15u);
        MD5_I_R(d_rev, a_rev, b_rev, c_rev, block[15], 0xFE2CE6E0u, 10u);
        MD5_I_R(a_rev, b_rev, c_rev, d_rev, block[8],  0x6FA87E4Fu, 6u);
        MD5_I_R(b_rev, c_rev, d_rev, a_rev, block[1],  0x85845DD1u, 21u);
        MD5_I_R(c_rev, d_rev, a_rev, b_rev, block[10], 0xFFEFF47Du, 15u);
        MD5_I_R(d_rev, a_rev, b_rev, c_rev, block[3],  0x8F0CCC92u, 10u);
        MD5_I_R(a_rev, b_rev, c_rev, d_rev, block[12], 0x655B59C3u, 6u);
        MD5_I_R(b_rev, c_rev, d_rev, a_rev, block[5],  0xFC93A039u, 21u);
        MD5_I_R(c_rev, d_rev, a_rev, b_rev, block[14], 0xAB9423A7u, 15u);
        MD5_I_R(d_rev, a_rev, b_rev, c_rev, block[7],  0x432AFF97u, 10u);
        // Reverse step 48 partially (skip w0 subtraction — step 48 uses w[0]):
        a_rev -= b_rev; a_rev = rotate(a_rev, 32u - 6u); a_rev -= (c_rev ^ (b_rev | (~d_rev))); a_rev -= 0xF4292244u;
        // Reverse 2 H-round steps (steps 48->46):
        #define MD5_H_R(a,b,c,d,x,K,s) { a -= b; a = rotate(a, (32u - s)); a -= x; a -= K; }
        uint pre_cd = c_rev ^ d_rev;
        MD5_H_R(b_rev, c_rev, d_rev, a_rev, block[2],  0xC4AC5665u, 23u);
        MD5_H_R(c_rev, d_rev, a_rev, b_rev, block[15], 0x1FA27CF8u, 16u);

        for (uint il_pos = 0; il_pos < il_cnt; il_pos++) {
            uint w0 = w0l | words_buf_r[il_pos];
            uint kc0  = 0xD76AA478u + w0;
            uint kc19 = 0xE9B6C7AAu + w0;
            uint kc41 = 0xEAA127FAu + w0;
            uint kc48 = 0xF4292244u + w0;

            uint pre_d = d_rev;
            uint pre_a = a_rev - w0;
            uint pre_b = b_rev - (pre_a ^ pre_cd);
            uint pre_c = c_rev - (pre_a ^ pre_b ^ pre_d);

            uint a = H0, b = H1, c = H2, d = H3;
            uint ht;
            // F round (16 steps)
            a = a + bitselect(d, c, b) + kc0;  a = rotate(a, 7u);  a += b;
            d = d + bitselect(c, b, a) + kc1;  d = rotate(d, 12u); d += a;
            c = c + bitselect(b, a, d) + kc2;  c = rotate(c, 17u); c += d;
            b = b + bitselect(a, d, c) + kc3;  b = rotate(b, 22u); b += c;
            a = a + bitselect(d, c, b) + kc4;  a = rotate(a, 7u);  a += b;
            d = d + bitselect(c, b, a) + kc5;  d = rotate(d, 12u); d += a;
            c = c + bitselect(b, a, d) + kc6;  c = rotate(c, 17u); c += d;
            b = b + bitselect(a, d, c) + kc7;  b = rotate(b, 22u); b += c;
            a = a + bitselect(d, c, b) + kc8;  a = rotate(a, 7u);  a += b;
            d = d + bitselect(c, b, a) + kc9;  d = rotate(d, 12u); d += a;
            c = c + bitselect(b, a, d) + kc10; c = rotate(c, 17u); c += d;
            b = b + bitselect(a, d, c) + kc11; b = rotate(b, 22u); b += c;
            a = a + bitselect(d, c, b) + kc12; a = rotate(a, 7u);  a += b;
            d = d + bitselect(c, b, a) + kc13; d = rotate(d, 12u); d += a;
            c = c + bitselect(b, a, d) + kc14; c = rotate(c, 17u); c += d;
            b = b + bitselect(a, d, c) + kc15; b = rotate(b, 22u); b += c;
            // G round (16 steps)
            a = a + bitselect(c, b, d) + kc16; a = rotate(a, 5u);  a += b;
            d = d + bitselect(b, a, c) + kc17; d = rotate(d, 9u);  d += a;
            c = c + bitselect(a, d, b) + kc18; c = rotate(c, 14u); c += d;
            b = b + bitselect(d, c, a) + kc19; b = rotate(b, 20u); b += c;
            a = a + bitselect(c, b, d) + kc20; a = rotate(a, 5u);  a += b;
            d = d + bitselect(b, a, c) + kc21; d = rotate(d, 9u);  d += a;
            c = c + bitselect(a, d, b) + kc22; c = rotate(c, 14u); c += d;
            b = b + bitselect(d, c, a) + kc23; b = rotate(b, 20u); b += c;
            a = a + bitselect(c, b, d) + kc24; a = rotate(a, 5u);  a += b;
            d = d + bitselect(b, a, c) + kc25; d = rotate(d, 9u);  d += a;
            c = c + bitselect(a, d, b) + kc26; c = rotate(c, 14u); c += d;
            b = b + bitselect(d, c, a) + kc27; b = rotate(b, 20u); b += c;
            a = a + bitselect(c, b, d) + kc28; a = rotate(a, 5u);  a += b;
            d = d + bitselect(b, a, c) + kc29; d = rotate(d, 9u);  d += a;
            c = c + bitselect(a, d, b) + kc30; c = rotate(c, 14u); c += d;
            b = b + bitselect(d, c, a) + kc31; b = rotate(b, 20u); b += c;
            // H round (11 steps) with H1/H2 pattern
            ht = b ^ c; a = a + (ht ^ d) + kc32; a = rotate(a, 4u);  a += b;
            d = d + (a ^ ht) + kc33;           d = rotate(d, 11u); d += a;
            ht = d ^ a; c = c + (ht ^ b) + kc34; c = rotate(c, 16u); c += d;
            b = b + (c ^ ht) + kc35;           b = rotate(b, 23u); b += c;
            ht = b ^ c; a = a + (ht ^ d) + kc36; a = rotate(a, 4u);  a += b;
            d = d + (a ^ ht) + kc37;           d = rotate(d, 11u); d += a;
            ht = d ^ a; c = c + (ht ^ b) + kc38; c = rotate(c, 16u); c += d;
            b = b + (c ^ ht) + kc39;           b = rotate(b, 23u); b += c;
            ht = b ^ c; a = a + (ht ^ d) + kc40; a = rotate(a, 4u);  a += b;
            d = d + (a ^ ht) + kc41;           d = rotate(d, 11u); d += a;
            ht = d ^ a; c = c + (ht ^ b) + kc42; c = rotate(c, 16u); c += d;
            // === EARLY EXIT CHECK 1 — 99.99%+ skip remaining 22 steps ===
            if (pre_c != c) continue;
            // H round (3 more steps)
            b = b + (c ^ ht) + kc43;           b = rotate(b, 23u); b += c;
            ht = b ^ c; a = a + (ht ^ d) + kc44; a = rotate(a, 4u);  a += b;
            d = d + (a ^ ht) + kc45;           d = rotate(d, 11u); d += a;
            // === EARLY EXIT CHECK 2 ===
            if (pre_d != d) continue;
    uint kc46 = 0x1FA27CF8u + block[15];
    uint kc47 = 0xC4AC5665u + block[2];
    uint kc49 = 0x432AFF97u + block[7];
    uint kc50 = 0xAB9423A7u + block[14];
    uint kc51 = 0xFC93A039u + block[5];
    uint kc52 = 0x655B59C3u + block[12];
    uint kc53 = 0x8F0CCC92u + block[3];
    uint kc54 = 0xFFEFF47Du + block[10];
    uint kc55 = 0x85845DD1u + block[1];
    uint kc56 = 0x6FA87E4Fu + block[8];
    uint kc57 = 0xFE2CE6E0u + block[15];
    uint kc58 = 0xA3014314u + block[6];
    uint kc59 = 0x4E0811A1u + block[13];
    uint kc60 = 0xF7537E82u + block[4];
    uint kc61 = 0xBD3AF235u + block[11];
    uint kc62 = 0x2AD7D2BBu + block[2];
    uint kc63 = 0xEB86D391u + block[9];
            // H round (2 more) + I round (16 steps)
            ht = d ^ a; c = c + (ht ^ b) + kc46; c = rotate(c, 16u); c += d;
            b = b + (c ^ ht) + kc47;           b = rotate(b, 23u); b += c;
            a = a + (c ^ (b | (~d))) + kc48; a = rotate(a, 6u);  a += b;
            d = d + (b ^ (a | (~c))) + kc49; d = rotate(d, 10u); d += a;
            c = c + (a ^ (d | (~b))) + kc50; c = rotate(c, 15u); c += d;
            b = b + (d ^ (c | (~a))) + kc51; b = rotate(b, 21u); b += c;
            a = a + (c ^ (b | (~d))) + kc52; a = rotate(a, 6u);  a += b;
            d = d + (b ^ (a | (~c))) + kc53; d = rotate(d, 10u); d += a;
            c = c + (a ^ (d | (~b))) + kc54; c = rotate(c, 15u); c += d;
            b = b + (d ^ (c | (~a))) + kc55; b = rotate(b, 21u); b += c;
            a = a + (c ^ (b | (~d))) + kc56; a = rotate(a, 6u);  a += b;
            d = d + (b ^ (a | (~c))) + kc57; d = rotate(d, 10u); d += a;
            c = c + (a ^ (d | (~b))) + kc58; c = rotate(c, 15u); c += d;
            b = b + (d ^ (c | (~a))) + kc59; b = rotate(b, 21u); b += c;
            a = a + (c ^ (b | (~d))) + kc60; a = rotate(a, 6u);  a += b;
            d = d + (b ^ (a | (~c))) + kc61; d = rotate(d, 10u); d += a;
            c = c + (a ^ (d | (~b))) + kc62; c = rotate(c, 15u); c += d;
            b = b + (d ^ (c | (~a))) + kc63; b = rotate(b, 21u); b += c;
            // === COMPARE (IV pre-subtracted from target) ===
            if (a == search0 && b == search1 && c == search2 && d == search3)
                record_match(found, found_idx, found_hash_idx, match_count, match_gids, match_hash_idxs, gid * il_cnt + il_pos, 0);
        }
    } else {
    uint kc46 = 0x1FA27CF8u + block[15];
    uint kc47 = 0xC4AC5665u + block[2];
    uint kc49 = 0x432AFF97u + block[7];
    uint kc50 = 0xAB9423A7u + block[14];
    uint kc51 = 0xFC93A039u + block[5];
    uint kc52 = 0x655B59C3u + block[12];
    uint kc53 = 0x8F0CCC92u + block[3];
    uint kc54 = 0xFFEFF47Du + block[10];
    uint kc55 = 0x85845DD1u + block[1];
    uint kc56 = 0x6FA87E4Fu + block[8];
    uint kc57 = 0xFE2CE6E0u + block[15];
    uint kc58 = 0xA3014314u + block[6];
    uint kc59 = 0x4E0811A1u + block[13];
    uint kc60 = 0xF7537E82u + block[4];
    uint kc61 = 0xBD3AF235u + block[11];
    uint kc62 = 0x2AD7D2BBu + block[2];
    uint kc63 = 0xEB86D391u + block[9];
        // === MULTI-HASH FULL PATH (existing code) ===
    for (uint il_pos = 0; il_pos < il_cnt; il_pos++) {
        block[0] = w0l | words_buf_r[il_pos];
        uint kc0 = 0xD76AA478u + block[0];
        uint kc19 = 0xE9B6C7AAu + block[0];
        uint kc41 = 0xEAA127FAu + block[0];
        uint kc48 = 0xF4292244u + block[0];
        uint a = H0, b = H1, c = H2, d = H3;
        uint ht; // shared temp for H1/H2 round 3 optimization
        a = a + bitselect(d, c, b) + kc0; a = rotate(a, 7u); a += b;
        d = d + bitselect(c, b, a) + kc1; d = rotate(d, 12u); d += a;
        c = c + bitselect(b, a, d) + kc2; c = rotate(c, 17u); c += d;
        b = b + bitselect(a, d, c) + kc3; b = rotate(b, 22u); b += c;
        a = a + bitselect(d, c, b) + kc4; a = rotate(a, 7u); a += b;
        d = d + bitselect(c, b, a) + kc5; d = rotate(d, 12u); d += a;
        c = c + bitselect(b, a, d) + kc6; c = rotate(c, 17u); c += d;
        b = b + bitselect(a, d, c) + kc7; b = rotate(b, 22u); b += c;
        a = a + bitselect(d, c, b) + kc8; a = rotate(a, 7u); a += b;
        d = d + bitselect(c, b, a) + kc9; d = rotate(d, 12u); d += a;
        c = c + bitselect(b, a, d) + kc10; c = rotate(c, 17u); c += d;
        b = b + bitselect(a, d, c) + kc11; b = rotate(b, 22u); b += c;
        a = a + bitselect(d, c, b) + kc12; a = rotate(a, 7u); a += b;
        d = d + bitselect(c, b, a) + kc13; d = rotate(d, 12u); d += a;
        c = c + bitselect(b, a, d) + kc14; c = rotate(c, 17u); c += d;
        b = b + bitselect(a, d, c) + kc15; b = rotate(b, 22u); b += c;
        a = a + bitselect(c, b, d) + kc16; a = rotate(a, 5u); a += b;
        d = d + bitselect(b, a, c) + kc17; d = rotate(d, 9u); d += a;
        c = c + bitselect(a, d, b) + kc18; c = rotate(c, 14u); c += d;
        b = b + bitselect(d, c, a) + kc19; b = rotate(b, 20u); b += c;
        a = a + bitselect(c, b, d) + kc20; a = rotate(a, 5u); a += b;
        d = d + bitselect(b, a, c) + kc21; d = rotate(d, 9u); d += a;
        c = c + bitselect(a, d, b) + kc22; c = rotate(c, 14u); c += d;
        b = b + bitselect(d, c, a) + kc23; b = rotate(b, 20u); b += c;
        a = a + bitselect(c, b, d) + kc24; a = rotate(a, 5u); a += b;
        d = d + bitselect(b, a, c) + kc25; d = rotate(d, 9u); d += a;
        c = c + bitselect(a, d, b) + kc26; c = rotate(c, 14u); c += d;
        b = b + bitselect(d, c, a) + kc27; b = rotate(b, 20u); b += c;
        a = a + bitselect(c, b, d) + kc28; a = rotate(a, 5u); a += b;
        d = d + bitselect(b, a, c) + kc29; d = rotate(d, 9u); d += a;
        c = c + bitselect(a, d, b) + kc30; c = rotate(c, 14u); c += d;
        b = b + bitselect(d, c, a) + kc31; b = rotate(b, 20u); b += c;
        ht = b ^ c; a = a + (ht ^ d) + kc32; a = rotate(a, 4u); a += b;
        d = d + (a ^ ht) + kc33; d = rotate(d, 11u); d += a;
        ht = d ^ a; c = c + (ht ^ b) + kc34; c = rotate(c, 16u); c += d;
        b = b + (c ^ ht) + kc35; b = rotate(b, 23u); b += c;
        ht = b ^ c; a = a + (ht ^ d) + kc36; a = rotate(a, 4u); a += b;
        d = d + (a ^ ht) + kc37; d = rotate(d, 11u); d += a;
        ht = d ^ a; c = c + (ht ^ b) + kc38; c = rotate(c, 16u); c += d;
        b = b + (c ^ ht) + kc39; b = rotate(b, 23u); b += c;
        ht = b ^ c; a = a + (ht ^ d) + kc40; a = rotate(a, 4u); a += b;
        d = d + (a ^ ht) + kc41; d = rotate(d, 11u); d += a;
        ht = d ^ a; c = c + (ht ^ b) + kc42; c = rotate(c, 16u); c += d;
        b = b + (c ^ ht) + kc43; b = rotate(b, 23u); b += c;
        ht = b ^ c; a = a + (ht ^ d) + kc44; a = rotate(a, 4u); a += b;
        d = d + (a ^ ht) + kc45; d = rotate(d, 11u); d += a;
        ht = d ^ a; c = c + (ht ^ b) + kc46; c = rotate(c, 16u); c += d;
        b = b + (c ^ ht) + kc47; b = rotate(b, 23u); b += c;
        a = a + (c ^ (b | (~d))) + kc48; a = rotate(a, 6u); a += b;
        d = d + (b ^ (a | (~c))) + kc49; d = rotate(d, 10u); d += a;
        c = c + (a ^ (d | (~b))) + kc50; c = rotate(c, 15u); c += d;
        b = b + (d ^ (c | (~a))) + kc51; b = rotate(b, 21u); b += c;
        a = a + (c ^ (b | (~d))) + kc52; a = rotate(a, 6u); a += b;
        d = d + (b ^ (a | (~c))) + kc53; d = rotate(d, 10u); d += a;
        c = c + (a ^ (d | (~b))) + kc54; c = rotate(c, 15u); c += d;
        b = b + (d ^ (c | (~a))) + kc55; b = rotate(b, 21u); b += c;
        a = a + (c ^ (b | (~d))) + kc56; a = rotate(a, 6u); a += b;
        d = d + (b ^ (a | (~c))) + kc57; d = rotate(d, 10u); d += a;
        c = c + (a ^ (d | (~b))) + kc58; c = rotate(c, 15u); c += d;
        b = b + (d ^ (c | (~a))) + kc59; b = rotate(b, 21u); b += c;
        a = a + (c ^ (b | (~d))) + kc60; a = rotate(a, 6u); a += b;
        d = d + (b ^ (a | (~c))) + kc61; d = rotate(d, 10u); d += a;
        c = c + (a ^ (d | (~b))) + kc62; c = rotate(c, 15u); c += d;
        b = b + (d ^ (c | (~a))) + kc63; b = rotate(b, 21u); b += c;
        uint dg0 = H0+a, dg1 = H1+b, dg2 = H2+c, dg3 = H3+d;
        // Bloom filter: fast reject non-matches in O(1) instead of O(n_hashes)
        if (num_hashes > 1 && bloom_check(bloom_s1, bloom_s2, dg0, dg1, dg2, dg3, 0, 0, 0, 0, 4)) {
            for (uint h = 0; h < num_hashes; h++) {
                if (dg0 == targets[h*4] && dg1 == targets[h*4+1] && dg2 == targets[h*4+2] && dg3 == targets[h*4+3])
                    record_match(found, found_idx, found_hash_idx, match_count, match_gids, match_hash_idxs, gid * il_cnt + il_pos, h);
            }
        } else if (num_hashes == 1) {
            // Single-hash: direct comparison (bloom filter not needed)
            if (dg0 == targets[0] && dg1 == targets[1] && dg2 == targets[2] && dg3 == targets[3])
                record_match(found, found_idx, found_hash_idx, match_count, match_gids, match_hash_idxs, gid * il_cnt + il_pos, 0);
        }
    }
}
    }

__kernel void crack_ntlm_mask(
    __global const uchar *charset, __global const uint *cs_offsets, __global const uint *cs_lengths,
    __constant uint *words_buf_r, const uint il_cnt,
    const ulong start_base_idx, const uint num_expanded_positions, const uint num_positions,
    __constant uint *targets,
    __global int *found, __global uint *found_idx, __global uint *found_hash_idx,
    __global uint *match_count, __global uint *match_gids, __global uint *match_hash_idxs,
    const uint batch_size, const uint num_hashes,
    __constant uint *bloom_s1, __constant uint *bloom_s2)
{
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;
    // Generate base password from thread index — replaces host DMA.
    uint pw[16];
    for (int i = 0; i < 16; i++) pw[i] = 0;
    ulong base_idx = start_base_idx + (ulong)gid;
    for (uint pos = num_expanded_positions; pos < num_positions; pos++) {
        uint cs_len = cs_lengths[pos];
        uint char_idx = (uint)(base_idx % (ulong)cs_len);
        base_idx /= (ulong)cs_len;
        uchar c = charset[cs_offsets[pos] + char_idx];
        pw[pos >> 2] |= (uint)c << ((pos & 3) * 8);
    }
    uint pw_len = num_positions;
    if (pw_len > 27) return; // FIX: single-block NTLM limit (27*2=54 + 1 pad + 8 len = 63 ≤ 64 bytes)
    uint w0l = pw[0];
    uint utf16_len = pw_len * 2, pad_word = utf16_len >> 2, pad_shift = (utf16_len & 3) * 8, bits = utf16_len * 8;
    // MD4 IV same as MD5 IV — H0/H1/H2/H3 already defined

    // Pre-compute UTF-16LE encoding and MD4 block outside the inner loop.
    // Only w[0] and w[1] change per il_pos (from pw[0] = w0l | words_buf_r[il_pos]).
    // w[2-15] are unchanged and pre-computed here.
    uint w[16];
    for (int i = 0; i < 16; i++) w[i] = 0;
    // Encode pw[1-7] into UTF-16LE (these don't change)
    for (int i = 1; i < 8 && i * 4 < pw_len; i++) {
        uint v = pw[i];
        w[i*2] = (v & 0xFF) | ((v & 0xFF00) << 8);
        w[i*2+1] = ((v >> 16) & 0xFF) | ((v & 0xFF000000) >> 8);
    }
    // Pre-compute pw[0] UTF-16LE (will be overwritten with mask in loop)
    uint v0 = pw[0];
    uint w0_pre = (v0 & 0xFF) | ((v0 & 0xFF00) << 8);
    uint w1_pre = ((v0 >> 16) & 0xFF) | ((v0 & 0xFF000000) >> 8);
    w[0] = w0_pre;
    w[1] = w1_pre;
    w[pad_word] |= (0x80u << pad_shift);
    if (pad_word < 14) { w[14] = bits; w[15] = 0; } // FIX N9: don't overwrite padding

    // Pre-compute K + w[i] for MD4 rounds 2 and 3 (hashcat optimization).
    // Round 1 has no K constant. Only w[0] changes per il_pos.
    // Round 2: K2=0x5A827999, schedule=[0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]
    uint nkc2_1  = 0x5A827999u + w[4];
    uint nkc2_2  = 0x5A827999u + w[8];
    uint nkc2_3  = 0x5A827999u + w[12];
    // nkc2_4 moved inside loop (w[1] changes)
    uint nkc2_5  = 0x5A827999u + w[5];
    uint nkc2_6  = 0x5A827999u + w[9];
    uint nkc2_7  = 0x5A827999u + w[13];
    uint nkc2_8  = 0x5A827999u + w[2];
    uint nkc2_9  = 0x5A827999u + w[6];
    uint nkc2_10 = 0x5A827999u + w[10];
    uint nkc2_11 = 0x5A827999u + w[14];
    uint nkc2_12 = 0x5A827999u + w[3];
    uint nkc2_13 = 0x5A827999u + w[7];
    uint nkc2_14 = 0x5A827999u + w[11];
    uint nkc2_15 = 0x5A827999u + w[15];
    // Round 3: K3=0x6ED9EBA1, schedule=[0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
    uint nkc3_1  = 0x6ED9EBA1u + w[8];
    uint nkc3_2  = 0x6ED9EBA1u + w[4];
    uint nkc3_3  = 0x6ED9EBA1u + w[12];
    uint nkc3_4  = 0x6ED9EBA1u + w[2];
    uint nkc3_5  = 0x6ED9EBA1u + w[10];
    uint nkc3_6  = 0x6ED9EBA1u + w[6];
    uint nkc3_7  = 0x6ED9EBA1u + w[14];
    // nkc3_8 moved inside loop (w[1] changes)
    uint nkc3_9  = 0x6ED9EBA1u + w[9];
    uint nkc3_10 = 0x6ED9EBA1u + w[5];
    uint nkc3_11 = 0x6ED9EBA1u + w[13];
    uint nkc3_12 = 0x6ED9EBA1u + w[3];
    uint nkc3_13 = 0x6ED9EBA1u + w[11];
    uint nkc3_14 = 0x6ED9EBA1u + w[7];
    uint nkc3_15 = 0x6ED9EBA1u + w[15];

    // MD4 reverse macros (no a+=b, simpler than MD5)
    #define MD4_H_REV(a,b,c,d,x,K,s) { a = rotate(a, (32u - s)); a -= (b ^ c ^ d); a -= x; a -= K; }
    #define MD4_G_REV1(a,b,c,d,x,K,s) { a = rotate(a, (32u - s)); a -= x; a -= K; }

    if (num_hashes == 1 && il_cnt > 1) {
        uint search0 = targets[0] - 0x67452301u, search1 = targets[1] - 0xefcdab89u, search2 = targets[2] - 0x98badcfeu, search3 = targets[3] - 0x10325476u;

        // Reverse 16 H-round steps (steps 48->32)
        uint a_rev = search0, b_rev = search1, c_rev = search2, d_rev = search3;
        MD4_H_REV(b_rev, c_rev, d_rev, a_rev, w[15], 0x6ED9EBA1u, 15u);
        MD4_H_REV(c_rev, d_rev, a_rev, b_rev, w[7], 0x6ED9EBA1u, 11u);
        MD4_H_REV(d_rev, a_rev, b_rev, c_rev, w[11], 0x6ED9EBA1u, 9u);
        MD4_H_REV(a_rev, b_rev, c_rev, d_rev, w[3], 0x6ED9EBA1u, 3u);
        MD4_H_REV(b_rev, c_rev, d_rev, a_rev, w[13], 0x6ED9EBA1u, 15u);
        MD4_H_REV(c_rev, d_rev, a_rev, b_rev, w[5], 0x6ED9EBA1u, 11u);
        MD4_H_REV(d_rev, a_rev, b_rev, c_rev, w[9], 0x6ED9EBA1u, 9u);
        MD4_H_REV(a_rev, b_rev, c_rev, d_rev, w[1], 0x6ED9EBA1u, 3u);
        MD4_H_REV(b_rev, c_rev, d_rev, a_rev, w[14], 0x6ED9EBA1u, 15u);
        MD4_H_REV(c_rev, d_rev, a_rev, b_rev, w[6], 0x6ED9EBA1u, 11u);
        MD4_H_REV(d_rev, a_rev, b_rev, c_rev, w[10], 0x6ED9EBA1u, 9u);
        MD4_H_REV(a_rev, b_rev, c_rev, d_rev, w[2], 0x6ED9EBA1u, 3u);
        MD4_H_REV(b_rev, c_rev, d_rev, a_rev, w[12], 0x6ED9EBA1u, 15u);
        MD4_H_REV(c_rev, d_rev, a_rev, b_rev, w[4], 0x6ED9EBA1u, 11u);
        MD4_H_REV(d_rev, a_rev, b_rev, c_rev, w[8], 0x6ED9EBA1u, 9u);
        a_rev = rotate(a_rev, 32u - 3u); a_rev -= (b_rev ^ c_rev ^ d_rev); a_rev -= 0x6ED9EBA1u;
        // Save c BEFORE G14 reverse (needed for pre_b's G function)
        uint sav_c_pre = c_rev;
        // Save state before G15/G14 reverses
        uint sav_a = a_rev, sav_d = d_rev;
        MD4_G_REV1(b_rev, c_rev, d_rev, a_rev, w[15], 0x5A827999u, 13u);
        MD4_G_REV1(c_rev, d_rev, a_rev, b_rev, w[11], 0x5A827999u, 9u);
        uint sav_b = b_rev; // after G15 reverse
        uint sav_c = c_rev; // after G14 reverse

        for (uint il_pos = 0; il_pos < il_cnt; il_pos++) {
            uint v = w0l | words_buf_r[il_pos];
            w[0] = (v & 0xFF) | ((v & 0xFF00) << 8);
            w[1] = ((v >> 16) & 0xFF) | ((v & 0xFF000000) >> 8);
            if (pad_word == 0) w[0] |= (0x80u << pad_shift);
            else if (pad_word == 1) w[1] |= (0x80u << pad_shift);
            uint nkc2_0 = 0x5A827999u + w[0];
            uint nkc3_0 = 0x6ED9EBA1u + w[0];
            uint nkc2_4 = 0x5A827999u + w[1];
            uint nkc3_8 = 0x6ED9EBA1u + w[1];
            uint pre_a = sav_a - w[0];
            uint pre_b = sav_b - bitselect(sav_c_pre, sav_d, sav_c_pre ^ pre_a);
            uint pre_c = sav_c - bitselect(sav_d, pre_a, sav_d ^ pre_b);

            uint a = H0, b = H1, c = H2, d = H3;
            uint ht;
            a = a + bitselect(d, c, b) + w[0]; a = rotate(a, 3u);
            d = d + bitselect(c, b, a) + w[1]; d = rotate(d, 7u);
            c = c + bitselect(b, a, d) + w[2]; c = rotate(c, 11u);
            b = b + bitselect(a, d, c) + w[3]; b = rotate(b, 19u);
            a = a + bitselect(d, c, b) + w[4]; a = rotate(a, 3u);
            d = d + bitselect(c, b, a) + w[5]; d = rotate(d, 7u);
            c = c + bitselect(b, a, d) + w[6]; c = rotate(c, 11u);
            b = b + bitselect(a, d, c) + w[7]; b = rotate(b, 19u);
            a = a + bitselect(d, c, b) + w[8]; a = rotate(a, 3u);
            d = d + bitselect(c, b, a) + w[9]; d = rotate(d, 7u);
            c = c + bitselect(b, a, d) + w[10]; c = rotate(c, 11u);
            b = b + bitselect(a, d, c) + w[11]; b = rotate(b, 19u);
            a = a + bitselect(d, c, b) + w[12]; a = rotate(a, 3u);
            d = d + bitselect(c, b, a) + w[13]; d = rotate(d, 7u);
            c = c + bitselect(b, a, d) + w[14]; c = rotate(c, 11u);
            b = b + bitselect(a, d, c) + w[15]; b = rotate(b, 19u);
            a = a + bitselect(b, c, b ^ d) + nkc2_0; a = rotate(a, 3u);
            d = d + bitselect(a, b, a ^ c) + nkc2_1; d = rotate(d, 5u);
            c = c + bitselect(d, a, d ^ b) + nkc2_2; c = rotate(c, 9u);
            b = b + bitselect(c, d, c ^ a) + nkc2_3; b = rotate(b, 13u);
            a = a + bitselect(b, c, b ^ d) + nkc2_4; a = rotate(a, 3u);
            d = d + bitselect(a, b, a ^ c) + nkc2_5; d = rotate(d, 5u);
            c = c + bitselect(d, a, d ^ b) + nkc2_6; c = rotate(c, 9u);
            b = b + bitselect(c, d, c ^ a) + nkc2_7; b = rotate(b, 13u);
            a = a + bitselect(b, c, b ^ d) + nkc2_8; a = rotate(a, 3u);
            d = d + bitselect(a, b, a ^ c) + nkc2_9; d = rotate(d, 5u);
            c = c + bitselect(d, a, d ^ b) + nkc2_10; c = rotate(c, 9u);
            if (pre_c != c) continue;
            b = b + bitselect(c, d, c ^ a) + nkc2_11; b = rotate(b, 13u);
            if (pre_b != b) continue;
            a = a + bitselect(b, c, b ^ d) + nkc2_12; a = rotate(a, 3u);
            if (pre_a != a) continue;
            d = d + bitselect(a, b, a ^ c) + nkc2_13; d = rotate(d, 5u);
            c = c + bitselect(d, a, d ^ b) + nkc2_14; c = rotate(c, 9u);
            b = b + bitselect(c, d, c ^ a) + nkc2_15; b = rotate(b, 13u);
            ht = b ^ c; a = a + (ht ^ d) + nkc3_0; a = rotate(a, 3u);
            d = d + (a ^ ht) + nkc3_1; d = rotate(d, 9u);
            ht = d ^ a; c = c + (ht ^ b) + nkc3_2; c = rotate(c, 11u);
            b = b + (c ^ ht) + nkc3_3; b = rotate(b, 15u);
            ht = b ^ c; a = a + (ht ^ d) + nkc3_4; a = rotate(a, 3u);
            d = d + (a ^ ht) + nkc3_5; d = rotate(d, 9u);
            ht = d ^ a; c = c + (ht ^ b) + nkc3_6; c = rotate(c, 11u);
            b = b + (c ^ ht) + nkc3_7; b = rotate(b, 15u);
            ht = b ^ c; a = a + (ht ^ d) + nkc3_8; a = rotate(a, 3u);
            d = d + (a ^ ht) + nkc3_9; d = rotate(d, 9u);
            ht = d ^ a; c = c + (ht ^ b) + nkc3_10; c = rotate(c, 11u);
            b = b + (c ^ ht) + nkc3_11; b = rotate(b, 15u);
            ht = b ^ c; a = a + (ht ^ d) + nkc3_12; a = rotate(a, 3u);
            d = d + (a ^ ht) + nkc3_13; d = rotate(d, 9u);
            ht = d ^ a; c = c + (ht ^ b) + nkc3_14; c = rotate(c, 11u);
            b = b + (c ^ ht) + nkc3_15; b = rotate(b, 15u);
            if (a == search0 && b == search1 && c == search2 && d == search3)
                record_match(found, found_idx, found_hash_idx, match_count, match_gids, match_hash_idxs, gid * il_cnt + il_pos, 0);
        }
    } else {
        for (uint il_pos = 0; il_pos < il_cnt; il_pos++) {
            // Only update w[0] and w[1] — the rest is pre-computed
            uint v = w0l | words_buf_r[il_pos];
            w[0] = (v & 0xFF) | ((v & 0xFF00) << 8);
            w[1] = ((v >> 16) & 0xFF) | ((v & 0xFF000000) >> 8);
            // Restore padding (in case w[0] or w[1] overlap with pad_word)
            if (pad_word == 0) w[0] |= (0x80u << pad_shift);
            else if (pad_word == 1) w[1] |= (0x80u << pad_shift);
            // Re-compute K+w[0] for rounds 2 and 3 (w[0] changed)
            uint nkc2_0 = 0x5A827999u + w[0];
            uint nkc3_0 = 0x6ED9EBA1u + w[0];
            uint nkc2_4 = 0x5A827999u + w[1];
            uint nkc3_8 = 0x6ED9EBA1u + w[1];
            uint a = H0, b = H1, c = H2, d = H3;
            uint ht; // shared temp for H1/H2 round 3 optimization
            a = a + bitselect(d, c, b) + w[0]; a = rotate(a, 3u);
            d = d + bitselect(c, b, a) + w[1]; d = rotate(d, 7u);
            c = c + bitselect(b, a, d) + w[2]; c = rotate(c, 11u);
            b = b + bitselect(a, d, c) + w[3]; b = rotate(b, 19u);
            a = a + bitselect(d, c, b) + w[4]; a = rotate(a, 3u);
            d = d + bitselect(c, b, a) + w[5]; d = rotate(d, 7u);
            c = c + bitselect(b, a, d) + w[6]; c = rotate(c, 11u);
            b = b + bitselect(a, d, c) + w[7]; b = rotate(b, 19u);
            a = a + bitselect(d, c, b) + w[8]; a = rotate(a, 3u);
            d = d + bitselect(c, b, a) + w[9]; d = rotate(d, 7u);
            c = c + bitselect(b, a, d) + w[10]; c = rotate(c, 11u);
            b = b + bitselect(a, d, c) + w[11]; b = rotate(b, 19u);
            a = a + bitselect(d, c, b) + w[12]; a = rotate(a, 3u);
            d = d + bitselect(c, b, a) + w[13]; d = rotate(d, 7u);
            c = c + bitselect(b, a, d) + w[14]; c = rotate(c, 11u);
            b = b + bitselect(a, d, c) + w[15]; b = rotate(b, 19u);
            a = a + bitselect(b, c, b ^ d) + nkc2_0; a = rotate(a, 3u);
            d = d + bitselect(a, b, a ^ c) + nkc2_1; d = rotate(d, 5u);
            c = c + bitselect(d, a, d ^ b) + nkc2_2; c = rotate(c, 9u);
            b = b + bitselect(c, d, c ^ a) + nkc2_3; b = rotate(b, 13u);
            a = a + bitselect(b, c, b ^ d) + nkc2_4; a = rotate(a, 3u);
            d = d + bitselect(a, b, a ^ c) + nkc2_5; d = rotate(d, 5u);
            c = c + bitselect(d, a, d ^ b) + nkc2_6; c = rotate(c, 9u);
            b = b + bitselect(c, d, c ^ a) + nkc2_7; b = rotate(b, 13u);
            a = a + bitselect(b, c, b ^ d) + nkc2_8; a = rotate(a, 3u);
            d = d + bitselect(a, b, a ^ c) + nkc2_9; d = rotate(d, 5u);
            c = c + bitselect(d, a, d ^ b) + nkc2_10; c = rotate(c, 9u);
            b = b + bitselect(c, d, c ^ a) + nkc2_11; b = rotate(b, 13u);
            a = a + bitselect(b, c, b ^ d) + nkc2_12; a = rotate(a, 3u);
            d = d + bitselect(a, b, a ^ c) + nkc2_13; d = rotate(d, 5u);
            c = c + bitselect(d, a, d ^ b) + nkc2_14; c = rotate(c, 9u);
            b = b + bitselect(c, d, c ^ a) + nkc2_15; b = rotate(b, 13u);
            ht = b ^ c; a = a + (ht ^ d) + nkc3_0; a = rotate(a, 3u);
            d = d + (a ^ ht) + nkc3_1; d = rotate(d, 9u);
            ht = d ^ a; c = c + (ht ^ b) + nkc3_2; c = rotate(c, 11u);
            b = b + (c ^ ht) + nkc3_3; b = rotate(b, 15u);
            ht = b ^ c; a = a + (ht ^ d) + nkc3_4; a = rotate(a, 3u);
            d = d + (a ^ ht) + nkc3_5; d = rotate(d, 9u);
            ht = d ^ a; c = c + (ht ^ b) + nkc3_6; c = rotate(c, 11u);
            b = b + (c ^ ht) + nkc3_7; b = rotate(b, 15u);
            ht = b ^ c; a = a + (ht ^ d) + nkc3_8; a = rotate(a, 3u);
            d = d + (a ^ ht) + nkc3_9; d = rotate(d, 9u);
            ht = d ^ a; c = c + (ht ^ b) + nkc3_10; c = rotate(c, 11u);
            b = b + (c ^ ht) + nkc3_11; b = rotate(b, 15u);
            ht = b ^ c; a = a + (ht ^ d) + nkc3_12; a = rotate(a, 3u);
            d = d + (a ^ ht) + nkc3_13; d = rotate(d, 9u);
            ht = d ^ a; c = c + (ht ^ b) + nkc3_14; c = rotate(c, 11u);
            b = b + (c ^ ht) + nkc3_15; b = rotate(b, 15u);
            uint dg0 = H0+a, dg1 = H1+b, dg2 = H2+c, dg3 = H3+d;
            // Bloom filter: fast reject non-matches in O(1) instead of O(n_hashes)
            if (num_hashes > 1 && bloom_check(bloom_s1, bloom_s2, dg0, dg1, dg2, dg3, 0, 0, 0, 0, 4)) {
                for (uint h = 0; h < num_hashes; h++) {
                    if (dg0 == targets[h*4] && dg1 == targets[h*4+1] && dg2 == targets[h*4+2] && dg3 == targets[h*4+3])
                        record_match(found, found_idx, found_hash_idx, match_count, match_gids, match_hash_idxs, gid * il_cnt + il_pos, h);
                }
            } else if (num_hashes == 1) {
                if (dg0 == targets[0] && dg1 == targets[1] && dg2 == targets[2] && dg3 == targets[3])
                    record_match(found, found_idx, found_hash_idx, match_count, match_gids, match_hash_idxs, gid * il_cnt + il_pos, 0);
            }
        }
    }
}
// === PMK LOOKUP KERNEL ===
// Takes precomputed PMKs directly, computes PMKID/EAPOL, compares.
// Skips PBKDF2 entirely — ~4000x faster than full cracking.
// Used with --pmk-input mode for repeated ESSID targets.
__kernel void crack_pmk_lookup(
    __global const uchar *pmks,
    __global const uchar *ap_macs,
    __global const uchar *sta_macs,
    __global const uint *targets,
    __global int *found,
    __global uint *found_idx,
    __global uint *found_hash_idx,
    __global uint *match_count,
    __global uint *match_gids,
    __global uint *match_hash_idxs,
    const uint batch_size,
    const uint num_hashes,
    __global const uint *hash_types,
    __global const uchar *anonces,
    __global const uchar *eapol_frames,
    __global const uint *eapol_lens,
    __global const uint *keyvers
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;

    uchar pmk[32];
    for (int i = 0; i < 32; i++) pmk[i] = pmks[gid * 32 + i];

    for (uint h = 0; h < num_hashes; h++) {
        uint ht = hash_types[h];
        if (ht >= 2) continue; // skip MD5/NTLM

        uchar ap_mac_local[6];
        uchar sta_mac_local[6];
        for (int i = 0; i < 6; i++) {
            ap_mac_local[i] = ap_macs[h * 6 + i];
            sta_mac_local[i] = sta_macs[h * 6 + i];
        }

        if (ht == 1) {
            // EAPOL mode: PMK → PTK → KCK → MIC
            uint eapol_len = eapol_lens[h];
            if (eapol_len < 97 || eapol_len > 256) continue;
            uint packed = keyvers[h];
            uint keyver = packed & 0xFF;
            if (keyver != 1 && keyver != 2 && keyver != 3) continue;
            uint nec = (packed >> 8) & 0xFF;
            uint det_le = (packed >> 16) & 1;
            uint det_be = (packed >> 17) & 1;
            if (nec > 128) nec = 128;
            uchar anonce_local[32];
            uchar eapol_local[256];
            for (int i = 0; i < 32; i++) anonce_local[i] = anonces[h * 32 + i];
            for (uint i = 0; i < eapol_len; i++) eapol_local[i] = eapol_frames[h * 256 + i];

            uint bo_loops = det_le + det_be;
            if (bo_loops == 0) bo_loops = 2;
            uint anonce_tail =
                ((uint)anonce_local[28] << 24) |
                ((uint)anonce_local[29] << 16) |
                ((uint)anonce_local[30] <<  8) |
                (uint)anonce_local[31];

            __global const uint *target_h = targets + h * 4;

            for (uint nec_iter = 0; nec_iter <= nec; nec_iter++) {
                for (uint bo_pos = 0; bo_pos < bo_loops; bo_pos++) {
                    uint t = anonce_tail;
                    if (bo_loops == 1) {
                        if (det_le == 1) { t -= nec / 2; t += nec_iter; }
                        else { t = MD5_SWAP32(t); t -= nec / 2; t += nec_iter; t = MD5_SWAP32(t); }
                    } else {
                        if (bo_pos == 0) { t -= nec / 2; t += nec_iter; }
                        else { t = MD5_SWAP32(t); t -= nec / 2; t += nec_iter; t = MD5_SWAP32(t); }
                    }
                    anonce_local[28] = (uchar)((t >> 24) & 0xFF);
                    anonce_local[29] = (uchar)((t >> 16) & 0xFF);
                    anonce_local[30] = (uchar)((t >>  8) & 0xFF);
                    anonce_local[31] = (uchar)(t & 0xFF);

                    uint result[4];
                    compute_eapol_mic(pmk, ap_mac_local, sta_mac_local,
                                      anonce_local, eapol_local, eapol_len,
                                      keyver, result);

                    if (result[0] == target_h[0] && result[1] == target_h[1] &&
                        result[2] == target_h[2] && result[3] == target_h[3]) {
                        record_match(found, found_idx, found_hash_idx,
                                    match_count, match_gids, match_hash_idxs, gid, h);
                    }
                }
            }
        } else {
            // PMKID mode: PMK → PMKID
            uint result[4];
            compute_pmkid(pmk, ap_mac_local, sta_mac_local, result);
            if (result[0] == targets[h*4] && result[1] == targets[h*4+1] &&
                result[2] == targets[h*4+2] && result[3] == targets[h*4+3]) {
                record_match(found, found_idx, found_hash_idx,
                            match_count, match_gids, match_hash_idxs, gid, h);
            }
        }
    }
}

// SHA-256 hash a password (wordlist mode).
static inline void sha256_hash_password(const uchar *pw, uint pw_len, uint *out_h) {
    sha256_state_t s;
    sha256_init(&s);
    
    uint offset = 0;
    while (offset + 64 <= pw_len) {
        uint block[16];
        bytes_to_block_sha256(pw + offset, 64, block);
        sha256_process_block(&s, block);
        offset += 64;
    }
    
    uint rem = pw_len - offset;
    ulong bits = (ulong)pw_len * 8;
    if (rem > 55) {
        uint block1[16];
        bytes_to_block_sha256(pw + offset, rem, block1);
        uint pw_w = rem >> 2, pw_b = rem & 3;
        block1[pw_w] |= ((uint)0x80) << (24 - pw_b * 8);
        sha256_process_block(&s, block1);
        
        uint block2[16];
        for (int i = 0; i < 16; i++) block2[i] = 0;
        block2[14] = (uint)(bits >> 32);
        block2[15] = (uint)bits;
        sha256_process_block(&s, block2);
    } else {
        uint block[16];
        bytes_to_block_sha256(pw + offset, rem, block);
        uint pw_w = rem >> 2, pw_b = rem & 3;
        block[pw_w] |= ((uint)0x80) << (24 - pw_b * 8);
        block[14] = (uint)(bits >> 32);
        block[15] = (uint)bits;
        sha256_process_block(&s, block);
    }
    
    for (int i = 0; i < 8; i++) out_h[i] = s.h[i];
}

// SHA-256 wordlist kernel (mode 1400)
__kernel void crack_sha256(
    __global const uchar *passwords,
    __global const uint *pw_lens,
    __global const uint *targets,
    __global int *found,
    __global uint *found_idx,
    __global uint *found_hash_idx,
    __global uint *match_count,
    __global uint *match_gids,
    __global uint *match_hash_idxs,
    const uint batch_size,
    const uint num_hashes
) {
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;
    
    uchar pw[64];
    for (int i = 0; i < 64; i++) pw[i] = passwords[gid * 64 + i];
    uint pw_len = pw_lens[gid];
    if (pw_len > 64) pw_len = 64; // R105 LOW-2: clamp like every sibling kernel —
                                  // a bit-flipped length > 64 would read past pw[]
    
    uint digest[8];
    sha256_hash_password(pw, pw_len, digest);
    
    for (uint h = 0; h < num_hashes; h++) {
        if (digest[0] == targets[h*8]   && digest[1] == targets[h*8+1] &&
            digest[2] == targets[h*8+2] && digest[3] == targets[h*8+3] &&
            digest[4] == targets[h*8+4] && digest[5] == targets[h*8+5] &&
            digest[6] == targets[h*8+6] && digest[7] == targets[h*8+7]) {
            record_match(found, found_idx, found_hash_idx,
                        match_count, match_gids, match_hash_idxs, gid, h);
        }
    }
}

// SHA-256 mask kernel (mode 1400) — no early-exit.
// GPU generates candidates from thread ID + mask structure.
// SHA-256 has recursive message schedule (w[0] propagates to all rounds),
// so early-exit is not feasible. All 64 rounds computed per candidate.
__kernel void crack_sha256_mask(
    __global const uchar *charset, __global const uint *cs_offsets, __global const uint *cs_lengths,
    __constant uint *words_buf_r, const uint il_cnt,
    const ulong start_base_idx, const uint num_expanded_positions, const uint num_positions,
    __constant uint *targets,
    __global int *found, __global uint *found_idx, __global uint *found_hash_idx,
    __global uint *match_count, __global uint *match_gids, __global uint *match_hash_idxs,
    const uint batch_size, const uint num_hashes,
    __constant uint *bloom_s1, __constant uint *bloom_s2)
{
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;

    // Generate base password from thread index — replaces host DMA.
    // Pack as LE u32s, then byte-swap to BE for SHA-256.
    uint w[16];
    for (int i = 0; i < 16; i++) w[i] = 0;
    ulong base_idx = start_base_idx + (ulong)gid;
    for (uint pos = num_expanded_positions; pos < num_positions; pos++) {
        uint cs_len = cs_lengths[pos];
        uint char_idx = (uint)(base_idx % (ulong)cs_len);
        base_idx /= (ulong)cs_len;
        uchar c = charset[cs_offsets[pos] + char_idx];
        w[pos >> 2] |= (uint)c << ((pos & 3) * 8);
    }
    for (int i = 0; i < 16; i++) w[i] = MD5_SWAP32(w[i]);
    uint pw_len = num_positions;
    if (pw_len > 55) return; // single-block SHA-256 limit

    // Only w[0] changes per il_pos
    uint w0l = w[0];

    // Pre-compute padding outside the inner loop
    uint pad_word = pw_len >> 2;
    uint pad_shift = 24 - (pw_len & 3) * 8; // BE: byte 0 = MSB = shift 24
    ulong bits = (ulong)pw_len * 8;

    // Build the fixed block (everything except w[0])
    uint block[16];
    for (int i = 1; i < 16; i++) block[i] = w[i];
    if (pad_word > 0) {
        block[pad_word] = w[pad_word] | (0x80u << pad_shift);
    }
    // For pad_word == 0, the 0x80 is OR'd into block[0] inside the loop
    block[14] = (uint)(bits >> 32);
    block[15] = (uint)bits;

    // SHA-256 IV
    uint iv0 = 0x6a09e667u, iv1 = 0xbb67ae85u;
    uint iv2 = 0x3c6ef372u, iv3 = 0xa54ff53au;
    uint iv4 = 0x510e527fu, iv5 = 0x9b05688cu;
    uint iv6 = 0x1f83d9abu, iv7 = 0x5be0cd19u;

    for (uint il_pos = 0; il_pos < il_cnt; il_pos++) {
        // Only w[0] changes: combine base word with mask character
        // words_buf_r is LE-packed, byte-swap to get BE position
        block[0] = w0l | MD5_SWAP32(words_buf_r[il_pos]);
        if (pad_word == 0) {
            block[0] |= (0x80u << pad_shift); // BE padding in word 0
        }

        // Inline SHA-256 compression (all 64 rounds)
        uint a = iv0, b = iv1, c = iv2, d = iv3;
        uint e = iv4, f = iv5, g = iv6, h = iv7;

        // Working copy of block (message schedule modifies in place)
        uint ws[16];
        for (int i = 0; i < 16; i++) ws[i] = block[i];

        for (int i = 0; i < 64; i++) {
            if (i >= 16) {
                ws[i & 15] += SHA256_SIG0(ws[(i + 1) & 15])
                            + ws[(i + 9) & 15]
                            + SHA256_SIG1(ws[(i + 14) & 15]);
            }
            uint t1 = h + SHA256_EP1(e) + SHA256_CH(e, f, g) + SHA256_K[i] + ws[i & 15];
            uint t2 = SHA256_EP0(a) + SHA256_MA(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        uint d0 = a + iv0, d1 = b + iv1, d2 = c + iv2, d3 = d + iv3;
        uint d4 = e + iv4, d5 = f + iv5, d6 = g + iv6, d7 = h + iv7;

        // Bloom filter: fast reject non-matches in O(1) instead of O(n_hashes)
        if (num_hashes > 1 && bloom_check(bloom_s1, bloom_s2, d0, d1, d2, d3, d4, d5, d6, d7, 8)) {
            for (uint h_idx = 0; h_idx < num_hashes; h_idx++) {
                if (d0 == targets[h_idx * 8]     && d1 == targets[h_idx * 8 + 1] &&
                    d2 == targets[h_idx * 8 + 2] && d3 == targets[h_idx * 8 + 3] &&
                    d4 == targets[h_idx * 8 + 4] && d5 == targets[h_idx * 8 + 5] &&
                    d6 == targets[h_idx * 8 + 6] && d7 == targets[h_idx * 8 + 7]) {
                    record_match(found, found_idx, found_hash_idx,
                                match_count, match_gids, match_hash_idxs,
                                gid * il_cnt + il_pos, h_idx);
                }
            }
        } else if (num_hashes == 1) {
            if (d0 == targets[0] && d1 == targets[1] &&
                d2 == targets[2] && d3 == targets[3] &&
                d4 == targets[4] && d5 == targets[5] &&
                d6 == targets[6] && d7 == targets[7]) {
                record_match(found, found_idx, found_hash_idx,
                            match_count, match_gids, match_hash_idxs,
                            gid * il_cnt + il_pos, 0);
            }
        }
    }
}
// Blowfish constants for bcrypt (from Schneier's reference)

__constant uint c_bf_pbox[18] = {
    0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344,
    0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,
    0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c,
    0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917,
    0x9216d5d9, 0x8979fb1b
};

__constant uint c_bf_sbox0[256] = {
    0xd1310ba6, 0x98dfb5ac, 0x2ffd72db, 0xd01adfb7,
    0xb8e1afed, 0x6a267e96, 0xba7c9045, 0xf12c7f99,
    0x24a19947, 0xb3916cf7, 0x0801f2e2, 0x858efc16,
    0x636920d8, 0x71574e69, 0xa458fea3, 0xf4933d7e,
    0x0d95748f, 0x728eb658, 0x718bcd58, 0x82154aee,
    0x7b54a41d, 0xc25a59b5, 0x9c30d539, 0x2af26013,
    0xc5d1b023, 0x286085f0, 0xca417918, 0xb8db38ef,
    0x8e79dcb0, 0x603a180e, 0x6c9e0e8b, 0xb01e8a3e,
    0xd71577c1, 0xbd314b27, 0x78af2fda, 0x55605c60,
    0xe65525f3, 0xaa55ab94, 0x57489862, 0x63e81440,
    0x55ca396a, 0x2aab10b6, 0xb4cc5c34, 0x1141e8ce,
    0xa15486af, 0x7c72e993, 0xb3ee1411, 0x636fbc2a,
    0x2ba9c55d, 0x741831f6, 0xce5c3e16, 0x9b87931e,
    0xafd6ba33, 0x6c24cf5c, 0x7a325381, 0x28958677,
    0x3b8f4898, 0x6b4bb9af, 0xc4bfe81b, 0x66282193,
    0x61d809cc, 0xfb21a991, 0x487cac60, 0x5dec8032,
    0xef845d5d, 0xe98575b1, 0xdc262302, 0xeb651b88,
    0x23893e81, 0xd396acc5, 0x0f6d6ff3, 0x83f44239,
    0x2e0b4482, 0xa4842004, 0x69c8f04a, 0x9e1f9b5e,
    0x21c66842, 0xf6e96c9a, 0x670c9c61, 0xabd388f0,
    0x6a51a0d2, 0xd8542f68, 0x960fa728, 0xab5133a3,
    0x6eef0b6c, 0x137a3be4, 0xba3bf050, 0x7efb2a98,
    0xa1f1651d, 0x39af0176, 0x66ca593e, 0x82430e88,
    0x8cee8619, 0x456f9fb4, 0x7d84a5c3, 0x3b8b5ebe,
    0xe06f75d8, 0x85c12073, 0x401a449f, 0x56c16aa6,
    0x4ed3aa62, 0x363f7706, 0x1bfedf72, 0x429b023d,
    0x37d0d724, 0xd00a1248, 0xdb0fead3, 0x49f1c09b,
    0x075372c9, 0x80991b7b, 0x25d479d8, 0xf6e8def7,
    0xe3fe501a, 0xb6794c3b, 0x976ce0bd, 0x04c006ba,
    0xc1a94fb6, 0x409f60c4, 0x5e5c9ec2, 0x196a2463,
    0x68fb6faf, 0x3e6c53b5, 0x1339b2eb, 0x3b52ec6f,
    0x6dfc511f, 0x9b30952c, 0xcc814544, 0xaf5ebd09,
    0xbee3d004, 0xde334afd, 0x660f2807, 0x192e4bb3,
    0xc0cba857, 0x45c8740f, 0xd20b5f39, 0xb9d3fbdb,
    0x5579c0bd, 0x1a60320a, 0xd6a100c6, 0x402c7279,
    0x679f25fe, 0xfb1fa3cc, 0x8ea5e9f8, 0xdb3222f8,
    0x3c7516df, 0xfd616b15, 0x2f501ec8, 0xad0552ab,
    0x323db5fa, 0xfd238760, 0x53317b48, 0x3e00df82,
    0x9e5c57bb, 0xca6f8ca0, 0x1a87562e, 0xdf1769db,
    0xd542a8f6, 0x287effc3, 0xac6732c6, 0x8c4f5573,
    0x695b27b0, 0xbbca58c8, 0xe1ffa35d, 0xb8f011a0,
    0x10fa3d98, 0xfd2183b8, 0x4afcb56c, 0x2dd1d35b,
    0x9a53e479, 0xb6f84565, 0xd28e49bc, 0x4bfb9790,
    0xe1ddf2da, 0xa4cb7e33, 0x62fb1341, 0xcee4c6e8,
    0xef20cada, 0x36774c01, 0xd07e9efe, 0x2bf11fb4,
    0x95dbda4d, 0xae909198, 0xeaad8e71, 0x6b93d5a0,
    0xd08ed1d0, 0xafc725e0, 0x8e3c5b2f, 0x8e7594b7,
    0x8ff6e2fb, 0xf2122b64, 0x8888b812, 0x900df01c,
    0x4fad5ea0, 0x688fc31c, 0xd1cff191, 0xb3a8c1ad,
    0x2f2f2218, 0xbe0e1777, 0xea752dfe, 0x8b021fa1,
    0xe5a0cc0f, 0xb56f74e8, 0x18acf3d6, 0xce89e299,
    0xb4a84fe0, 0xfd13e0b7, 0x7cc43b81, 0xd2ada8d9,
    0x165fa266, 0x80957705, 0x93cc7314, 0x211a1477,
    0xe6ad2065, 0x77b5fa86, 0xc75442f5, 0xfb9d35cf,
    0xebcdaf0c, 0x7b3e89a0, 0xd6411bd3, 0xae1e7e49,
    0x00250e2d, 0x2071b35e, 0x226800bb, 0x57b8e0af,
    0x2464369b, 0xf009b91e, 0x5563911d, 0x59dfa6aa,
    0x78c14389, 0xd95a537f, 0x207d5ba2, 0x02e5b9c5,
    0x83260376, 0x6295cfa9, 0x11c81968, 0x4e734a41,
    0xb3472dca, 0x7b14a94a, 0x1b510052, 0x9a532915,
    0xd60f573f, 0xbc9bc6e4, 0x2b60a476, 0x81e67400,
    0x08ba6fb5, 0x571be91f, 0xf296ec6b, 0x2a0dd915,
    0xb6636521, 0xe7b9f9b6, 0xff34052e, 0xc5855664,
    0x53b02d5d, 0xa99f8fa1, 0x08ba4799, 0x6e85076a
};

__constant uint c_bf_sbox1[256] = {
    0x4b7a70e9, 0xb5b32944, 0xdb75092e, 0xc4192623,
    0xad6ea6b0, 0x49a7df7d, 0x9cee60b8, 0x8fedb266,
    0xecaa8c71, 0x699a17ff, 0x5664526c, 0xc2b19ee1,
    0x193602a5, 0x75094c29, 0xa0591340, 0xe4183a3e,
    0x3f54989a, 0x5b429d65, 0x6b8fe4d6, 0x99f73fd6,
    0xa1d29c07, 0xefe830f5, 0x4d2d38e6, 0xf0255dc1,
    0x4cdd2086, 0x8470eb26, 0x6382e9c6, 0x021ecc5e,
    0x09686b3f, 0x3ebaefc9, 0x3c971814, 0x6b6a70a1,
    0x687f3584, 0x52a0e286, 0xb79c5305, 0xaa500737,
    0x3e07841c, 0x7fdeae5c, 0x8e7d44ec, 0x5716f2b8,
    0xb03ada37, 0xf0500c0d, 0xf01c1f04, 0x0200b3ff,
    0xae0cf51a, 0x3cb574b2, 0x25837a58, 0xdc0921bd,
    0xd19113f9, 0x7ca92ff6, 0x94324773, 0x22f54701,
    0x3ae5e581, 0x37c2dadc, 0xc8b57634, 0x9af3dda7,
    0xa9446146, 0x0fd0030e, 0xecc8c73e, 0xa4751e41,
    0xe238cd99, 0x3bea0e2f, 0x3280bba1, 0x183eb331,
    0x4e548b38, 0x4f6db908, 0x6f420d03, 0xf60a04bf,
    0x2cb81290, 0x24977c79, 0x5679b072, 0xbcaf89af,
    0xde9a771f, 0xd9930810, 0xb38bae12, 0xdccf3f2e,
    0x5512721f, 0x2e6b7124, 0x501adde6, 0x9f84cd87,
    0x7a584718, 0x7408da17, 0xbc9f9abc, 0xe94b7d8c,
    0xec7aec3a, 0xdb851dfa, 0x63094366, 0xc464c3d2,
    0xef1c1847, 0x3215d908, 0xdd433b37, 0x24c2ba16,
    0x12a14d43, 0x2a65c451, 0x50940002, 0x133ae4dd,
    0x71dff89e, 0x10314e55, 0x81ac77d6, 0x5f11199b,
    0x043556f1, 0xd7a3c76b, 0x3c11183b, 0x5924a509,
    0xf28fe6ed, 0x97f1fbfa, 0x9ebabf2c, 0x1e153c6e,
    0x86e34570, 0xeae96fb1, 0x860e5e0a, 0x5a3e2ab3,
    0x771fe71c, 0x4e3d06fa, 0x2965dcb9, 0x99e71d0f,
    0x803e89d6, 0x5266c825, 0x2e4cc978, 0x9c10b36a,
    0xc6150eba, 0x94e2ea78, 0xa5fc3c53, 0x1e0a2df4,
    0xf2f74ea7, 0x361d2b3d, 0x1939260f, 0x19c27960,
    0x5223a708, 0xf71312b6, 0xebadfe6e, 0xeac31f66,
    0xe3bc4595, 0xa67bc883, 0xb17f37d1, 0x018cff28,
    0xc332ddef, 0xbe6c5aa5, 0x65582185, 0x68ab9802,
    0xeecea50f, 0xdb2f953b, 0x2aef7dad, 0x5b6e2f84,
    0x1521b628, 0x29076170, 0xecdd4775, 0x619f1510,
    0x13cca830, 0xeb61bd96, 0x0334fe1e, 0xaa0363cf,
    0xb5735c90, 0x4c70a239, 0xd59e9e0b, 0xcbaade14,
    0xeecc86bc, 0x60622ca7, 0x9cab5cab, 0xb2f3846e,
    0x648b1eaf, 0x19bdf0ca, 0xa02369b9, 0x655abb50,
    0x40685a32, 0x3c2ab4b3, 0x319ee9d5, 0xc021b8f7,
    0x9b540b19, 0x875fa099, 0x95f7997e, 0x623d7da8,
    0xf837889a, 0x97e32d77, 0x11ed935f, 0x16681281,
    0x0e358829, 0xc7e61fd6, 0x96dedfa1, 0x7858ba99,
    0x57f584a5, 0x1b227263, 0x9b83c3ff, 0x1ac24696,
    0xcdb30aeb, 0x532e3054, 0x8fd948e4, 0x6dbc3128,
    0x58ebf2ef, 0x34c6ffea, 0xfe28ed61, 0xee7c3c73,
    0x5d4a14d9, 0xe864b7e3, 0x42105d14, 0x203e13e0,
    0x45eee2b6, 0xa3aaabea, 0xdb6c4f15, 0xfacb4fd0,
    0xc742f442, 0xef6abbb5, 0x654f3b1d, 0x41cd2105,
    0xd81e799e, 0x86854dc7, 0xe44b476a, 0x3d816250,
    0xcf62a1f2, 0x5b8d2646, 0xfc8883a0, 0xc1c7b6a3,
    0x7f1524c3, 0x69cb7492, 0x47848a0b, 0x5692b285,
    0x095bbf00, 0xad19489d, 0x1462b174, 0x23820e00,
    0x58428d2a, 0x0c55f5ea, 0x1dadf43e, 0x233f7061,
    0x3372f092, 0x8d937e41, 0xd65fecf1, 0x6c223bdb,
    0x7cde3759, 0xcbee7460, 0x4085f2a7, 0xce77326e,
    0xa6078084, 0x19f8509e, 0xe8efd855, 0x61d99735,
    0xa969a7aa, 0xc50c06c2, 0x5a04abfc, 0x800bcadc,
    0x9e447a2e, 0xc3453484, 0xfdd56705, 0x0e1e9ec9,
    0xdb73dbd3, 0x105588cd, 0x675fda79, 0xe3674340,
    0xc5c43465, 0x713e38d8, 0x3d28f89e, 0xf16dff20,
    0x153e21e7, 0x8fb03d4a, 0xe6e39f2b, 0xdb83adf7
};

__constant uint c_bf_sbox2[256] = {
    0xe93d5a68, 0x948140f7, 0xf64c261c, 0x94692934,
    0x411520f7, 0x7602d4f7, 0xbcf46b2e, 0xd4a20068,
    0xd4082471, 0x3320f46a, 0x43b7d4b7, 0x500061af,
    0x1e39f62e, 0x97244546, 0x14214f74, 0xbf8b8840,
    0x4d95fc1d, 0x96b591af, 0x70f4ddd3, 0x66a02f45,
    0xbfbc09ec, 0x03bd9785, 0x7fac6dd0, 0x31cb8504,
    0x96eb27b3, 0x55fd3941, 0xda2547e6, 0xabca0a9a,
    0x28507825, 0x530429f4, 0x0a2c86da, 0xe9b66dfb,
    0x68dc1462, 0xd7486900, 0x680ec0a4, 0x27a18dee,
    0x4f3ffea2, 0xe887ad8c, 0xb58ce006, 0x7af4d6b6,
    0xaace1e7c, 0xd3375fec, 0xce78a399, 0x406b2a42,
    0x20fe9e35, 0xd9f385b9, 0xee39d7ab, 0x3b124e8b,
    0x1dc9faf7, 0x4b6d1856, 0x26a36631, 0xeae397b2,
    0x3a6efa74, 0xdd5b4332, 0x6841e7f7, 0xca7820fb,
    0xfb0af54e, 0xd8feb397, 0x454056ac, 0xba489527,
    0x55533a3a, 0x20838d87, 0xfe6ba9b7, 0xd096954b,
    0x55a867bc, 0xa1159a58, 0xcca92963, 0x99e1db33,
    0xa62a4a56, 0x3f3125f9, 0x5ef47e1c, 0x9029317c,
    0xfdf8e802, 0x04272f70, 0x80bb155c, 0x05282ce3,
    0x95c11548, 0xe4c66d22, 0x48c1133f, 0xc70f86dc,
    0x07f9c9ee, 0x41041f0f, 0x404779a4, 0x5d886e17,
    0x325f51eb, 0xd59bc0d1, 0xf2bcc18f, 0x41113564,
    0x257b7834, 0x602a9c60, 0xdff8e8a3, 0x1f636c1b,
    0x0e12b4c2, 0x02e1329e, 0xaf664fd1, 0xcad18115,
    0x6b2395e0, 0x333e92e1, 0x3b240b62, 0xeebeb922,
    0x85b2a20e, 0xe6ba0d99, 0xde720c8c, 0x2da2f728,
    0xd0127845, 0x95b794fd, 0x647d0862, 0xe7ccf5f0,
    0x5449a36f, 0x877d48fa, 0xc39dfd27, 0xf33e8d1e,
    0x0a476341, 0x992eff74, 0x3a6f6eab, 0xf4f8fd37,
    0xa812dc60, 0xa1ebddf8, 0x991be14c, 0xdb6e6b0d,
    0xc67b5510, 0x6d672c37, 0x2765d43b, 0xdcd0e804,
    0xf1290dc7, 0xcc00ffa3, 0xb5390f92, 0x690fed0b,
    0x667b9ffb, 0xcedb7d9c, 0xa091cf0b, 0xd9155ea3,
    0xbb132f88, 0x515bad24, 0x7b9479bf, 0x763bd6eb,
    0x37392eb3, 0xcc115979, 0x8026e297, 0xf42e312d,
    0x6842ada7, 0xc66a2b3b, 0x12754ccc, 0x782ef11c,
    0x6a124237, 0xb79251e7, 0x06a1bbe6, 0x4bfb6350,
    0x1a6b1018, 0x11caedfa, 0x3d25bdd8, 0xe2e1c3c9,
    0x44421659, 0x0a121386, 0xd90cec6e, 0xd5abea2a,
    0x64af674e, 0xda86a85f, 0xbebfe988, 0x64e4c3fe,
    0x9dbc8057, 0xf0f7c086, 0x60787bf8, 0x6003604d,
    0xd1fd8346, 0xf6381fb0, 0x7745ae04, 0xd736fccc,
    0x83426b33, 0xf01eab71, 0xb0804187, 0x3c005e5f,
    0x77a057be, 0xbde8ae24, 0x55464299, 0xbf582e61,
    0x4e58f48f, 0xf2ddfda2, 0xf474ef38, 0x8789bdc2,
    0x5366f9c3, 0xc8b38e74, 0xb475f255, 0x46fcd9b9,
    0x7aeb2661, 0x8b1ddf84, 0x846a0e79, 0x915f95e2,
    0x466e598e, 0x20b45770, 0x8cd55591, 0xc902de4c,
    0xb90bace1, 0xbb8205d0, 0x11a86248, 0x7574a99e,
    0xb77f19b6, 0xe0a9dc09, 0x662d09a1, 0xc4324633,
    0xe85a1f02, 0x09f0be8c, 0x4a99a025, 0x1d6efe10,
    0x1ab93d1d, 0x0ba5a4df, 0xa186f20f, 0x2868f169,
    0xdcb7da83, 0x573906fe, 0xa1e2ce9b, 0x4fcd7f52,
    0x50115e01, 0xa70683fa, 0xa002b5c4, 0x0de6d027,
    0x9af88c27, 0x773f8641, 0xc3604c06, 0x61a806b5,
    0xf0177a28, 0xc0f586e0, 0x006058aa, 0x30dc7d62,
    0x11e69ed7, 0x2338ea63, 0x53c2dd94, 0xc2c21634,
    0xbbcbee56, 0x90bcb6de, 0xebfc7da1, 0xce591d76,
    0x6f05e409, 0x4b7c0188, 0x39720a3d, 0x7c927c24,
    0x86e3725f, 0x724d9db9, 0x1ac15bb4, 0xd39eb8fc,
    0xed545578, 0x08fca5b5, 0xd83d7cd3, 0x4dad0fc4,
    0x1e50ef5e, 0xb161e6f8, 0xa28514d9, 0x6c51133c,
    0x6fd5c7e7, 0x56e14ec4, 0x362abfce, 0xddc6c837,
    0xd79a3234, 0x92638212, 0x670efa8e, 0x406000e0
};

__constant uint c_bf_sbox3[256] = {
    0x3a39ce37, 0xd3faf5cf, 0xabc27737, 0x5ac52d1b,
    0x5cb0679e, 0x4fa33742, 0xd3822740, 0x99bc9bbe,
    0xd5118e9d, 0xbf0f7315, 0xd62d1c7e, 0xc700c47b,
    0xb78c1b6b, 0x21a19045, 0xb26eb1be, 0x6a366eb4,
    0x5748ab2f, 0xbc946e79, 0xc6a376d2, 0x6549c2c8,
    0x530ff8ee, 0x468dde7d, 0xd5730a1d, 0x4cd04dc6,
    0x2939bbdb, 0xa9ba4650, 0xac9526e8, 0xbe5ee304,
    0xa1fad5f0, 0x6a2d519a, 0x63ef8ce2, 0x9a86ee22,
    0xc089c2b8, 0x43242ef6, 0xa51e03aa, 0x9cf2d0a4,
    0x83c061ba, 0x9be96a4d, 0x8fe51550, 0xba645bd6,
    0x2826a2f9, 0xa73a3ae1, 0x4ba99586, 0xef5562e9,
    0xc72fefd3, 0xf752f7da, 0x3f046f69, 0x77fa0a59,
    0x80e4a915, 0x87b08601, 0x9b09e6ad, 0x3b3ee593,
    0xe990fd5a, 0x9e34d797, 0x2cf0b7d9, 0x022b8b51,
    0x96d5ac3a, 0x017da67d, 0xd1cf3ed6, 0x7c7d2d28,
    0x1f9f25cf, 0xadf2b89b, 0x5ad6b472, 0x5a88f54c,
    0xe029ac71, 0xe019a5e6, 0x47b0acfd, 0xed93fa9b,
    0xe8d3c48d, 0x283b57cc, 0xf8d56629, 0x79132e28,
    0x785f0191, 0xed756055, 0xf7960e44, 0xe3d35e8c,
    0x15056dd4, 0x88f46dba, 0x03a16125, 0x0564f0bd,
    0xc3eb9e15, 0x3c9057a2, 0x97271aec, 0xa93a072a,
    0x1b3f6d9b, 0x1e6321f5, 0xf59c66fb, 0x26dcf319,
    0x7533d928, 0xb155fdf5, 0x03563482, 0x8aba3cbb,
    0x28517711, 0xc20ad9f8, 0xabcc5167, 0xccad925f,
    0x4de81751, 0x3830dc8e, 0x379d5862, 0x9320f991,
    0xea7a90c2, 0xfb3e7bce, 0x5121ce64, 0x774fbe32,
    0xa8b6e37e, 0xc3293d46, 0x48de5369, 0x6413e680,
    0xa2ae0810, 0xdd6db224, 0x69852dfd, 0x09072166,
    0xb39a460a, 0x6445c0dd, 0x586cdecf, 0x1c20c8ae,
    0x5bbef7dd, 0x1b588d40, 0xccd2017f, 0x6bb4e3bb,
    0xdda26a7e, 0x3a59ff45, 0x3e350a44, 0xbcb4cdd5,
    0x72eacea8, 0xfa6484bb, 0x8d6612ae, 0xbf3c6f47,
    0xd29be463, 0x542f5d9e, 0xaec2771b, 0xf64e6370,
    0x740e0d8d, 0xe75b1357, 0xf8721671, 0xaf537d5d,
    0x4040cb08, 0x4eb4e2cc, 0x34d2466a, 0x0115af84,
    0xe1b00428, 0x95983a1d, 0x06b89fb4, 0xce6ea048,
    0x6f3f3b82, 0x3520ab82, 0x011a1d4b, 0x277227f8,
    0x611560b1, 0xe7933fdc, 0xbb3a792b, 0x344525bd,
    0xa08839e1, 0x51ce794b, 0x2f32c9b7, 0xa01fbac9,
    0xe01cc87e, 0xbcc7d1f6, 0xcf0111c3, 0xa1e8aac7,
    0x1a908749, 0xd44fbd9a, 0xd0dadecb, 0xd50ada38,
    0x0339c32a, 0xc6913667, 0x8df9317c, 0xe0b12b4f,
    0xf79e59b7, 0x43f5bb3a, 0xf2d519ff, 0x27d9459c,
    0xbf97222c, 0x15e6fc2a, 0x0f91fc71, 0x9b941525,
    0xfae59361, 0xceb69ceb, 0xc2a86459, 0x12baa8d1,
    0xb6c1075e, 0xe3056a0c, 0x10d25065, 0xcb03a442,
    0xe0ec6e0e, 0x1698db3b, 0x4c98a0be, 0x3278e964,
    0x9f1f9532, 0xe0d392df, 0xd3a0342b, 0x8971f21e,
    0x1b0a7441, 0x4ba3348c, 0xc5be7120, 0xc37632d8,
    0xdf359f8d, 0x9b992f2e, 0xe60b6f47, 0x0fe3f11d,
    0xe54cda54, 0x1edad891, 0xce6279cf, 0xcd3e7e6f,
    0x1618b166, 0xfd2c1d05, 0x848fd2c5, 0xf6fb2299,
    0xf523f357, 0xa6327623, 0x93a83531, 0x56cccd02,
    0xacf08162, 0x5a75ebb5, 0x6e163697, 0x88d273cc,
    0xde966292, 0x81b949d0, 0x4c50901b, 0x71c65614,
    0xe6c6c7bd, 0x327a140a, 0x45e1d006, 0xc3f27b9a,
    0xc9aa53fd, 0x62a80f00, 0xbb25bfe2, 0x35bdd2f6,
    0x71126905, 0xb2040222, 0xb6cbcf7c, 0xcd769c2b,
    0x53113ec0, 0x1640e3d3, 0x38abbd60, 0x2547adf0,
    0xba38209c, 0xf746ce76, 0x77afa1c5, 0x20756060,
    0x85cbfe4e, 0x8ae88dd8, 0x7aaaf9b0, 0x4cf9aa7e,
    0x1948c25c, 0x02fb8a8c, 0x01c36ae4, 0xd6ebe1f9,
    0x90d4f869, 0xa65cdea0, 0x3f09252d, 0xc208e69f,
    0xb74e6132, 0xce77e25b, 0x578fdfe3, 0x3ac372e6
};




// --- Blowfish macros ---
// Optimized BF_ROUND: explicitly separate the 4 LDS reads from the ALU chain
// to help the compiler issue all 4 reads concurrently (Mesa NIR may not
// reorder pointer reads through += dependencies without this hint).
// Bank-conflict swizzle: XOR lane ID into index bits so each work-item's
// S-box reads land in different banks. Since 256 % 32 == 0, without swizzle
// all lanes reading the same byte value hit the same bank (2-4 way conflict).
// The swizzle is transparent: same value stored/read, just at a different
// offset within the work-item's 256-entry slice.
#define BF_SBOX_IDX(idx, lid) ((idx) ^ (lid))

#define BF_ROUND(L, R, N) {             \
    uint _a0 = ((L) >> 24) & 0xff;       \
    uint _a1 = ((L) >> 16) & 0xff;       \
    uint _a2 = ((L) >> 8)  & 0xff;       \
    uint _a3 = (L)         & 0xff;       \
    uint _s0v = S0[BF_SBOX_IDX(_a0, _lid)]; \
    uint _s1v = S1[BF_SBOX_IDX(_a1, _lid)]; \
    uint _s2v = S2[BF_SBOX_IDX(_a2, _lid)]; \
    uint _s3v = S3[BF_SBOX_IDX(_a3, _lid)]; \
    uint _tmp = _s0v;                   \
    _tmp += _s1v;                        \
    _tmp ^= _s2v;                        \
    _tmp += _s3v;                        \
    R ^= _tmp ^ P[(N)];                  \
}

#define BF_ENCRYPT(L, R) {     L ^= P[0];     BF_ROUND(L, R,  1);     BF_ROUND(R, L,  2);     BF_ROUND(L, R,  3);     BF_ROUND(R, L,  4);     BF_ROUND(L, R,  5);     BF_ROUND(R, L,  6);     BF_ROUND(L, R,  7);     BF_ROUND(R, L,  8);     BF_ROUND(L, R,  9);     BF_ROUND(R, L, 10);     BF_ROUND(L, R, 11);     BF_ROUND(R, L, 12);     BF_ROUND(L, R, 13);     BF_ROUND(R, L, 14);     BF_ROUND(L, R, 15);     BF_ROUND(R, L, 16);     uint bf_tmp = R; R = L; L = bf_tmp;     L ^= P[17]; }

// stream2word: read 4 bytes as big-endian u32 from byte stream with wrapping
// Matches OpenBSD Blowfish_stream2word
uint stream2word(__global const uchar *data, const uint databytes, __private uint *current)
{
    uint temp = 0;
    uint j = *current;
    for (int i = 0; i < 4; i++, j++) {
        if (j >= databytes) j = 0;
        temp = (temp << 8) | (uint)data[j];
    }
    *current = j;
    return temp;
}

// stream2word for private data (salt in private memory)
uint stream2word_p(__private const uchar *data, const uint databytes, __private uint *current)
{
    uint temp = 0;
    uint j = *current;
    for (int i = 0; i < 4; i++, j++) {
        if (j >= databytes) j = 0;
        temp = (temp << 8) | (uint)data[j];
    }
    *current = j;
    return temp;
}

// Optimized version: takes precomputed word arrays instead of byte streams.
// Eliminates 1042 branchy stream2word_p calls per expand0state by precomputing
// the cyclic key/salt words once per work-item (byte-level wrapping semantics
// are pinned by test_stream2word_byte_level_wrapping in main.rs).
// key_w: precomputed big-endian u32 words from the key (byte-level wrap)
// keycnt: number of words before the wrap resets _j (18 for the P XOR phase)
#define BF_EXPAND0STATE_W(key_w, keycnt, P, S0, S1, S2, S3) { \
    uint _j = 0; \
    for (int _i = 0; _i < 18; _i++) { \
        P[_i] ^= key_w[_j]; \
        _j++; if (_j >= keycnt) _j = 0; \
    } \
    uint L = 0, R = 0; \
    for (int _i = 0; _i < 18; _i += 2) { \
        BF_ENCRYPT(L, R); \
        P[_i] = L; P[_i + 1] = R; \
    } \
    for (int _i = 0; _i < 256; _i += 2) { BF_ENCRYPT(L, R); S0[BF_SBOX_IDX(_i, _lid)] = L; S0[BF_SBOX_IDX(_i+1, _lid)] = R; } \
    for (int _i = 0; _i < 256; _i += 2) { BF_ENCRYPT(L, R); S1[BF_SBOX_IDX(_i, _lid)] = L; S1[BF_SBOX_IDX(_i+1, _lid)] = R; } \
    for (int _i = 0; _i < 256; _i += 2) { BF_ENCRYPT(L, R); S2[BF_SBOX_IDX(_i, _lid)] = L; S2[BF_SBOX_IDX(_i+1, _lid)] = R; } \
    for (int _i = 0; _i < 256; _i += 2) { BF_ENCRYPT(L, R); S3[BF_SBOX_IDX(_i, _lid)] = L; S3[BF_SBOX_IDX(_i+1, _lid)] = R; } \
}

// BF_EXPANDSTATE: inlined bf_expandstate with bank-conflict swizzle on S-box writes.
// XOR P with key (global), then encrypt P+S with salt XOR (private).
// S-box writes use BF_SBOX_IDX so local storage is swizzled.
#define BF_EXPANDSTATE(salt, saltbytes, key, keybytes, P, S0, S1, S2, S3) { \
    uint _j = 0; \
    for (int _i = 0; _i < 18; _i++) { \
        P[_i] ^= stream2word(key, keybytes, &_j); \
    } \
    uint _k = 0; \
    uint L = 0, R = 0; \
    for (int _i = 0; _i < 18; _i += 2) { \
        L ^= stream2word_p(salt, saltbytes, &_k); \
        R ^= stream2word_p(salt, saltbytes, &_k); \
        BF_ENCRYPT(L, R); \
        P[_i] = L; P[_i + 1] = R; \
    } \
    for (int _i = 0; _i < 256; _i += 2) { \
        L ^= stream2word_p(salt, saltbytes, &_k); \
        R ^= stream2word_p(salt, saltbytes, &_k); \
        BF_ENCRYPT(L, R); S0[BF_SBOX_IDX(_i, _lid)] = L; S0[BF_SBOX_IDX(_i+1, _lid)] = R; \
    } \
    for (int _i = 0; _i < 256; _i += 2) { \
        L ^= stream2word_p(salt, saltbytes, &_k); \
        R ^= stream2word_p(salt, saltbytes, &_k); \
        BF_ENCRYPT(L, R); S1[BF_SBOX_IDX(_i, _lid)] = L; S1[BF_SBOX_IDX(_i+1, _lid)] = R; \
    } \
    for (int _i = 0; _i < 256; _i += 2) { \
        L ^= stream2word_p(salt, saltbytes, &_k); \
        R ^= stream2word_p(salt, saltbytes, &_k); \
        BF_ENCRYPT(L, R); S2[BF_SBOX_IDX(_i, _lid)] = L; S2[BF_SBOX_IDX(_i+1, _lid)] = R; \
    } \
    for (int _i = 0; _i < 256; _i += 2) { \
        L ^= stream2word_p(salt, saltbytes, &_k); \
        R ^= stream2word_p(salt, saltbytes, &_k); \
        BF_ENCRYPT(L, R); S3[BF_SBOX_IDX(_i, _lid)] = L; S3[BF_SBOX_IDX(_i+1, _lid)] = R; \
    } \
}

// --- bcrypt kernels ---
// State per work-item: P[18] + S0[256] + S1[256] + S2[256] + S3[256] + pw[18] + pwlen[1] = 1061 u32s
#define BF_STATE_UINTS 1061
#define BF_PW_OFFSET 1042   // offset of pw_bytes in state
#define BF_PWLEN_OFFSET 1060 // offset of pw_len in state

// BF_LWS: bcrypt local work size (must match the lws passed at dispatch).
// Default 4 (optimal for Vega 3 GPU). Override with -DBF_LWS=N at compile time
// for CPU (PoCL) where larger lws (16-64) enables SIMD vectorization.
#ifndef BF_LWS
#define BF_LWS 4
#endif

// Magic string "OrpheanBeholderScryDoubt" = 24 bytes = 6 u32s (big-endian via stream2word)
__constant uchar c_bcrypt_magic[24] = {
    'O','r','p','h','e','a','n','B','e','h','o','l',
    'd','e','r','S','c','r','y','D','o','u','b','t'
};

// Kernel 1: Initialize Blowfish state from password + salt
__kernel void crack_bcrypt_init(
    __global const uchar *pws, __global const uint *pw_lens,
    __constant uchar *salt_buf, const uint salt_len,
    __global uint *state,
    const uint batch_size)
{
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;
    
    uint pw_len = pw_lens[gid];
    __global const uchar *pw = &pws[gid * 64]; // PW_SLOT_LEN = 64
    
    // Load salt to private memory (16 bytes)
    uchar salt[16];
    for (int i = 0; i < 16; i++) salt[i] = salt_buf[i];
    
    // Allocate S-boxes in __local memory
    uint _lid = get_local_id(0);  // _lid used by BF_ROUND bank-conflict swizzle
    __local uint l_sbox0[256 * BF_LWS];
    __local uint l_sbox1[256 * BF_LWS];
    __local uint l_sbox2[256 * BF_LWS];
    __local uint l_sbox3[256 * BF_LWS];
    __local uint *S0 = &l_sbox0[_lid * 256];
    __local uint *S1 = &l_sbox1[_lid * 256];
    __local uint *S2 = &l_sbox2[_lid * 256];
    __local uint *S3 = &l_sbox3[_lid * 256];
    
    // InitState: P and S from pi constants (swizzled for bank conflict avoidance)
    uint P[18];
    for (int i = 0; i < 18; i++) P[i] = c_bf_pbox[i];
    for (int i = 0; i < 256; i++) { S0[BF_SBOX_IDX(i, _lid)] = c_bf_sbox0[i]; S1[BF_SBOX_IDX(i, _lid)] = c_bf_sbox1[i]; S2[BF_SBOX_IDX(i, _lid)] = c_bf_sbox2[i]; S3[BF_SBOX_IDX(i, _lid)] = c_bf_sbox3[i]; }
    
    // bcrypt includes the null terminator in the key: key_len = strlen + 1
    uint bcrypt_key_len = pw_len + 1;
    if (bcrypt_key_len > 72) bcrypt_key_len = 72;
    
    // ExpandState(salt, key): init with salt + password (key includes null terminator)
    // Inlined as macro for bank-conflict swizzle on __local S-box writes
    BF_EXPANDSTATE(salt, 16, pw, bcrypt_key_len, P, S0, S1, S2, S3);
    
    // Store state to global memory (global is ALWAYS sequential/un-swizzled)
    __global uint *s = &state[gid * BF_STATE_UINTS];
    for (int i = 0; i < 18; i++) s[i] = P[i];
    for (int i = 0; i < 256; i++) s[18 + i] = S0[BF_SBOX_IDX(i, _lid)];
    for (int i = 0; i < 256; i++) s[274 + i] = S1[BF_SBOX_IDX(i, _lid)];
    for (int i = 0; i < 256; i++) s[530 + i] = S2[BF_SBOX_IDX(i, _lid)];
    for (int i = 0; i < 256; i++) s[786 + i] = S3[BF_SBOX_IDX(i, _lid)];
    
    // Store password bytes (as u32s) and pw_len for loop kernel
    // bcrypt_key_len already includes null terminator (computed above)
    // Copy bytes into state u32 array (72 bytes = 18 u32s)
    for (int i = 0; i < 18; i++) {
        uint val = 0;
        for (int b = 0; b < 4; b++) {
            int byte_idx = i * 4 + b;
            if (byte_idx < (int)bcrypt_key_len) {
                // pw has the password bytes, pw[bcrypt_key_len-1] is 0 (null from buffer)
                val = (val << 8) | (uint)pw[byte_idx];
            } else {
                val = val << 8;
            }
        }
        s[BF_PW_OFFSET + i] = val;
    }
    s[BF_PWLEN_OFFSET] = bcrypt_key_len;
}

// Kernel 2: Run bcrypt iterations (chunked via loop_cnt)
// Each iteration: expand0state(key) + expand0state(salt)
__kernel void crack_bcrypt_loop(
    __global uint *state,
    __global const uint *pw_lens,  // unused
    __constant uchar *salt_buf, const uint salt_len,
    const uint batch_size,
    const uint loop_cnt)
{
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;
    
    __global uint *s = &state[gid * BF_STATE_UINTS];
    
    // Load salt to private memory
    uchar salt[16];
    for (int i = 0; i < 16; i++) salt[i] = salt_buf[i];
    
    // Load password from state (stored as big-endian u32s)
    uint pw_len = s[BF_PWLEN_OFFSET];
    // Extract bytes from u32 array (big-endian packing)
    uchar pw[72];
    for (int i = 0; i < 18; i++) {
        uint val = s[BF_PW_OFFSET + i];
        pw[i * 4 + 0] = (uchar)((val >> 24) & 0xff);
        pw[i * 4 + 1] = (uchar)((val >> 16) & 0xff);
        pw[i * 4 + 2] = (uchar)((val >> 8) & 0xff);
        pw[i * 4 + 3] = (uchar)(val & 0xff);
    }
    
    // Load P + S into __local memory
    uint _lid = get_local_id(0);  // _lid used by BF_ROUND bank-conflict swizzle
    __local uint l_sbox0[256 * BF_LWS];
    __local uint l_sbox1[256 * BF_LWS];
    __local uint l_sbox2[256 * BF_LWS];
    __local uint l_sbox3[256 * BF_LWS];
    __local uint *S0 = &l_sbox0[_lid * 256];
    __local uint *S1 = &l_sbox1[_lid * 256];
    __local uint *S2 = &l_sbox2[_lid * 256];
    __local uint *S3 = &l_sbox3[_lid * 256];
    
    uint P[18];
    for (int i = 0; i < 18; i++) P[i] = s[i];
    // Load from global (sequential) to local (swizzled)
    for (int i = 0; i < 256; i++) S0[BF_SBOX_IDX(i, _lid)] = s[18 + i];
    for (int i = 0; i < 256; i++) S1[BF_SBOX_IDX(i, _lid)] = s[274 + i];
    for (int i = 0; i < 256; i++) S2[BF_SBOX_IDX(i, _lid)] = s[530 + i];
    for (int i = 0; i < 256; i++) S3[BF_SBOX_IDX(i, _lid)] = s[786 + i];
    
    // Precompute cyclic key and salt word arrays once per work-item.
    // This eliminates 1042 branchy stream2word_p calls per expand0state
    // (CC analysis: ~5% ALU reduction in the loop kernel hot path).
    // IMPORTANT: stream2word_p wraps at the BYTE level (period = keybytes),
    // not the word level. We must reproduce the exact same byte-level wrapping
    // to get correct hash values. Salt is always 16 bytes = 4 words (no wrapping).
    uint key_w[18];  // precomputed words matching stream2word_p output
    {
        uint j = 0;
        for (int w = 0; w < 18; w++) {
            uint word = 0;
            for (int b = 0; b < 4; b++) {
                if (j >= pw_len) j = 0;
                word = (word << 8) | (uint)pw[j];
                j++;
            }
            key_w[w] = word;
        }
    }
    uint keycnt = 18;  // stream2word_p always produces 18 words for P XOR
    
    // Precompute salt words (always 16 bytes = 4 u32 words, big-endian)
    uint salt_w[4];
    for (int i = 0; i < 4; i++) {
        salt_w[i] = ((uint)salt[i * 4] << 24) | ((uint)salt[i * 4 + 1] << 16) |
                    ((uint)salt[i * 4 + 2] << 8) | (uint)salt[i * 4 + 3];
    }
    
    // Run loop_cnt iterations: expand0state(key) + expand0state(salt)
    for (uint iter = 0; iter < loop_cnt; iter++) {
        BF_EXPAND0STATE_W(key_w, keycnt, P, S0, S1, S2, S3);
        BF_EXPAND0STATE_W(salt_w, 4, P, S0, S1, S2, S3);
    }
    
    // Store P + S back to global (global is ALWAYS sequential/un-swizzled)
    for (int i = 0; i < 18; i++) s[i] = P[i];
    for (int i = 0; i < 256; i++) s[18 + i] = S0[BF_SBOX_IDX(i, _lid)];
    for (int i = 0; i < 256; i++) s[274 + i] = S1[BF_SBOX_IDX(i, _lid)];
    for (int i = 0; i < 256; i++) s[530 + i] = S2[BF_SBOX_IDX(i, _lid)];
    for (int i = 0; i < 256; i++) s[786 + i] = S3[BF_SBOX_IDX(i, _lid)];
}

// Kernel 3: Encrypt magic string, compare digest
__kernel void crack_bcrypt_comp(
    __global uint *state,
    __constant uint *targets,
    __global int *found, __global uint *found_idx, __global uint *found_hash_idx,
    __global uint *match_count, __global uint *match_gids, __global uint *match_hash_idxs,
    const uint batch_size, const uint num_hashes)
{
    uint gid = get_global_id(0);
    if (gid >= batch_size) return;
    
    __global uint *s = &state[gid * BF_STATE_UINTS];
    
    // Load P + S into __local memory
    uint _lid = get_local_id(0);  // _lid used by BF_ROUND bank-conflict swizzle
    __local uint l_sbox0[256 * BF_LWS];
    __local uint l_sbox1[256 * BF_LWS];
    __local uint l_sbox2[256 * BF_LWS];
    __local uint l_sbox3[256 * BF_LWS];
    __local uint *S0 = &l_sbox0[_lid * 256];
    __local uint *S1 = &l_sbox1[_lid * 256];
    __local uint *S2 = &l_sbox2[_lid * 256];
    __local uint *S3 = &l_sbox3[_lid * 256];
    
    uint P[18];
    for (int i = 0; i < 18; i++) P[i] = s[i];
    // Load from global (sequential) to local (swizzled)
    for (int i = 0; i < 256; i++) S0[BF_SBOX_IDX(i, _lid)] = s[18 + i];
    for (int i = 0; i < 256; i++) S1[BF_SBOX_IDX(i, _lid)] = s[274 + i];
    for (int i = 0; i < 256; i++) S2[BF_SBOX_IDX(i, _lid)] = s[530 + i];
    for (int i = 0; i < 256; i++) S3[BF_SBOX_IDX(i, _lid)] = s[786 + i];
    
    // Convert magic string to 6 u32s via stream2word (big-endian, wrapping)
    uint j = 0;
    uint cdata[6];
    for (int i = 0; i < 6; i++) {
        cdata[i] = 0;
        for (int b = 0; b < 4; b++, j++) {
            if (j >= 24) j = 0;
            cdata[i] = (cdata[i] << 8) | (uint)c_bcrypt_magic[j];
        }
    }
    
    // Encrypt 64 times (3 blocks per blf_enc = 6 words total per call)
    for (int k = 0; k < 64; k++) {
        // blf_enc: encrypt 3 blocks (6 words)
        for (int blk = 0; blk < 3; blk++) {
            uint L = cdata[blk * 2];
            uint R = cdata[blk * 2 + 1];
            BF_ENCRYPT(L, R);
            cdata[blk * 2] = L;
            cdata[blk * 2 + 1] = R;
        }
    }
    
    // Convert u32s back to bytes (big-endian) for comparison
    // The target is stored as 6 big-endian u32s
    // cdata[5] has only 1 valid byte (23rd), mask with 0xffffff00
    cdata[5] &= 0xffffff00u;
    
    // Compare against targets (6 u32s per hash, big-endian)
    for (uint h = 0; h < num_hashes; h++) {
        if (cdata[0] == targets[h * 6] &&
            cdata[1] == targets[h * 6 + 1] &&
            cdata[2] == targets[h * 6 + 2] &&
            cdata[3] == targets[h * 6 + 3] &&
            cdata[4] == targets[h * 6 + 4] &&
            cdata[5] == targets[h * 6 + 5]) {
            record_match(found, found_idx, found_hash_idx,
                        match_count, match_gids, match_hash_idxs,
                        gid, h);
        }
    }
}
