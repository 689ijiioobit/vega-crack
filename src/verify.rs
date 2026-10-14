#![allow(
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::needless_range_loop,
    clippy::manual_rotate,
    clippy::too_many_lines
)]
//! Host-side PMKID verification — recomputes PMKID for a candidate password
//! and compares against the target. Used to verify GPU-reported matches
//! before declaring success, guarding against GPU bit flips / memory corruption.
//! Only runs once per found match (not in the hot loop), so simplicity > speed.
//
// clippy: this is a compact crypto implementation using single-char variable
// names (a,b,c,d,e,h,k,w) which is idiomatic for SHA1.

// SHA1 constants
const SHA1_K: [u32; 4] = [0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xCA62C1D6];

// === MD5 constants (for EAPOL keyver 1 host verification) ===

// MD5 sine table: T[i] = floor(2^32 * abs(sin(i+1)))
const MD5_T: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

// MD5 shift amounts per round
const MD5_S: [u32; 64] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9,
    14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6, 10, 15,
    21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
];

#[inline]
fn rotl(x: u32, n: u32) -> u32 {
    (x << n) | (x >> (32 - n))
}

#[allow(clippy::similar_names)]
fn sha1_compress(h: &mut [u32; 5], block: &[u32; 16]) {
    let mut w = [0u32; 80];
    w[..16].copy_from_slice(block);
    for i in 16..80 {
        w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    }

    let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);

    for i in 0..80 {
        let (f, k) = match i {
            0..=19 => ((b & c) | (!b & d), SHA1_K[0]),
            20..=39 => (b ^ c ^ d, SHA1_K[1]),
            40..=59 => ((b & c) | (b & d) | (c & d), SHA1_K[2]),
            _ => (b ^ c ^ d, SHA1_K[3]),
        };
        let temp = rotl(a, 5)
            .wrapping_add(f)
            .wrapping_add(e)
            .wrapping_add(k)
            .wrapping_add(w[i]);
        e = d;
        d = c;
        c = rotl(b, 30);
        b = a;
        a = temp;
    }

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
}

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let len = data.len();
    let bit_len = (len as u64) * 8;

    // Process full blocks
    let mut offset = 0;
    while offset + 64 <= len {
        let mut block = [0u32; 16];
        for i in 0..16 {
            block[i] = u32::from_be_bytes([
                data[offset + i * 4],
                data[offset + i * 4 + 1],
                data[offset + i * 4 + 2],
                data[offset + i * 4 + 3],
            ]);
        }
        sha1_compress(&mut h, &block);
        offset += 64;
    }

    // Final block(s) with padding
    let mut final_block = [0u8; 128];
    let remaining = len - offset;
    final_block[..remaining].copy_from_slice(&data[offset..]);
    final_block[remaining] = 0x80;

    let total_blocks = if remaining < 56 { 1 } else { 2 };
    let len_offset = if remaining < 56 { 56 } else { 120 };

    final_block[len_offset..len_offset + 8].copy_from_slice(&bit_len.to_be_bytes());

    for block_idx in 0..total_blocks {
        let mut block = [0u32; 16];
        for i in 0..16 {
            let base = block_idx * 64 + i * 4;
            block[i] = u32::from_be_bytes([
                final_block[base],
                final_block[base + 1],
                final_block[base + 2],
                final_block[base + 3],
            ]);
        }
        sha1_compress(&mut h, &block);
    }

    let mut out = [0u8; 20];
    for i in 0..5 {
        out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

fn hmac_sha1(key: &[u8], msg: &[u8]) -> [u8; 20] {
    // Pad key to 64 bytes
    let mut k = [0u8; 64];
    let key: Vec<u8> = if key.len() > 64 {
        sha1(key).to_vec() // hash oversized keys
    } else {
        key.to_vec()
    };
    k[..key.len()].copy_from_slice(&key);

    let mut ipad = [0u8; 64];
    let mut opad = [0u8; 64];
    for i in 0..64 {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }

    // inner = SHA1(ipad || msg)
    let mut inner_input = Vec::with_capacity(64 + msg.len());
    inner_input.extend_from_slice(&ipad);
    inner_input.extend_from_slice(msg);
    let inner = sha1(&inner_input);

    // outer = SHA1(opad || inner)
    let mut outer_input = [0u8; 84];
    outer_input[..64].copy_from_slice(&opad);
    outer_input[64..].copy_from_slice(&inner);
    sha1(&outer_input)
}

// === MD5 implementation (for EAPOL keyver 1 host verification) ===
// MD5 uses little-endian word ordering (unlike SHA1's big-endian).

#[allow(clippy::similar_names)]
fn md5_compress(h: &mut [u32; 4], block: &[u32; 16]) {
    let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);

    for i in 0..64 {
        let (f, g) = if i < 16 {
            ((b & c) | (!b & d), i)
        } else if i < 32 {
            ((d & b) | (!d & c), (5 * i + 1) % 16)
        } else if i < 48 {
            (b ^ c ^ d, (3 * i + 5) % 16)
        } else {
            (c ^ (b | !d), (7 * i) % 16)
        };

        let temp = d;
        d = c;
        c = b;
        b = b.wrapping_add(rotl(
            a.wrapping_add(f)
                .wrapping_add(MD5_T[i])
                .wrapping_add(block[g]),
            MD5_S[i],
        ));
        a = temp;
    }

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
}

// === NTLM (mode 1000): MD4(UTF-16LE(password)) ===

fn md4_compress(h: &mut [u32; 4], w: &[u32; 16]) {
    let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);

    // Round 1: F(b,c,d) = (b&c)|(~b&d), rotate 3/7/11/19
    a = rotl(a.wrapping_add((b & c) | (!b & d)).wrapping_add(w[0]), 3);
    d = rotl(d.wrapping_add((a & b) | (!a & c)).wrapping_add(w[1]), 7);
    c = rotl(c.wrapping_add((d & a) | (!d & b)).wrapping_add(w[2]), 11);
    b = rotl(b.wrapping_add((c & d) | (!c & a)).wrapping_add(w[3]), 19);
    a = rotl(a.wrapping_add((b & c) | (!b & d)).wrapping_add(w[4]), 3);
    d = rotl(d.wrapping_add((a & b) | (!a & c)).wrapping_add(w[5]), 7);
    c = rotl(c.wrapping_add((d & a) | (!d & b)).wrapping_add(w[6]), 11);
    b = rotl(b.wrapping_add((c & d) | (!c & a)).wrapping_add(w[7]), 19);
    a = rotl(a.wrapping_add((b & c) | (!b & d)).wrapping_add(w[8]), 3);
    d = rotl(d.wrapping_add((a & b) | (!a & c)).wrapping_add(w[9]), 7);
    c = rotl(c.wrapping_add((d & a) | (!d & b)).wrapping_add(w[10]), 11);
    b = rotl(b.wrapping_add((c & d) | (!c & a)).wrapping_add(w[11]), 19);
    a = rotl(a.wrapping_add((b & c) | (!b & d)).wrapping_add(w[12]), 3);
    d = rotl(d.wrapping_add((a & b) | (!a & c)).wrapping_add(w[13]), 7);
    c = rotl(c.wrapping_add((d & a) | (!d & b)).wrapping_add(w[14]), 11);
    b = rotl(b.wrapping_add((c & d) | (!c & a)).wrapping_add(w[15]), 19);

    // Round 2: G(b,c,d) = (b&c)|(b&d)|(c&d), constant 0x5A827999, rotate 3/5/9/13
    // Permutation: [0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15]
    a = rotl(
        a.wrapping_add((b & c) | (b & d) | (c & d))
            .wrapping_add(w[0])
            .wrapping_add(0x5A827999),
        3,
    );
    d = rotl(
        d.wrapping_add((a & b) | (a & c) | (b & c))
            .wrapping_add(w[4])
            .wrapping_add(0x5A827999),
        5,
    );
    c = rotl(
        c.wrapping_add((d & a) | (d & b) | (a & b))
            .wrapping_add(w[8])
            .wrapping_add(0x5A827999),
        9,
    );
    b = rotl(
        b.wrapping_add((c & d) | (c & a) | (d & a))
            .wrapping_add(w[12])
            .wrapping_add(0x5A827999),
        13,
    );
    a = rotl(
        a.wrapping_add((b & c) | (b & d) | (c & d))
            .wrapping_add(w[1])
            .wrapping_add(0x5A827999),
        3,
    );
    d = rotl(
        d.wrapping_add((a & b) | (a & c) | (b & c))
            .wrapping_add(w[5])
            .wrapping_add(0x5A827999),
        5,
    );
    c = rotl(
        c.wrapping_add((d & a) | (d & b) | (a & b))
            .wrapping_add(w[9])
            .wrapping_add(0x5A827999),
        9,
    );
    b = rotl(
        b.wrapping_add((c & d) | (c & a) | (d & a))
            .wrapping_add(w[13])
            .wrapping_add(0x5A827999),
        13,
    );
    a = rotl(
        a.wrapping_add((b & c) | (b & d) | (c & d))
            .wrapping_add(w[2])
            .wrapping_add(0x5A827999),
        3,
    );
    d = rotl(
        d.wrapping_add((a & b) | (a & c) | (b & c))
            .wrapping_add(w[6])
            .wrapping_add(0x5A827999),
        5,
    );
    c = rotl(
        c.wrapping_add((d & a) | (d & b) | (a & b))
            .wrapping_add(w[10])
            .wrapping_add(0x5A827999),
        9,
    );
    b = rotl(
        b.wrapping_add((c & d) | (c & a) | (d & a))
            .wrapping_add(w[14])
            .wrapping_add(0x5A827999),
        13,
    );
    a = rotl(
        a.wrapping_add((b & c) | (b & d) | (c & d))
            .wrapping_add(w[3])
            .wrapping_add(0x5A827999),
        3,
    );
    d = rotl(
        d.wrapping_add((a & b) | (a & c) | (b & c))
            .wrapping_add(w[7])
            .wrapping_add(0x5A827999),
        5,
    );
    c = rotl(
        c.wrapping_add((d & a) | (d & b) | (a & b))
            .wrapping_add(w[11])
            .wrapping_add(0x5A827999),
        9,
    );
    b = rotl(
        b.wrapping_add((c & d) | (c & a) | (d & a))
            .wrapping_add(w[15])
            .wrapping_add(0x5A827999),
        13,
    );

    // Round 3: H(b,c,d) = b^c^d, constant 0x6ED9EBA1, rotate 3/9/11/15
    // Permutation: [0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15]
    a = rotl(
        a.wrapping_add(b ^ c ^ d)
            .wrapping_add(w[0])
            .wrapping_add(0x6ED9EBA1),
        3,
    );
    d = rotl(
        d.wrapping_add(a ^ b ^ c)
            .wrapping_add(w[8])
            .wrapping_add(0x6ED9EBA1),
        9,
    );
    c = rotl(
        c.wrapping_add(d ^ a ^ b)
            .wrapping_add(w[4])
            .wrapping_add(0x6ED9EBA1),
        11,
    );
    b = rotl(
        b.wrapping_add(c ^ d ^ a)
            .wrapping_add(w[12])
            .wrapping_add(0x6ED9EBA1),
        15,
    );
    a = rotl(
        a.wrapping_add(b ^ c ^ d)
            .wrapping_add(w[2])
            .wrapping_add(0x6ED9EBA1),
        3,
    );
    d = rotl(
        d.wrapping_add(a ^ b ^ c)
            .wrapping_add(w[10])
            .wrapping_add(0x6ED9EBA1),
        9,
    );
    c = rotl(
        c.wrapping_add(d ^ a ^ b)
            .wrapping_add(w[6])
            .wrapping_add(0x6ED9EBA1),
        11,
    );
    b = rotl(
        b.wrapping_add(c ^ d ^ a)
            .wrapping_add(w[14])
            .wrapping_add(0x6ED9EBA1),
        15,
    );
    a = rotl(
        a.wrapping_add(b ^ c ^ d)
            .wrapping_add(w[1])
            .wrapping_add(0x6ED9EBA1),
        3,
    );
    d = rotl(
        d.wrapping_add(a ^ b ^ c)
            .wrapping_add(w[9])
            .wrapping_add(0x6ED9EBA1),
        9,
    );
    c = rotl(
        c.wrapping_add(d ^ a ^ b)
            .wrapping_add(w[5])
            .wrapping_add(0x6ED9EBA1),
        11,
    );
    b = rotl(
        b.wrapping_add(c ^ d ^ a)
            .wrapping_add(w[13])
            .wrapping_add(0x6ED9EBA1),
        15,
    );
    a = rotl(
        a.wrapping_add(b ^ c ^ d)
            .wrapping_add(w[3])
            .wrapping_add(0x6ED9EBA1),
        3,
    );
    d = rotl(
        d.wrapping_add(a ^ b ^ c)
            .wrapping_add(w[11])
            .wrapping_add(0x6ED9EBA1),
        9,
    );
    c = rotl(
        c.wrapping_add(d ^ a ^ b)
            .wrapping_add(w[7])
            .wrapping_add(0x6ED9EBA1),
        11,
    );
    b = rotl(
        b.wrapping_add(c ^ d ^ a)
            .wrapping_add(w[15])
            .wrapping_add(0x6ED9EBA1),
        15,
    );

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
}

/// Compute NTLM hash: MD4(UTF-16LE(password)).
/// Returns 16-byte digest.
pub fn ntlm(pw: &[u8]) -> [u8; 16] {
    // Convert to UTF-16LE
    let mut utf16: Vec<u8> = Vec::with_capacity(pw.len() * 2);
    for &b in pw {
        utf16.push(b);
        utf16.push(0);
    }

    let mut h: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];
    let len = utf16.len();
    let bit_len = (len as u64) * 8;

    let mut offset = 0;
    while offset + 64 <= len {
        let mut block = [0u32; 16];
        for i in 0..16 {
            block[i] = u32::from_le_bytes([
                utf16[offset + i * 4],
                utf16[offset + i * 4 + 1],
                utf16[offset + i * 4 + 2],
                utf16[offset + i * 4 + 3],
            ]);
        }
        md4_compress(&mut h, &block);
        offset += 64;
    }

    // Final block with padding
    let rem = len - offset;
    let mut final_block = [0u8; 128];
    final_block[..rem].copy_from_slice(&utf16[offset..]);
    final_block[rem] = 0x80;

    let total_blocks = if rem > 55 { 2 } else { 1 };
    let length_offset = if rem > 55 { 64 + 56 } else { 56 };
    final_block[length_offset..length_offset + 8].copy_from_slice(&bit_len.to_le_bytes());

    for block_idx in 0..total_blocks {
        let mut block = [0u32; 16];
        for i in 0..16 {
            let base = block_idx * 64 + i * 4;
            block[i] = u32::from_le_bytes([
                final_block[base],
                final_block[base + 1],
                final_block[base + 2],
                final_block[base + 3],
            ]);
        }
        md4_compress(&mut h, &block);
    }

    let mut result = [0u8; 16];
    for i in 0..4 {
        result[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_le_bytes());
    }
    result
}

pub fn md5(data: &[u8]) -> [u8; 16] {
    let mut h: [u32; 4] = [0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476];
    let len = data.len();
    let bit_len = (len as u64) * 8;

    // Process full blocks
    let mut offset = 0;
    while offset + 64 <= len {
        let mut block = [0u32; 16];
        for i in 0..16 {
            // Little-endian byte packing (unlike SHA1's big-endian)
            block[i] = u32::from_le_bytes([
                data[offset + i * 4],
                data[offset + i * 4 + 1],
                data[offset + i * 4 + 2],
                data[offset + i * 4 + 3],
            ]);
        }
        md5_compress(&mut h, &block);
        offset += 64;
    }

    // Final block(s) with padding
    let mut final_block = [0u8; 128];
    let remaining = len - offset;
    final_block[..remaining].copy_from_slice(&data[offset..]);
    final_block[remaining] = 0x80;

    let total_blocks = if remaining < 56 { 1 } else { 2 };
    let len_offset = if remaining < 56 { 56 } else { 120 };

    // Little-endian length field (unlike SHA1's big-endian)
    final_block[len_offset..len_offset + 8].copy_from_slice(&bit_len.to_le_bytes());

    for block_idx in 0..total_blocks {
        let mut block = [0u32; 16];
        for i in 0..16 {
            let base = block_idx * 64 + i * 4;
            block[i] = u32::from_le_bytes([
                final_block[base],
                final_block[base + 1],
                final_block[base + 2],
                final_block[base + 3],
            ]);
        }
        md5_compress(&mut h, &block);
    }

    let mut out = [0u8; 16];
    for i in 0..4 {
        // Little-endian output (unlike SHA1's big-endian)
        out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_le_bytes());
    }
    out
}

fn hmac_md5(key: &[u8], msg: &[u8]) -> [u8; 16] {
    // Pad key to 64 bytes
    let mut k = [0u8; 64];
    let key: Vec<u8> = if key.len() > 64 {
        md5(key).to_vec()
    } else {
        key.to_vec()
    };
    k[..key.len()].copy_from_slice(&key);

    let mut ipad = [0u8; 64];
    let mut opad = [0u8; 64];
    for i in 0..64 {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }

    // inner = MD5(ipad || msg)
    let mut inner_input = Vec::with_capacity(64 + msg.len());
    inner_input.extend_from_slice(&ipad);
    inner_input.extend_from_slice(msg);
    let inner = md5(&inner_input);

    // outer = MD5(opad || inner)
    let mut outer_input = [0u8; 80];
    outer_input[..64].copy_from_slice(&opad);
    outer_input[64..].copy_from_slice(&inner);
    md5(&outer_input)
}

// === AES-128 + AES-128-CMAC (for EAPOL keyver 3 host verification) ===
// References: FIPS-197 (AES), RFC 4493 (AES-CMAC).
// This is a compact byte-oriented implementation — clarity over speed,
// since host verification runs once per found match, not in the hot loop.

// AES S-box (256 entries)
const AES_SBOX: [u8; 256] = [
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
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

// GF(2^8) multiplication by 2 (xtime)
#[inline]
fn aes_gmul2(a: u8) -> u8 {
    if a & 0x80 != 0 {
        (a << 1) ^ 0x1b
    } else {
        a << 1
    }
}

// GF(2^8) multiplication by 3
#[inline]
fn aes_gmul3(a: u8) -> u8 {
    aes_gmul2(a) ^ a
}

// AES-128 key expansion: 16-byte key → 176-byte expanded key
fn aes128_key_expansion(key: &[u8; 16]) -> [u8; 176] {
    let mut ek = [0u8; 176];
    ek[..16].copy_from_slice(key);
    let rcon: [u8; 10] = [0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36];
    for i in 4..44 {
        let mut temp = [
            ek[(i - 1) * 4],
            ek[(i - 1) * 4 + 1],
            ek[(i - 1) * 4 + 2],
            ek[(i - 1) * 4 + 3],
        ];
        if i % 4 == 0 {
            // RotWord
            temp = [temp[1], temp[2], temp[3], temp[0]];
            // SubWord
            temp = [
                AES_SBOX[temp[0] as usize],
                AES_SBOX[temp[1] as usize],
                AES_SBOX[temp[2] as usize],
                AES_SBOX[temp[3] as usize],
            ];
            // XOR rcon
            temp[0] ^= rcon[i / 4 - 1];
        }
        for j in 0..4 {
            ek[i * 4 + j] = ek[(i - 4) * 4 + j] ^ temp[j];
        }
    }
    ek
}

// AES-128 encrypt one 16-byte block
fn aes128_encrypt_block(input: &[u8; 16], ek: &[u8; 176]) -> [u8; 16] {
    let mut state = *input;

    // AddRoundKey (round 0)
    for i in 0..16 {
        state[i] ^= ek[i];
    }

    // Rounds 1-9: SubBytes, ShiftRows, MixColumns, AddRoundKey
    for rnd in 1..10 {
        // SubBytes
        for i in 0..16 {
            state[i] = AES_SBOX[state[i] as usize];
        }
        // ShiftRows (state is column-major: state[col*4 + row])
        let temp = state;
        for row in 0..4 {
            for col in 0..4 {
                state[col * 4 + row] = temp[((col + row) % 4) * 4 + row];
            }
        }
        // MixColumns
        for col in 0..4 {
            let s0 = state[col * 4];
            let s1 = state[col * 4 + 1];
            let s2 = state[col * 4 + 2];
            let s3 = state[col * 4 + 3];
            state[col * 4] = aes_gmul2(s0) ^ aes_gmul3(s1) ^ s2 ^ s3;
            state[col * 4 + 1] = s0 ^ aes_gmul2(s1) ^ aes_gmul3(s2) ^ s3;
            state[col * 4 + 2] = s0 ^ s1 ^ aes_gmul2(s2) ^ aes_gmul3(s3);
            state[col * 4 + 3] = aes_gmul3(s0) ^ s1 ^ s2 ^ aes_gmul2(s3);
        }
        // AddRoundKey
        for i in 0..16 {
            state[i] ^= ek[rnd * 16 + i];
        }
    }

    // Round 10: SubBytes, ShiftRows, AddRoundKey (no MixColumns)
    for i in 0..16 {
        state[i] = AES_SBOX[state[i] as usize];
    }
    let temp = state;
    for row in 0..4 {
        for col in 0..4 {
            state[col * 4 + row] = temp[((col + row) % 4) * 4 + row];
        }
    }
    for i in 0..16 {
        state[i] ^= ek[160 + i];
    }

    state
}

// Left-shift a 16-byte big-endian value by 1 bit, returning (result, carry_out)
fn aes_left_shift_1(input: &[u8; 16]) -> ([u8; 16], u8) {
    let mut result = [0u8; 16];
    let mut carry: u8 = 0;
    // Shift from LSB (byte 15) to MSB (byte 0)
    for i in (0..16).rev() {
        result[i] = (input[i] << 1) | carry;
        carry = input[i] >> 7;
    }
    (result, carry)
}

// AES-128-CMAC per RFC 4493. Returns 16-byte tag.
fn aes128_cmac(key: &[u8; 16], msg: &[u8]) -> [u8; 16] {
    let ek = aes128_key_expansion(key);

    // Generate subkeys K1, K2
    let l = aes128_encrypt_block(&[0u8; 16], &ek);
    let (mut k1, carry_l) = aes_left_shift_1(&l);
    if carry_l != 0 {
        k1[15] ^= 0x87;
    }
    let (mut k2, carry_k1) = aes_left_shift_1(&k1);
    if carry_k1 != 0 {
        k2[15] ^= 0x87;
    }

    // Determine number of blocks
    let msg_len = msg.len();
    let n = if msg_len == 0 {
        1
    } else {
        msg_len.div_ceil(16)
    };

    // CBC-MAC over all blocks except the last
    let mut x = [0u8; 16];
    for i in 0..n.saturating_sub(1) {
        let mut y = [0u8; 16];
        for j in 0..16 {
            y[j] = x[j] ^ msg[i * 16 + j];
        }
        x = aes128_encrypt_block(&y, &ek);
    }

    // Prepare the last block
    let last_start = (n - 1) * 16;
    let last_len = msg_len - last_start;
    let mut last_block = [0u8; 16];

    // Copy remaining message bytes
    last_block[..last_len].copy_from_slice(&msg[last_start..last_start + last_len]);

    // Choose subkey and apply padding if needed
    if msg_len > 0 && msg_len.is_multiple_of(16) {
        // Complete last block → XOR with K1
        for j in 0..16 {
            last_block[j] ^= k1[j];
        }
    } else {
        // Incomplete (or empty) last block → pad with 0x80 and XOR with K2
        last_block[last_len % 16] = 0x80;
        for j in 0..16 {
            last_block[j] ^= k2[j];
        }
    }

    // Final: Y = X XOR last_block; T = AES-128(K, Y)
    let mut y = [0u8; 16];
    for j in 0..16 {
        y[j] = x[j] ^ last_block[j];
    }
    aes128_encrypt_block(&y, &ek)
}

pub fn pbkdf2_sha1(password: &[u8], salt: &[u8], iterations: u32, out_len: usize) -> Vec<u8> {
    let mut result = Vec::with_capacity(out_len);
    let mut block_num: u32 = 1;

    while result.len() < out_len {
        // U1 = HMAC(password, salt || INT(block_num))
        let mut msg = Vec::with_capacity(salt.len() + 4);
        msg.extend_from_slice(salt);
        msg.extend_from_slice(&block_num.to_be_bytes());
        let mut u = hmac_sha1(password, &msg);
        let mut t = u;

        for _ in 1..iterations {
            u = hmac_sha1(password, &u);
            for i in 0..20 {
                t[i] ^= u[i];
            }
        }

        result.extend_from_slice(&t);
        block_num += 1;
    }

    result.truncate(out_len);
    result
}

/// Verify a GPU-reported match by recomputing PMKID on the host.
/// Returns true if the password genuinely matches the target PMKID.
pub fn verify_pmkid(
    password: &[u8],
    essid: &[u8],
    ap_mac: &[u8],
    sta_mac: &[u8],
    target_pmkid: &[u8; 16],
) -> bool {
    let pmk = pbkdf2_sha1(password, essid, 4096, 32);
    let mut msg = Vec::with_capacity(20);
    msg.extend_from_slice(b"PMK Name");
    msg.extend_from_slice(ap_mac);
    msg.extend_from_slice(sta_mac);
    let pmkid = hmac_sha1(&pmk, &msg);
    &pmkid[..16] == target_pmkid
}

/// PRF-512: derive 64-byte PTK from PMK using HMAC-SHA1
fn prf_512(pmk: &[u8], label: &[u8], data: &[u8]) -> [u8; 64] {
    let mut ptk = [0u8; 64];
    for i in 0..4u8 {
        let mut msg = Vec::with_capacity(label.len() + data.len() + 1);
        msg.extend_from_slice(label);
        msg.extend_from_slice(data);
        msg.push(i);
        let mac = hmac_sha1(pmk, &msg);
        ptk[i as usize * 16..(i as usize + 1) * 16].copy_from_slice(&mac[..16]);
    }
    ptk
}

// === SHA-256 (for keyver 3 KDF-256 host verification) ===

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

#[inline]
fn sha256_rotr(x: u32, n: u32) -> u32 {
    (x >> n) | (x << (32 - n))
}

#[allow(clippy::similar_names)]
fn sha256_compress(h: &mut [u32; 8], block: &[u32; 16]) {
    let mut w = [0u32; 64];
    w[..16].copy_from_slice(block);
    for i in 16..64 {
        let sig1 = sha256_rotr(w[i - 2], 17) ^ sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        let sig0 = sha256_rotr(w[i - 15], 7) ^ sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        w[i] = sig0
            .wrapping_add(w[i - 16])
            .wrapping_add(sig1)
            .wrapping_add(w[i - 7]);
    }

    let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
    let (mut e, mut f, mut g, mut hh) = (h[4], h[5], h[6], h[7]);

    for i in 0..64 {
        let s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^ sha256_rotr(e, 25);
        let ch = (e & f) ^ (!e & g);
        let t1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[i])
            .wrapping_add(w[i]);
        let s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^ sha256_rotr(a, 22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let t2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let len = data.len();
    let bit_len = (len as u64) * 8;

    let mut offset = 0;
    while offset + 64 <= len {
        let mut block = [0u32; 16];
        for i in 0..16 {
            block[i] = u32::from_be_bytes([
                data[offset + i * 4],
                data[offset + i * 4 + 1],
                data[offset + i * 4 + 2],
                data[offset + i * 4 + 3],
            ]);
        }
        sha256_compress(&mut h, &block);
        offset += 64;
    }

    let mut final_block = [0u8; 128];
    let remaining = len - offset;
    final_block[..remaining].copy_from_slice(&data[offset..]);
    final_block[remaining] = 0x80;

    let total_blocks = if remaining < 56 { 1 } else { 2 };
    let len_offset = if remaining < 56 { 56 } else { 120 };
    final_block[len_offset..len_offset + 8].copy_from_slice(&bit_len.to_be_bytes());

    for block_idx in 0..total_blocks {
        let mut block = [0u32; 16];
        for i in 0..16 {
            let base = block_idx * 64 + i * 4;
            block[i] = u32::from_be_bytes([
                final_block[base],
                final_block[base + 1],
                final_block[base + 2],
                final_block[base + 3],
            ]);
        }
        sha256_compress(&mut h, &block);
    }

    let mut out = [0u8; 32];
    for i in 0..8 {
        out[i * 4..i * 4 + 4].copy_from_slice(&h[i].to_be_bytes());
    }
    out
}

fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut k = [0u8; 64];
    let key: Vec<u8> = if key.len() > 64 {
        sha256(key).to_vec()
    } else {
        key.to_vec()
    };
    k[..key.len()].copy_from_slice(&key);

    let mut ipad = [0u8; 64];
    let mut opad = [0u8; 64];
    for i in 0..64 {
        ipad[i] = k[i] ^ 0x36;
        opad[i] = k[i] ^ 0x5c;
    }

    let mut inner_input = Vec::with_capacity(64 + msg.len());
    inner_input.extend_from_slice(&ipad);
    inner_input.extend_from_slice(msg);
    let inner = sha256(&inner_input);

    let mut outer_input = [0u8; 96];
    outer_input[..64].copy_from_slice(&opad);
    outer_input[64..].copy_from_slice(&inner);
    sha256(&outer_input)
}

/// KDF-256: derive KCK for keyver 3 using HMAC-SHA-256.
/// hashcat's PKE format: [0x01, 0x00] + "Pairwise key expansion" + B + [0x80, 0x01]
/// KCK = HMAC-SHA-256(PMK, PKE)[0:16]
fn kdf_256(pmk: &[u8], ap_mac: &[u8], sta_mac: &[u8], anonce: &[u8], snonce: &[u8]) -> [u8; 16] {
    let mac_min = if ap_mac < sta_mac { ap_mac } else { sta_mac };
    let mac_max = if ap_mac < sta_mac { sta_mac } else { ap_mac };
    let nonce_min = if anonce < snonce { anonce } else { snonce };
    let nonce_max = if anonce < snonce { snonce } else { anonce };

    let mut pke = Vec::with_capacity(102);
    pke.extend_from_slice(&[0x01, 0x00]);
    pke.extend_from_slice(b"Pairwise key expansion");
    pke.extend_from_slice(mac_min);
    pke.extend_from_slice(mac_max);
    pke.extend_from_slice(nonce_min);
    pke.extend_from_slice(nonce_max);
    pke.extend_from_slice(&[0x80, 0x01]);

    let mac = hmac_sha256(pmk, &pke);
    let mut kck = [0u8; 16];
    kck.copy_from_slice(&mac[..16]);
    kck
}

/// Verify an EAPOL match by recomputing PTK and MIC on the host.
/// Returns true if the password genuinely matches the target MIC.
/// `key_version`: 1 = HMAC-MD5 (WPA1/TKIP), 2 = HMAC-SHA1 (WPA2/CCMP),
/// 3 = AES-128-CMAC (WPA2).
pub fn verify_eapol(
    password: &[u8],
    essid: &[u8],
    ap_mac: &[u8],
    sta_mac: &[u8],
    anonce: &[u8],
    eapol: &[u8],
    target_mic: &[u8; 16],
    key_version: u8,
    nonce_error_corrections: u32,
    detected_le: u32,
    detected_be: u32,
) -> bool {
    let pmk = pbkdf2_sha1(password, essid, 4096, 32);

    // SNonce is at offset 17 in the EAPOL frame
    if eapol.len() < 49 {
        return false;
    }

    // Zero the MIC field in EAPOL (offset 81, 16 bytes)
    let mut eapol_zeroed = eapol.to_vec();
    if eapol_zeroed.len() >= 97 {
        for i in 81..97 {
            eapol_zeroed[i] = 0;
        }
    }

    let mac_min = if ap_mac < sta_mac { ap_mac } else { sta_mac };
    let mac_max = if ap_mac < sta_mac { sta_mac } else { ap_mac };

    // Nonce error corrections (hashcat parity).
    // When NEC=0, the loop runs once with the original ANonce.
    let nec = nonce_error_corrections.min(128);
    let bo_loops = if detected_le + detected_be == 0 {
        2
    } else {
        detected_le + detected_be
    };
    let det_le = if detected_le + detected_be == 0 {
        1
    } else {
        detected_le
    };
    let det_be = if detected_le + detected_be == 0 {
        1
    } else {
        detected_be
    };

    // Extract last 4 bytes of ANonce as big-endian u32
    let anonce_tail = (u32::from(anonce[28]) << 24)
        | (u32::from(anonce[29]) << 16)
        | (u32::from(anonce[30]) << 8)
        | u32::from(anonce[31]);

    for nec_iter in 0..=nec {
        for bo_pos in 0..bo_loops {
            let mut t = anonce_tail;
            if bo_loops == 1 {
                if det_le == 1 {
                    t = t.wrapping_sub(nec / 2);
                    t = t.wrapping_add(nec_iter);
                } else if det_be == 1 {
                    // BE byte order
                    t = u32::swap_bytes(t);
                    t = t.wrapping_sub(nec / 2);
                    t = t.wrapping_add(nec_iter);
                    t = u32::swap_bytes(t);
                }
            } else if bo_pos == 0 {
                t = t.wrapping_sub(nec / 2);
                t = t.wrapping_add(nec_iter);
            } else {
                t = u32::swap_bytes(t);
                t = t.wrapping_sub(nec / 2);
                t = t.wrapping_add(nec_iter);
                t = u32::swap_bytes(t);
            }

            let mut anonce_mod = [0u8; 32];
            anonce_mod.copy_from_slice(anonce);
            anonce_mod[28] = (t >> 24) as u8;
            anonce_mod[29] = (t >> 16) as u8;
            anonce_mod[30] = (t >> 8) as u8;
            anonce_mod[31] = t as u8;

            let snonce = &eapol[17..49];
            let nonce_min = if anonce_mod.as_slice() < snonce {
                &anonce_mod
            } else {
                snonce
            };
            let nonce_max = if anonce_mod.as_slice() < snonce {
                snonce
            } else {
                &anonce_mod
            };

            let mut b = Vec::with_capacity(76);
            b.extend_from_slice(mac_min);
            b.extend_from_slice(mac_max);
            b.extend_from_slice(nonce_min);
            b.extend_from_slice(nonce_max);

            // PRF-512 label per IEEE 802.11-2020 §12.7.1.3:
            // R = Label || 0x00 || B — the 0x00 null separator is mandatory.
            // Verified against a real hashcat-cracked capture.
            let kck = if key_version == 3 {
                // keyver 3: KDF-256 (HMAC-SHA-256) — matches hashcat
                kdf_256(&pmk, ap_mac, sta_mac, &anonce_mod, snonce)
            } else {
                // keyver 1/2: PRF-512 (HMAC-SHA-1)
                let ptk = prf_512(&pmk, b"Pairwise key expansion\x00", &b);
                let mut k = [0u8; 16];
                k.copy_from_slice(&ptk[..16]);
                k
            };

            let matched = if key_version == 1 {
                let mic = hmac_md5(&kck, &eapol_zeroed);
                &mic[..16] == target_mic
            } else if key_version == 3 {
                let mic = aes128_cmac(&kck, &eapol_zeroed);
                &mic[..16] == target_mic
            } else {
                let mic = hmac_sha1(&kck, &eapol_zeroed);
                &mic[..16] == target_mic
            };

            if matched {
                return true;
            }
        }
    }

    false
}
/// Verify a PMKID directly from a PMK (no PBKDF2). Used to host-verify GPU
/// matches in --pmk-input lookup mode before they reach the potfile —
/// a GPU false positive written to the potfile would permanently filter
/// the hash out of every future run (R106 MEDIUM-1 fix).
pub fn verify_pmkid_from_pmk(
    pmk: &[u8],
    ap_mac: &[u8],
    sta_mac: &[u8],
    target_pmkid: &[u8; 16],
) -> bool {
    if pmk.len() != 32 {
        return false;
    }
    let mut msg = Vec::with_capacity(20);
    msg.extend_from_slice(b"PMK Name");
    msg.extend_from_slice(ap_mac);
    msg.extend_from_slice(sta_mac);
    let pmkid = hmac_sha1(pmk, &msg);
    &pmkid[..16] == target_pmkid
}

/// Verify an EAPOL MIC directly from a PMK (no PBKDF2). Same purpose as
/// verify_pmkid_from_pmk — host gate for GPU matches in PMK lookup mode.
pub fn verify_eapol_from_pmk(
    pmk: &[u8],
    ap_mac: &[u8],
    sta_mac: &[u8],
    anonce: &[u8],
    eapol: &[u8],
    target_mic: &[u8; 16],
    key_version: u8,
    nonce_error_corrections: u32,
    detected_le: u32,
    detected_be: u32,
) -> bool {
    if pmk.len() != 32 || anonce.len() != 32 || eapol.len() < 49 {
        return false;
    }

    // Zero the MIC field (offset 81, 16 bytes)
    let mut eapol_zeroed = eapol.to_vec();
    if eapol_zeroed.len() >= 97 {
        for i in 81..97 {
            eapol_zeroed[i] = 0;
        }
    }

    let mac_min = if ap_mac < sta_mac { ap_mac } else { sta_mac };
    let mac_max = if ap_mac < sta_mac { sta_mac } else { ap_mac };

    let nec = nonce_error_corrections.min(128);
    let bo_loops = if detected_le + detected_be == 0 {
        2
    } else {
        detected_le + detected_be
    };
    let det_le = if detected_le + detected_be == 0 {
        1
    } else {
        detected_le
    };
    let det_be = if detected_le + detected_be == 0 {
        1
    } else {
        detected_be
    };

    let anonce_tail = (u32::from(anonce[28]) << 24)
        | (u32::from(anonce[29]) << 16)
        | (u32::from(anonce[30]) << 8)
        | u32::from(anonce[31]);

    for nec_iter in 0..=nec {
        for bo_pos in 0..bo_loops {
            let mut t = anonce_tail;
            if bo_loops == 1 {
                if det_le == 1 {
                    t = t.wrapping_sub(nec / 2);
                    t = t.wrapping_add(nec_iter);
                } else if det_be == 1 {
                    t = u32::swap_bytes(t);
                    t = t.wrapping_sub(nec / 2);
                    t = t.wrapping_add(nec_iter);
                    t = u32::swap_bytes(t);
                }
            } else if bo_pos == 0 {
                t = t.wrapping_sub(nec / 2);
                t = t.wrapping_add(nec_iter);
            } else {
                t = u32::swap_bytes(t);
                t = t.wrapping_sub(nec / 2);
                t = t.wrapping_add(nec_iter);
                t = u32::swap_bytes(t);
            }

            let mut anonce_mod = [0u8; 32];
            anonce_mod.copy_from_slice(anonce);
            anonce_mod[28] = (t >> 24) as u8;
            anonce_mod[29] = (t >> 16) as u8;
            anonce_mod[30] = (t >> 8) as u8;
            anonce_mod[31] = t as u8;

            let snonce = &eapol[17..49];
            let nonce_min = if anonce_mod.as_slice() < snonce {
                &anonce_mod
            } else {
                snonce
            };
            let nonce_max = if anonce_mod.as_slice() < snonce {
                snonce
            } else {
                &anonce_mod
            };

            let mut b = Vec::with_capacity(76);
            b.extend_from_slice(mac_min);
            b.extend_from_slice(mac_max);
            b.extend_from_slice(nonce_min);
            b.extend_from_slice(nonce_max);

            let kck = if key_version == 3 {
                kdf_256(pmk, ap_mac, sta_mac, &anonce_mod, snonce)
            } else {
                let ptk = prf_512(pmk, b"Pairwise key expansion\x00", &b);
                let mut k = [0u8; 16];
                k.copy_from_slice(&ptk[..16]);
                k
            };

            let matched = if key_version == 1 {
                let mic = hmac_md5(&kck, &eapol_zeroed);
                &mic[..16] == target_mic
            } else if key_version == 3 {
                let mic = aes128_cmac(&kck, &eapol_zeroed);
                &mic[..16] == target_mic
            } else {
                let mic = hmac_sha1(&kck, &eapol_zeroed);
                &mic[..16] == target_mic
            };

            if matched {
                return true;
            }
        }
    }

    false
}

#[cfg(test)]
mod wpa_kav_tests {
    use super::*;

    // Vectors generated per hashcat's reference (tools/test_modules/m22000.py)
    // and cracked by hashcat -m 22000 on this machine (PoCL) — ground truth.
    // password = "password", ESSID = "testessid".
    // PMK = PBKDF2-HMAC-SHA1("password", "testessid", 4096, 32)
    const PMK_HEX: &str = "3217e2dee2d6e749bbb802187c63d92343f5170abe173bdcf360eecdb6fc3a90";
    // PMKID = HMAC-SHA1(PMK, "PMK Name" || AA || SPA)[:16]
    const PMKID_HEX: &str = "f5118890ca0ee1aae16aa0df187e4fbc";

    fn pmk() -> Vec<u8> {
        let mut v = Vec::with_capacity(32);
        for i in 0..32 {
            v.push(u8::from_str_radix(&PMK_HEX[i * 2..i * 2 + 2], 16).unwrap());
        }
        v
    }

    #[test]
    fn test_pmkid_from_pmk_kav() {
        let aa = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        let spa = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let mut target = [0u8; 16];
        for i in 0..16 {
            target[i] = u8::from_str_radix(&PMKID_HEX[i * 2..i * 2 + 2], 16).unwrap();
        }
        assert!(
            verify_pmkid_from_pmk(&pmk(), &aa, &spa, &target),
            "PMKID from-PMK verification must match the hashcat-cracked vector"
        );
        // Negative control: corrupted target must fail
        let mut bad = target;
        bad[0] ^= 0x01;
        assert!(!verify_pmkid_from_pmk(&pmk(), &aa, &spa, &bad));
        // Negative control: wrong MAC
        let wrong_mac = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(!verify_pmkid_from_pmk(&pmk(), &wrong_mac, &spa, &target));
    }

    #[test]
    fn test_pbkdf2_pmk_kav() {
        // The PMK above is PBKDF2("password", "testessid", 4096)
        let derived = pbkdf2_sha1(b"password", b"testessid", 4096, 32);
        assert_eq!(derived, pmk(), "PBKDF2 must reproduce the vector PMK");
    }

    #[test]
    fn test_eapol_kv2_from_pmk_kav() {
        // EAPOL keyver 2 (HMAC-SHA1 MIC) vector, cracked by hashcat on this machine.
        // anonce = 0x00..0x1f, snonce = 0x20..0x3f, AA=AABBCCDDEEFF, SPA=112233445566
        let anonce: [u8; 32] = core::array::from_fn(|i| i as u8);
        let snonce: [u8; 32] = core::array::from_fn(|i| (i + 32) as u8);
        let aa = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        let spa = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];

        // Frame per hashcat's m22000 generator (keyver 2):
        // version=1, type=3, length=117, descriptor=1, key_info=0x010a
        // (MIC|pairwise|keyver2), key_length=0, replay=1, snonce, iv/rsc/id
        // zeros, MIC zeros, kd_len=22, RSN IE.
        let mut eapol: Vec<u8> = Vec::with_capacity(121);
        eapol.extend_from_slice(&[0x01, 0x03]); // version, type
        eapol.extend_from_slice(&117u16.to_be_bytes()); // length
        eapol.push(0x01); // descriptor
        eapol.extend_from_slice(&0x010au16.to_be_bytes()); // key_info
        eapol.extend_from_slice(&0u16.to_be_bytes()); // key_length
        eapol.extend_from_slice(&1u64.to_be_bytes()); // replay counter
        eapol.extend_from_slice(&snonce);
        eapol.extend_from_slice(&[0u8; 16]); // key IV
        eapol.extend_from_slice(&[0u8; 8]); // key RSC
        eapol.extend_from_slice(&[0u8; 8]); // key ID
        eapol.extend_from_slice(&[0u8; 16]); // MIC (zeroed)
        eapol.extend_from_slice(&22u16.to_be_bytes()); // key data len = 22
        eapol.extend_from_slice(&[
            0x30, 0x14, 0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00, 0x00, 0x0f, 0xac, 0x04,
            0x01, 0x00, 0x00, 0x0f, 0xac, 0x02, 0x00, 0x00,
        ]);
        assert_eq!(eapol.len(), 121);
        assert_eq!(&eapol[17..49], &snonce[..]); // SNonce must land at offset 17

        // MIC from the hashcat-cracked line
        let mic_hex = "dc75e4021c75074291553e882689b948";
        let mut mic = [0u8; 16];
        for i in 0..16 {
            mic[i] = u8::from_str_radix(&mic_hex[i * 2..i * 2 + 2], 16).unwrap();
        }

        assert!(
            verify_eapol_from_pmk(
                &pmk(), &aa, &spa, &anonce, &eapol, &mic, 2, 0, 0, 0
            ),
            "EAPOL kv2 from-PMK verification must match the hashcat-cracked vector"
        );
        // Negative control
        let mut bad = mic;
        bad[15] ^= 0x01;
        assert!(!verify_eapol_from_pmk(
            &pmk(), &aa, &spa, &anonce, &eapol, &bad, 2, 0, 0, 0
        ));
    }
}

// Blowfish constants for bcrypt CPU verification (from OpenBSD reference)
const BF_PBOX: [u32; 18] = [
    0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, 0x082efa98, 0xec4e6c89,
    0x452821e6, 0x38d01377, 0xbe5466cf, 0x34e90c6c, 0xc0ac29b7, 0xc97c50dd, 0x3f84d5b5, 0xb5470917,
    0x9216d5d9, 0x8979fb1b,
];

const BF_SBOX_0: [u32; 256] = [
    0xd1310ba6, 0x98dfb5ac, 0x2ffd72db, 0xd01adfb7, 0xb8e1afed, 0x6a267e96, 0xba7c9045, 0xf12c7f99,
    0x24a19947, 0xb3916cf7, 0x0801f2e2, 0x858efc16, 0x636920d8, 0x71574e69, 0xa458fea3, 0xf4933d7e,
    0x0d95748f, 0x728eb658, 0x718bcd58, 0x82154aee, 0x7b54a41d, 0xc25a59b5, 0x9c30d539, 0x2af26013,
    0xc5d1b023, 0x286085f0, 0xca417918, 0xb8db38ef, 0x8e79dcb0, 0x603a180e, 0x6c9e0e8b, 0xb01e8a3e,
    0xd71577c1, 0xbd314b27, 0x78af2fda, 0x55605c60, 0xe65525f3, 0xaa55ab94, 0x57489862, 0x63e81440,
    0x55ca396a, 0x2aab10b6, 0xb4cc5c34, 0x1141e8ce, 0xa15486af, 0x7c72e993, 0xb3ee1411, 0x636fbc2a,
    0x2ba9c55d, 0x741831f6, 0xce5c3e16, 0x9b87931e, 0xafd6ba33, 0x6c24cf5c, 0x7a325381, 0x28958677,
    0x3b8f4898, 0x6b4bb9af, 0xc4bfe81b, 0x66282193, 0x61d809cc, 0xfb21a991, 0x487cac60, 0x5dec8032,
    0xef845d5d, 0xe98575b1, 0xdc262302, 0xeb651b88, 0x23893e81, 0xd396acc5, 0x0f6d6ff3, 0x83f44239,
    0x2e0b4482, 0xa4842004, 0x69c8f04a, 0x9e1f9b5e, 0x21c66842, 0xf6e96c9a, 0x670c9c61, 0xabd388f0,
    0x6a51a0d2, 0xd8542f68, 0x960fa728, 0xab5133a3, 0x6eef0b6c, 0x137a3be4, 0xba3bf050, 0x7efb2a98,
    0xa1f1651d, 0x39af0176, 0x66ca593e, 0x82430e88, 0x8cee8619, 0x456f9fb4, 0x7d84a5c3, 0x3b8b5ebe,
    0xe06f75d8, 0x85c12073, 0x401a449f, 0x56c16aa6, 0x4ed3aa62, 0x363f7706, 0x1bfedf72, 0x429b023d,
    0x37d0d724, 0xd00a1248, 0xdb0fead3, 0x49f1c09b, 0x075372c9, 0x80991b7b, 0x25d479d8, 0xf6e8def7,
    0xe3fe501a, 0xb6794c3b, 0x976ce0bd, 0x04c006ba, 0xc1a94fb6, 0x409f60c4, 0x5e5c9ec2, 0x196a2463,
    0x68fb6faf, 0x3e6c53b5, 0x1339b2eb, 0x3b52ec6f, 0x6dfc511f, 0x9b30952c, 0xcc814544, 0xaf5ebd09,
    0xbee3d004, 0xde334afd, 0x660f2807, 0x192e4bb3, 0xc0cba857, 0x45c8740f, 0xd20b5f39, 0xb9d3fbdb,
    0x5579c0bd, 0x1a60320a, 0xd6a100c6, 0x402c7279, 0x679f25fe, 0xfb1fa3cc, 0x8ea5e9f8, 0xdb3222f8,
    0x3c7516df, 0xfd616b15, 0x2f501ec8, 0xad0552ab, 0x323db5fa, 0xfd238760, 0x53317b48, 0x3e00df82,
    0x9e5c57bb, 0xca6f8ca0, 0x1a87562e, 0xdf1769db, 0xd542a8f6, 0x287effc3, 0xac6732c6, 0x8c4f5573,
    0x695b27b0, 0xbbca58c8, 0xe1ffa35d, 0xb8f011a0, 0x10fa3d98, 0xfd2183b8, 0x4afcb56c, 0x2dd1d35b,
    0x9a53e479, 0xb6f84565, 0xd28e49bc, 0x4bfb9790, 0xe1ddf2da, 0xa4cb7e33, 0x62fb1341, 0xcee4c6e8,
    0xef20cada, 0x36774c01, 0xd07e9efe, 0x2bf11fb4, 0x95dbda4d, 0xae909198, 0xeaad8e71, 0x6b93d5a0,
    0xd08ed1d0, 0xafc725e0, 0x8e3c5b2f, 0x8e7594b7, 0x8ff6e2fb, 0xf2122b64, 0x8888b812, 0x900df01c,
    0x4fad5ea0, 0x688fc31c, 0xd1cff191, 0xb3a8c1ad, 0x2f2f2218, 0xbe0e1777, 0xea752dfe, 0x8b021fa1,
    0xe5a0cc0f, 0xb56f74e8, 0x18acf3d6, 0xce89e299, 0xb4a84fe0, 0xfd13e0b7, 0x7cc43b81, 0xd2ada8d9,
    0x165fa266, 0x80957705, 0x93cc7314, 0x211a1477, 0xe6ad2065, 0x77b5fa86, 0xc75442f5, 0xfb9d35cf,
    0xebcdaf0c, 0x7b3e89a0, 0xd6411bd3, 0xae1e7e49, 0x00250e2d, 0x2071b35e, 0x226800bb, 0x57b8e0af,
    0x2464369b, 0xf009b91e, 0x5563911d, 0x59dfa6aa, 0x78c14389, 0xd95a537f, 0x207d5ba2, 0x02e5b9c5,
    0x83260376, 0x6295cfa9, 0x11c81968, 0x4e734a41, 0xb3472dca, 0x7b14a94a, 0x1b510052, 0x9a532915,
    0xd60f573f, 0xbc9bc6e4, 0x2b60a476, 0x81e67400, 0x08ba6fb5, 0x571be91f, 0xf296ec6b, 0x2a0dd915,
    0xb6636521, 0xe7b9f9b6, 0xff34052e, 0xc5855664, 0x53b02d5d, 0xa99f8fa1, 0x08ba4799, 0x6e85076a,
];

const BF_SBOX_1: [u32; 256] = [
    0x4b7a70e9, 0xb5b32944, 0xdb75092e, 0xc4192623, 0xad6ea6b0, 0x49a7df7d, 0x9cee60b8, 0x8fedb266,
    0xecaa8c71, 0x699a17ff, 0x5664526c, 0xc2b19ee1, 0x193602a5, 0x75094c29, 0xa0591340, 0xe4183a3e,
    0x3f54989a, 0x5b429d65, 0x6b8fe4d6, 0x99f73fd6, 0xa1d29c07, 0xefe830f5, 0x4d2d38e6, 0xf0255dc1,
    0x4cdd2086, 0x8470eb26, 0x6382e9c6, 0x021ecc5e, 0x09686b3f, 0x3ebaefc9, 0x3c971814, 0x6b6a70a1,
    0x687f3584, 0x52a0e286, 0xb79c5305, 0xaa500737, 0x3e07841c, 0x7fdeae5c, 0x8e7d44ec, 0x5716f2b8,
    0xb03ada37, 0xf0500c0d, 0xf01c1f04, 0x0200b3ff, 0xae0cf51a, 0x3cb574b2, 0x25837a58, 0xdc0921bd,
    0xd19113f9, 0x7ca92ff6, 0x94324773, 0x22f54701, 0x3ae5e581, 0x37c2dadc, 0xc8b57634, 0x9af3dda7,
    0xa9446146, 0x0fd0030e, 0xecc8c73e, 0xa4751e41, 0xe238cd99, 0x3bea0e2f, 0x3280bba1, 0x183eb331,
    0x4e548b38, 0x4f6db908, 0x6f420d03, 0xf60a04bf, 0x2cb81290, 0x24977c79, 0x5679b072, 0xbcaf89af,
    0xde9a771f, 0xd9930810, 0xb38bae12, 0xdccf3f2e, 0x5512721f, 0x2e6b7124, 0x501adde6, 0x9f84cd87,
    0x7a584718, 0x7408da17, 0xbc9f9abc, 0xe94b7d8c, 0xec7aec3a, 0xdb851dfa, 0x63094366, 0xc464c3d2,
    0xef1c1847, 0x3215d908, 0xdd433b37, 0x24c2ba16, 0x12a14d43, 0x2a65c451, 0x50940002, 0x133ae4dd,
    0x71dff89e, 0x10314e55, 0x81ac77d6, 0x5f11199b, 0x043556f1, 0xd7a3c76b, 0x3c11183b, 0x5924a509,
    0xf28fe6ed, 0x97f1fbfa, 0x9ebabf2c, 0x1e153c6e, 0x86e34570, 0xeae96fb1, 0x860e5e0a, 0x5a3e2ab3,
    0x771fe71c, 0x4e3d06fa, 0x2965dcb9, 0x99e71d0f, 0x803e89d6, 0x5266c825, 0x2e4cc978, 0x9c10b36a,
    0xc6150eba, 0x94e2ea78, 0xa5fc3c53, 0x1e0a2df4, 0xf2f74ea7, 0x361d2b3d, 0x1939260f, 0x19c27960,
    0x5223a708, 0xf71312b6, 0xebadfe6e, 0xeac31f66, 0xe3bc4595, 0xa67bc883, 0xb17f37d1, 0x018cff28,
    0xc332ddef, 0xbe6c5aa5, 0x65582185, 0x68ab9802, 0xeecea50f, 0xdb2f953b, 0x2aef7dad, 0x5b6e2f84,
    0x1521b628, 0x29076170, 0xecdd4775, 0x619f1510, 0x13cca830, 0xeb61bd96, 0x0334fe1e, 0xaa0363cf,
    0xb5735c90, 0x4c70a239, 0xd59e9e0b, 0xcbaade14, 0xeecc86bc, 0x60622ca7, 0x9cab5cab, 0xb2f3846e,
    0x648b1eaf, 0x19bdf0ca, 0xa02369b9, 0x655abb50, 0x40685a32, 0x3c2ab4b3, 0x319ee9d5, 0xc021b8f7,
    0x9b540b19, 0x875fa099, 0x95f7997e, 0x623d7da8, 0xf837889a, 0x97e32d77, 0x11ed935f, 0x16681281,
    0x0e358829, 0xc7e61fd6, 0x96dedfa1, 0x7858ba99, 0x57f584a5, 0x1b227263, 0x9b83c3ff, 0x1ac24696,
    0xcdb30aeb, 0x532e3054, 0x8fd948e4, 0x6dbc3128, 0x58ebf2ef, 0x34c6ffea, 0xfe28ed61, 0xee7c3c73,
    0x5d4a14d9, 0xe864b7e3, 0x42105d14, 0x203e13e0, 0x45eee2b6, 0xa3aaabea, 0xdb6c4f15, 0xfacb4fd0,
    0xc742f442, 0xef6abbb5, 0x654f3b1d, 0x41cd2105, 0xd81e799e, 0x86854dc7, 0xe44b476a, 0x3d816250,
    0xcf62a1f2, 0x5b8d2646, 0xfc8883a0, 0xc1c7b6a3, 0x7f1524c3, 0x69cb7492, 0x47848a0b, 0x5692b285,
    0x095bbf00, 0xad19489d, 0x1462b174, 0x23820e00, 0x58428d2a, 0x0c55f5ea, 0x1dadf43e, 0x233f7061,
    0x3372f092, 0x8d937e41, 0xd65fecf1, 0x6c223bdb, 0x7cde3759, 0xcbee7460, 0x4085f2a7, 0xce77326e,
    0xa6078084, 0x19f8509e, 0xe8efd855, 0x61d99735, 0xa969a7aa, 0xc50c06c2, 0x5a04abfc, 0x800bcadc,
    0x9e447a2e, 0xc3453484, 0xfdd56705, 0x0e1e9ec9, 0xdb73dbd3, 0x105588cd, 0x675fda79, 0xe3674340,
    0xc5c43465, 0x713e38d8, 0x3d28f89e, 0xf16dff20, 0x153e21e7, 0x8fb03d4a, 0xe6e39f2b, 0xdb83adf7,
];

const BF_SBOX_2: [u32; 256] = [
    0xe93d5a68, 0x948140f7, 0xf64c261c, 0x94692934, 0x411520f7, 0x7602d4f7, 0xbcf46b2e, 0xd4a20068,
    0xd4082471, 0x3320f46a, 0x43b7d4b7, 0x500061af, 0x1e39f62e, 0x97244546, 0x14214f74, 0xbf8b8840,
    0x4d95fc1d, 0x96b591af, 0x70f4ddd3, 0x66a02f45, 0xbfbc09ec, 0x03bd9785, 0x7fac6dd0, 0x31cb8504,
    0x96eb27b3, 0x55fd3941, 0xda2547e6, 0xabca0a9a, 0x28507825, 0x530429f4, 0x0a2c86da, 0xe9b66dfb,
    0x68dc1462, 0xd7486900, 0x680ec0a4, 0x27a18dee, 0x4f3ffea2, 0xe887ad8c, 0xb58ce006, 0x7af4d6b6,
    0xaace1e7c, 0xd3375fec, 0xce78a399, 0x406b2a42, 0x20fe9e35, 0xd9f385b9, 0xee39d7ab, 0x3b124e8b,
    0x1dc9faf7, 0x4b6d1856, 0x26a36631, 0xeae397b2, 0x3a6efa74, 0xdd5b4332, 0x6841e7f7, 0xca7820fb,
    0xfb0af54e, 0xd8feb397, 0x454056ac, 0xba489527, 0x55533a3a, 0x20838d87, 0xfe6ba9b7, 0xd096954b,
    0x55a867bc, 0xa1159a58, 0xcca92963, 0x99e1db33, 0xa62a4a56, 0x3f3125f9, 0x5ef47e1c, 0x9029317c,
    0xfdf8e802, 0x04272f70, 0x80bb155c, 0x05282ce3, 0x95c11548, 0xe4c66d22, 0x48c1133f, 0xc70f86dc,
    0x07f9c9ee, 0x41041f0f, 0x404779a4, 0x5d886e17, 0x325f51eb, 0xd59bc0d1, 0xf2bcc18f, 0x41113564,
    0x257b7834, 0x602a9c60, 0xdff8e8a3, 0x1f636c1b, 0x0e12b4c2, 0x02e1329e, 0xaf664fd1, 0xcad18115,
    0x6b2395e0, 0x333e92e1, 0x3b240b62, 0xeebeb922, 0x85b2a20e, 0xe6ba0d99, 0xde720c8c, 0x2da2f728,
    0xd0127845, 0x95b794fd, 0x647d0862, 0xe7ccf5f0, 0x5449a36f, 0x877d48fa, 0xc39dfd27, 0xf33e8d1e,
    0x0a476341, 0x992eff74, 0x3a6f6eab, 0xf4f8fd37, 0xa812dc60, 0xa1ebddf8, 0x991be14c, 0xdb6e6b0d,
    0xc67b5510, 0x6d672c37, 0x2765d43b, 0xdcd0e804, 0xf1290dc7, 0xcc00ffa3, 0xb5390f92, 0x690fed0b,
    0x667b9ffb, 0xcedb7d9c, 0xa091cf0b, 0xd9155ea3, 0xbb132f88, 0x515bad24, 0x7b9479bf, 0x763bd6eb,
    0x37392eb3, 0xcc115979, 0x8026e297, 0xf42e312d, 0x6842ada7, 0xc66a2b3b, 0x12754ccc, 0x782ef11c,
    0x6a124237, 0xb79251e7, 0x06a1bbe6, 0x4bfb6350, 0x1a6b1018, 0x11caedfa, 0x3d25bdd8, 0xe2e1c3c9,
    0x44421659, 0x0a121386, 0xd90cec6e, 0xd5abea2a, 0x64af674e, 0xda86a85f, 0xbebfe988, 0x64e4c3fe,
    0x9dbc8057, 0xf0f7c086, 0x60787bf8, 0x6003604d, 0xd1fd8346, 0xf6381fb0, 0x7745ae04, 0xd736fccc,
    0x83426b33, 0xf01eab71, 0xb0804187, 0x3c005e5f, 0x77a057be, 0xbde8ae24, 0x55464299, 0xbf582e61,
    0x4e58f48f, 0xf2ddfda2, 0xf474ef38, 0x8789bdc2, 0x5366f9c3, 0xc8b38e74, 0xb475f255, 0x46fcd9b9,
    0x7aeb2661, 0x8b1ddf84, 0x846a0e79, 0x915f95e2, 0x466e598e, 0x20b45770, 0x8cd55591, 0xc902de4c,
    0xb90bace1, 0xbb8205d0, 0x11a86248, 0x7574a99e, 0xb77f19b6, 0xe0a9dc09, 0x662d09a1, 0xc4324633,
    0xe85a1f02, 0x09f0be8c, 0x4a99a025, 0x1d6efe10, 0x1ab93d1d, 0x0ba5a4df, 0xa186f20f, 0x2868f169,
    0xdcb7da83, 0x573906fe, 0xa1e2ce9b, 0x4fcd7f52, 0x50115e01, 0xa70683fa, 0xa002b5c4, 0x0de6d027,
    0x9af88c27, 0x773f8641, 0xc3604c06, 0x61a806b5, 0xf0177a28, 0xc0f586e0, 0x006058aa, 0x30dc7d62,
    0x11e69ed7, 0x2338ea63, 0x53c2dd94, 0xc2c21634, 0xbbcbee56, 0x90bcb6de, 0xebfc7da1, 0xce591d76,
    0x6f05e409, 0x4b7c0188, 0x39720a3d, 0x7c927c24, 0x86e3725f, 0x724d9db9, 0x1ac15bb4, 0xd39eb8fc,
    0xed545578, 0x08fca5b5, 0xd83d7cd3, 0x4dad0fc4, 0x1e50ef5e, 0xb161e6f8, 0xa28514d9, 0x6c51133c,
    0x6fd5c7e7, 0x56e14ec4, 0x362abfce, 0xddc6c837, 0xd79a3234, 0x92638212, 0x670efa8e, 0x406000e0,
];

const BF_SBOX_3: [u32; 256] = [
    0x3a39ce37, 0xd3faf5cf, 0xabc27737, 0x5ac52d1b, 0x5cb0679e, 0x4fa33742, 0xd3822740, 0x99bc9bbe,
    0xd5118e9d, 0xbf0f7315, 0xd62d1c7e, 0xc700c47b, 0xb78c1b6b, 0x21a19045, 0xb26eb1be, 0x6a366eb4,
    0x5748ab2f, 0xbc946e79, 0xc6a376d2, 0x6549c2c8, 0x530ff8ee, 0x468dde7d, 0xd5730a1d, 0x4cd04dc6,
    0x2939bbdb, 0xa9ba4650, 0xac9526e8, 0xbe5ee304, 0xa1fad5f0, 0x6a2d519a, 0x63ef8ce2, 0x9a86ee22,
    0xc089c2b8, 0x43242ef6, 0xa51e03aa, 0x9cf2d0a4, 0x83c061ba, 0x9be96a4d, 0x8fe51550, 0xba645bd6,
    0x2826a2f9, 0xa73a3ae1, 0x4ba99586, 0xef5562e9, 0xc72fefd3, 0xf752f7da, 0x3f046f69, 0x77fa0a59,
    0x80e4a915, 0x87b08601, 0x9b09e6ad, 0x3b3ee593, 0xe990fd5a, 0x9e34d797, 0x2cf0b7d9, 0x022b8b51,
    0x96d5ac3a, 0x017da67d, 0xd1cf3ed6, 0x7c7d2d28, 0x1f9f25cf, 0xadf2b89b, 0x5ad6b472, 0x5a88f54c,
    0xe029ac71, 0xe019a5e6, 0x47b0acfd, 0xed93fa9b, 0xe8d3c48d, 0x283b57cc, 0xf8d56629, 0x79132e28,
    0x785f0191, 0xed756055, 0xf7960e44, 0xe3d35e8c, 0x15056dd4, 0x88f46dba, 0x03a16125, 0x0564f0bd,
    0xc3eb9e15, 0x3c9057a2, 0x97271aec, 0xa93a072a, 0x1b3f6d9b, 0x1e6321f5, 0xf59c66fb, 0x26dcf319,
    0x7533d928, 0xb155fdf5, 0x03563482, 0x8aba3cbb, 0x28517711, 0xc20ad9f8, 0xabcc5167, 0xccad925f,
    0x4de81751, 0x3830dc8e, 0x379d5862, 0x9320f991, 0xea7a90c2, 0xfb3e7bce, 0x5121ce64, 0x774fbe32,
    0xa8b6e37e, 0xc3293d46, 0x48de5369, 0x6413e680, 0xa2ae0810, 0xdd6db224, 0x69852dfd, 0x09072166,
    0xb39a460a, 0x6445c0dd, 0x586cdecf, 0x1c20c8ae, 0x5bbef7dd, 0x1b588d40, 0xccd2017f, 0x6bb4e3bb,
    0xdda26a7e, 0x3a59ff45, 0x3e350a44, 0xbcb4cdd5, 0x72eacea8, 0xfa6484bb, 0x8d6612ae, 0xbf3c6f47,
    0xd29be463, 0x542f5d9e, 0xaec2771b, 0xf64e6370, 0x740e0d8d, 0xe75b1357, 0xf8721671, 0xaf537d5d,
    0x4040cb08, 0x4eb4e2cc, 0x34d2466a, 0x0115af84, 0xe1b00428, 0x95983a1d, 0x06b89fb4, 0xce6ea048,
    0x6f3f3b82, 0x3520ab82, 0x011a1d4b, 0x277227f8, 0x611560b1, 0xe7933fdc, 0xbb3a792b, 0x344525bd,
    0xa08839e1, 0x51ce794b, 0x2f32c9b7, 0xa01fbac9, 0xe01cc87e, 0xbcc7d1f6, 0xcf0111c3, 0xa1e8aac7,
    0x1a908749, 0xd44fbd9a, 0xd0dadecb, 0xd50ada38, 0x0339c32a, 0xc6913667, 0x8df9317c, 0xe0b12b4f,
    0xf79e59b7, 0x43f5bb3a, 0xf2d519ff, 0x27d9459c, 0xbf97222c, 0x15e6fc2a, 0x0f91fc71, 0x9b941525,
    0xfae59361, 0xceb69ceb, 0xc2a86459, 0x12baa8d1, 0xb6c1075e, 0xe3056a0c, 0x10d25065, 0xcb03a442,
    0xe0ec6e0e, 0x1698db3b, 0x4c98a0be, 0x3278e964, 0x9f1f9532, 0xe0d392df, 0xd3a0342b, 0x8971f21e,
    0x1b0a7441, 0x4ba3348c, 0xc5be7120, 0xc37632d8, 0xdf359f8d, 0x9b992f2e, 0xe60b6f47, 0x0fe3f11d,
    0xe54cda54, 0x1edad891, 0xce6279cf, 0xcd3e7e6f, 0x1618b166, 0xfd2c1d05, 0x848fd2c5, 0xf6fb2299,
    0xf523f357, 0xa6327623, 0x93a83531, 0x56cccd02, 0xacf08162, 0x5a75ebb5, 0x6e163697, 0x88d273cc,
    0xde966292, 0x81b949d0, 0x4c50901b, 0x71c65614, 0xe6c6c7bd, 0x327a140a, 0x45e1d006, 0xc3f27b9a,
    0xc9aa53fd, 0x62a80f00, 0xbb25bfe2, 0x35bdd2f6, 0x71126905, 0xb2040222, 0xb6cbcf7c, 0xcd769c2b,
    0x53113ec0, 0x1640e3d3, 0x38abbd60, 0x2547adf0, 0xba38209c, 0xf746ce76, 0x77afa1c5, 0x20756060,
    0x85cbfe4e, 0x8ae88dd8, 0x7aaaf9b0, 0x4cf9aa7e, 0x1948c25c, 0x02fb8a8c, 0x01c36ae4, 0xd6ebe1f9,
    0x90d4f869, 0xa65cdea0, 0x3f09252d, 0xc208e69f, 0xb74e6132, 0xce77e25b, 0x578fdfe3, 0x3ac372e6,
];

// === bcrypt CPU verification (matches OpenBSD reference) ===
// Only runs once per found match — simplicity > speed.

struct BlowfishCtx {
    p: [u32; 18],
    s: [[u32; 256]; 4],
}

fn bf_encipher(ctx: &BlowfishCtx, xl: &mut u32, xr: &mut u32) {
    let mut l = *xl;
    let mut r = *xr;
    let s0 = &ctx.s[0];
    let s1 = &ctx.s[1];
    let s2 = &ctx.s[2];
    let s3 = &ctx.s[3];
    let p = &ctx.p;

    l ^= p[0];
    // 16 rounds (unrolled F function)
    // F(s, x) = ((s0[byte3] + s1[byte2]) ^ s2[byte1]) + s3[byte0]
    macro_rules! bf_round {
        ($l:ident, $r:ident, $n:expr) => {{
            // F(s, x) = ((s0[byte3] + s1[byte2]) ^ s2[byte1]) + s3[byte0]
            let f = (s0[((($l >> 24) & 0xff) as usize)]
                .wrapping_add(s1[((($l >> 16) & 0xff) as usize)]))
                ^ s2[((($l >> 8) & 0xff) as usize)];
            let tmp = f.wrapping_add(s3[($l & 0xff) as usize]);
            $r ^= tmp ^ p[$n];
        }};
    }
    bf_round!(l, r, 1);
    bf_round!(r, l, 2);
    bf_round!(l, r, 3);
    bf_round!(r, l, 4);
    bf_round!(l, r, 5);
    bf_round!(r, l, 6);
    bf_round!(l, r, 7);
    bf_round!(r, l, 8);
    bf_round!(l, r, 9);
    bf_round!(r, l, 10);
    bf_round!(l, r, 11);
    bf_round!(r, l, 12);
    bf_round!(l, r, 13);
    bf_round!(r, l, 14);
    bf_round!(l, r, 15);
    bf_round!(r, l, 16);

    *xl = r ^ p[17];
    *xr = l;
}

/// stream2word: read 4 bytes as big-endian u32 from data with wrapping
fn stream2word(data: &[u8], current: &mut usize) -> u32 {
    let mut temp: u32 = 0;
    let mut j = *current;
    for _ in 0..4 {
        if j >= data.len() {
            j = 0;
        }
        temp = (temp << 8) | data[j] as u32;
        j += 1;
    }
    *current = j;
    temp
}

/// Blowfish_expandstate: XOR P with key, encrypt P+S with salt XOR
fn bf_expandstate(ctx: &mut BlowfishCtx, data: &[u8], key: &[u8]) {
    // XOR P with key
    let mut j = 0;
    for i in 0..18 {
        ctx.p[i] ^= stream2word(key, &mut j);
    }
    // Encrypt P+S with data (salt) XOR, continuous L,R chain
    let mut k = 0;
    let mut l: u32 = 0;
    let mut r: u32 = 0;
    for i in (0..18).step_by(2) {
        l ^= stream2word(data, &mut k);
        r ^= stream2word(data, &mut k);
        bf_encipher(ctx, &mut l, &mut r);
        ctx.p[i] = l;
        ctx.p[i + 1] = r;
    }
    for si in 0..4 {
        for ki in (0..256).step_by(2) {
            l ^= stream2word(data, &mut k);
            r ^= stream2word(data, &mut k);
            bf_encipher(ctx, &mut l, &mut r);
            ctx.s[si][ki] = l;
            ctx.s[si][ki + 1] = r;
        }
    }
}

/// Blowfish_expand0state: XOR P with key, encrypt P+S (no salt)
fn bf_expand0state(ctx: &mut BlowfishCtx, key: &[u8]) {
    let mut j = 0;
    for i in 0..18 {
        ctx.p[i] ^= stream2word(key, &mut j);
    }
    let mut l: u32 = 0;
    let mut r: u32 = 0;
    for i in (0..18).step_by(2) {
        bf_encipher(ctx, &mut l, &mut r);
        ctx.p[i] = l;
        ctx.p[i + 1] = r;
    }
    for si in 0..4 {
        for ki in (0..256).step_by(2) {
            bf_encipher(ctx, &mut l, &mut r);
            ctx.s[si][ki] = l;
            ctx.s[si][ki + 1] = r;
        }
    }
}

/// Compute bcrypt hash for a password + salt + cost.
/// Returns the 24-byte digest (only 23 bytes are valid, byte 23 = 0).
/// Matches OpenBSD bcrypt_hashpass reference.
pub fn bcrypt(pw: &[u8], salt: &[u8], cost: u32) -> [u8; 24] {
    // key_len includes null terminator (matching OpenBSD $2a$/$2b$ behavior)
    let mut key = [0u8; 72];
    let key_len = (pw.len() + 1).min(72);
    key[..pw.len().min(72)].copy_from_slice(&pw[..pw.len().min(72)]);
    // key[key_len-1] is already 0 (null terminator from vec init)

    let rounds = 1u32 << cost.min(31);

    // InitState
    let mut ctx = BlowfishCtx {
        p: BF_PBOX,
        s: [BF_SBOX_0, BF_SBOX_1, BF_SBOX_2, BF_SBOX_3],
    };

    // ExpandState(salt, key)
    bf_expandstate(&mut ctx, salt, &key[..key_len]);

    // Main loop: expand0state(key) + expand0state(salt) per round
    for _ in 0..rounds {
        bf_expand0state(&mut ctx, &key[..key_len]);
        bf_expand0state(&mut ctx, salt);
    }

    // Encrypt "OrpheanBeholderScryDoubt" 64 times
    let magic: [u8; 24] = *b"OrpheanBeholderScryDoubt";
    let mut cdata = [0u32; 6];
    let mut j = 0;
    for i in 0..6 {
        cdata[i] = stream2word(&magic, &mut j);
    }

    for _ in 0..64 {
        // blf_enc: 3 blocks (6 words)
        for blk in 0..3 {
            let mut l = cdata[blk * 2];
            let mut r = cdata[blk * 2 + 1];
            bf_encipher(&ctx, &mut l, &mut r);
            cdata[blk * 2] = l;
            cdata[blk * 2 + 1] = r;
        }
    }

    // Convert u32s back to bytes (big-endian)
    let mut result = [0u8; 24];
    for i in 0..6 {
        result[4 * i] = (cdata[i] >> 24) as u8;
        result[4 * i + 1] = (cdata[i] >> 16) as u8;
        result[4 * i + 2] = (cdata[i] >> 8) as u8;
        result[4 * i + 3] = cdata[i] as u8;
    }
    // Only 23 bytes are valid — mask the 24th
    result[23] = 0;
    result
}
