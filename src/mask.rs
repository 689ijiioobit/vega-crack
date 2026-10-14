//! Mask attack: generate password candidates from a mask pattern.
//!
//! Mask syntax (compatible with hashcat -a 3):
//!   ?l = lowercase a-z          ?u = uppercase A-Z
//!   ?d = digits 0-9             ?s = special chars
//!   ?h = hex lowercase           ?H = hex uppercase
//!   ?a = all printable ASCII     ?b = all bytes 0x00-0xFF
//!   ?1..?8 = custom charsets (-1..-8 flags)
//!   ?? = literal question mark
//!   Any other char = literal
//!
//! --increment: try mask lengths from min to full mask length
// (truncates the mask from the right, adding one position each iteration)

#![allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::needless_pass_by_value
)]

const CHARSET_LOWER: &[u8] = b"abcdefghijklmnopqrstuvwxyz";
const CHARSET_UPPER: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const CHARSET_DIGIT: &[u8] = b"0123456789";
const CHARSET_SPECIAL: &[u8] = b" !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const CHARSET_HEX_LOWER: &[u8] = b"0123456789abcdef";
const CHARSET_HEX_UPPER: &[u8] = b"0123456789ABCDEF";
const CHARSET_ALL: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const CHARSET_ALL_BYTES: &[u8] = b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x20\x21\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\x30\x31\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f\x40\x41\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f\x50\x51\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f\x60\x61\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f\x70\x71\x72\x73\x74\x75\x76\x77\x78\x79\x7a\x7b\x7c\x7d\x7e\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xfb\xfc\xfd\xfe\xff";

/// A parsed mask: list of charsets, one per position
#[derive(Clone)]
pub struct Mask {
    pub positions: Vec<Vec<u8>>, // charset for each position
    total_candidates: u64,
}

impl Mask {
    /// Parse a mask string with a configurable minimum length. Used by hybrid
    /// mode where the mask side alone may be shorter than 8 (the combined
    /// word+mask candidate is length-checked instead). `min_len` of 1 disables
    /// the minimum-length check; the maximum of 63 always applies. A `min_len`
    /// of 8 gives the WPA behavior.
    pub fn parse_with_min_len(
        mask: &str,
        custom_charsets: &[Vec<u8>; 8],
        min_len: usize,
    ) -> Result<Self, String> {
        let mut positions = Vec::new();
        let bytes = mask.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if bytes[i] == b'?' && i + 1 < bytes.len() {
                let next = bytes[i + 1];
                let charset: &[u8] = match next {
                    b'l' => CHARSET_LOWER,
                    b'u' => CHARSET_UPPER,
                    b'd' => CHARSET_DIGIT,
                    b's' => CHARSET_SPECIAL,
                    b'h' => CHARSET_HEX_LOWER,
                    b'H' => CHARSET_HEX_UPPER,
                    b'a' => CHARSET_ALL,
                    b'b' => CHARSET_ALL_BYTES,
                    b'?' => {
                        // Literal question mark
                        positions.push(vec![b'?']);
                        i += 2;
                        continue;
                    }
                    b'1' => &custom_charsets[0],
                    b'2' => &custom_charsets[1],
                    b'3' => &custom_charsets[2],
                    b'4' => &custom_charsets[3],
                    b'5' => &custom_charsets[4],
                    b'6' => &custom_charsets[5],
                    b'7' => &custom_charsets[6],
                    b'8' => &custom_charsets[7],
                    _ => return Err(format!("unknown charset ?{next} in mask")),
                };
                if charset.is_empty() {
                    return Err(format!("charset ?{next} is empty"));
                }
                positions.push(charset.to_vec());
                i += 2;
            } else {
                // Literal character
                positions.push(vec![bytes[i]]);
                i += 1;
            }
        }

        if positions.is_empty() {
            return Err("mask is empty".to_string());
        }

        // Check length bounds — minimum is configurable (hybrid uses 1),
        // maximum is always 63 (WPA spec).
        let mask_len: usize = positions.len();
        if mask_len < min_len {
            return Err(format!(
                "mask produces {mask_len}-char passwords, but minimum is {min_len}. Use a longer mask."
            ));
        }
        if mask_len > 63 {
            return Err(format!(
                "mask produces {mask_len}-char passwords, but WPA maximum is 63"
            ));
        }

        // Calculate total candidates — use checked_mul to detect overflow
        let total: u64 = positions
            .iter()
            .map(|p| p.len() as u64)
            .try_fold(1u64, u64::checked_mul)
            .ok_or_else(|| "Mask keyspace exceeds 2^64 — too large to enumerate".to_string())?;

        Ok(Self {
            positions,
            total_candidates: total,
        })
    }

    /// Get a truncated mask (first n positions) for --increment
    pub fn truncate(&self, len: usize) -> Self {
        let positions = self.positions[..len.min(self.positions.len())].to_vec();
        // Use saturating_mul here — truncate is only called with valid lengths
        // from the increment loop, and the full mask already passed checked_mul
        let total: u64 = positions
            .iter()
            .map(|p| p.len() as u64)
            .fold(1u64, u64::saturating_mul);
        Self {
            positions,
            total_candidates: total,
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn total_candidates(&self) -> u64 {
        self.total_candidates
    }

    /// Generate the next batch of candidates from this mask.
    /// `start_index` is the candidate index to start from (odometer position).
    /// Returns (candidates, next_index) where next_index is the next candidate
    /// index to process, or None if exhausted.
    pub fn generate_batch(
        &self,
        start_index: u64,
        batch_size: usize,
        pw_batch: &mut [u8],
        pw_lens: &mut [u32],
        slot_len: usize,
        max_pw_len: usize,
        indices_buf: &mut [usize],
    ) -> (usize, Option<u64>) {
        let num_positions = self.positions.len();
        let pw_len = num_positions.min(max_pw_len);
        let mut count = 0usize;
        let mut current_idx = start_index;

        // Odigit indices: which character in each position's charset
        // Reuse caller-provided buffer instead of allocating per batch.
        let indices = &mut indices_buf[..num_positions];

        // Convert flat index to odometer position
        // Position 0 is least significant (changes fastest)
        let mut temp = current_idx;
        for pos in 0..num_positions {
            let charset_len = self.positions[pos].len() as u64;
            indices[pos] = (temp % charset_len) as usize;
            temp /= charset_len;
        }

        while count < batch_size && current_idx < self.total_candidates {
            // Generate password from current odometer position
            let offset = count * slot_len;
            for pos in 0..pw_len {
                let charset = &self.positions[pos];
                let char_idx = indices[pos];
                if offset + pos < pw_batch.len() {
                    pw_batch[offset + pos] = charset[char_idx];
                }
            }
            // Zero the rest of the slot — only on first candidate of this batch.
            // In mask mode pw_len is constant, so tail bytes are already zero
            // from the previous batch's fill (buffer is reused). Skipping this
            // saves ~87% of memory writes for 8-char masks (56 of 64 bytes).
            if count == 0 {
                for i in pw_len..slot_len {
                    if offset + i < pw_batch.len() {
                        pw_batch[offset + i] = 0;
                    }
                }
            }
            pw_lens[count] = pw_len as u32;
            count += 1;
            current_idx += 1;

            // Increment odometer (position 0 = least significant)
            let mut pos = 0;
            loop {
                indices[pos] += 1;
                if indices[pos] < self.positions[pos].len() {
                    break;
                }
                indices[pos] = 0;
                pos += 1;
                if pos >= num_positions {
                    // Exhausted
                    return (count, None);
                }
            }
        }

        let next = if current_idx >= self.total_candidates {
            None
        } else {
            Some(current_idx)
        };
        (count, next)
    }
}

/// Parse a custom charset string like "?l?d" or "abcdef" or "?lABC123".
/// When `hex_charset` is true, literal characters are decoded as hex pairs
/// (e.g. "414243" produces bytes [0x41, 0x42, 0x43] = "ABC").
pub fn parse_custom_charset(
    s: &str,
    existing: &[Vec<u8>; 8],
    hex_charset: bool,
) -> Result<Vec<u8>, String> {
    let mut result = Vec::new();
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'?' && i + 1 < bytes.len() {
            let next = bytes[i + 1];
            match next {
                b'l' => result.extend_from_slice(CHARSET_LOWER),
                b'u' => result.extend_from_slice(CHARSET_UPPER),
                b'd' => result.extend_from_slice(CHARSET_DIGIT),
                b's' => result.extend_from_slice(CHARSET_SPECIAL),
                b'h' => result.extend_from_slice(CHARSET_HEX_LOWER),
                b'H' => result.extend_from_slice(CHARSET_HEX_UPPER),
                b'a' => result.extend_from_slice(CHARSET_ALL),
                b'b' => result.extend_from_slice(CHARSET_ALL_BYTES),
                b'?' => result.push(b'?'),
                b'1' => result.extend_from_slice(&existing[0]),
                b'2' => result.extend_from_slice(&existing[1]),
                b'3' => result.extend_from_slice(&existing[2]),
                b'4' => result.extend_from_slice(&existing[3]),
                b'5' => result.extend_from_slice(&existing[4]),
                b'6' => result.extend_from_slice(&existing[5]),
                b'7' => result.extend_from_slice(&existing[6]),
                b'8' => result.extend_from_slice(&existing[7]),
                _ => {
                    return Err(format!(
                        "Unknown charset placeholder '?{}' in custom charset",
                        next as char
                    ));
                }
            }
            i += 2;
        } else if hex_charset {
            // Hex charset mode: decode pairs of hex characters as bytes
            if i + 1 >= bytes.len() {
                return Err("hex charset has odd number of literal characters".to_string());
            }
            let hex_pair = &bytes[i..i + 2];
            let byte =
                u8::from_str_radix(&String::from_utf8_lossy(hex_pair), 16).map_err(|_| {
                    format!(
                        "invalid hex pair in custom charset: {}{}",
                        hex_pair[0] as char, hex_pair[1] as char
                    )
                })?;
            result.push(byte);
            i += 2;
        } else {
            result.push(bytes[i]);
            i += 1;
        }
    }
    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    result.retain(|b| seen.insert(*b));
    Ok(result)
}

/// Display a mask's keyspace info
pub fn mask_info(mask: &Mask) -> String {
    let len = mask.len();
    let total = mask.total_candidates();
    let mut parts = Vec::new();
    for pos in &mask.positions {
        if pos.len() == 1 {
            parts.push(format!("'{}'", pos[0] as char));
        } else {
            parts.push(format!("[{}]", pos.len()));
        }
    }
    format!(
        "Mask: {} ({len} chars, {:.0} candidates)",
        parts.join(" "),
        total as f64
    )
}
