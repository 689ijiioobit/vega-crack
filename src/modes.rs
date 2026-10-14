// Hash mode registry — supports multiple hash types (WPA, MD5, SHA256, etc.)
// Each mode implements HashMode trait: kernel source, hash parsing, buffer setup.

use crate::multihash::{HashGroup, ParsedHash};

/// Supported hash modes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum HashModeId {
    Wpa22000,
    Md5,
    Sha256,
    Ntlm,
    Bcrypt,
}

impl HashModeId {
    /// Parse a hash file for this mode. Returns groups of hashes.
    /// For WPA, delegates to multihash::parse_hash_file (groups by ESSID).
    /// For MD5/NTLM, parses 32-char hex lines into a single group (no salt).
    pub fn parse_hash_file(&self, content: &str) -> Result<Vec<HashGroup>, String> {
        match self {
            HashModeId::Wpa22000 => crate::multihash::parse_hash_file(content),
            HashModeId::Md5 => parse_md5_hashes(content),
            HashModeId::Sha256 => parse_sha256_hashes(content),
            HashModeId::Ntlm => parse_ntlm_hashes(content),
            HashModeId::Bcrypt => parse_bcrypt_hashes(content),
        }
    }
}

/// Parse MD5 hash file: one 32-char hex hash per line.
/// All hashes go into a single group (no salt/ESSID).
/// Empty lines and lines starting with # are skipped.
fn parse_md5_hashes(content: &str) -> Result<Vec<HashGroup>, String> {
    let mut hashes = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // MD5 hash: exactly 32 hex characters
        if trimmed.len() != 32 {
            return Err(format!(
                "MD5 parse error line {}: expected 32 hex chars, got {} chars: '{}'",
                line_num + 1,
                trimmed.len(),
                trimmed
            ));
        }

        if !trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!(
                "MD5 parse error line {}: non-hex character in hash: '{}'",
                line_num + 1,
                trimmed
            ));
        }

        // Decode 32 hex chars → 16 bytes → target
        let mut target = [0u8; 16];
        for i in 0..16 {
            target[i] = u8::from_str_radix(&trimmed[i * 2..i * 2 + 2], 16).map_err(|e| {
                format!(
                    "MD5 parse error line {}: hex decode failed: {e}",
                    line_num + 1
                )
            })?;
        }

        hashes.push(ParsedHash {
            target,
            ap_mac: [0u8; 6],
            sta_mac: [0u8; 6],
            hash_type: 2, // 2 = MD5 (0=PMKID, 1=EAPOL, 2=MD5)
            eapol: None,
            bcrypt_cost: 0,
            source: trimmed.to_string(),
        });
    }

    if hashes.is_empty() {
        return Err("MD5 hash file contains no valid hashes".into());
    }

    // All MD5 hashes share one group (no salt)
    Ok(vec![HashGroup {
        essid: Vec::new(), // MD5 has no ESSID/salt
        hashes,
    }])
}

/// Parse SHA-256 hash file (mode 1400): one 64-char hex hash per line.
/// SHA-256 produces 32 bytes = 8 big-endian u32s.
fn parse_sha256_hashes(content: &str) -> Result<Vec<HashGroup>, String> {
    let mut hashes = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.len() != 64 {
            return Err(format!(
                "SHA-256 parse error line {}: expected 64 hex chars, got {} chars: '{}'",
                line_num + 1,
                trimmed.len(),
                trimmed
            ));
        }

        if !trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!(
                "SHA-256 parse error line {}: non-hex character in hash: '{}'",
                line_num + 1,
                trimmed
            ));
        }

        // Decode 64 hex chars → 32 bytes → target (pad to 16 with zeros)
        let mut target = [0u8; 16];
        let mut full = [0u8; 32];
        for i in 0..32 {
            full[i] = u8::from_str_radix(&trimmed[i * 2..i * 2 + 2], 16).map_err(|e| {
                format!(
                    "SHA-256 parse error line {}: hex decode failed: {e}",
                    line_num + 1
                )
            })?;
        }
        // ParsedHash.target is [u8; 16], but SHA-256 needs 32 bytes.
        // We store the first 16 bytes in target, and the rest in eapol (hack).
        // Actually, we need to extend ParsedHash or use a different approach.
        // For now, store first 16 bytes in target. The kernel will need 8 u32s.
        target.copy_from_slice(&full[..16]);

        // Store the full 32-byte hash in eapol as a workaround
        let eapol_data = crate::multihash::EapolData {
            anonce: {
                let mut a = [0u8; 32];
                a[..16].copy_from_slice(&full[16..32]);
                a
            },
            eapol: Vec::new(),
            key_version: 0,
            nonce_error_corrections: 0,
            detected_le: 0,
            detected_be: 0,
        };

        hashes.push(ParsedHash {
            target,
            ap_mac: [0u8; 6],
            sta_mac: [0u8; 6],
            hash_type: 4, // 4 = SHA-256
            eapol: Some(eapol_data),
            bcrypt_cost: 0,
            source: trimmed.to_string(),
        });
    }

    if hashes.is_empty() {
        return Err("No valid SHA-256 hashes found in file".into());
    }

    Ok(vec![HashGroup {
        essid: Vec::new(),
        hashes,
    }])
}

/// Detect hash mode from file content. Auto-detection for common formats.
/// Checks the first line only — multi-line files are parsed line-by-line later.
pub fn detect_mode(content: &str) -> Result<HashModeId, String> {
    let first_line = content.trim().lines().next().unwrap_or("");

    // WPA 22000 format: WPA*01*... or WPA*02*...
    if first_line.starts_with("WPA*") {
        return Ok(HashModeId::Wpa22000);
    }

    // WPA 16800 deprecated format: hex_pmkid*mac_ap*mac_sta*essid (3-4 hex fields)
    // Field constraints match is_16800_line() in multihash.rs:
    //   pmkid: 32 hex chars, mac_ap: 12 hex chars, mac_sta: 12 hex chars,
    //   essid: even-length hex (or absent for 3-field form)
    let parts: Vec<&str> = first_line.split('*').collect();
    if parts.len() == 4 || parts.len() == 3 {
        let is_hex =
            |s: &str, len: usize| s.len() == len && s.chars().all(|c| c.is_ascii_hexdigit());
        let pmkid_ok = is_hex(parts[0], 32);
        let mac_ap_ok = is_hex(parts[1], 12);
        let mac_sta_ok = is_hex(parts[2], 12);
        let essid_ok = if parts.len() == 4 {
            // Empty ESSID (trailing *) or valid even-length hex
            parts[3].is_empty()
                || (parts[3].len().is_multiple_of(2)
                    && parts[3].chars().all(|c| c.is_ascii_hexdigit()))
        } else {
            true // 3-field form: no ESSID
        };
        if pmkid_ok && mac_ap_ok && mac_sta_ok && essid_ok {
            return Ok(HashModeId::Wpa22000);
        }
    }

    // bcrypt: $2a$, $2b$, $2x$, or $2y$ prefix
    if first_line.starts_with("$2a$")
        || first_line.starts_with("$2b$")
        || first_line.starts_with("$2x$")
        || first_line.starts_with("$2y$")
    {
        return Ok(HashModeId::Bcrypt);
    }

    // SHA-256: 64 hex chars. Must check BEFORE MD5/NTLM (32 chars).
    if first_line.len() == 64 && first_line.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(HashModeId::Sha256);
    }

    // MD5 / NTLM: both are 32 hex chars. Default to MD5 for auto-detect.
    // User can override with --mode ntlm.
    if first_line.len() == 32 && first_line.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(HashModeId::Md5);
    }

    Err(format!(
        "Could not detect hash mode from content. First 50 chars: '{}'. \
         Use --mode ntlm for NTLM hashes (32 hex chars, same format as MD5). \
         Use --mode sha256 for SHA-256 hashes (64 hex chars).",
        first_line.chars().take(50).collect::<String>()
    ))
}

/// Parse NTLM hashes (mode 1000): 32-char hex lines → single group.
/// NTLM = MD4(UTF-16LE(password)), 16-byte digest.
/// Same format as MD5 — user selects via --mode ntlm.
fn parse_ntlm_hashes(content: &str) -> Result<Vec<HashGroup>, String> {
    let mut hashes = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if trimmed.len() != 32 {
            return Err(format!(
                "NTLM parse error line {}: expected 32 hex chars, got {} chars: '{}'",
                line_num + 1,
                trimmed.len(),
                trimmed
            ));
        }

        if !trimmed.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(format!(
                "NTLM parse error line {}: non-hex character in hash: '{}'",
                line_num + 1,
                trimmed
            ));
        }

        let mut target = [0u8; 16];
        for i in 0..16 {
            target[i] = u8::from_str_radix(&trimmed[i * 2..i * 2 + 2], 16).map_err(|e| {
                format!(
                    "NTLM parse error line {}: hex decode failed: {e}",
                    line_num + 1
                )
            })?;
        }

        hashes.push(ParsedHash {
            target,
            ap_mac: [0u8; 6],
            sta_mac: [0u8; 6],
            hash_type: 3, // 3 = NTLM
            eapol: None,
            bcrypt_cost: 0,
            source: trimmed.to_string(),
        });
    }

    if hashes.is_empty() {
        return Err("No valid NTLM hashes found in file".into());
    }

    Ok(vec![HashGroup {
        essid: Vec::new(),
        hashes,
    }])
}

// ==================== bcrypt (mode 3200) ====================

/// bcrypt custom base64 alphabet: ./A-Za-z0-9 (different from standard base64)
const BCRYPT_BASE64: &[u8] = b"./ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

/// Decode a character using bcrypt's custom base64 alphabet.
fn bcrypt_b64_decode_char(c: u8) -> Option<u8> {
    BCRYPT_BASE64.iter().position(|&b| b == c).map(|p| p as u8)
}

/// Decode bcrypt base64 string to raw bytes.
/// bcrypt base64 uses a different alphabet than standard base64: ./A-Za-z0-9
/// No padding characters — the output length is determined by the input length.
pub(crate) fn bcrypt_b64_decode(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    let mut i = 0;
    while i + 4 <= input.len() {
        let c0 = bcrypt_b64_decode_char(input[i]).ok_or("invalid bcrypt base64 char")?;
        let c1 = bcrypt_b64_decode_char(input[i + 1]).ok_or("invalid bcrypt base64 char")?;
        let c2 = bcrypt_b64_decode_char(input[i + 2]).ok_or("invalid bcrypt base64 char")?;
        let c3 = bcrypt_b64_decode_char(input[i + 3]).ok_or("invalid bcrypt base64 char")?;

        // 24 bits → 3 bytes
        let b0 = (c0 << 2) | (c1 >> 4);
        let b1 = ((c1 & 0x0f) << 4) | (c2 >> 2);
        let b2 = ((c2 & 0x03) << 6) | c3;

        output.push(b0);
        output.push(b1);
        output.push(b2);
        i += 4;
    }

    // Handle remaining 2-3 characters
    let remaining = input.len() - i;
    if remaining == 3 {
        let c0 = bcrypt_b64_decode_char(input[i]).ok_or("invalid bcrypt base64 char")?;
        let c1 = bcrypt_b64_decode_char(input[i + 1]).ok_or("invalid bcrypt base64 char")?;
        let c2 = bcrypt_b64_decode_char(input[i + 2]).ok_or("invalid bcrypt base64 char")?;

        let b0 = (c0 << 2) | (c1 >> 4);
        let b1 = ((c1 & 0x0f) << 4) | (c2 >> 2);
        output.push(b0);
        output.push(b1);
    } else if remaining == 2 {
        let c0 = bcrypt_b64_decode_char(input[i]).ok_or("invalid bcrypt base64 char")?;
        let c1 = bcrypt_b64_decode_char(input[i + 1]).ok_or("invalid bcrypt base64 char")?;

        let b0 = (c0 << 2) | (c1 >> 4);
        output.push(b0);
    } else if remaining == 1 {
        return Err("invalid bcrypt base64: trailing single char".into());
    }

    Ok(output)
}

/// Parse bcrypt hash file: one $2*$ hash per line.
/// Groups by salt (each hash typically has a unique salt → one hash per group).
/// Format: $2a$05$<22-char salt><31-char hash>
///   - Signature: $2a$, $2b$, $2x$, or $2y$
///   - Cost: 2-digit decimal (04-31)
///   - Salt: 22 chars bcrypt-base64 → 16 bytes
///   - Hash: 31 chars bcrypt-base64 → 23 bytes (6 u32s, last byte masked with 0xffffff00)
fn parse_bcrypt_hashes(content: &str) -> Result<Vec<HashGroup>, String> {
    let mut groups: std::collections::BTreeMap<Vec<u8>, Vec<ParsedHash>> =
        std::collections::BTreeMap::new();

    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Guard: bcrypt hash is at least 60 chars ($2a$05$ + 22 salt + 31 hash)
        // Check before slicing to avoid panic on short/malformed lines
        if trimmed.len() < 60 {
            return Err(format!(
                "bcrypt parse error line {}: line too short ({} chars, need >= 60): '{}'",
                line_num + 1,
                trimmed.len(),
                trimmed
            ));
        }

        // Validate signature (bytes — a multi-byte UTF-8 start would make
        // &trimmed[..4] panic on a non-char boundary; R105 LOW-1 fix)
        let sig: [u8; 4] = trimmed.as_bytes()[..4].try_into().unwrap();
        if sig != *b"$2a$" && sig != *b"$2b$" && sig != *b"$2x$" && sig != *b"$2y$" {
            return Err(format!(
                "bcrypt parse error line {}: invalid signature '{}', expected $2a$/$2b$/$2x$/$2y$",
                line_num + 1,
                String::from_utf8_lossy(&sig)
            ));
        }
        // $2x$ uses signed-char Blowfish (old buggy variant). Our GPU uses unsigned uchar.
        // Passwords with bytes >= 0x80 would produce wrong hashes. Warn but don't reject.
        if sig == *b"$2x$" {
            eprintln!("[!] Warning: $2x$ signature uses signed-char Blowfish variant.");
            eprintln!("[!] Passwords with bytes >= 0x80 may not crack correctly.");
        }

        // Format: $2a$05$<22 salt chars><31 hash chars>
        // Total: 4 (sig) + 2 (cost) + 1 ($) + 22 (salt) + 31 (hash) = 60 chars
        // (Length already validated above)

        // Parse cost (2 digits after $2a$)
        let cost_str = &trimmed[4..6];
        let cost: u32 = cost_str.parse().map_err(|_| {
            format!(
                "bcrypt parse error line {}: invalid cost '{}'",
                line_num + 1,
                cost_str
            )
        })?;
        if cost > 31 {
            return Err(format!(
                "bcrypt parse error line {}: cost {} exceeds max 31",
                line_num + 1,
                cost
            ));
        }

        // Validate the $ separator after cost
        if &trimmed[6..7] != "$" {
            return Err(format!(
                "bcrypt parse error line {}: expected '$' after cost, got '{}'",
                line_num + 1,
                &trimmed[6..7]
            ));
        }

        // Extract salt (22 chars) and hash (31 chars)
        let salt_b64 = &trimmed.as_bytes()[7..29];
        let hash_b64 = &trimmed.as_bytes()[29..60];

        // Decode salt: 22 base64 chars → 16 bytes
        let salt = bcrypt_b64_decode(salt_b64).map_err(|e| {
            format!(
                "bcrypt parse error line {}: salt decode failed: {e}",
                line_num + 1
            )
        })?;
        if salt.len() < 16 {
            return Err(format!(
                "bcrypt parse error line {}: salt too short ({} bytes, expected 16)",
                line_num + 1,
                salt.len()
            ));
        }

        // Decode hash: 31 base64 chars → 23 bytes
        let digest = bcrypt_b64_decode(hash_b64).map_err(|e| {
            format!(
                "bcrypt parse error line {}: hash decode failed: {e}",
                line_num + 1
            )
        })?;
        if digest.len() < 23 {
            return Err(format!(
                "bcrypt parse error line {}: digest too short ({} bytes, expected 23)",
                line_num + 1,
                digest.len()
            ));
        }

        // Encode into ParsedHash fields:
        // - target[0..16]: first 16 bytes of 24-byte digest (pad to 24 with 0)
        // - ap_mac[0..6]: digest bytes 16-21
        // - sta_mac[0..2]: digest bytes 22-23 (rest 0)
        // - hash_type: 5 (bcrypt)
        // - bcrypt_cost: cost (the raw cost value, not 2^cost)
        // - essid (HashGroup): 16-byte salt
        let mut target = [0u8; 16];
        target.copy_from_slice(&digest[..16]);

        let mut ap_mac = [0u8; 6];
        ap_mac.copy_from_slice(&digest[16..22]);

        let mut sta_mac = [0u8; 6];
        sta_mac[0] = digest[22];

        let salt16 = &salt[..16];

        let parsed = ParsedHash {
            target,
            ap_mac,
            sta_mac,
            hash_type: 5, // bcrypt
            eapol: None,
            bcrypt_cost: cost,
            source: trimmed.to_string(),
        };

        // Group by (salt, cost) — hashes with the same salt but different
        // costs need different iteration counts and must be in separate groups.
        let mut group_key = salt16.to_vec();
        group_key.push(cost as u8);
        groups.entry(group_key).or_default().push(parsed);
    }

    if groups.is_empty() {
        return Err("No valid bcrypt hashes found in file".into());
    }

    let result: Vec<HashGroup> = groups
        .into_iter()
        .map(|(key, hashes)| HashGroup {
            // essid = first 16 bytes of key (the salt), without the cost byte
            essid: key[..16].to_vec(),
            hashes,
        })
        .collect();

    Ok(result)
}
