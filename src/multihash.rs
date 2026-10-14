//! Multi-hash parsing: parse all hashes from a WPA hash file.
//!
//! All hashes sharing the same ESSID (PBKDF2 salt) are grouped together
//! so the GPU computes PBKDF2 once per password and compares against
//! all targets in the group.

#![allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::needless_pass_by_value
)]

use std::collections::BTreeMap;

const MAX_HEX_LEN: usize = 1024;

/// A parsed WPA hash — PMKID or EAPOL.
#[derive(Clone, Debug)]
pub struct ParsedHash {
    /// 16-byte target: PMKID (type 01) or MIC (type 02), or first 16 bytes of bcrypt digest
    pub target: [u8; 16],
    /// AP MAC (6 bytes), or bcrypt digest bytes 16-21
    pub ap_mac: [u8; 6],
    /// Station MAC (6 bytes), or bcrypt digest bytes 22-23 + 4 padding
    pub sta_mac: [u8; 6],
    /// Hash type: 0 = PMKID, 1 = EAPOL, 2 = MD5, 3 = NTLM, 4 = SHA-256, 5 = bcrypt
    pub hash_type: u32,
    /// EAPOL data (only for type 1 = EAPOL)
    pub eapol: Option<EapolData>,
    /// bcrypt cost: 2^cost iterations (0 for non-bcrypt)
    #[allow(dead_code)]
    pub bcrypt_cost: u32,
    /// Source line (for display/error messages)
    #[allow(dead_code)]
    pub source: String,
}

/// EAPOL-specific data for a parsed EAPOL hash.
#[derive(Clone, Debug)]
pub struct EapolData {
    pub anonce: [u8; 32],
    /// EAPOL frame with MIC field zeroed (bytes 81-96)
    pub eapol: Vec<u8>,
    /// EAPOL key descriptor version (1=HMAC-MD5/WPA1, 2=HMAC-SHA1/WPA2, 3=AES-128-CMAC/WPA2)
    pub key_version: u8,
    /// Nonce error corrections count (from message_pair, or --nonce-error-corrections override)
    pub nonce_error_corrections: u32,
    /// 1 if little-endian replay counter detected (message_pair bit 5)
    pub detected_le: u32,
    /// 1 if big-endian replay counter detected (message_pair bit 6)
    pub detected_be: u32,
}

/// A group of hashes sharing the same ESSID (salt).
/// The GPU computes PBKDF2 once per password, then compares against all hashes.
#[derive(Clone, Debug)]
pub struct HashGroup {
    pub essid: Vec<u8>,
    pub hashes: Vec<ParsedHash>,
}

/// Parse all hash lines from a WPA hash file content.
///
/// Groups hashes by ESSID. If all hashes share the same ESSID, returns
/// a single group. If multiple ESSIDs are found, returns an error
/// listing the groups so the caller can run separately for each.
pub fn parse_hash_file(content: &str) -> Result<Vec<HashGroup>, String> {
    let mut groups: BTreeMap<Vec<u8>, Vec<ParsedHash>> = BTreeMap::new();
    let mut skipped = 0usize;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match parse_hash_line(trimmed) {
            Ok((parsed, essid)) => {
                groups.entry(essid).or_default().push(parsed);
            }
            Err(e) => {
                if !trimmed.is_empty() {
                    skipped += 1;
                    eprintln!("[!] Skipping line: {e}");
                }
            }
        }
    }

    if groups.is_empty() {
        return Err("No PMKID/EAPOL hash found in hash file".to_string());
    }

    if skipped > 0 {
        eprintln!("[*] Skipped {skipped} unparseable line(s)");
    }

    // Report ESSID groups
    if groups.len() > 1 {
        eprintln!(
            "[!] Warning: {essid_count} different ESSIDs found — will crack each separately",
            essid_count = groups.len()
        );
        for (essid, hashes) in &groups {
            let essid_display = String::from_utf8_lossy(essid);
            eprintln!(
                "    ESSID '{essid_display}': {n} hash(es)",
                n = hashes.len()
            );
        }
    }

    let result: Vec<HashGroup> = groups
        .into_iter()
        .map(|(essid, hashes)| HashGroup { essid, hashes })
        .collect();

    Ok(result)
}

/// Parse a single WPA hash line (22000 or 16800 format).
/// Returns (ParsedHash, ESSID) on success.
fn parse_hash_line(line: &str) -> Result<(ParsedHash, Vec<u8>), String> {
    if line.len() > 4096 {
        return Err(format!(
            "hash line too long ({} chars, max 4096)",
            line.len()
        ));
    }

    // WPA*01*PMKID*MAC_AP*MAC_STA*ESSID*** (22000 PMKID)
    // WPA*02*MIC*MAC_AP*MAC_STA*ESSID*ANONCE*EAPOL*MESSAGEPAIR (22000 EAPOL)
    if line.starts_with("WPA*") {
        return parse_22000(line);
    }

    // PMKID*MAC_AP*MAC_STA*ESSID (deprecated 16800)
    if is_16800_line(line) {
        eprintln!("[*] Detected deprecated 16800 format (PMKID*MAC_AP*MAC_STA*ESSID)");
        eprintln!("[!] Consider converting to 22000 format: hcxpcapngtool -o out.22000 <capture>");
        return parse_16800(line);
    }

    Err(format!("unrecognized hash format: {line}"))
}

/// Parse a 22000-format hash line.
fn parse_22000(line: &str) -> Result<(ParsedHash, Vec<u8>), String> {
    let parts: Vec<&str> = line.split('*').collect();
    if parts.len() < 6 {
        return Err(format!("22000 line has too few fields: {line}"));
    }

    let hash_type_str = parts[1];
    let target_hex = parts[2];
    let ap_mac_hex = parts[3];
    let sta_mac_hex = parts[4];
    let essid_hex = parts[5];

    let target = hex_decode_16(target_hex)?;
    let ap_mac = hex_decode_6(ap_mac_hex)?;
    let sta_mac = hex_decode_6(sta_mac_hex)?;
    let essid = hex_decode(essid_hex)?;

    if essid.len() > 64 {
        return Err(format!("ESSID too long ({} bytes, max 64)", essid.len()));
    }

    let parsed = if hash_type_str == "01" {
        // PMKID
        eprintln!("[*] Detected PMKID hash (type 01)");
        ParsedHash {
            target,
            ap_mac,
            sta_mac,
            hash_type: 0,
            eapol: None,
            bcrypt_cost: 0,
            source: line.to_string(),
        }
    } else if hash_type_str == "02" {
        // EAPOL
        eprintln!("[*] Detected EAPOL hash (type 02) — 4-way handshake cracking mode");
        if parts.len() < 8 {
            return Err(format!(
                "EAPOL line needs >=8 fields (WPA*02*MIC*AP*STA*ESSID*ANONCE*EAPOL[+MP]), got {}",
                parts.len()
            ));
        }

        let anonce_hex = parts[6];
        let eapol_hex = parts[7];

        let anonce = hex_decode(anonce_hex)?;
        if anonce.len() != 32 {
            return Err(format!(
                "ANonce must be 32 bytes (64 hex chars), got {} bytes",
                anonce.len()
            ));
        }

        let mut eapol = hex_decode(eapol_hex)?;

        // Validate EAPOL length — minimum complete EAPOL-Key frame is 99 bytes
        // (81 pre-MIC + 16 MIC + 2 Key Data Length). Frames of 97-98 are malformed.
        if eapol.len() < 99 {
            return Err(format!(
                "EAPOL frame too short ({} bytes, min 99)",
                eapol.len()
            ));
        }
        if eapol.len() > 256 {
            return Err(format!(
                "EAPOL frame too long ({} bytes, max 256)",
                eapol.len()
            ));
        }

        // Check key descriptor version (bytes 5-6, big-endian)
        let key_info = (u16::from(eapol[5]) << 8) | u16::from(eapol[6]);
        let key_version = key_info & 0x7;
        if key_version != 1 && key_version != 2 && key_version != 3 {
            return Err(format!(
                "EAPOL key descriptor version {key_version} is not supported — only version 1 (HMAC-MD5/WPA1), version 2 (HMAC-SHA1/WPA2-CCMP), and version 3 (AES-128-CMAC/WPA2)"
            ));
        }

        // Zero the MIC field (offset 81, 16 bytes)
        for i in 81..97 {
            eapol[i] = 0;
        }

        let mut anonce_arr = [0u8; 32];
        anonce_arr.copy_from_slice(&anonce);

        let keyver_msg = match key_version {
            1 => "HMAC-MD5 (WPA1/TKIP)",
            2 => "HMAC-SHA1 (WPA2/CCMP)",
            3 => "AES-128-CMAC (WPA2)",
            _ => "unknown",
        };
        eprintln!("[*] EAPOL key version: {key_version} ({keyver_msg})");

        // Parse message_pair field (field 8) to determine nonce error corrections.
        // message_pair is a single byte encoded as 2 hex chars.
        // Bit 4: ap-less attack → NEC = 0
        // Bit 5: LE replay counter detected → detected_le = 1
        // Bit 6: BE replay counter detected → detected_be = 1
        // Bit 7: replaycount NOT checked → NEC = 3 (default)
        // If bit 7 = 0 (replaycount checked) → NEC = 0
        // If neither bit 5 nor 6 → try both byte orders (detected_le = detected_be = 1)
        const DEFAULT_NEC: u32 = 3;
        let message_pair: u8 = if parts.len() >= 9 && !parts[8].is_empty() {
            u8::from_str_radix(parts[8], 16)
                .map_err(|e| format!("invalid message_pair hex '{}': {e}", parts[8]))?
        } else {
            0 // Missing/empty → assume replaycount was checked
        };

        let bit4 = message_pair & (1 << 4) != 0; // ap-less
        let bit5 = message_pair & (1 << 5) != 0; // LE detected
        let bit6 = message_pair & (1 << 6) != 0; // BE detected
        let bit7 = message_pair & (1 << 7) != 0; // replaycount not checked

        let nec = if bit4 || !bit7 { 0 } else { DEFAULT_NEC };

        // If neither LE nor BE detected, try both (hashcat default).
        // Hashcat uses if/else-if priority: bit5 wins over bit6.
        let (det_le, det_be) = if bit5 {
            (1u32, 0u32)
        } else if bit6 {
            (0u32, 1u32)
        } else {
            (1u32, 1u32)
        };

        if nec > 0 {
            eprintln!("[*] Nonce error corrections: {nec} (replaycount not checked in capture)");
        }

        ParsedHash {
            target,
            ap_mac,
            sta_mac,
            hash_type: 1,
            eapol: Some(EapolData {
                anonce: anonce_arr,
                eapol,
                key_version: key_version as u8,
                nonce_error_corrections: nec,
                detected_le: det_le,
                detected_be: det_be,
            }),
            bcrypt_cost: 0,
            source: line.to_string(),
        }
    } else {
        return Err(format!(
            "Expected PMKID (type 01) or EAPOL (type 02), got type: {hash_type_str}"
        ));
    };

    Ok((parsed, essid))
}

/// Parse a deprecated 16800-format hash line.
fn parse_16800(line: &str) -> Result<(ParsedHash, Vec<u8>), String> {
    let parts: Vec<&str> = line.split('*').collect();
    if parts.len() < 3 {
        return Err(format!("16800 line has too few fields: {line}"));
    }

    let pmkid_hex = parts[0];
    let ap_mac_hex = parts[1];
    let sta_mac_hex = parts[2];
    let essid_hex = if parts.len() >= 4 { parts[3] } else { "" };

    let target = hex_decode_16(pmkid_hex)?;
    let ap_mac = hex_decode_6(ap_mac_hex)?;
    let sta_mac = hex_decode_6(sta_mac_hex)?;
    let essid = hex_decode(essid_hex)?;

    if essid.len() > 64 {
        return Err(format!("ESSID too long ({} bytes, max 64)", essid.len()));
    }

    let parsed = ParsedHash {
        target,
        ap_mac,
        sta_mac,
        hash_type: 0,
        eapol: None,
        bcrypt_cost: 0,
        source: line.to_string(),
    };

    Ok((parsed, essid))
}

/// Detect a deprecated 16800-format PMKID line.
fn is_16800_line(line: &str) -> bool {
    if line.is_empty() || line.starts_with("WPA*") {
        return false;
    }
    let parts: Vec<&str> = line.split('*').collect();
    let is_hex = |s: &str, len: usize| s.len() == len && s.chars().all(|c| c.is_ascii_hexdigit());
    parts.len() >= 3
        && is_hex(parts[0], 32)
        && is_hex(parts[1], 12)
        && is_hex(parts[2], 12)
        && (parts.len() == 3
            || (parts[3].len().is_multiple_of(2)
                && parts[3].chars().all(|c| c.is_ascii_hexdigit())))
}

// === hccapx binary format ===

/// hccapx record size (packed struct, no padding).
const HCCAPX_RECORD_SIZE: usize = 393;

/// hccapx magic: "HCPX" as a little-endian u32.
const HCCAPX_SIGNATURE: u32 = 0x58504348;

/// hccapx format version.
const HCCAPX_VERSION: u32 = 4;

/// Check whether a byte slice starts with a valid hccapx header
/// (signature 0x58504348). Only the signature is checked for file-type
/// detection, matching hashcat's is_hccapx() (line 140-155). Version
/// is validated per-record in parse_hccapx_to_22000().
pub fn is_hccapx(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    let signature = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    signature == HCCAPX_SIGNATURE
}

/// Convert hccapx binary data into 22000 text format, one line per record.
///
/// Each 393-byte record becomes:
/// `WPA*02*<keymic>*<mac_ap>*<mac_sta>*<essid>*<anonce>*<eapol>*<message_pair>`
///
/// Follows hashcat module_22000.c validation: essid_len must be <= 32,
/// eapol_len must be 1..=256. Bad records are silently skipped (matching
/// hashcat's `if (parser_status != PARSER_OK) continue` behavior) so a
/// partially corrupted file still yields the valid hashes it contains.
pub fn parse_hccapx_to_22000(data: &[u8]) -> Result<String, String> {
    if !is_hccapx(data) {
        return Err("not a valid hccapx file (signature/version mismatch)".to_string());
    }

    let mut lines = Vec::new();
    let mut offset = 0usize;
    let mut skipped = 0u32;

    while offset < data.len() {
        let remaining = data.len() - offset;
        // Truncated trailing data — stop (hashcat silently stops on short fread).
        if remaining < HCCAPX_RECORD_SIZE {
            if !lines.is_empty() {
                if skipped > 0 {
                    eprintln!(
                        "[*] hccapx: {skipped} record(s) skipped, {remaining} trailing bytes"
                    );
                }
                return Ok(lines.join("\n"));
            }
            return Err(format!(
                "truncated hccapx record: {remaining} bytes remaining, need {HCCAPX_RECORD_SIZE}"
            ));
        }

        let rec = &data[offset..offset + HCCAPX_RECORD_SIZE];

        // Per-record signature/version validation (matches hashcat's per-record check).
        let sig = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
        let ver = u32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
        if sig != HCCAPX_SIGNATURE || ver != HCCAPX_VERSION {
            skipped += 1;
            offset += HCCAPX_RECORD_SIZE;
            continue;
        }

        // Field offsets (packed struct, little-endian multi-byte).
        let essid_len = rec[9] as usize;
        let eapol_len = u16::from_le_bytes([rec[135], rec[136]]) as usize;

        // Skip records with invalid field lengths (hashcat: continue on parse error).
        if essid_len > 32 || !(1..=256).contains(&eapol_len) {
            skipped += 1;
            offset += HCCAPX_RECORD_SIZE;
            continue;
        }

        let keymic = &rec[43..59];
        let mac_ap = &rec[59..65];
        let essid = &rec[10..10 + essid_len];
        let nonce_ap = &rec[65..97];
        let mac_sta = &rec[97..103];
        let eapol = &rec[137..137 + eapol_len];
        let message_pair = rec[8];

        let line = format!(
            "WPA*02*{}*{}*{}*{}*{}*{}*{:02x}",
            hex_encode(keymic),
            hex_encode(mac_ap),
            hex_encode(mac_sta),
            hex_encode(essid),
            hex_encode(nonce_ap),
            hex_encode(eapol),
            message_pair,
        );

        lines.push(line);
        offset += HCCAPX_RECORD_SIZE;
    }

    if lines.is_empty() {
        return Err("hccapx file contained no valid records".to_string());
    }
    if skipped > 0 {
        eprintln!("[*] hccapx: {skipped} record(s) skipped (invalid/corrupted)");
    }
    Ok(lines.join("\n"))
}

/// Hex-encode a byte slice to a lowercase string.
fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// === Hex helpers ===

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err(format!("hex string has odd length ({})", s.len()));
    }
    if !s.is_ascii() {
        return Err("hex string contains non-ASCII characters".to_string());
    }
    if s.len() > MAX_HEX_LEN {
        return Err(format!(
            "hex string too long ({} chars, max {MAX_HEX_LEN})",
            s.len()
        ));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| format!("hex decode error at {i}: {e}"))
        })
        .collect()
}

fn hex_decode_16(s: &str) -> Result<[u8; 16], String> {
    let bytes = hex_decode(s)?;
    if bytes.len() != 16 {
        return Err(format!(
            "expected 16 bytes (32 hex chars), got {} bytes",
            bytes.len()
        ));
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

fn hex_decode_6(s: &str) -> Result<[u8; 6], String> {
    let bytes = hex_decode(s)?;
    if bytes.len() != 6 {
        return Err(format!(
            "expected 6 bytes (12 hex chars), got {} bytes",
            bytes.len()
        ));
    }
    let mut arr = [0u8; 6];
    arr.copy_from_slice(&bytes);
    Ok(arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pmkid_22000() {
        let line = "WPA*01*4d4fe7aac3a2cecab195321ceb99a7d0*fc690c158264*f4747f87f9f4*686173686361742d6573736964***";
        let (parsed, essid) = parse_hash_line(line).unwrap();
        assert_eq!(parsed.hash_type, 0);
        assert_eq!(essid, b"hashcat-essid");
        assert!(parsed.eapol.is_none());
    }

    #[test]
    fn test_parse_16800() {
        let line =
            "4d4fe7aac3a2cecab195321ceb99a7d0*fc690c158264*f4747f87f9f4*686173686361742d6573736964";
        let (parsed, essid) = parse_hash_line(line).unwrap();
        assert_eq!(parsed.hash_type, 0);
        assert_eq!(essid, b"hashcat-essid");
    }

    // EAPOL test vector: password123 + PACE969, keyver 2
    // 99-byte EAPOL-Key frame, MIC zeroed in parsed data.
    const EAPOL_HEX: &str = "0103005f02008a00100100000000000000112233445566778899aabbccddeeff00112233445566778899aabbccddeeff000000000000000000000000000000000000000000000000000000000000000000d7dd0c4dde57de1a0e751d18c314aca80000";
    const ANONCE_HEX: &str = "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899";
    const MIC_HEX: &str = "d7dd0c4dde57de1a0e751d18c314aca8";

    fn eapol_line(mp: &str) -> String {
        format!("WPA*02*{MIC_HEX}*4604ba734d4e*89acf0e761f4*50414345393639*{ANONCE_HEX}*{EAPOL_HEX}*{mp}")
    }

    #[test]
    fn test_parse_eapol_nec_from_message_pair() {
        // message_pair=80 (bit7: replaycount NOT checked → NEC=3, no LE/BE → both)
        let (parsed, essid) = parse_hash_line(&eapol_line("80")).unwrap();
        assert_eq!(parsed.hash_type, 1);
        assert_eq!(essid, b"PACE969");
        let e = parsed.eapol.unwrap();
        assert_eq!(e.key_version, 2);
        assert_eq!(e.nonce_error_corrections, 3);
        assert_eq!(e.detected_le, 1);
        assert_eq!(e.detected_be, 1);
    }

    #[test]
    fn test_parse_eapol_nec_zero_when_replaycount_checked() {
        // message_pair=00 (bit7=0: replaycount checked → NEC=0)
        let (parsed, _) = parse_hash_line(&eapol_line("00")).unwrap();
        let e = parsed.eapol.unwrap();
        assert_eq!(e.nonce_error_corrections, 0);
    }

    #[test]
    fn test_parse_eapol_nec_zero_ap_less() {
        // message_pair=10 (bit4: ap-less attack → NEC=0)
        let (parsed, _) = parse_hash_line(&eapol_line("10")).unwrap();
        let e = parsed.eapol.unwrap();
        assert_eq!(e.nonce_error_corrections, 0);
    }

    #[test]
    fn test_parse_eapol_nec_le_detected() {
        // message_pair=a0 (bit7=1 + bit5=1: LE detected, NEC=3, det_le=1, det_be=0)
        let (parsed, _) = parse_hash_line(&eapol_line("a0")).unwrap();
        let e = parsed.eapol.unwrap();
        assert_eq!(e.nonce_error_corrections, 3);
        assert_eq!(e.detected_le, 1);
        assert_eq!(e.detected_be, 0);
    }

    #[test]
    fn test_parse_eapol_nec_be_detected() {
        // message_pair=c0 (bit7=1 + bit6=1: BE detected, NEC=3, det_le=0, det_be=1)
        let (parsed, _) = parse_hash_line(&eapol_line("c0")).unwrap();
        let e = parsed.eapol.unwrap();
        assert_eq!(e.nonce_error_corrections, 3);
        assert_eq!(e.detected_le, 0);
        assert_eq!(e.detected_be, 1);
    }

    #[test]
    fn test_parse_eapol_no_message_pair() {
        // No message_pair field (only 8 fields) → default to 0 (NEC=0)
        let line = format!(
            "WPA*02*{MIC_HEX}*4604ba734d4e*89acf0e761f4*50414345393639*{ANONCE_HEX}*{EAPOL_HEX}"
        );
        let (parsed, _) = parse_hash_line(&line).unwrap();
        let e = parsed.eapol.unwrap();
        assert_eq!(e.nonce_error_corrections, 0);
    }

    // === hccapx binary format tests ===

    /// Build a 393-byte hccapx record for testing.
    fn make_hccapx_record(
        essid: &[u8],
        keymic: &[u8; 16],
        mac_ap: &[u8; 6],
        mac_sta: &[u8; 6],
        nonce_ap: &[u8; 32],
        eapol: &[u8],
        message_pair: u8,
        keyver: u8,
    ) -> Vec<u8> {
        assert!(essid.len() <= 32);
        assert!(eapol.len() <= 256);
        let mut rec = vec![0u8; HCCAPX_RECORD_SIZE];
        // signature (offset 0, 4 bytes LE)
        rec[0..4].copy_from_slice(&HCCAPX_SIGNATURE.to_le_bytes());
        // version (offset 4, 4 bytes LE)
        rec[4..8].copy_from_slice(&HCCAPX_VERSION.to_le_bytes());
        // message_pair (offset 8)
        rec[8] = message_pair;
        // essid_len (offset 9)
        rec[9] = essid.len() as u8;
        // essid (offset 10, 32 bytes, null-padded)
        rec[10..10 + essid.len()].copy_from_slice(essid);
        // keyver (offset 42)
        rec[42] = keyver;
        // keymic (offset 43, 16 bytes)
        rec[43..59].copy_from_slice(keymic);
        // mac_ap (offset 59, 6 bytes)
        rec[59..65].copy_from_slice(mac_ap);
        // nonce_ap / ANonce (offset 65, 32 bytes)
        rec[65..97].copy_from_slice(nonce_ap);
        // mac_sta (offset 97, 6 bytes)
        rec[97..103].copy_from_slice(mac_sta);
        // nonce_sta / SNonce (offset 103, 32 bytes) — left as zeros for test
        // eapol_len (offset 135, 2 bytes LE)
        rec[135..137].copy_from_slice(&(eapol.len() as u16).to_le_bytes());
        // eapol (offset 137, 256 bytes, null-padded)
        rec[137..137 + eapol.len()].copy_from_slice(eapol);
        rec
    }

    #[test]
    fn test_hccapx_valid_single_record() {
        let essid = b"MyNet";
        let keymic = [0xaa; 16];
        let mac_ap = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
        let mac_sta = [0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc];
        let nonce_ap = [0x42u8; 32];
        // 99-byte EAPOL with keyver 2 (bits 5-6 = 0x0002)
        let mut eapol = vec![0u8; 99];
        eapol[0] = 0x01; // version
        eapol[1] = 0x03; // type
        eapol[2] = 0x00;
        eapol[3] = 0x5f; // length
        eapol[5] = 0x00; // key_info high byte
        eapol[6] = 0x02; // key_info low byte → keyver = 2

        let rec = make_hccapx_record(
            essid, &keymic, &mac_ap, &mac_sta, &nonce_ap, &eapol, 0x80, 2,
        );

        assert!(is_hccapx(&rec));
        let text = parse_hccapx_to_22000(&rec).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 1);

        // Verify the line starts with WPA*02*
        assert!(lines[0].starts_with("WPA*02*"));

        // Parse it through the normal 22000 parser to ensure round-trip works
        let (parsed, parsed_essid) = parse_hash_line(lines[0]).unwrap();
        assert_eq!(parsed.hash_type, 1); // EAPOL
        assert_eq!(parsed_essid, essid);
        let e = parsed.eapol.unwrap();
        assert_eq!(e.key_version, 2);
    }

    #[test]
    fn test_hccapx_multiple_records() {
        let essid = b"TestAP";
        let keymic = [0xbb; 16];
        let mac_ap = [0xde, 0xad, 0xbe, 0xef, 0x00, 0x01];
        let mac_sta = [0xca, 0xfe, 0xba, 0xbe, 0x02, 0x03];
        let nonce_ap = [0x55u8; 32];
        let mut eapol = vec![0u8; 99];
        eapol[5] = 0x00;
        eapol[6] = 0x02; // keyver 2

        let rec1 = make_hccapx_record(
            essid, &keymic, &mac_ap, &mac_sta, &nonce_ap, &eapol, 0x80, 2,
        );
        let rec2 = make_hccapx_record(
            essid,
            &[0xcc; 16],
            &mac_ap,
            &mac_sta,
            &nonce_ap,
            &eapol,
            0x80,
            2,
        );

        let mut data = rec1;
        data.extend_from_slice(&rec2);

        assert!(is_hccapx(&data));
        let text = parse_hccapx_to_22000(&data).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 2);

        // Both lines should parse successfully
        for line in &lines {
            let (parsed, _) = parse_hash_line(line).unwrap();
            assert_eq!(parsed.hash_type, 1);
        }
    }

    #[test]
    fn test_hccapx_invalid_signature() {
        let mut rec = vec![0u8; HCCAPX_RECORD_SIZE];
        // Wrong signature
        rec[0..4].copy_from_slice(&0x12345678u32.to_le_bytes());
        rec[4..8].copy_from_slice(&HCCAPX_VERSION.to_le_bytes());
        assert!(!is_hccapx(&rec));
        assert!(parse_hccapx_to_22000(&rec).is_err());
    }

    #[test]
    fn test_hccapx_invalid_version() {
        let mut rec = vec![0u8; HCCAPX_RECORD_SIZE];
        rec[0..4].copy_from_slice(&HCCAPX_SIGNATURE.to_le_bytes());
        rec[4..8].copy_from_slice(&3u32.to_le_bytes()); // version 3, not 4
                                                        // is_hccapx only checks signature (matching hashcat), so it returns true
        assert!(is_hccapx(&rec));
        // parse_hccapx_to_22000 skips records with wrong version → no valid records → Err
        assert!(parse_hccapx_to_22000(&rec).is_err());
    }

    #[test]
    fn test_hccapx_truncated_record() {
        let essid = b"Short";
        let keymic = [0xdd; 16];
        let mac_ap = [0x01; 6];
        let mac_sta = [0x02; 6];
        let nonce_ap = [0x03u8; 32];
        let mut eapol = vec![0u8; 99];
        eapol[5] = 0x00;
        eapol[6] = 0x02;

        let rec = make_hccapx_record(
            essid, &keymic, &mac_ap, &mac_sta, &nonce_ap, &eapol, 0x80, 2,
        );

        // Truncate by 10 bytes
        let truncated = &rec[..HCCAPX_RECORD_SIZE - 10];
        assert!(is_hccapx(truncated)); // header is still valid
        assert!(parse_hccapx_to_22000(truncated).is_err());
    }

    #[test]
    fn test_hccapx_essid_len_validation() {
        let mut rec = vec![0u8; HCCAPX_RECORD_SIZE];
        rec[0..4].copy_from_slice(&HCCAPX_SIGNATURE.to_le_bytes());
        rec[4..8].copy_from_slice(&HCCAPX_VERSION.to_le_bytes());
        rec[9] = 33; // essid_len > 32 → invalid
                     // Set valid eapol_len
        rec[135..137].copy_from_slice(&100u16.to_le_bytes());

        assert!(is_hccapx(&rec));
        // Bad records are now skipped (matching hashcat) → no valid records → Err
        let result = parse_hccapx_to_22000(&rec);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("no valid records"),
            "error should mention no valid records: {err}"
        );
    }

    #[test]
    fn test_hccapx_eapol_len_validation() {
        let mut rec = vec![0u8; HCCAPX_RECORD_SIZE];
        rec[0..4].copy_from_slice(&HCCAPX_SIGNATURE.to_le_bytes());
        rec[4..8].copy_from_slice(&HCCAPX_VERSION.to_le_bytes());
        rec[9] = 5; // valid essid_len
                    // eapol_len = 0 → invalid (must be 1..=256)
        rec[135..137].copy_from_slice(&0u16.to_le_bytes());

        assert!(is_hccapx(&rec));
        // Bad records are now skipped (matching hashcat) → no valid records → Err
        let result = parse_hccapx_to_22000(&rec);
        assert!(result.is_err());

        // Also test eapol_len > 256
        rec[135..137].copy_from_slice(&257u16.to_le_bytes());
        let result2 = parse_hccapx_to_22000(&rec);
        assert!(result2.is_err());
    }

    /// Skip-and-continue: a file with one valid + one invalid record should
    /// return the valid record's text (matching hashcat's behavior).
    #[test]
    fn test_hccapx_skip_and_continue() {
        // Build one valid record
        let mut valid = vec![0u8; HCCAPX_RECORD_SIZE];
        valid[0..4].copy_from_slice(&HCCAPX_SIGNATURE.to_le_bytes());
        valid[4..8].copy_from_slice(&HCCAPX_VERSION.to_le_bytes());
        valid[8] = 0x02; // message_pair
        valid[9] = 5; // essid_len
        valid[10..15].copy_from_slice(b"Test1");
        valid[42] = 2; // keyver
        valid[135..137].copy_from_slice(&100u16.to_le_bytes()); // eapol_len

        // Build one invalid record (bad version)
        let mut invalid = vec![0u8; HCCAPX_RECORD_SIZE];
        invalid[0..4].copy_from_slice(&HCCAPX_SIGNATURE.to_le_bytes());
        invalid[4..8].copy_from_slice(&99u32.to_le_bytes()); // wrong version

        // Concatenate: valid + invalid + trailing 100 bytes
        let mut data = valid.clone();
        data.extend_from_slice(&invalid);
        data.extend_from_slice(&[0u8; 100]);

        let result = parse_hccapx_to_22000(&data).unwrap();
        assert!(
            result.contains("WPA*02*"),
            "should contain the valid record"
        );
        assert_eq!(
            result.lines().count(),
            1,
            "should have exactly 1 valid line (invalid record skipped, trailing bytes ignored)"
        );
    }
}
