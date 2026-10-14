//! Potfile: track cracked hashes to skip already-cracked ones.
//!
//! Design:
//! - `--potfile <path>`: enables potfile. When a password is found, the
//!   hash:password pair is written to the potfile.
//! - On startup, the potfile is read. Hashes matching entries in the potfile
//!   are marked as cracked and skipped during cracking.
//! - `--potfile-disable`: disables potfile even if a default exists.
//! - `--show`: print cracked hashes from potfile and exit.
//! - `--left`: print uncracked hashes and exit.
//!
//! Potfile format (per line): `<hash_line>:<password>`
//! Where `<hash_line>` is the original 22000/16800 hash line and `<password>`
//! is the cracked password (hex-encoded if non-UTF8).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// Maximum potfile size (1 MB) — prevents DoS from malicious potfiles.
const MAX_POTFILE_SIZE: u64 = 1_048_576;

/// Read a potfile and return the set of cracked hash source lines.
/// Each line in the potfile is `hash_source:password`. We extract the
/// hash_source part to match against the input hash file.
pub fn load_potfile(path: &str) -> Result<HashSet<String>, String> {
    if !Path::new(path).exists() {
        return Ok(HashSet::new());
    }
    let meta = fs::metadata(path).map_err(|e| format!("failed to stat potfile: {e}"))?;
    if meta.len() > MAX_POTFILE_SIZE {
        return Err(format!(
            "potfile too large ({} bytes, max {MAX_POTFILE_SIZE})",
            meta.len()
        ));
    }
    if !meta.is_file() {
        return Err("potfile is not a regular file".into());
    }
    let content = fs::read_to_string(path).map_err(|e| format!("failed to read potfile: {e}"))?;
    let mut cracked = HashSet::new();
    for line in content.lines() {
        // Find the last colon — the password may contain colons
        if let Some(idx) = line.rfind(':') {
            let hash_line = &line[..idx];
            cracked.insert(hash_line.to_string());
        }
    }
    Ok(cracked)
}

/// Load full potfile entries (hash:password) as a Vec of (hash_source, password) pairs.
/// Used by --loopback to extract cracked passwords for re-running.
/// Unlike load_potfile (which returns only hash sources), this preserves passwords.
pub fn load_potfile_passwords(path: &str) -> Result<Vec<(String, String)>, String> {
    if !Path::new(path).exists() {
        return Ok(Vec::new());
    }
    let meta = fs::metadata(path).map_err(|e| format!("failed to stat potfile: {e}"))?;
    if meta.len() > MAX_POTFILE_SIZE {
        return Err(format!(
            "potfile too large ({} bytes, max {MAX_POTFILE_SIZE})",
            meta.len()
        ));
    }
    if !meta.is_file() {
        return Err("potfile is not a regular file".into());
    }
    let content = fs::read_to_string(path).map_err(|e| format!("failed to read potfile: {e}"))?;
    let mut entries = Vec::new();
    for line in content.lines() {
        if line.is_empty() {
            continue;
        }
        if let Some(idx) = line.rfind(':') {
            entries.push((line[..idx].to_string(), line[idx + 1..].to_string()));
        }
    }
    Ok(entries)
}

/// Append a cracked hash:password entry to the potfile.
/// The hash_line is the original source line from the hash file.
/// The password is hex-encoded if it contains non-printable bytes.
pub fn write_potfile_entry(path: &str, hash_line: &str, password: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("failed to open potfile for writing: {e}"))?;
    let pw_str = if std::str::from_utf8(password).is_ok()
        && !password.iter().any(|&b| b < 0x20 || b == 0x7f)
    {
        // Printable UTF-8 — write as-is (but wrap in $HEX[...] if it
        // contains a colon, which would break the potfile format)
        let s = String::from_utf8_lossy(password);
        // Also hex-encode if the password starts with $HEX[ to avoid
        // ambiguity on reload (decode_hex_password would misinterpret it).
        if s.contains(':') || s.contains('\n') || s.contains('\r') || s.starts_with("$HEX[") {
            format!("$HEX[{}]", hex_encode(password))
        } else {
            s.into_owned()
        }
    } else {
        // Non-printable — hex-encode with $HEX[...] wrapper
        format!("$HEX[{}]", hex_encode(password))
    };
    writeln!(file, "{hash_line}:{pw_str}")
        .map_err(|e| format!("failed to write potfile entry: {e}"))?;
    file.sync_all()
        .map_err(|e| format!("failed to sync potfile: {e}"))?;
    Ok(())
}

/// Decode a $HEX[...] wrapped password back to raw bytes.
/// If the password doesn't have the $HEX[] wrapper, return as-is.
#[allow(dead_code)] // used in tests only
pub fn decode_hex_password(s: &str) -> Vec<u8> {
    if let Some(rest) = s.strip_prefix("$HEX[") {
        if let Some(hex) = rest.strip_suffix(']') {
            return hex_decode(hex).unwrap_or_else(|_| s.as_bytes().to_vec());
        }
    }
    s.as_bytes().to_vec()
}

/// Show cracked hashes: read potfile, print matching entries from the hash file.
/// Builds a HashMap from the potfile (hash_line → password) for O(n+m) lookup.
pub fn show_cracked(potfile_path: &str, hash_content: &str) -> Result<Vec<String>, String> {
    // Build a HashMap of hash_line → password from the potfile (single read).
    // Uses the same size + is_file guards as load_potfile for defense-in-depth.
    let mut potfile_map: HashMap<String, String> = HashMap::new();
    if Path::new(potfile_path).exists() {
        let meta =
            fs::metadata(potfile_path).map_err(|e| format!("failed to stat potfile: {e}"))?;
        if meta.len() > MAX_POTFILE_SIZE {
            return Err(format!(
                "potfile too large ({} bytes, max {MAX_POTFILE_SIZE})",
                meta.len()
            ));
        }
        if !meta.is_file() {
            return Err("potfile is not a regular file".into());
        }
        let content =
            fs::read_to_string(potfile_path).map_err(|e| format!("failed to read potfile: {e}"))?;
        for line in content.lines() {
            if let Some(idx) = line.rfind(':') {
                potfile_map.insert(line[..idx].to_string(), line[idx + 1..].to_string());
            }
        }
    }
    let mut results = Vec::new();
    for line in hash_content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(pw) = potfile_map.get(trimmed) {
            results.push(format!("{trimmed}:{pw}"));
        }
    }
    Ok(results)
}

/// Show uncracked hashes: print hashes NOT in the potfile.
pub fn show_left(potfile_path: &str, hash_content: &str) -> Result<Vec<String>, String> {
    let cracked = load_potfile(potfile_path)?;
    let mut results = Vec::new();
    for line in hash_content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !cracked.contains(trimmed) {
            results.push(trimmed.to_string());
        }
    }
    Ok(results)
}

fn hex_encode(data: &[u8]) -> String {
    let mut s = String::with_capacity(data.len() * 2);
    use std::fmt::Write;
    for &b in data {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[allow(dead_code)] // used in tests via decode_hex_password
fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("odd length".into());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| format!("hex decode: {e}")))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: create a temp file path in the system temp dir.
    fn temp_path(name: &str) -> String {
        let pid = std::process::id();
        format!("/tmp/vega_potfile_test_{pid}_{name}")
    }

    /// Write/load round-trip: write two entries, load them back, verify the
    /// cracked set contains both hash source lines.
    #[test]
    fn test_write_load_round_trip() {
        let path = temp_path("roundtrip");
        let _ = std::fs::remove_file(&path);

        let hash1 = "WPA*01*aa*bb*cc*dd";
        let hash2 = "WPA*02*ee*ff*00*11";

        write_potfile_entry(&path, hash1, b"password1").unwrap();
        write_potfile_entry(&path, hash2, b"password2").unwrap();

        let cracked = load_potfile(&path).unwrap();
        assert!(cracked.contains(hash1), "hash1 should be in cracked set");
        assert!(cracked.contains(hash2), "hash2 should be in cracked set");
        assert_eq!(cracked.len(), 2, "should have exactly 2 entries");

        std::fs::remove_file(&path).ok();
    }

    /// $HEX encoding: a password with non-printable bytes should be hex-encoded
    /// with the $HEX[...] wrapper. decode_hex_password should round-trip it.
    #[test]
    fn test_hex_encoding_round_trip() {
        let path = temp_path("hexenc");
        let _ = std::fs::remove_file(&path);

        let hash_line = "WPA*01*ab*cd*ef*01";
        // Password with a null byte (non-printable)
        let pw: &[u8] = b"pass\x00word";

        write_potfile_entry(&path, hash_line, pw).unwrap();

        // Read the potfile back and verify the $HEX[...] wrapper is present
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(
            content.contains("$HEX["),
            "potfile should contain $HEX[...] for non-printable password"
        );

        // Extract the password part (after last colon) and decode it
        let pw_part = content.rfind(':').map(|i| &content[i + 1..]).unwrap();
        let decoded = decode_hex_password(pw_part.trim());
        assert_eq!(
            decoded, pw,
            "decoded password should match original after $HEX round-trip"
        );

        std::fs::remove_file(&path).ok();
    }

    /// $HEX encoding for password containing a colon (which would break the
    /// potfile format). The colon triggers $HEX[] wrapping even though the
    /// password is printable.
    #[test]
    fn test_hex_encoding_for_colon_password() {
        let path = temp_path("colonpw");
        let _ = std::fs::remove_file(&path);

        let hash_line = "WPA*01*ab*cd*ef*02";
        let pw: &[u8] = b"pass:word";

        write_potfile_entry(&path, hash_line, pw).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        // The password part (after the hash_line's colon) should be $HEX[...]
        let prefix = format!("{hash_line}:");
        let pw_part = &content[prefix.len()..];
        assert!(
            pw_part.starts_with("$HEX["),
            "colon-containing password should be $HEX-encoded, got: {pw_part}"
        );

        let decoded = decode_hex_password(pw_part.trim());
        assert_eq!(decoded, pw, "decoded colon-password should match original");

        std::fs::remove_file(&path).ok();
    }

    /// show_cracked: given a potfile with one cracked entry and a hash file
    /// with two hashes, show_cracked should return only the cracked one.
    #[test]
    fn test_show_cracked() {
        let path = temp_path("showcracked");
        let _ = std::fs::remove_file(&path);

        let hash1 = "WPA*01*aa*bb*cc*dd";
        let hash2 = "WPA*02*ee*ff*00*11";
        let hash_content = format!("{hash1}\n{hash2}\n");

        // Only hash1 is cracked
        write_potfile_entry(&path, hash1, b"secret123").unwrap();

        let results = show_cracked(&path, &hash_content).unwrap();
        assert_eq!(results.len(), 1, "should show exactly 1 cracked hash");
        assert!(
            results[0].starts_with(hash1),
            "cracked result should start with hash1, got: {}",
            results[0]
        );
        assert!(
            results[0].contains("secret123"),
            "cracked result should contain the password"
        );

        std::fs::remove_file(&path).ok();
    }

    /// show_left: given a potfile with one cracked entry and a hash file with
    /// two hashes, show_left should return only the uncracked one.
    #[test]
    fn test_show_left() {
        let path = temp_path("showleft");
        let _ = std::fs::remove_file(&path);

        let hash1 = "WPA*01*aa*bb*cc*dd";
        let hash2 = "WPA*02*ee*ff*00*11";
        let hash_content = format!("{hash1}\n{hash2}\n");

        // Only hash1 is cracked
        write_potfile_entry(&path, hash1, b"secret123").unwrap();

        let results = show_left(&path, &hash_content).unwrap();
        assert_eq!(results.len(), 1, "should show exactly 1 uncracked hash");
        assert_eq!(results[0], hash2, "uncracked result should be hash2");

        std::fs::remove_file(&path).ok();
    }

    /// load_potfile on a non-existent path should return an empty set (not error).
    #[test]
    fn test_load_potfile_nonexistent() {
        let path = temp_path("nonexistent_file_999");
        let _ = std::fs::remove_file(&path);

        let cracked = load_potfile(&path).unwrap();
        assert!(
            cracked.is_empty(),
            "non-existent potfile should yield empty set"
        );
    }

    /// load_potfile on an empty file should return an empty set.
    #[test]
    fn test_load_potfile_empty() {
        let path = temp_path("empty");
        let _ = std::fs::remove_file(&path);
        std::fs::write(&path, "").unwrap();

        let cracked = load_potfile(&path).unwrap();
        assert!(cracked.is_empty(), "empty potfile should yield empty set");

        std::fs::remove_file(&path).ok();
    }

    /// load_potfile should handle blank lines gracefully (skip them).
    #[test]
    fn test_load_potfile_blank_lines() {
        let path = temp_path("blanklines");
        let _ = std::fs::remove_file(&path);

        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "hash1:pass1").unwrap();
        writeln!(f).unwrap();
        writeln!(f, "hash2:pass2").unwrap();
        drop(f);

        let cracked = load_potfile(&path).unwrap();
        assert_eq!(
            cracked.len(),
            2,
            "should have 2 entries (blank line skipped)"
        );
        assert!(cracked.contains("hash1"));
        assert!(cracked.contains("hash2"));

        std::fs::remove_file(&path).ok();
    }

    /// write_potfile_entry should append (not overwrite) — multiple writes
    /// accumulate in the file.
    #[test]
    fn test_write_appends() {
        let path = temp_path("append");
        let _ = std::fs::remove_file(&path);

        write_potfile_entry(&path, "hash_a", b"pw_a").unwrap();
        write_potfile_entry(&path, "hash_b", b"pw_b").unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2, "file should have 2 lines after 2 appends");
        assert!(lines[0].starts_with("hash_a:"));
        assert!(lines[1].starts_with("hash_b:"));

        std::fs::remove_file(&path).ok();
    }

    /// $HEX prefix ambiguity: a password starting with "$HEX[" should be
    /// hex-encoded to avoid being misinterpreted on reload.
    #[test]
    fn test_hex_prefix_ambiguity() {
        let path = temp_path("hexprefix");
        let _ = std::fs::remove_file(&path);

        let hash_line = "WPA*01*ab*cd*ef*03";
        let pw: &[u8] = b"$HEX[4142]";

        write_potfile_entry(&path, hash_line, pw).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        let pw_part = content.rfind(':').map(|i| &content[i + 1..]).unwrap();
        // Should be double-encoded: $HEX[24484558...], not the raw $HEX[4142]
        let decoded = decode_hex_password(pw_part.trim());
        assert_eq!(
            decoded, pw,
            "double-encoded $HEX prefix password should round-trip"
        );

        std::fs::remove_file(&path).ok();
    }
}
