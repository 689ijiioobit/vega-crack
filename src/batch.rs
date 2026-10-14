//! Multi-target batch cracking: load all .hc22000/.hc16800 files from a directory
//! and process them with a single wordlist run.

use crate::modes::HashModeId;
use crate::multihash::HashGroup;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct BatchCapture {
    pub path: String,
    pub name: String,
    pub groups: Vec<HashGroup>,
    pub status: BatchStatus,
    /// Raw cracked password bytes (for correct export encoding).
    /// Set when status is Cracked; None otherwise.
    pub cracked_pw_bytes: Option<Vec<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BatchStatus {
    Pending,
    Cracked { password: String, essid: String },
    NotCracked,
    Error(String),
}

/// Scan a directory for .hc22000 and .hc16800 files.
/// Returns a Vec of BatchCapture, sorted by filename.
pub fn load_batch_dir(dir: &str, mode_override: &str) -> Result<Vec<BatchCapture>, String> {
    let dir_path = Path::new(dir);
    if !dir_path.is_dir() {
        return Err(format!("'{dir}' is not a directory"));
    }

    let hash_mode = if mode_override.eq_ignore_ascii_case("ntlm") {
        HashModeId::Ntlm
    } else if mode_override.eq_ignore_ascii_case("md5") {
        HashModeId::Md5
    } else if mode_override.eq_ignore_ascii_case("sha256") {
        HashModeId::Sha256
    } else {
        // Default for batch-dir: WPA (covers both "wpa" and unrecognized)
        HashModeId::Wpa22000
    };

    let mut captures = Vec::new();

    let entries = fs::read_dir(dir).map_err(|e| format!("Cannot read directory: {e}"))?;
    for entry in entries {
        let entry = entry.map_err(|e| format!("Dir entry error: {e}"))?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().to_string();

        // Filter for hash files
        let is_hash = name.ends_with(".hc22000")
            || name.ends_with(".hc16800")
            || name.ends_with(".22000")
            || name.ends_with(".16800")
            // FIX F11: Support MD5/NTLM hash files in batch mode
            || name.ends_with(".txt")
            || name.ends_with(".hash")
            || name.ends_with(".md5")
            || name.ends_with(".ntlm")
            || (!name.contains('.') && !name.starts_with('.')); // extensionless files

        if !is_hash {
            continue;
        }

        // Guard: skip oversized files and non-regular files (symlinks, dirs, etc.)
        let meta = match fs::metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[!] Warning: cannot stat {name}: {e}");
                continue;
            }
        };
        if !meta.is_file() {
            continue;
        }
        if meta.len() > 1_000_000 {
            eprintln!(
                "[!] Warning: {name} is too large ({} bytes, max 1 MB) — skipping",
                meta.len()
            );
            continue;
        }

        let content = match fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[!] Warning: cannot read {name}: {e}");
                continue;
            }
        };

        let groups = match hash_mode.parse_hash_file(&content) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("[!] Warning: cannot parse {name}: {e}");
                continue;
            }
        };

        if groups.is_empty() {
            continue;
        }

        captures.push(BatchCapture {
            path: path.to_string_lossy().to_string(),
            name,
            groups,
            status: BatchStatus::Pending,
            cracked_pw_bytes: None,
        });
    }

    // Sort by filename for deterministic ordering
    captures.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(captures)
}

/// Print the batch summary report.
pub fn print_batch_report(captures: &[BatchCapture]) {
    let cracked = captures
        .iter()
        .filter(|c| matches!(c.status, BatchStatus::Cracked { .. }))
        .count();
    let not_cracked = captures
        .iter()
        .filter(|c| c.status == BatchStatus::NotCracked)
        .count();
    let errors = captures
        .iter()
        .filter(|c| matches!(c.status, BatchStatus::Error(_)))
        .count();

    eprintln!("\n=== BATCH CRACK SUMMARY ===");
    eprintln!(
        "Captures: {} | Cracked: {} | Not found: {} | Errors: {}",
        captures.len(),
        cracked,
        not_cracked,
        errors
    );

    for cap in captures {
        match &cap.status {
            BatchStatus::Cracked { password, essid } => {
                eprintln!("  {}     → {} (ESSID: {})", cap.name, password, essid);
            }
            BatchStatus::NotCracked => {
                eprintln!("  {}     → NOT CRACKED", cap.name);
            }
            BatchStatus::Error(e) => {
                eprintln!("  {}     → ERROR: {}", cap.name, e);
            }
            BatchStatus::Pending => {
                eprintln!("  {}     → (not processed)", cap.name);
            }
        }
    }
}
