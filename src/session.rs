#![allow(dead_code, unused_assignments)]
//! Session/restore: save and resume cracking position.
//!
//! Design:
//! - `--session <name>` enables periodic state saving to a restore file.
//! - The restore file is a simple JSON document containing:
//!   - The full original command-line args (for reconstruction)
//!   - Total candidates tested so far (for progress display)
//!   - Per-group state: which ESSID group index, which increment length,
//!     and the skip offset to resume at
//! - `--restore <name>` loads the restore file, injects the saved skip
//!   offset as `--skip`, and re-runs with the original args.
//! - State is saved every 30 seconds during cracking (checkpoint interval).
//! - The restore file is written atomically (write to .tmp, rename).

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Restore file version — bumped on format changes.
const RESTORE_VERSION: u32 = 1;

/// Checkpoint interval: how often to save state during cracking.
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(30);

/// Per-group restore state (for multi-ESSID groups + increment mode).
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct GroupRestoreState {
    /// ESSID of this group (hex-encoded for safe serialization).
    pub essid_hex: String,
    /// Current increment length (0 if not in increment mode).
    pub increment_length: usize,
    /// Skip offset to resume at for this group+length.
    pub skip: u64,
    /// Total candidates tested in this group so far.
    pub tested: u64,
}

/// Full restore state — serialized to JSON.
#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct RestoreState {
    pub version: u32,
    /// Full original command-line args.
    pub args: Vec<String>,
    /// Working directory at session start.
    pub cwd: String,
    /// Which group index we're currently processing.
    pub group_index: usize,
    /// Per-group states (keyed by group index as string).
    pub groups: HashMap<String, GroupRestoreState>,
    /// Timestamp of last save (Unix epoch seconds).
    pub saved_at: u64,
}

impl RestoreState {
    /// Create a new restore state from the current command-line args.
    pub fn new(args: &[String]) -> Self {
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        Self {
            version: RESTORE_VERSION,
            args: args.to_vec(),
            cwd,
            group_index: 0,
            groups: HashMap::new(),
            saved_at: 0,
        }
    }

    /// Get the restore file path for a given session name.
    pub fn restore_file_path(session_name: &str) -> Result<PathBuf, String> {
        if session_name.is_empty() {
            return Err("session name cannot be empty".into());
        }
        // Cap session name length to prevent excessively long path names.
        if session_name.len() > 64 {
            return Err("session name may be at most 64 characters".into());
        }
        // Sanitize session name: only allow alphanumeric, dash, underscore
        if !session_name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err("session name may only contain alphanumeric, dash, or underscore".into());
        }
        let home =
            std::env::var("HOME").map_err(|_| "HOME environment variable not set".to_string())?;
        let dir = format!("{home}/.vega-crack/restore");
        Ok(PathBuf::from(format!("{dir}/{session_name}.restore")))
    }

    /// Save the restore state to disk atomically.
    pub fn save(&mut self, session_name: &str) -> Result<(), String> {
        let path = Self::restore_file_path(session_name)?;
        let dir = path.parent().ok_or("cannot determine restore directory")?;
        fs::create_dir_all(dir).map_err(|e| format!("failed to create restore directory: {e}"))?;

        self.saved_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("failed to serialize restore state: {e}"))?;

        // Atomic write: write to .tmp, then rename
        let tmp_path = path.with_extension("restore.tmp");
        let mut f = fs::File::create(&tmp_path)
            .map_err(|e| format!("failed to create temp restore file: {e}"))?;
        f.write_all(json.as_bytes())
            .map_err(|e| format!("failed to write restore file: {e}"))?;
        f.sync_all()
            .map_err(|e| format!("failed to sync restore file: {e}"))?;
        drop(f);

        fs::rename(&tmp_path, &path).map_err(|e| format!("failed to rename restore file: {e}"))?;

        Ok(())
    }

    /// Load a restore state from disk.
    pub fn load(session_name: &str) -> Result<Self, String> {
        let path = Self::restore_file_path(session_name)?;
        if !path.exists() {
            return Err(format!("restore file not found: {}", path.display()));
        }
        let content =
            fs::read_to_string(&path).map_err(|e| format!("failed to read restore file: {e}"))?;
        let state: Self = serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse restore file: {e}"))?;

        if state.version != RESTORE_VERSION {
            return Err(format!(
                "restore file version mismatch: file={}, expected={RESTORE_VERSION}",
                state.version
            ));
        }

        Ok(state)
    }

    /// Delete the restore file (called after successful completion).
    pub fn delete(session_name: &str) -> Result<(), String> {
        let path = Self::restore_file_path(session_name)?;
        if path.exists() {
            fs::remove_file(&path).map_err(|e| format!("failed to delete restore file: {e}"))?;
        }
        Ok(())
    }

    /// Get or create the group state for the given group index.
    pub fn get_or_create_group(
        &mut self,
        group_index: usize,
        essid: &[u8],
    ) -> &mut GroupRestoreState {
        let key = group_index.to_string();
        self.groups.entry(key).or_insert(GroupRestoreState {
            essid_hex: essid.iter().map(|b| format!("{b:02x}")).collect(),
            increment_length: 0,
            skip: 0,
            tested: 0,
        })
    }
}

/// Checkpoint timer — tracks when the next save should happen.
pub struct CheckpointTimer {
    last_save: Instant,
    interval: Duration,
}

impl CheckpointTimer {
    pub fn new() -> Self {
        Self {
            last_save: Instant::now(),
            interval: CHECKPOINT_INTERVAL,
        }
    }

    /// Returns true if it's time to save (and resets the timer).
    pub fn should_save(&self) -> bool {
        self.last_save.elapsed() >= self.interval
    }
    pub fn mark_saved(&mut self) {
        self.last_save = Instant::now();
    }

    /// Force reset (e.g., after a manual save).
    pub fn reset(&mut self) {
        self.last_save = Instant::now();
    }
}
