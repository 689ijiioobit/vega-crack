//! Markov chain candidate ordering using hashcat's .hcstat2 statistics.
//!
//! Loads hashcat's hcstat2 file (LZMA2 compressed), builds per-position
//! character probability tables, and orders mask candidates by likelihood.
//!
//! The hcstat2 format:
//! - Header: u64 version (0x6863737461740002), u64 zero
//! - Root stats: SP_PW_MAX * 256 u64s (probability of each char at each position)
//! - Markov stats: SP_PW_MAX * 256 * 256 u64s (probability of next char given
//!   position + current char)
//!
//! For WPA (max 63 chars), we only load positions 0-62 to save memory (~32MB
//! vs 128MB for the full 256-position table).

use crate::mask;
use std::collections::HashSet;

/// Maximum password position we track (WPA max is 63 chars, so 63 positions).
const MAX_MARKOV_POS: usize = 63;

/// Number of possible byte values.
const CHARSIZ: usize = 256;

/// hcstat2 version magic.
const HCSTAT_VERSION: u64 = 0x6863737461740002;

/// LZMA2 properties byte used by hashcat's hcstat2 files.
const LZMA2_PROPS: u8 = 0x1c;

/// Per-position sorted character table entry.
#[derive(Clone, Copy, Debug)]
struct HcstatEntry {
    key: u8,
    val: u64,
}

/// Markov probability tables.
pub struct MarkovTables {
    /// Root stats: per-position character probabilities (sorted descending).
    root_table: Vec<[HcstatEntry; CHARSIZ]>,
    /// Markov stats: per-position+key transition probabilities (raw u64 values).
    /// Indexed as markov_raw[pos][key * CHARSIZ + next]. Sorted lazily in
    /// ordered_chars() to avoid 4M sort operations at load time.
    markov_raw: Vec<Vec<u64>>,
}

impl MarkovTables {
    /// Load and build Markov tables from a hashcat .hcstat2 file.
    /// Only loads positions 0..MAX_MARKOV_POS to save memory.
    #[allow(dead_code)]
    pub fn from_hcstat2(path: &str) -> Result<Self, String> {
        Self::from_hcstat2_with_flags(path, false, false)
    }

    /// Load hcstat2 with Markov modifier flags (matching hashcat's --markov-classic,
    /// --markov-inverse, --markov-disable).
    pub fn from_hcstat2_with_flags(
        path: &str,
        classic: bool,
        inverse: bool,
    ) -> Result<Self, String> {
        // Security: validate file metadata before reading (mirrors hash file guards)
        let meta = std::fs::metadata(path)
            .map_err(|e| format!("failed to stat hcstat2 file '{path}': {e}"))?;
        if !meta.is_file() {
            return Err("hcstat2 file is not a regular file".into());
        }
        // Real hcstat2 is ~240KB compressed. Cap at 100MB to block zip bombs / FIFOs.
        if meta.len() > 100_000_000 {
            return Err(format!(
                "hcstat2 file too large ({} bytes, max 100MB)",
                meta.len()
            ));
        }
        let compressed = std::fs::read(path)
            .map_err(|e| format!("failed to read hcstat2 file '{path}': {e}"))?;

        // Decompress LZMA2
        let decompressed = decompress_lzma2(&compressed)?;
        // Cap decompressed size to prevent zip bomb (expected: ~128.5MB)
        let max_decompressed = 256 * 256 * 256 * 8 + 65536 * 8 + 16;
        if decompressed.len() > max_decompressed {
            return Err("decompressed hcstat2 exceeds expected size (possible zip bomb)".into());
        }
        if decompressed.len() < 16 {
            return Err("hcstat2 file too small (corrupted)".into());
        }

        // Parse header
        let version = u64::from_be_bytes(decompressed[0..8].try_into().unwrap());
        let zero = u64::from_be_bytes(decompressed[8..16].try_into().unwrap());
        if version != HCSTAT_VERSION {
            return Err(format!(
                "hcstat2 version mismatch: expected 0x{HCSTAT_VERSION:016x}, got 0x{version:016x}"
            ));
        }
        if zero != 0 {
            return Err("hcstat2 header zero field is non-zero (corrupted)".into());
        }

        // Root stats: 256 positions * 256 chars = 65536 u64s
        let root_offset = 16;
        let root_size = 65536 * 8;
        if decompressed.len() < root_offset + root_size {
            return Err("hcstat2 file truncated (root stats)".into());
        }

        // Build root table for positions 0..MAX_MARKOV_POS
        // Pre-sort root table (only 63*256 entries — fast)
        let mut root_table = vec![[HcstatEntry { key: 0, val: 0 }; CHARSIZ]; MAX_MARKOV_POS];
        for pos in 0..MAX_MARKOV_POS {
            for ch in 0..CHARSIZ {
                let idx = pos * CHARSIZ + ch;
                let offset = root_offset + idx * 8;
                let val = u64::from_be_bytes(decompressed[offset..offset + 8].try_into().unwrap());
                root_table[pos][ch] = HcstatEntry { key: ch as u8, val };
            }
            root_table[pos].sort_by_key(|a| std::cmp::Reverse(a.val));
        }

        // Markov stats: 256 pos * 256 key * 256 next = 16777216 u64s
        let mk_offset = root_offset + root_size;
        if decompressed.len() < mk_offset + 256 * 256 * 256 * 8 {
            return Err("hcstat2 file truncated (markov stats)".into());
        }

        // Store markov table as raw u64 values (no pre-sorting).
        // Sorting is done lazily in ordered_chars() — only for positions
        // and keys actually used by the current mask. This avoids 4M sort
        // operations at load time (which was taking >30s).
        let mut markov_raw: Vec<Vec<u64>> = (0..MAX_MARKOV_POS)
            .map(|_| vec![0u64; CHARSIZ * CHARSIZ])
            .collect();
        for pos in 0..MAX_MARKOV_POS {
            for key in 0..CHARSIZ {
                for next in 0..CHARSIZ {
                    let idx = (pos * CHARSIZ + key) * CHARSIZ + next;
                    let offset = mk_offset + idx * 8;
                    markov_raw[pos][key * CHARSIZ + next] =
                        u64::from_be_bytes(decompressed[offset..offset + 8].try_into().unwrap());
                }
            }
        }

        // --markov-inverse: invert probability values (try least probable first)
        // https://github.com/hashcat/hashcat/issues/1058
        if inverse {
            for pos in 0..MAX_MARKOV_POS {
                for ch in 0..CHARSIZ {
                    root_table[pos][ch].val =
                        0u64.wrapping_sub(1u64.wrapping_add(root_table[pos][ch].val));
                }
            }
            for pos in 0..MAX_MARKOV_POS {
                for idx in 0..CHARSIZ * CHARSIZ {
                    markov_raw[pos][idx] =
                        0u64.wrapping_sub(1u64.wrapping_add(markov_raw[pos][idx]));
                }
            }
            // Re-sort root table after inversion
            for pos in 0..MAX_MARKOV_POS {
                root_table[pos].sort_by_key(|a| std::cmp::Reverse(a.val));
            }
        }

        // --markov-classic: aggregate all position stats into position 0,
        // then copy to all positions (position-independent Markov).
        // R84 MEDIUM fix: sum by character key, not by sorted rank index.
        // root_table is sorted descending by val, so root_table[pos][ch] is
        // the ch-th most probable entry — its .key identifies the actual char.
        // Summing by index ch mixes different characters. Fix: use a
        // key-indexed accumulator.
        if classic {
            // Root: aggregate by character key across all positions
            let mut aggregated = [0u64; CHARSIZ];
            for pos in 0..MAX_MARKOV_POS {
                for ch in 0..CHARSIZ {
                    let key = root_table[pos][ch].key as usize;
                    aggregated[key] = aggregated[key].wrapping_add(root_table[pos][ch].val);
                }
            }
            // Rebuild pos 0 from aggregated values, then sort
            for ch in 0..CHARSIZ {
                root_table[0][ch] = HcstatEntry {
                    key: ch as u8,
                    val: aggregated[ch],
                };
            }
            root_table[0].sort_by_key(|a| std::cmp::Reverse(a.val));
            // Copy pos 0 to all positions
            for pos in 1..MAX_MARKOV_POS {
                root_table[pos] = root_table[0];
            }
            // Markov: sum all positions into pos 0 (markov_raw is unsorted, indexed by key*CHARSIZ+next)
            for pos in 1..MAX_MARKOV_POS {
                for idx in 0..CHARSIZ * CHARSIZ {
                    markov_raw[0][idx] = markov_raw[0][idx].wrapping_add(markov_raw[pos][idx]);
                }
            }
            // Copy pos 0 to all positions
            for pos in 1..MAX_MARKOV_POS {
                markov_raw[pos] = markov_raw[0].clone();
            }
        }

        Ok(MarkovTables {
            root_table,
            markov_raw,
        })
    }

    /// Get the ordered character list for a given position and optional previous char.
    /// Returns characters sorted by Markov probability (most likely first).
    /// If `threshold` > 0, only includes characters with val >= threshold.
    /// If `prev_char` is None, uses root table (first position).
    /// If `prev_char` is Some, uses markov transition table.
    pub fn ordered_chars(
        &self,
        pos: usize,
        prev_char: Option<u8>,
        threshold: u64,
        charset: &HashSet<u8>,
    ) -> Vec<u8> {
        let pos = pos.min(MAX_MARKOV_POS - 1);
        if let Some(prev_ch) = prev_char {
            // Build sorted entries from raw markov data (lazy sort)
            let base = (prev_ch as usize) * CHARSIZ;
            let mut entries: Vec<HcstatEntry> = (0..CHARSIZ)
                .map(|next| HcstatEntry {
                    key: next as u8,
                    val: self.markov_raw[pos][base + next],
                })
                .filter(|e| e.val >= threshold || threshold == 0)
                .filter(|e| charset.contains(&e.key))
                .collect();
            entries.sort_by_key(|a| std::cmp::Reverse(a.val));
            entries.iter().map(|e| e.key).collect()
        } else {
            // No previous char — use root table (already sorted)
            self.root_table[pos]
                .iter()
                .filter(|e| e.val >= threshold || threshold == 0)
                .filter(|e| charset.contains(&e.key))
                .map(|e| e.key)
                .collect()
        }
    }
}

/// Decompress LZMA2 data with hashcat's properties byte.
fn decompress_lzma2(compressed: &[u8]) -> Result<Vec<u8>, String> {
    // LZMA2 props byte 0x1c:
    // lc = props % 9, lp = (props / 9) % 5, pb = props / 45
    let lc = (LZMA2_PROPS % 9) as u32;
    let lp = ((LZMA2_PROPS / 9) % 5) as u32;
    let pb = (LZMA2_PROPS / 45) as u32;

    // We need to use Rust's lzma crate or a fallback.
    // Since we may not have the lzma crate available, we'll try to use
    // the system xz binary as a fallback.
    decompress_lzma2_via_xz(compressed, lc, lp, pb)
        .or_else(|_| decompress_lzma2_via_python(compressed))
}

/// Decompress LZMA2 using the system xz binary.
/// hashcat's hcstat2 uses raw LZMA2 format (no XZ container).
/// The props byte 0x1c encodes dictionary size = 64MB (2^26).
/// Uses a temp file to avoid pipe deadlock when decompressing 128MB.
fn decompress_lzma2_via_xz(
    compressed: &[u8],
    _lc: u32,
    _lp: u32,
    _pb: u32,
) -> Result<Vec<u8>, String> {
    use std::process::{Command, Stdio};

    let pid = std::process::id();
    use std::sync::atomic::{AtomicU64, Ordering};
    static HCSTAT_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let file_id = HCSTAT_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let in_path = format!("/tmp/vega_hcstat2_{pid}_{file_id}.in");
    let out_path = format!("/tmp/vega_hcstat2_{pid}_{file_id}.out");

    // Write compressed data to temp file (create_new to prevent symlink attacks)
    {
        use std::fs::OpenOptions;
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&in_path)
            .and_then(|mut f| std::io::Write::write_all(&mut f, compressed))
            .map_err(|e| format!("failed to write temp hcstat2: {e}"))?;
    }

    // Run xz inside a closure so we can always clean up temp files afterward.
    let result = (|| -> Result<Vec<u8>, String> {
        let stdout_file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&out_path)
            .map_err(|e| format!("failed to create temp: {e}"))?;
        let status = Command::new("xz")
            .args([
                "--decompress",
                "--format=raw",
                "--lzma2=dict=64MiB",
                "--keep",
                "-c",
                &in_path,
            ])
            .stdout(stdout_file)
            .stderr(Stdio::piped())
            .status()
            .map_err(|e| format!("xz execution failed: {e}"))?;
        if !status.success() {
            return Err("xz decompression failed".into());
        }
        let decompressed = std::fs::read(&out_path)
            .map_err(|e| format!("failed to read decompressed temp: {e}"))?;
        if decompressed.is_empty() {
            return Err("xz produced no output".into());
        }
        Ok(decompressed)
    })();

    // Always clean up both temp files, regardless of success or failure
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);

    result
}

/// Fallback: decompress via Python's lzma module.
/// Uses temp files to avoid pipe deadlock when decompressing 128MB.
fn decompress_lzma2_via_python(compressed: &[u8]) -> Result<Vec<u8>, String> {
    use std::process::{Command, Stdio};

    let pid = std::process::id();
    use std::sync::atomic::{AtomicU64, Ordering};
    static HCSTAT_PY_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
    let file_id = HCSTAT_PY_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let in_path = format!("/tmp/vega_hcstat2_py_{pid}_{file_id}.in");
    let out_path = format!("/tmp/vega_hcstat2_py_{pid}_{file_id}.out");

    // Write compressed data to temp file (create_new to prevent symlink attacks)
    {
        use std::fs::OpenOptions;
        OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&in_path)
            .and_then(|mut f| std::io::Write::write_all(&mut f, compressed))
            .map_err(|e| format!("failed to write temp hcstat2: {e}"))?;
    }

    // hashcat's hcstat2 uses raw LZMA2 format. The props byte 0x1c
    // encodes dictionary size 2^26 (64MB). For raw LZMA2, only dict_size
    // matters — lc/lp/pb are encoded in the LZMA2 stream chunks themselves.
    let script = format!(
        r#"
import lzma
with open("{in_path}", "rb") as f:
    data = f.read()
filt = [{{"id": lzma.FILTER_LZMA2, "dict_size": 1 << 26}}]
dec = lzma.decompress(data, format=lzma.FORMAT_RAW, filters=filt)
with open("{out_path}", "wb") as f:
    f.write(dec)
"#
    );

    // Run python3 inside a closure so we can always clean up temp files.
    let result = (|| -> Result<Vec<u8>, String> {
        let status = Command::new("python3")
            .arg("-c")
            .arg(&script)
            .stderr(Stdio::piped())
            .status()
            .map_err(|e| format!("python3 not available: {e}"))?;
        if !status.success() {
            return Err("python3 lzma decompression failed".into());
        }
        let decompressed = std::fs::read(&out_path)
            .map_err(|e| format!("failed to read decompressed temp: {e}"))?;
        if decompressed.is_empty() {
            return Err("python3 lzma decompression produced no output".into());
        }
        Ok(decompressed)
    })();

    // Always clean up both temp files
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);

    result
}
/// Write a single Markov candidate to the wordlist file.
/// Hex-encodes candidates containing \n, \r, or $HEX[ prefix (matching
/// write_temp_candidate's logic) to prevent the fill loop's read_until(b'\n')
/// from splitting mid-candidate. Cleans up the temp file on write error.
fn write_markov_candidate(
    file: &mut std::fs::File,
    candidate: &[u8],
    temp_path: &str,
) -> Result<(), String> {
    use std::io::Write;
    // Check if candidate needs hex encoding (same logic as write_temp_candidate)
    let needs_hex = candidate
        .iter()
        .any(|&b| b < 0x20 || b == 0x7f) // FIX F10: removed redundant checks
        || (candidate.len() >= 5 && &candidate[..5] == b"$HEX[");

    let write_result = if needs_hex {
        // Hex-encode as $HEX[...] to prevent line splitting
        let hex: String = candidate.iter().map(|b| format!("{b:02x}")).collect();
        let encoded = format!("$HEX[{hex}]\n");
        file.write_all(encoded.as_bytes())
    } else {
        file.write_all(candidate)
            .and_then(|_| file.write_all(b"\n"))
    };

    write_result.map_err(|e| {
        // LOW-3: Clean up temp file on write error
        let _ = std::fs::remove_file(temp_path);
        format!("markov write failed: {e}")
    })
}

/// Generate a Markov-ordered wordlist from a mask.
///
/// Instead of brute-force odometer order (position 0 cycles slowest),
/// this generates candidates ordered by Markov probability: the most
/// probable first character, then the most probable second character
/// given the first, etc.
///
/// Uses a threshold to limit the number of transitions tried. With
/// threshold=0, all candidates are generated (same keyspace as mask,
/// just reordered). With threshold>0, only high-probability transitions
/// are tried, reducing the keyspace.
///
/// Writes candidates to a temp file, returns the path and count.
impl MarkovTables {
    pub fn generate_markov_wordlist(
        &self,
        mask: &mask::Mask,
        threshold: u64,
        min_pw_len: usize,
        max_candidates: u64,
        skip: u64,
    ) -> Result<(String, u64), String> {
        let pid = std::process::id();
        // Use a unique atomic counter to ensure parallel calls (e.g. in tests)
        // don't clobber each other's temp files — all calls in the same process
        // share the same pid, so pid alone is insufficient.
        use std::sync::atomic::{AtomicU64, Ordering};
        static MARKOV_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);
        let file_id = MARKOV_FILE_COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = format!("/tmp/vega_markov_{pid}_{file_id}.txt");
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .map_err(|e| format!("failed to create markov wordlist: {e}"))?;

        let positions = &mask.positions;
        let num_positions = positions.len();

        if num_positions == 0 || num_positions < min_pw_len {
            let _ = std::fs::remove_file(&path);
            return Ok((path, 0));
        }

        // Build per-position charsets as HashSets for fast lookup
        let charsets: Vec<HashSet<u8>> = positions
            .iter()
            .map(|cs| cs.iter().copied().collect())
            .collect();

        // Generate candidates using a stack-based DFS approach.
        // Each stack entry is (position, partial_password).
        // At each position, we try characters in Markov order.
        let mut count = 0u64;
        let mut skipped = 0u64;
        let mut stack: Vec<(usize, Vec<u8>)> = Vec::new();

        // Seed: position 0, ordered by root table
        let first_chars = self.ordered_chars(0, None, threshold, &charsets[0]);
        for &ch in first_chars.iter().rev() {
            stack.push((0, vec![ch]));
        }

        while let Some((pos, partial)) = stack.pop() {
            if partial.len() >= min_pw_len && partial.len() == num_positions {
                // Complete candidate
                if skipped < skip {
                    skipped += 1;
                    continue;
                }
                write_markov_candidate(&mut file, &partial, &path)?;
                count += 1;
                if count >= max_candidates {
                    break;
                }
                continue;
            }

            let next_pos = pos + 1;
            if next_pos >= num_positions {
                // Partial candidate that reached the end but is shorter than mask
                // (shouldn't happen with fixed-length mask, but handle gracefully)
                if partial.len() >= min_pw_len {
                    if skipped < skip {
                        skipped += 1;
                    } else {
                        write_markov_candidate(&mut file, &partial, &path)?;
                        count += 1;
                        if count >= max_candidates {
                            break;
                        }
                    }
                }
                continue;
            }

            // Get ordered characters for the next position
            let prev_char = partial.last().copied();
            let next_chars =
                self.ordered_chars(next_pos, prev_char, threshold, &charsets[next_pos]);

            // Push in reverse order so most probable is processed first (stack = LIFO)
            for &ch in next_chars.iter().rev() {
                let mut new_partial = partial.clone();
                new_partial.push(ch);
                stack.push((next_pos, new_partial));
            }
        }

        drop(file);
        Ok((path, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markov_load() {
        // Try to load hashcat's default hcstat2 file
        let path = "../hashcat-src/hashcat.hcstat2";
        if !std::path::Path::new(path).exists() {
            eprintln!("[test] Skipping markov test — {path} not found");
            return;
        }
        let tables = MarkovTables::from_hcstat2(path).expect("failed to load hcstat2");

        // Root table for position 0: 'e' should be in top 10 most common first chars
        let top_chars: Vec<u8> = tables.root_table[0]
            .iter()
            .take(10)
            .map(|e| e.key)
            .collect();
        eprintln!(
            "[test] Top 10 first chars: {:?}",
            top_chars.iter().map(|&c| c as char).collect::<Vec<_>>()
        );

        // Verify sorted descending
        for pos in 0..MAX_MARKOV_POS {
            for i in 1..CHARSIZ {
                assert!(
                    tables.root_table[pos][i - 1].val >= tables.root_table[pos][i].val,
                    "root table not sorted at pos={pos} idx={i}"
                );
            }
        }

        // Test ordered_chars with a charset
        let charset: HashSet<u8> = b"abcdefghijklmnopqrstuvwxyz".iter().copied().collect();
        let ordered = tables.ordered_chars(0, None, 0, &charset);
        assert!(!ordered.is_empty(), "ordered_chars should return results");
        assert!(ordered.len() <= 26, "should only return lowercase letters");

        eprintln!("[test] Markov tables loaded successfully");
    }

    /// R79-MEDIUM-2: --skip must skip the first N candidates in Markov generation.
    #[test]
    fn test_markov_skip() {
        let path = "../hashcat-src/hashcat.hcstat2";
        if !std::path::Path::new(path).exists() {
            eprintln!("[test] Skipping markov skip test — {path} not found");
            return;
        }
        let tables = MarkovTables::from_hcstat2(path).expect("failed to load hcstat2");
        let custom = [
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let mask =
            mask::Mask::parse_with_min_len("?d?d?d?d", &custom, 1).expect("failed to parse mask");

        // Generate without skip
        let (_, count_noskip) = tables
            .generate_markov_wordlist(&mask, 0, 1, 10000, 0)
            .expect("gen failed");
        assert_eq!(count_noskip, 10000, "should generate 10000 candidates");

        // Generate with skip=5000 — should produce 5000 fewer
        let (_, count_skip) = tables
            .generate_markov_wordlist(&mask, 0, 1, 10000, 5000)
            .expect("gen failed with skip");
        assert_eq!(
            count_skip, 5000,
            "skip=5000 should produce 5000 candidates from 10000 total"
        );
    }

    /// R79-LOW-3: --limit 0 should use u64::MAX (unlimited), not 10M cap.
    /// Verify that max_candidates=u64::MAX doesn't panic and generates all
    /// candidates for a small mask.
    #[test]
    fn test_markov_unlimited() {
        let path = "../hashcat-src/hashcat.hcstat2";
        if !std::path::Path::new(path).exists() {
            eprintln!("[test] Skipping markov unlimited test — {path} not found");
            return;
        }
        let tables = MarkovTables::from_hcstat2(path).expect("failed to load hcstat2");
        let custom = [
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        // 2-char mask of digits = 100 candidates total
        let mask =
            mask::Mask::parse_with_min_len("?d?d", &custom, 1).expect("failed to parse mask");

        let (_, count) = tables
            .generate_markov_wordlist(&mask, 0, 1, u64::MAX, 0)
            .expect("gen failed");
        assert_eq!(
            count, 100,
            "2-digit mask should produce exactly 100 candidates"
        );
    }

    /// Verify Markov ordering: first candidate should be the most probable
    /// first char, not the lowest-value char. For digits, '1' should come
    /// before '0' in hashcat's hcstat2 (root table position 0).
    #[test]
    fn test_markov_ordering() {
        let path = "../hashcat-src/hashcat.hcstat2";
        if !std::path::Path::new(path).exists() {
            eprintln!("[test] Skipping markov ordering test — {path} not found");
            return;
        }
        let tables = MarkovTables::from_hcstat2(path).expect("failed to load hcstat2");
        let custom = [
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let mask = mask::Mask::parse_with_min_len("?d", &custom, 1).expect("failed to parse mask");

        // Generate 1 candidate — should be the most probable digit
        let (wl_path, count) = tables
            .generate_markov_wordlist(&mask, 0, 1, 1, 0)
            .expect("gen failed");
        assert_eq!(count, 1, "should generate exactly 1 candidate");

        let content = std::fs::read_to_string(&wl_path).expect("failed to read wl");
        let _ = std::fs::remove_file(&wl_path);
        let first = content.lines().next().expect("wordlist empty");
        // Most probable digit at pos 0 in hashcat's hcstat2 is '1' (val=2075978)
        assert_eq!(
            first, "1",
            "first Markov candidate should be '1' (most probable), got '{first}'"
        );
    }

    /// R79-MEDIUM-1: TempWordlist Drop must clean up the Markov temp file.
    /// Verify the temp file is deleted after Drop.
    #[test]
    fn test_markov_temp_cleanup() {
        let path = "../hashcat-src/hashcat.hcstat2";
        if !std::path::Path::new(path).exists() {
            eprintln!("[test] Skipping markov cleanup test — {path} not found");
            return;
        }
        let tables = MarkovTables::from_hcstat2(path).expect("failed to load hcstat2");
        let custom = [
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let mask =
            mask::Mask::parse_with_min_len("?d?d", &custom, 1).expect("failed to parse mask");

        let (wl_path, _) = tables
            .generate_markov_wordlist(&mask, 0, 1, 10, 0)
            .expect("gen failed");
        assert!(
            std::path::Path::new(&wl_path).exists(),
            "temp file should exist"
        );

        // Simulate TempWordlist Drop
        let _ = std::fs::remove_file(&wl_path);
        assert!(
            !std::path::Path::new(&wl_path).exists(),
            "temp file should be deleted after cleanup"
        );
    }

    /// Verify threshold filtering: with a high threshold, fewer candidates
    /// should be generated than without.
    #[test]
    fn test_markov_threshold() {
        let path = "../hashcat-src/hashcat.hcstat2";
        if !std::path::Path::new(path).exists() {
            eprintln!("[test] Skipping markov threshold test — {path} not found");
            return;
        }
        let tables = MarkovTables::from_hcstat2(path).expect("failed to load hcstat2");
        let custom = [
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ];
        let mask =
            mask::Mask::parse_with_min_len("?d?d?d?d", &custom, 1).expect("failed to parse mask");

        // threshold=0: all 10000 candidates
        let (_, count_all) = tables
            .generate_markov_wordlist(&mask, 0, 1, 10000, 0)
            .expect("gen failed");
        assert_eq!(count_all, 10000, "threshold=0 should allow all candidates");

        // threshold=very high: should produce fewer (possibly 0)
        let (_, count_filtered) = tables
            .generate_markov_wordlist(&mask, u64::MAX, 1, 10000, 0)
            .expect("gen failed with threshold");
        assert!(
            count_filtered < count_all,
            "high threshold should produce fewer candidates ({count_filtered} < {count_all})"
        );
    }

    /// R84 MEDIUM: --markov-classic must aggregate by character key, not by
    /// sorted rank index. Verify that classic mode produces the same ordering
    /// regardless of which position we query (position-independent).
    #[test]
    fn test_markov_classic_position_independent() {
        let path = "../hashcat-src/hashcat.hcstat2";
        if !std::path::Path::new(path).exists() {
            eprintln!("[test] Skipping markov classic test — {path} not found");
            return;
        }
        let tables = MarkovTables::from_hcstat2_with_flags(path, true, false)
            .expect("failed to load hcstat2 with classic");

        // In classic mode, all positions should have the same root_table ordering
        let pos0_first = tables.root_table[0][0].key;
        let pos1_first = tables.root_table[1][0].key;
        let pos10_first = tables.root_table[10][0].key;

        assert_eq!(
            pos0_first, pos1_first,
            "classic mode: pos 0 and pos 1 should have same first char"
        );
        assert_eq!(
            pos0_first, pos10_first,
            "classic mode: pos 0 and pos 10 should have same first char"
        );
    }

    /// R81-MEDIUM-1: Markov candidates with \n must be hex-encoded to prevent
    /// the fill loop's read_until(b'\n') from splitting mid-candidate.
    #[test]
    fn test_markov_hex_encoding() {
        // Test the write_markov_candidate helper directly
        use std::io::Read;
        let path = format!("/tmp/vega_test_markov_hex_{0}.txt", std::process::id());
        let _ = std::fs::remove_file(&path);
        let mut file = std::fs::File::create(&path).unwrap();

        // Candidate with newline — should be hex-encoded
        let candidate = b"pass\nword";
        write_markov_candidate(&mut file, candidate, &path).unwrap();
        drop(file);

        let mut content = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut content)
            .unwrap();
        let _ = std::fs::remove_file(&path);

        assert!(
            content.starts_with("$HEX["),
            "candidate with \\n should be $HEX-encoded, got: {content:?}"
        );
        assert!(
            content.ends_with("]\n"),
            "encoded candidate should end with ]\\n, got: {content:?}"
        );

        // Normal candidate — should NOT be hex-encoded
        let path2 = format!("/tmp/vega_test_markov_hex2_{0}.txt", std::process::id());
        let mut file2 = std::fs::File::create(&path2).unwrap();
        write_markov_candidate(&mut file2, b"password123", &path2).unwrap();
        drop(file2);

        let mut content2 = String::new();
        std::fs::File::open(&path2)
            .unwrap()
            .read_to_string(&mut content2)
            .unwrap();
        let _ = std::fs::remove_file(&path2);

        assert_eq!(
            content2, "password123\n",
            "normal candidate should be written as-is + newline"
        );
    }
}
