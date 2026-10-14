//! PRINCE (PRobability INfinite Chained Elements) attack mode.
//!
//! Takes words from a dictionary and concatenates 1-N of them to form
//! password candidates within a specified length range.
//! Compatible with hashcat's --prince attack mode.
//!
//! Algorithm:
//! 1. Read dictionary, extract unique words (binary-safe)
//! 2. Group words by length
//! 3. For K = 1 to max_words:
//!    For each ordered combination of K words:
//! 4. Deduplicate candidates

use std::collections::HashSet;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{Read, Write};

/// Dedup cap — mirrors main.rs DEDUP_CAP. Prevents OOM from unbounded HashSet growth.
const PRINCE_DEDUP_CAP: usize = 1_000_000;
use std::path::Path;

/// PRINCE attack context.
pub struct Prince {
    /// All unique words from the dictionary, grouped by length.
    /// words_by_len[L] = all words of length L (L = 1..=63)
    words_by_len: Vec<Vec<Vec<u8>>>,
    /// Minimum candidate length
    min_len: usize,
    /// Maximum candidate length
    max_len: usize,
    /// Maximum number of word segments to concatenate
    max_words: usize,
    /// Total number of unique words
    #[allow(dead_code)]
    word_count: usize,
    /// Total estimated candidates (may be capped by max_keyspace)
    #[allow(dead_code)]
    total_candidates: u64,
}

impl Prince {
    /// Load dictionary and build PRINCE context.
    pub fn load(
        dict_path: &str,
        min_len: usize,
        max_len: usize,
        max_words: usize,
    ) -> Result<Self, String> {
        let path = Path::new(dict_path);
        if !path.is_file() {
            return Err(format!("PRINCE dictionary not found: {dict_path}"));
        }

        // Size guard: cap at 100MB to prevent OOM on large dictionaries.
        let meta =
            std::fs::metadata(path).map_err(|e| format!("Failed to stat PRINCE dict: {e}"))?;
        if meta.len() > 100_000_000 {
            return Err(format!(
                "PRINCE dictionary too large ({} bytes, max 100MB)",
                meta.len()
            ));
        }

        // Read file as bytes (binary-safe, like wordlist mode)
        let mut file = File::open(path).map_err(|e| format!("Failed to open PRINCE dict: {e}"))?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)
            .map_err(|e| format!("Failed to read PRINCE dict: {e}"))?;

        // Parse words (split on newline, binary-safe with read_until pattern)
        let mut words: Vec<Vec<u8>> = Vec::new();
        let mut seen: HashSet<Vec<u8>> = HashSet::new();
        let mut start = 0;
        for i in 0..buf.len() {
            if buf[i] == b'\n' {
                let mut word = buf[start..i].to_vec();
                // Strip \r if present (Windows line endings)
                if word.last() == Some(&b'\r') {
                    word.pop();
                }
                if !word.is_empty() && word.len() <= 63 && !seen.contains(&word) {
                    seen.insert(word.clone());
                    words.push(word);
                }
                start = i + 1;
            }
        }
        // Handle last line (no trailing newline)
        if start < buf.len() {
            let mut word = buf[start..].to_vec();
            if word.last() == Some(&b'\r') {
                word.pop();
            }
            if !word.is_empty() && word.len() <= 63 && !seen.contains(&word) {
                seen.insert(word.clone());
                words.push(word);
            }
        }

        if words.is_empty() {
            return Err("PRINCE dictionary contains no valid words".into());
        }

        // Group words by length
        let max_word_len = words.iter().map(|w| w.len()).max().unwrap_or(0);
        let mut words_by_len: Vec<Vec<Vec<u8>>> = vec![Vec::new(); max_word_len + 1];
        for word in words {
            words_by_len[word.len()].push(word);
        }

        let word_count: usize = words_by_len.iter().map(|v| v.len()).sum();

        // Estimate total candidates (rough upper bound)
        // For K words: C = sum over all length combinations that sum to [min_len, max_len]
        // This is complex to compute exactly, so we estimate conservatively
        let total_candidates = estimate_candidates(&words_by_len, min_len, max_len, max_words);

        eprintln!(
            "[*] PRINCE: {} unique words, lengths 1-{}, candidates: {} (est), max_words: {}",
            word_count, max_word_len, total_candidates, max_words
        );

        Ok(Prince {
            words_by_len,
            min_len,
            max_len,
            max_words,
            word_count,
            total_candidates,
        })
    }

    /// Estimate total candidates (rough upper bound for progress display).
    #[allow(dead_code)]
    pub fn total_candidates(&self) -> u64 {
        self.total_candidates
    }

    /// Generate all PRINCE candidates to a temp file.
    /// Returns (path, count) for TempWordlist integration.
    pub fn generate_to_file(
        &self,
        skip: u64,
        limit: u64,
        dedup: bool,
        min_pw_len: usize,
    ) -> Result<(String, u64), Box<dyn std::error::Error>> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let pid = std::process::id();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!("vega_crack_prince_{pid}_{seq}.txt"));
        let path = p.to_string_lossy().to_string();

        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;

        // Wrap write loop in a closure so we can clean up the temp file on error.
        let result = (|| -> Result<u64, std::io::Error> {
            let mut seen: HashSet<Vec<u8>> = HashSet::new();
            let mut count = 0u64;
            let mut skipped = 0u64;

            // Build available lengths (sorted ascending)
            let avail_lens: Vec<usize> = (1..self.words_by_len.len())
                .filter(|&l| !self.words_by_len[l].is_empty())
                .collect();

            // For each number of word segments K = 1 to max_words
            for k in 1..=self.max_words {
                // Generate all ordered combinations of k lengths that sum to [min_len, max_len]
                let length_combos =
                    generate_length_combinations(&avail_lens, k, self.min_len, self.max_len);

                for lens in &length_combos {
                    // For each combination of words with these lengths
                    let word_lists: Vec<&Vec<Vec<u8>>> =
                        lens.iter().map(|&l| &self.words_by_len[l]).collect();

                    // Generate cartesian product via odometer (avoids materializing
                    // the full cross-product in memory — prevents OOM on large dicts).
                    let sizes: Vec<usize> = word_lists.iter().map(|l| l.len()).collect();
                    if sizes.is_empty() || sizes.contains(&0) {
                        continue;
                    }
                    // Compute total combinations with overflow check
                    let mut total_combos: u128 = 1;
                    for &s in &sizes {
                        total_combos = total_combos.saturating_mul(s as u128);
                    }
                    if total_combos == 0 || total_combos > u64::MAX as u128 {
                        // Too many combinations — skip this length combo
                        continue;
                    }

                    for flat_idx in 0..total_combos as u64 {
                        // Decode flat_idx into per-segment word indices (odometer)
                        let mut idx = flat_idx;
                        let mut pw: Vec<u8> = Vec::new();
                        for (seg_i, &seg_size) in sizes.iter().enumerate() {
                            let word_idx = idx % seg_size as u64;
                            idx /= seg_size as u64;
                            pw.extend_from_slice(&word_lists[seg_i][word_idx as usize]);
                        }

                        // Check length
                        if pw.len() < min_pw_len || pw.len() > self.max_len || pw.len() > 63 {
                            continue;
                        }

                        // Dedup (before skip/limit so all three count the same unit).
                        // Cap the HashSet to prevent OOM (mirrors main.rs DEDUP_CAP).
                        if dedup && seen.len() < PRINCE_DEDUP_CAP {
                            if !seen.insert(pw.clone()) {
                                continue;
                            }
                        } else if dedup && seen.len() >= PRINCE_DEDUP_CAP {
                            // Cap reached — dedup disabled for remaining candidates.
                            // Check if this is a duplicate of something already seen.
                            if !seen.contains(&pw) {
                                // Not a duplicate, but we can't track it. Accept it.
                            } else {
                                continue;
                            }
                        }

                        // Apply skip (post-dedup, consistent with limit)
                        if skipped < skip {
                            skipped += 1;
                            continue;
                        }

                        // Apply limit (post-dedup, consistent with skip)
                        if limit > 0 && count >= limit {
                            file.flush()?;
                            return Ok(count);
                        }

                        // Write to file (handle \n/\r with $HEX encoding)
                        let needs_hex = pw.iter().any(|&b| b == b'\n' || b == b'\r')
                            || (pw.len() >= 5 && &pw[..5] == b"$HEX[");
                        if needs_hex {
                            let hex: String = pw.iter().map(|b| format!("{:02x}", b)).collect();
                            file.write_all(b"$HEX[")?;
                            file.write_all(hex.as_bytes())?;
                            file.write_all(b"]\n")?;
                        } else {
                            file.write_all(&pw)?;
                            file.write_all(b"\n")?;
                        }
                        count += 1;
                    }
                }
            }

            file.flush()?;
            Ok(count)
        })();

        match result {
            Ok(count) => Ok((path, count)),
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                Err(format!("PRINCE write failed: {e}").into())
            }
        }
    }
}

/// Generate all ordered combinations of k lengths from avail_lens
/// that sum to a value in [min_sum, max_sum].
/// Caps at 1M combinations to prevent OOM (MEDIUM-3).
/// When capped, distributes the budget across all first-segment lengths
/// to avoid biased truncation (R89 LOW fix).
/// Uses a counting pass to avoid materializing all combos before truncating
/// (R90 MEDIUM: per_first materialization caused OOM regression).
fn generate_length_combinations(
    avail_lens: &[usize],
    k: usize,
    min_sum: usize,
    max_sum: usize,
) -> Vec<Vec<usize>> {
    const MAX_COMBOS: usize = 1_000_000;
    if k == 0 {
        return vec![vec![]];
    }

    if k == 1 {
        return avail_lens
            .iter()
            .filter(|&&l| l >= min_sum && l <= max_sum)
            .map(|&l| vec![l])
            .take(MAX_COMBOS)
            .collect();
    }

    // Counting pass: compute sub-combo counts per first_len WITHOUT materializing.
    // This avoids OOM from holding all groups simultaneously (R90 MEDIUM fix).
    let mut group_counts: Vec<(usize, usize)> = Vec::new(); // (first_len, sub_count)
    for &first_len in avail_lens {
        if first_len > max_sum {
            continue;
        }
        let remaining_min = min_sum.saturating_sub(first_len);
        let remaining_max = max_sum - first_len;
        let sub_count = count_length_combinations(avail_lens, k - 1, remaining_min, remaining_max);
        if sub_count > 0 {
            group_counts.push((first_len, sub_count));
        }
    }

    let total_subs: usize = group_counts.iter().map(|(_, c)| *c).sum();
    if total_subs <= MAX_COMBOS {
        // No cap needed — materialize all (same order as before)
        let mut result = Vec::with_capacity(total_subs);
        for &(first_len, _) in &group_counts {
            let remaining_min = min_sum.saturating_sub(first_len);
            let remaining_max = max_sum - first_len;
            let subs =
                generate_length_combinations(avail_lens, k - 1, remaining_min, remaining_max);
            for sub in subs {
                let mut combo = vec![first_len];
                combo.extend(sub);
                result.push(combo);
            }
        }
        return result;
    }

    // Cap needed: distribute budget proportionally across first_len groups.
    // Materialize one group at a time, take only the budget share — no
    // intermediate storage of all groups simultaneously.
    eprintln!(
        "[!] PRINCE: length combinations capped at {MAX_COMBOS} (k={k}). Reduce --prince-words for full coverage."
    );
    let num_groups = group_counts.len().max(1);
    let budget_per_group = MAX_COMBOS / num_groups;
    let mut result = Vec::with_capacity(MAX_COMBOS);
    for &(first_len, _) in &group_counts {
        let remaining_min = min_sum.saturating_sub(first_len);
        let remaining_max = max_sum - first_len;
        let subs = generate_length_combinations(avail_lens, k - 1, remaining_min, remaining_max);
        for sub in subs.iter().take(budget_per_group) {
            let mut combo = vec![first_len];
            combo.extend(sub.iter());
            result.push(combo);
            if result.len() >= MAX_COMBOS {
                return result;
            }
        }
    }
    result
}

/// Count combinations without materializing them (counting pass for cap check).
fn count_length_combinations(
    avail_lens: &[usize],
    k: usize,
    min_sum: usize,
    max_sum: usize,
) -> usize {
    const MAX_COMBOS: usize = 1_000_000;
    if k == 0 {
        return 1;
    }
    if k == 1 {
        return avail_lens
            .iter()
            .filter(|&&l| l >= min_sum && l <= max_sum)
            .take(MAX_COMBOS)
            .count();
    }
    let mut total = 0usize;
    for &first_len in avail_lens {
        if first_len > max_sum {
            continue;
        }
        let remaining_min = min_sum.saturating_sub(first_len);
        let remaining_max = max_sum - first_len;
        let sub_count = count_length_combinations(avail_lens, k - 1, remaining_min, remaining_max);
        total = total.saturating_add(sub_count);
        if total > MAX_COMBOS {
            return total; // early exit — we only need to know if > MAX_COMBOS
        }
    }
    total
}

/// Rough estimate of total candidates.
fn estimate_candidates(
    words_by_len: &[Vec<Vec<u8>>],
    min_len: usize,
    max_len: usize,
    max_words: usize,
) -> u64 {
    let avail_lens: Vec<usize> = (1..words_by_len.len())
        .filter(|&l| !words_by_len[l].is_empty())
        .collect();

    let mut total: u64 = 0;
    for k in 1..=max_words {
        let length_combos = generate_length_combinations(&avail_lens, k, min_len, max_len);
        for lens in &length_combos {
            let mut combo_count: u64 = 1;
            for &l in lens {
                combo_count = combo_count.saturating_mul(words_by_len[l].len() as u64);
            }
            total = total.saturating_add(combo_count);
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prince_single_word() {
        // Create a temp dictionary
        let dict_path = "/tmp/vega_crack_prince_test_dict.txt";
        std::fs::write(dict_path, "hello\nworld\ntest\n").unwrap();

        let prince = Prince::load(dict_path, 4, 8, 1).unwrap();
        assert_eq!(prince.word_count, 3);
        // With k=1, min=4, max=8: "hello"(5), "world"(5), "test"(4) all qualify
        let total = prince.total_candidates();
        assert_eq!(total, 3, "3 single words of length 4-5");

        let _ = std::fs::remove_file(dict_path);
    }

    #[test]
    fn test_prince_two_words() {
        let dict_path = "/tmp/vega_crack_prince_test_dict2.txt";
        std::fs::write(dict_path, "ab\ncd\nef\n").unwrap();

        let prince = Prince::load(dict_path, 4, 4, 2).unwrap();
        assert_eq!(prince.word_count, 3);
        // k=1: no words of length 4 (all length 2) → 0
        // k=2: all pairs of length-2 words → 3*3 = 9 candidates of length 4
        let total = prince.total_candidates();
        assert_eq!(total, 9, "9 two-word combinations of length 4");

        let _ = std::fs::remove_file(dict_path);
    }

    #[test]
    fn test_prince_generate() {
        let dict_path = "/tmp/vega_crack_prince_test_dict3.txt";
        std::fs::write(dict_path, "te\nst\n").unwrap();

        let prince = Prince::load(dict_path, 4, 4, 2).unwrap();
        let (file_path, count) = prince.generate_to_file(0, 0, true, 1).unwrap();

        // Should generate: test (te+st), stte (st+te), tete (te+te), stst (st+st)
        // With dedup: all 4 are unique
        assert_eq!(count, 4, "4 candidates: test, stte, tete, stst");

        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("test"), "should contain 'test'");
        assert!(content.contains("stte"), "should contain 'stte'");

        let _ = std::fs::remove_file(dict_path);
        let _ = std::fs::remove_file(&file_path);
    }

    #[test]
    fn test_prince_dedup() {
        let dict_path = "/tmp/vega_crack_prince_test_dict4.txt";
        std::fs::write(dict_path, "a\na\nb\n").unwrap();

        let prince = Prince::load(dict_path, 1, 2, 2).unwrap();
        // Dedup at load: only 2 unique words (a, b)
        assert_eq!(prince.word_count, 2, "dedup at load: 2 unique words");

        let _ = std::fs::remove_file(dict_path);
    }

    #[test]
    fn test_prince_length_filter() {
        let dict_path = "/tmp/vega_crack_prince_test_dict5.txt";
        std::fs::write(dict_path, "x\nhello\nworld\ntest\n").unwrap();

        // min=5, max=5: only "hello" and "world" qualify for k=1
        let prince = Prince::load(dict_path, 5, 5, 1).unwrap();
        let total = prince.total_candidates();
        assert_eq!(total, 2, "2 words of length 5: hello, world");

        let _ = std::fs::remove_file(dict_path);
    }

    #[test]
    fn test_length_combinations() {
        let avail = vec![2usize, 3];
        // k=2, min=4, max=5: (2,2)=4, (2,3)=5, (3,2)=5, (3,3)=6→excluded
        let combos = generate_length_combinations(&avail, 2, 4, 5);
        assert_eq!(combos.len(), 3, "3 valid length combinations");
    }

    #[test]
    fn test_prince_empty_dict() {
        let dict_path = "/tmp/vega_crack_prince_test_dict6.txt";
        std::fs::write(dict_path, "\n\n\n").unwrap();

        let result = Prince::load(dict_path, 1, 8, 2);
        assert!(result.is_err(), "empty dict should error");

        let _ = std::fs::remove_file(dict_path);
    }

    #[test]
    fn test_prince_skip_limit() {
        let dict_path = "/tmp/vega_crack_prince_test_dict7.txt";
        std::fs::write(dict_path, "aa\nbb\ncc\ndd\n").unwrap();

        let prince = Prince::load(dict_path, 2, 2, 1).unwrap();
        // 4 words of length 2, k=1: 4 candidates
        // Skip 1, limit 2: should return 2 candidates
        let (_path, count) = prince.generate_to_file(1, 2, false, 1).unwrap();
        assert_eq!(count, 2, "skip 1, limit 2 → 2 candidates");

        let _ = std::fs::remove_file(dict_path);
    }
}
