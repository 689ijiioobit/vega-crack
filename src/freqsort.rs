//! Wordlist frequency sorting.
//! Sorts a wordlist by password frequency (most common first) or by length (shorter first).

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::atomic::AtomicU64;
static FREQSORT_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Sort a wordlist by frequency ordering and write to a temp file.
/// If freq_file is provided, passwords are ordered by their position in the frequency file.
/// If no freq_file, sort by length (shorter first).
/// Returns the path to the sorted temp file and the line count.
pub fn sort_wordlist_by_frequency(
    wordlist_path: &str,
    freq_file: Option<&str>,
    _min_pw_len: usize,
) -> Result<(String, u64), String> {
    let file = File::open(wordlist_path).map_err(|e| format!("Cannot open wordlist: {e}"))?;
    let mut reader = BufReader::new(file);

    // Load all passwords into memory (use read_until for non-UTF-8 wordlist support)
    let mut entries: Vec<(Vec<u8>, u32)> = Vec::new();
    let mut idx: u32 = 0;
    let mut buf = Vec::with_capacity(256);
    loop {
        buf.clear();
        let n = reader
            .read_until(b'\n', &mut buf)
            .map_err(|e| format!("Read error: {e}"))?;
        if n == 0 {
            break;
        }
        // Strip \n and \r\n
        while buf.last() == Some(&b'\n') || buf.last() == Some(&b'\r') {
            buf.pop();
        }
        if buf.is_empty() {
            continue;
        }
        entries.push((buf.clone(), idx));
        idx += 1;
        // Cap at 2GB of entries to prevent OOM
        if entries.len() > 30_000_000 {
            eprintln!("[!] Wordlist exceeds 30M lines — truncating for freq-sort");
            break;
        }
    }

    if let Some(freq_path) = freq_file {
        // Frequency-based sort
        eprintln!("[*] Loading frequency table from {freq_path}...");
        let freq_map = build_freq_rank_map(freq_path)?;
        eprintln!("[*] Frequency table: {} entries", freq_map.len());
        entries.sort_by_key(|(pw, orig_idx)| {
            let rank = freq_map.get(pw).copied().unwrap_or(u32::MAX);
            (rank, *orig_idx)
        });
    } else {
        // Length-based sort (shorter first, stable)
        eprintln!("[*] Sorting by length (shorter first)...");
        entries.sort_by_key(|(pw, orig_idx)| (pw.len() as u32, *orig_idx));
    }

    // Write sorted wordlist to temp file
    let file_id = FREQSORT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let temp_path = format!(
        "/tmp/vega_freqsort_{}_{}_{}",
        std::process::id(),
        file_id,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    );
    let result: Result<(String, u64), String> = (|| {
        let mut out = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&temp_path)
            .map_err(|e| format!("Cannot create temp file: {e}"))?;
        let mut count: u64 = 0;
        for (pw, _) in &entries {
            out.write_all(pw).map_err(|e| format!("Write error: {e}"))?;
            out.write_all(b"\n")
                .map_err(|e| format!("Write error: {e}"))?;
            count += 1;
        }
        Ok((temp_path.clone(), count))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path); // FIX: cleanup on error (Finding 11)
    }
    result
}

/// Build a frequency rank map from a frequency file.
/// Returns HashMap<password_bytes, rank> where lower rank = more common.
fn build_freq_rank_map(freq_file: &str) -> Result<HashMap<Vec<u8>, u32>, String> {
    // FIX F7: Validate file size to prevent OOM
    if let Ok(meta) = std::fs::metadata(freq_file) {
        if meta.len() > 100_000_000 {
            return Err(format!(
                "freq file too large ({} bytes, max 100MB)",
                meta.len()
            ));
        }
    }
    let file = File::open(freq_file).map_err(|e| format!("Cannot open freq file: {e}"))?;
    let mut reader = BufReader::new(file);
    let mut map = HashMap::new();
    let mut rank: u32 = 0;
    let mut buf = Vec::with_capacity(256);
    loop {
        buf.clear();
        let n = reader
            .read_until(b'\n', &mut buf)
            .map_err(|e| format!("Read error: {e}"))?;
        if n == 0 {
            break;
        }
        while buf.last() == Some(&b'\n') || buf.last() == Some(&b'\r') {
            buf.pop();
        }
        if buf.is_empty() {
            continue;
        }
        map.entry(buf.clone()).or_insert(rank); // FIX: keep lowest rank for duplicates
        rank += 1;
    }
    Ok(map)
}
