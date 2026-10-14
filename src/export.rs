//! Export cracked credentials in various formats.

use crate::potfile;

/// Export cracked results from the potfile in the specified format.
/// Formats: csv, json, sql, metasploit, hashcat.
pub fn export_results(potfile_path: &str, format: &str, export_file: &str) {
    if potfile_path.is_empty() {
        eprintln!("[!] Export requires --potfile to collect results");
        return;
    }
    let entries = match potfile::load_potfile_passwords(potfile_path) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[!] Failed to read potfile for export: {e}");
            return;
        }
    };
    if entries.is_empty() {
        eprintln!("[*] No cracked credentials to export");
        return;
    }

    let output = match format {
        "csv" => export_csv(&entries),
        "json" => export_json(&entries),
        "sql" => export_sql(&entries),
        "metasploit" => export_metasploit(&entries),
        "hashcat" => export_hashcat(&entries),
        _ => {
            eprintln!(
                "[!] Unknown export format '{format}'. Valid: csv, json, sql, metasploit, hashcat"
            );
            return;
        }
    };

    if export_file.is_empty() {
        print!("{output}");
    } else {
        match std::fs::write(export_file, &output) {
            Ok(_) => eprintln!("[*] Export written to {export_file}"),
            Err(e) => eprintln!("[!] Failed to write export file: {e}"),
        }
    }
}

/// Escape a string for CSV (RFC 4180): wrap in quotes if it contains
/// comma, quote, or newline. Double internal quotes. Prefix with tab
/// to prevent formula injection in spreadsheet apps (DDE attacks).
fn csv_escape(s: &str) -> String {
    let needs_quoting = s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r');
    let needs_formula_guard =
        s.starts_with('=') || s.starts_with('+') || s.starts_with('-') || s.starts_with('@');
    let escaped = s.replace('"', "\"\"");
    if needs_formula_guard {
        // FIX: Tab prefix neutralizes formula injection in Excel/LibreOffice
        if needs_quoting {
            format!("\t\"{escaped}\"")
        } else {
            format!("\t{escaped}")
        }
    } else if needs_quoting {
        format!("\"{escaped}\"")
    } else {
        escaped
    }
}

fn export_csv(entries: &[(String, String)]) -> String {
    let mut s = String::from("password,hash\n");
    for (hash, pw) in entries {
        let pw_bytes = potfile::decode_hex_password(pw);
        let pw_str = String::from_utf8_lossy(&pw_bytes);
        s.push_str(&format!("{},{}\n", csv_escape(&pw_str), csv_escape(hash)));
    }
    s
}

/// Escape a string for JSON: backslash and double-quote.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn export_json(entries: &[(String, String)]) -> String {
    let mut s = String::from("[\n");
    for (i, (hash, pw)) in entries.iter().enumerate() {
        let pw_bytes = potfile::decode_hex_password(pw);
        let pw_hex: String = pw_bytes.iter().map(|b| format!("{:02x}", b)).collect();
        let pw_str = String::from_utf8_lossy(&pw_bytes);
        s.push_str(&format!(
            "  {{\"password\": \"{}\", \"password_hex\": \"{}\", \"hash\": \"{}\"}}",
            json_escape(&pw_str),
            pw_hex,
            json_escape(hash)
        ));
        if i + 1 < entries.len() {
            s.push(',');
        }
        s.push('\n');
    }
    s.push(']');
    s
}

/// Escape a string for SQL: double single quotes (SQL-standard string literals).
/// Backslash is NOT doubled — with standard_conforming_strings=on (PostgreSQL default),
/// backslash is a literal character. MySQL with NO_BACKSLASH_ESCAPES also treats it as literal.
/// Single-quote doubling prevents injection in all SQL dialects.
/// R105 LOW-5: the password VALUE itself is emitted as hex (password_hex), not
/// this escaped string — attacker-influenced passwords can end in a backslash
/// which would escape the closing quote under MySQL's default mode.
/// sql_escape is used only for the hash column (hex-only, safe) and comments.
fn sql_escape(s: &str) -> String {
    s.replace('\'', "''")
}

fn export_sql(entries: &[(String, String)]) -> String {
    let mut s = String::from("-- vega-crack export\n");
    s.push_str("-- SQL-standard string literals (standard_conforming_strings=on)\n");
    s.push_str(
        "-- Compatible with PostgreSQL (default since 9.1) and MySQL with NO_BACKSLASH_ESCAPES\n",
    );
    for (hash, pw) in entries {
        let pw_bytes = potfile::decode_hex_password(pw);
        let pw_hex: String = pw_bytes.iter().map(|b| format!("{:02x}", b)).collect();
        // R105 LOW-5: password column carries the HEX form (injection-proof in
        // every dialect — no quote/backslash can appear in hex); the readable
        // form goes in a trailing comment for humans. A password ending in
        // backslash previously escaped the closing quote under MySQL's default
        // backslash-escapes mode.
        let pw_str = String::from_utf8_lossy(&pw_bytes);
        s.push_str(&format!(
            "INSERT INTO credentials (password, password_hex, hash) VALUES ('{}', '{}', '{}'); -- pw: {}\n",
            pw_hex,
            pw_hex,
            sql_escape(hash),
            pw_str.escape_debug()
        ));
    }
    s
}

/// For hashcat and metasploit formats: keep $HEX[...] encoding if present
/// to preserve delimiter round-tripping. Only decode if not $HEX-encoded.
fn export_metasploit(entries: &[(String, String)]) -> String {
    let mut s = String::new();
    for (hash, pw) in entries {
        let pw_display = pw.clone(); // FIX: removed dead if/else (both branches identical)
        let essid = extract_essid(hash);
        s.push_str(&format!("{}:{}\n", essid, pw_display));
    }
    s
}

fn export_hashcat(entries: &[(String, String)]) -> String {
    let mut s = String::new();
    for (hash, pw) in entries {
        // Keep original potfile encoding ($HEX or plain) for round-trip safety
        s.push_str(&format!("{}:{}\n", hash, pw));
    }
    s
}

fn extract_essid(hash: &str) -> String {
    // 22000 format: WPA*01*pmkid*ap*sta*essid***01
    if hash.starts_with("WPA*") {
        let fields: Vec<&str> = hash.split('*').collect();
        if fields.len() > 5 {
            return decode_hex_essid(fields[5]);
        }
    }
    // 16800 format: PMKID*essid*ap*sta*pmkid (starts with "PMKID*")
    if hash.starts_with("PMKID*") {
        let fields: Vec<&str> = hash.split('*').collect();
        if fields.len() > 1 {
            return decode_hex_essid(fields[1]);
        }
    }
    // Legacy 16800: pmkid_hex*ap_mac*sta_mac*essid_hex
    // ESSID is field index 3, not 1 (field 1 is AP MAC)
    if hash.len() > 32 && hash.contains('*') {
        let fields: Vec<&str> = hash.split('*').collect();
        if fields.len() > 3 {
            return decode_hex_essid(fields[3]);
        }
    }
    "?".to_string()
}

fn decode_hex_essid(hex: &str) -> String {
    let mut bytes = Vec::with_capacity(hex.len() / 2);
    for i in (0..hex.len()).step_by(2) {
        if i + 2 <= hex.len() {
            if let Ok(b) = u8::from_str_radix(&hex[i..i + 2], 16) {
                bytes.push(b);
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
