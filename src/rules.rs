//! Rule-based attack: apply hashcat-compatible rules to wordlist candidates.
//!
//! Rules are CPU-side transformations applied to each wordlist word before
//! sending to the GPU. This gives a 10-100x multiplier on any wordlist with
//! zero VRAM cost.
//!
//! Rule syntax (compatible with hashcat -r):
//!   Each line in a .rule file is one rule — a sequence of operations.
//!   Operations are single-char codes, some followed by parameters.
//!
//! Mangle operations:
//!   :  = noop          l  = lowercase      u  = uppercase
//!   c  = capitalize     C  = upper first   t  = toggle case
//!   r  = reverse        d  = duplicate      f  = reflect
//!   {  = rotate left    }  = rotate right
//!   $X = append char X  ^X = prepend char X
//!   [  = delete first   ]  = delete last
//!   DN = delete at N    xNM = extract from N for M
//!   ONM = omit from N for M
//!   iNX = insert X at N  oNX = overstrike X at N
//!   'N = truncate at N  sAB = replace A with B
//!   @X = purge all X    q  = dupe every char
//!   zN = dupe first char N times
//!   ZN = dupe last char N times
//!   pN = dupe word N times
//!   yN = dupe first N chars
//!   YN = dupe last N chars
//!   k  = swap first two  K  = swap last two
//!   *NM = swap chars at N and M
//!   TN = toggle case at N
//!   E  = title case      eX = title case with sep X
//!   +N = incr char at N  -N = decr char at N
//!   BN = add N to char at pos (??)
//!   .N = replace pos+1 with pos  ,N = replace pos-1 with pos
//!   h  = to hex lower   H  = to hex upper
//!   LN = shift left at N RN = shift right at N
//!   vNX = insert X at every Nth position
//!
//! Reject operations:
//!   <N = reject len < N  >N = reject len > N
//!   _N = reject len == N  !X = reject if contains X
//!   /X = reject if not contains X  =NX = reject if char at N == X
//!   (X = reject if first == X  )X = reject if last == X
//!   %NX = reject if N occurrences of X
//!   Q  = reject if matches memorized
//!
//! Memory operations:
//!   M  = memorize word    4  = append memorized
//!   6  = prepend memorized XNML = extract from memory

#![allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    dead_code
)]

/// Maximum password length (WPA: 63 bytes)
const MAX_PW: usize = 63;

// ── Class helpers — match hashcat's rp.c class functions exactly ──

fn class_num(c: u8) -> bool {
    c.is_ascii_digit()
}

fn class_lower(c: u8) -> bool {
    c.is_ascii_lowercase()
}

fn class_upper(c: u8) -> bool {
    c.is_ascii_uppercase()
}

fn class_lower_hex(c: u8) -> bool {
    class_num(c) || (b'a'..=b'f').contains(&c)
}

fn class_upper_hex(c: u8) -> bool {
    class_num(c) || (b'A'..=b'F').contains(&c)
}

fn class_sym(c: u8) -> bool {
    c == b' '
        || (b'!'..=b'/').contains(&c)
        || (b':'..=b'@').contains(&c)
        || (b'['..=b'`').contains(&c)
        || (b'{'..=b'~').contains(&c)
}

fn class_alpha(c: u8) -> bool {
    class_lower(c) || class_upper(c)
}

/// Check if a byte belongs to the class identified by the given class char.
/// Dispatches like hashcat's per-class functions: l/u/d/h/H/s/a.
fn char_in_class(c: u8, class: u8) -> bool {
    match class {
        b'l' => class_lower(c),
        b'u' => class_upper(c),
        b'd' => class_num(c),
        b'h' => class_lower_hex(c),
        b'H' => class_upper_hex(c),
        b's' => class_sym(c),
        b'a' => class_alpha(c),
        _ => false,
    }
}

/// A parsed rule — a sequence of operations to apply to a candidate.
#[derive(Clone, Debug)]
pub struct Rule {
    ops: Vec<RuleOp>,
}

/// A single rule operation.
#[derive(Clone, Debug)]
enum RuleOp {
    // Mangle — no params
    Noop,         // :
    Lower,        // l
    Upper,        // u
    Cap,          // c
    UpperFirst,   // C
    ToggleAll,    // t
    Reverse,      // r
    Duplicate,    // d
    Reflect,      // f
    RotLeft,      // {
    RotRight,     // }
    DelFirst,     // [
    DelLast,      // ]
    DupeAllChars, // q
    SwapFirst,    // k
    SwapLast,     // K
    TitleCase,    // E
    ToHexLower,   // h
    ToHexUpper,   // H

    // Mangle — with position param
    ToggleAt(usize),     // TN
    DelAt(usize),        // DN
    TruncAt(usize),      // 'N
    IncAt(usize),        // +N
    DecAt(usize),        // -N
    ShiftLeftAt(usize),  // LN
    ShiftRightAt(usize), // RN

    // Mangle — with char param
    Append(u8),  // $X
    Prepend(u8), // ^X
    Purge(u8),   // @X

    // Mangle — with position + char
    Insert(usize, u8),     // iNX
    Overstrike(usize, u8), // oNX

    // Mangle — with two chars
    Replace(u8, u8), // sAB

    // Mangle — with position + count
    DupeFirstN(usize),    // zN
    DupeLastN(usize),     // ZN
    DupeWord(usize),      // pN
    DupFirstChars(usize), // yN
    DupLastChars(usize),  // YN

    // Mangle — with two positions
    Swap(usize, usize), // *NM

    // Mangle — extract range
    Extract(usize, usize), // xNM

    // Mangle — omit range
    Omit(usize, usize), // ONM

    // Mangle — title with separator
    TitleSep(u8), // eX

    // Mangle — replace adjacent
    ReplaceNext(usize), // .N
    ReplacePrev(usize), // ,N

    // Mangle — insert at every Nth
    InsertEvery(usize, u8), // vNX

    // Reject — with position
    RejectLess(usize),    // <N
    RejectGreater(usize), // >N
    RejectEqual(usize),   // _N

    // Reject — with char
    RejectContains(u8),    // !X
    RejectNotContains(u8), // /X
    RejectFirst(u8),       // (X
    RejectLast(u8),        // )X

    // Reject — with position + char
    RejectAt(usize, u8), // =NX

    // Reject — with count + char
    RejectContainsN(usize, u8), // %NX

    // Reject — memory
    RejectMemory, // Q

    // Memory ops
    Memorize,                        // M
    AppendMemory,                    // 4
    PrependMemory,                   // 6
    ExtractMem(usize, usize, usize), // XNML

    // Class-based operations (~ prefix)
    // Note: class 'a' (alpha) is a vega-crack extension — hashcat's class-based
    // ops only support l/u/d/h/H/s and noop for 'a'. We support 'a' as an
    // intentional extension for broader rule compatibility.
    ReplaceClass(u8, u8),   // ~s?CY — replace all chars in class C with char Y
    PurgeClass(u8),         // ~@?C  — purge all chars in class C
    TitleSepClass(u8),      // ~e?C  — title case with class-based separator
    RejectContainClass(u8), // ~!?C  — reject if contains any char in class C
    RejectNotContainClass(u8), // ~/?C  — reject if doesn't contain any char in class C
    RejectEqualFirstClass(u8), // ~(?C  — reject if first char NOT in class C (keep if IN)
    RejectEqualLastClass(u8), // ~)?C  — reject if last char NOT in class C (keep if IN)
    RejectEqualAtClass(usize, u8), // ~=N?C — reject if char at N NOT in class C (keep if IN)
    RejectContainsClass(usize, u8), // ~%N?C — reject if contains fewer than N chars in class C
}

/// Result of applying a rule to a candidate.
/// Ok(Some(bytes)) = accepted, transformed password
/// Ok(None) = rejected by a reject rule
pub fn apply_rule(rule: &Rule, word: &[u8], min_len: usize) -> Option<Vec<u8>> {
    let mut result = word.to_vec();
    let mut memory: Vec<u8> = Vec::new();

    for op in &rule.ops {
        match op {
            // No-param mangle
            RuleOp::Noop => {}
            RuleOp::Lower => {
                for b in &mut result {
                    if b.is_ascii_uppercase() {
                        *b += 32;
                    }
                }
            }
            RuleOp::Upper => {
                for b in &mut result {
                    if b.is_ascii_lowercase() {
                        *b -= 32;
                    }
                }
            }
            RuleOp::Cap => {
                if let Some(first) = result.first_mut() {
                    if first.is_ascii_lowercase() {
                        *first -= 32;
                    }
                }
                for b in result.iter_mut().skip(1) {
                    if b.is_ascii_uppercase() {
                        *b += 32;
                    }
                }
            }
            RuleOp::UpperFirst => {
                // hashcat C = UREST_LFIRST: uppercase entire word, then lowercase first char
                for b in &mut result {
                    if b.is_ascii_lowercase() {
                        *b -= 32;
                    }
                }
                if let Some(first) = result.first_mut() {
                    if first.is_ascii_uppercase() {
                        *first += 32;
                    }
                }
            }
            RuleOp::ToggleAll => {
                for b in &mut result {
                    if b.is_ascii_lowercase() {
                        *b -= 32;
                    } else if b.is_ascii_uppercase() {
                        *b += 32;
                    }
                }
            }
            RuleOp::Reverse => result.reverse(),
            RuleOp::Duplicate => {
                let len = result.len();
                if len * 2 <= MAX_PW {
                    result.extend_from_within(..len);
                }
            }
            RuleOp::Reflect => {
                let rev: Vec<u8> = result.iter().rev().copied().collect();
                if result.len() + rev.len() <= MAX_PW {
                    result.extend(rev);
                }
            }
            RuleOp::RotLeft => {
                if result.len() > 1 {
                    let first = result.remove(0);
                    result.push(first);
                }
            }
            RuleOp::RotRight => {
                if result.len() > 1 {
                    let last = result.pop().unwrap();
                    result.insert(0, last);
                }
            }
            RuleOp::DelFirst => {
                if !result.is_empty() {
                    result.remove(0);
                }
            }
            RuleOp::DelLast => {
                result.pop();
            }
            RuleOp::DupeAllChars => {
                let len = result.len();
                if len * 2 <= MAX_PW {
                    let mut duped = Vec::with_capacity(len * 2);
                    for &b in &result {
                        duped.push(b);
                        duped.push(b);
                    }
                    result = duped;
                }
            }
            RuleOp::SwapFirst => {
                if result.len() >= 2 {
                    result.swap(0, 1);
                }
            }
            RuleOp::SwapLast => {
                let len = result.len();
                if len >= 2 {
                    result.swap(len - 1, len - 2);
                }
            }
            // hashcat E = mangle_title_sep(out, len, ' ') — space is the only separator
            // Matches TitleSep with ' ' as separator exactly
            RuleOp::TitleCase => {
                let mut cap_next = true;
                for b in &mut result {
                    if *b == b' ' {
                        cap_next = true;
                        continue;
                    }
                    if cap_next {
                        cap_next = false;
                        if b.is_ascii_lowercase() {
                            *b -= 32;
                        }
                    } else if b.is_ascii_uppercase() {
                        *b += 32;
                    }
                }
                // hashcat: MANGLE_UPPER_AT(arr, 0) — unconditional uppercase of first char
                if let Some(first) = result.first_mut() {
                    if first.is_ascii_lowercase() {
                        *first -= 32;
                    }
                }
            }
            RuleOp::ToHexLower => {
                result = result
                    .iter()
                    .flat_map(|b| format!("{b:02x}").into_bytes())
                    .collect();
            }
            RuleOp::ToHexUpper => {
                result = result
                    .iter()
                    .flat_map(|b| format!("{b:02X}").into_bytes())
                    .collect();
            }

            // Position-param mangle
            RuleOp::ToggleAt(n) => {
                if *n < result.len() {
                    let b = &mut result[*n];
                    if b.is_ascii_lowercase() {
                        *b -= 32;
                    } else if b.is_ascii_uppercase() {
                        *b += 32;
                    }
                }
            }
            RuleOp::DelAt(n) => {
                if *n < result.len() {
                    result.remove(*n);
                }
            }
            RuleOp::TruncAt(n) => {
                if *n < result.len() {
                    result.truncate(*n);
                }
            }
            RuleOp::IncAt(n) => {
                if *n < result.len() {
                    result[*n] = result[*n].wrapping_add(1);
                }
            }
            RuleOp::DecAt(n) => {
                if *n < result.len() {
                    result[*n] = result[*n].wrapping_sub(1);
                }
            }
            RuleOp::ShiftLeftAt(n) => {
                if *n < result.len() {
                    result[*n] = result[*n].wrapping_shl(1);
                }
            }
            RuleOp::ShiftRightAt(n) => {
                if *n < result.len() {
                    // hashcat uses signed char >> 1 (arithmetic shift, sign-extends).
                    // For u8: simulate by preserving bit 7.
                    let b = result[*n];
                    result[*n] = (b >> 1) | (b & 0x80);
                }
            }

            // Char-param mangle
            RuleOp::Append(c) => {
                if result.len() < MAX_PW {
                    result.push(*c);
                }
            }
            RuleOp::Prepend(c) => {
                if result.len() < MAX_PW {
                    result.insert(0, *c);
                }
            }
            RuleOp::Purge(c) => {
                result.retain(|b| b != c);
            }

            // Position + char
            RuleOp::Insert(n, c) => {
                if *n <= result.len() && result.len() < MAX_PW {
                    result.insert(*n, *c);
                }
            }
            RuleOp::Overstrike(n, c) => {
                if *n < result.len() {
                    result[*n] = *c;
                }
            }

            // Two chars
            RuleOp::Replace(a, b) => {
                for byte in &mut result {
                    if *byte == *a {
                        *byte = *b;
                    }
                }
            }

            // Position + count
            RuleOp::DupeFirstN(n) => {
                if !result.is_empty() && result.len() + n <= MAX_PW {
                    // hashcat: insert N copies of first char at position 0 (PREPEND)
                    let c = result[0];
                    for _ in 0..*n {
                        result.insert(0, c);
                    }
                }
            }
            RuleOp::DupeLastN(n) => {
                if !result.is_empty() && result.len() + n <= MAX_PW {
                    let c = *result.last().unwrap();
                    let mut new = result.clone();
                    new.extend(vec![c; *n]);
                    result = new;
                }
            }
            RuleOp::DupeWord(n) => {
                let len = result.len();
                if len > 0 && len * (n + 1) <= MAX_PW {
                    for _ in 0..*n {
                        result.extend_from_within(..len);
                    }
                }
            }
            RuleOp::DupFirstChars(n) => {
                if *n > 0 && *n <= result.len() && result.len() + *n <= MAX_PW {
                    // hashcat: prepend first N chars (mangle_dupeblock_prepend)
                    let chars: Vec<u8> = result[..*n].to_vec();
                    for (i, &c) in chars.iter().enumerate() {
                        result.insert(i, c);
                    }
                }
            }
            RuleOp::DupLastChars(n) => {
                if *n > 0 && *n <= result.len() && result.len() + *n <= MAX_PW {
                    let chars: Vec<u8> = result[result.len() - *n..].to_vec();
                    result.extend(chars);
                }
            }

            // Two positions
            RuleOp::Swap(a, b) => {
                if *a < result.len() && *b < result.len() {
                    result.swap(*a, *b);
                }
            }

            // Extract range — hashcat xNM: noop if start >= len or start+len > len
            RuleOp::Extract(start, len) => {
                if *start >= result.len() || start.saturating_add(*len) > result.len() {
                    // hashcat noops on out-of-range — keep original unchanged
                } else {
                    result = result[*start..*start + *len].to_vec();
                }
            }

            // Omit range — hashcat ONM: noop if start >= len or start+len > len
            RuleOp::Omit(start, len) => {
                if *start >= result.len() || start.saturating_add(*len) > result.len() {
                    // hashcat noops on out-of-range — keep original unchanged
                } else {
                    result.drain(*start..*start + *len);
                }
            }

            // Title with separator — only capitalize after the specified separator
            // Matches hashcat: non-separator bytes consume cap_next even if non-alpha
            // hashcat also unconditionally uppercases position 0 after the loop
            RuleOp::TitleSep(sep) => {
                let mut cap_next = true;
                for b in &mut result {
                    if *b == *sep {
                        cap_next = true;
                        continue;
                    }
                    if cap_next {
                        cap_next = false;
                        if b.is_ascii_lowercase() {
                            *b -= 32;
                        }
                    } else if b.is_ascii_uppercase() {
                        *b += 32;
                    }
                }
                // hashcat: MANGLE_UPPER_AT(arr, 0) — unconditional uppercase of first char
                if let Some(first) = result.first_mut() {
                    if first.is_ascii_lowercase() {
                        *first -= 32;
                    }
                }
            }

            // Replace adjacent — hashcat: .N replaces char at N with char at N+1
            RuleOp::ReplaceNext(n) => {
                if *n + 1 < result.len() {
                    result[*n] = result[*n + 1];
                }
            }
            // hashcat: ,N replaces char at N with char at N-1
            RuleOp::ReplacePrev(n) => {
                if *n > 0 && *n < result.len() {
                    result[*n] = result[*n - 1];
                }
            }

            // Insert at every Nth
            RuleOp::InsertEvery(n, c) => {
                if *n > 0 && result.len() < MAX_PW {
                    let mut new = Vec::with_capacity(result.len() + result.len() / n);
                    for (i, &b) in result.iter().enumerate() {
                        new.push(b);
                        if (i + 1) % n == 0 && new.len() < MAX_PW {
                            new.push(*c);
                        }
                    }
                    result = new;
                }
            }

            // Reject operations — hashcat rejects (returns None) when condition is MET
            // hashcat <N (REJECT_LESS): reject if len > N → keep if len <= N
            RuleOp::RejectLess(n) => {
                if result.len() > *n {
                    return None;
                }
            }
            // hashcat >N (REJECT_GREATER): reject if len < N → keep if len >= N
            RuleOp::RejectGreater(n) => {
                if result.len() < *n {
                    return None;
                }
            }
            // hashcat _N (REJECT_EQUAL): reject if len != N → keep if len == N
            RuleOp::RejectEqual(n) => {
                if result.len() != *n {
                    return None;
                }
            }
            RuleOp::RejectContains(c) => {
                if result.contains(c) {
                    return None;
                }
            }
            RuleOp::RejectNotContains(c) => {
                if !result.contains(c) {
                    return None;
                }
            }
            // hashcat (X (REJECT_EQUAL_FIRST): reject if first != X → keep if first == X
            RuleOp::RejectFirst(c) => {
                if result.first() != Some(c) {
                    return None;
                }
            }
            // hashcat )X (REJECT_EQUAL_LAST): reject if last != X → keep if last == X
            RuleOp::RejectLast(c) => {
                if result.last() != Some(c) {
                    return None;
                }
            }
            // hashcat =NX (REJECT_EQUAL_AT): reject if char[N] != X → keep if char[N] == X
            // Also rejects if N >= len (out of range)
            RuleOp::RejectAt(n, c) => {
                if result.get(*n) != Some(c) {
                    return None;
                }
            }
            RuleOp::RejectContainsN(n, c) => {
                // hashcat early-reject: if n+1 > len, reject (rp_cpu.c:1599)
                if *n + 1 > result.len() {
                    return None;
                }
                let mut count = 0;
                for b in &result {
                    if *b == *c {
                        count += 1;
                    }
                }
                if count < *n {
                    return None;
                }
            }
            RuleOp::RejectMemory => {
                if result == memory {
                    return None;
                }
            }

            // Memory operations
            RuleOp::Memorize => {
                memory = result.clone();
            }
            RuleOp::AppendMemory => {
                // hashcat rejects when memory is empty (rp_cpu.c:1523)
                if memory.is_empty() {
                    return None;
                }
                // hashcat rejects on overflow (rp_cpu.c:1525) — not silent skip
                if result.len() + memory.len() > MAX_PW {
                    return None;
                }
                result.extend_from_slice(&memory);
            }
            RuleOp::PrependMemory => {
                // hashcat rejects when memory is empty (rp_cpu.c:1531)
                if memory.is_empty() {
                    return None;
                }
                // hashcat rejects on overflow (rp_cpu.c:1532) — not silent skip
                if result.len() + memory.len() > MAX_PW {
                    return None;
                }
                let mut new = memory.clone();
                new.extend_from_slice(&result);
                result = new;
            }
            RuleOp::ExtractMem(start, len, m) => {
                // hashcat rejects when memory is empty — matching that behavior
                if memory.is_empty() {
                    return None;
                }
                // hashcat rejects when start + len > memory.len() (over-range)
                if *start + *len > memory.len() {
                    return None;
                }
                // hashcat rejects when m > result.len() (insert position out of range)
                if *m > result.len() {
                    return None;
                }
                let extracted = &memory[*start..*start + *len];
                if extracted.is_empty() {
                    return None;
                }
                // Insert extracted bytes at position m in the current result
                let mut new = Vec::with_capacity(result.len() + extracted.len());
                new.extend_from_slice(&result[..*m]);
                new.extend_from_slice(extracted);
                new.extend_from_slice(&result[*m..]);
                // hashcat rejects on overflow (arr_len + arr2_cpy > RP_PASSWORD_SIZE)
                // instead of silently skipping — matching for exact parity
                if new.len() > MAX_PW {
                    return None;
                }
                result = new;
            }

            // ── Class-based operations (~ prefix) ──

            // ~s?CY — replace all chars in class C with char Y
            RuleOp::ReplaceClass(class, newc) => {
                for b in &mut result {
                    if char_in_class(*b, *class) {
                        *b = *newc;
                    }
                }
            }
            // ~@?C — purge all chars in class C
            RuleOp::PurgeClass(class) => {
                result.retain(|b| !char_in_class(*b, *class));
            }
            // ~e?C — title case using class chars as separators
            // For l/u/h/H: the separator char itself is lowercased and serves as
            // boundary marker (upper_next = class_match on that char).
            // For d/s: separator chars are skipped (continue), then uppercase/lowercase.
            // For a: class_alpha as separator.
            // hashcat also unconditionally uppercases position 0 after the loop for d/s.
            RuleOp::TitleSepClass(class) => {
                match *class {
                    b'l' | b'u' | b'h' | b'H' => {
                        let class_fn = |c: u8| -> bool { char_in_class(c, *class) };
                        let mut upper_next = true;
                        for b in &mut result {
                            let upper = upper_next;
                            upper_next = class_fn(*b);
                            // lowercase first, then uppercase if needed
                            if b.is_ascii_uppercase() {
                                *b += 32;
                            }
                            if upper && b.is_ascii_lowercase() {
                                *b -= 32;
                            }
                        }
                    }
                    b'd' | b's' | b'a' => {
                        let class_fn = |c: u8| -> bool { char_in_class(c, *class) };
                        let mut upper_next = true;
                        for b in &mut result {
                            if class_fn(*b) {
                                upper_next = true;
                                continue;
                            }
                            if upper_next {
                                upper_next = false;
                                if b.is_ascii_lowercase() {
                                    *b -= 32;
                                }
                            } else if b.is_ascii_uppercase() {
                                *b += 32;
                            }
                        }
                        // hashcat: MANGLE_UPPER_AT(arr, 0) — unconditional uppercase of first char
                        if let Some(first) = result.first_mut() {
                            if first.is_ascii_lowercase() {
                                *first -= 32;
                            }
                        }
                    }
                    _ => {}
                }
            }
            // ~!?C — reject if contains any char in class C
            RuleOp::RejectContainClass(class) => {
                if result.iter().any(|b| char_in_class(*b, *class)) {
                    return None;
                }
            }
            // ~/?C — reject if doesn't contain any char in class C
            RuleOp::RejectNotContainClass(class) => {
                if !result.iter().any(|b| char_in_class(*b, *class)) {
                    return None;
                }
            }
            // ~(?C — reject if first char NOT in class C (keep if IN)
            RuleOp::RejectEqualFirstClass(class) => {
                if result.is_empty() || !char_in_class(result[0], *class) {
                    return None;
                }
            }
            // ~)?C — reject if last char NOT in class C (keep if IN)
            RuleOp::RejectEqualLastClass(class) => {
                if result.is_empty() || !char_in_class(*result.last().unwrap(), *class) {
                    return None;
                }
            }
            // ~=N?C — reject if char at N NOT in class C (keep if IN)
            RuleOp::RejectEqualAtClass(n, class) => {
                if n + 1 > result.len() || !char_in_class(result[*n], *class) {
                    return None;
                }
            }
            // ~%N?C — reject if contains fewer than N chars in class C
            // hashcat: returns true (reject) if cnt < upos, i.e. reject if count < N
            RuleOp::RejectContainsClass(n, class) => {
                if *n + 1 > result.len() {
                    return None;
                }
                let mut count = 0;
                for b in &result {
                    if char_in_class(*b, *class) {
                        count += 1;
                    }
                }
                if count < *n {
                    return None;
                }
            }
        }
    }

    // Minimum length check (8 for WPA, 1 for MD5/other modes)
    if result.len() < min_len || result.len() > MAX_PW {
        return None;
    }

    Some(result)
}

/// Parse a single rule line into a Rule.
/// Returns Err with a description if the rule is malformed.
pub fn parse_rule_line(line: &str) -> Result<Rule, String> {
    let bytes = line.as_bytes();
    let mut ops = Vec::new();
    let mut i = 0;

    while i < bytes.len() {
        let c = bytes[i];

        match c {
            // No-param mangle
            b':' => ops.push(RuleOp::Noop),
            b'l' => ops.push(RuleOp::Lower),
            b'u' => ops.push(RuleOp::Upper),
            b'c' => ops.push(RuleOp::Cap),
            b'C' => ops.push(RuleOp::UpperFirst),
            b't' => ops.push(RuleOp::ToggleAll),
            b'r' => ops.push(RuleOp::Reverse),
            b'd' => ops.push(RuleOp::Duplicate),
            b'f' => ops.push(RuleOp::Reflect),
            b'{' => ops.push(RuleOp::RotLeft),
            b'}' => ops.push(RuleOp::RotRight),
            b'[' => ops.push(RuleOp::DelFirst),
            b']' => ops.push(RuleOp::DelLast),
            b'q' => ops.push(RuleOp::DupeAllChars),
            b'k' => ops.push(RuleOp::SwapFirst),
            b'K' => ops.push(RuleOp::SwapLast),
            b'E' => ops.push(RuleOp::TitleCase),
            b'h' => ops.push(RuleOp::ToHexLower),
            b'H' => ops.push(RuleOp::ToHexUpper),

            // Param mangle — char
            b'$' | b'^' | b'@' => {
                if i + 1 >= bytes.len() {
                    return Err(format!("'{c}' at end of rule needs a char parameter"));
                }
                let ch = bytes[i + 1];
                match c {
                    b'$' => ops.push(RuleOp::Append(ch)),
                    b'^' => ops.push(RuleOp::Prepend(ch)),
                    b'@' => ops.push(RuleOp::Purge(ch)),
                    _ => unreachable!(),
                }
                i += 1;
            }

            // Param mangle — position (single digit)
            b'T' | b'D' | b'\'' | b'+' | b'-' | b'L' | b'R' => {
                if i + 1 >= bytes.len() {
                    return Err(format!("'{c}' needs a position parameter"));
                }
                let n = parse_pos(bytes[i + 1])?;
                match c {
                    b'T' => ops.push(RuleOp::ToggleAt(n)),
                    b'D' => ops.push(RuleOp::DelAt(n)),
                    b'\'' => ops.push(RuleOp::TruncAt(n)),
                    b'+' => ops.push(RuleOp::IncAt(n)),
                    b'-' => ops.push(RuleOp::DecAt(n)),
                    b'L' => ops.push(RuleOp::ShiftLeftAt(n)),
                    b'R' => ops.push(RuleOp::ShiftRightAt(n)),
                    _ => unreachable!(),
                }
                i += 1;
            }

            // Two-param mangle — sAB (replace)
            b's' => {
                if i + 2 >= bytes.len() {
                    return Err("'s' needs two char parameters".to_string());
                }
                ops.push(RuleOp::Replace(bytes[i + 1], bytes[i + 2]));
                i += 2;
            }

            // Position + char — iNX, oNX, =NX
            b'i' | b'o' => {
                if i + 2 >= bytes.len() {
                    return Err(format!("'{c}' needs position + char parameters"));
                }
                let n = parse_pos(bytes[i + 1])?;
                let ch = bytes[i + 2];
                match c {
                    b'i' => ops.push(RuleOp::Insert(n, ch)),
                    b'o' => ops.push(RuleOp::Overstrike(n, ch)),
                    _ => unreachable!(),
                }
                i += 2;
            }

            // Position + count — zN, ZN, pN, yN, YN
            b'z' | b'Z' | b'p' | b'y' | b'Y' => {
                if i + 1 >= bytes.len() {
                    return Err(format!("'{c}' needs a count parameter"));
                }
                let n = parse_pos(bytes[i + 1])?;
                match c {
                    b'z' => ops.push(RuleOp::DupeFirstN(n)),
                    b'Z' => ops.push(RuleOp::DupeLastN(n)),
                    b'p' => ops.push(RuleOp::DupeWord(n)),
                    b'y' => ops.push(RuleOp::DupFirstChars(n)),
                    b'Y' => ops.push(RuleOp::DupLastChars(n)),
                    _ => unreachable!(),
                }
                i += 1;
            }

            // Two positions — *NM
            b'*' => {
                if i + 2 >= bytes.len() {
                    return Err("'*' needs two position parameters".to_string());
                }
                let a = parse_pos(bytes[i + 1])?;
                let b = parse_pos(bytes[i + 2])?;
                ops.push(RuleOp::Swap(a, b));
                i += 2;
            }

            // Extract range — xNM
            b'x' => {
                if i + 2 >= bytes.len() {
                    return Err("'x' needs two position parameters".to_string());
                }
                let start = parse_pos(bytes[i + 1])?;
                let len = parse_pos(bytes[i + 2])?;
                ops.push(RuleOp::Extract(start, len));
                i += 2;
            }

            // Omit range — ONM
            b'O' => {
                if i + 2 >= bytes.len() {
                    return Err("'O' needs two position parameters".to_string());
                }
                let start = parse_pos(bytes[i + 1])?;
                let len = parse_pos(bytes[i + 2])?;
                ops.push(RuleOp::Omit(start, len));
                i += 2;
            }

            // Title with separator — eX
            b'e' => {
                if i + 1 >= bytes.len() {
                    return Err("'e' needs a separator char".to_string());
                }
                ops.push(RuleOp::TitleSep(bytes[i + 1]));
                i += 1;
            }

            // Replace adjacent — .N ,N
            b'.' | b',' => {
                if i + 1 >= bytes.len() {
                    return Err(format!("'{c}' needs a position parameter"));
                }
                let n = parse_pos(bytes[i + 1])?;
                match c {
                    b'.' => ops.push(RuleOp::ReplaceNext(n)),
                    b',' => ops.push(RuleOp::ReplacePrev(n)),
                    _ => unreachable!(),
                }
                i += 1;
            }

            // Insert at every Nth — vNX
            b'v' => {
                if i + 2 >= bytes.len() {
                    return Err("'v' needs position + char parameters".to_string());
                }
                let n = parse_pos(bytes[i + 1])?;
                let ch = bytes[i + 2];
                ops.push(RuleOp::InsertEvery(n, ch));
                i += 2;
            }

            // Reject — position
            b'<' | b'>' | b'_' => {
                if i + 1 >= bytes.len() {
                    return Err(format!("'{c}' needs a length parameter"));
                }
                let n = parse_pos(bytes[i + 1])?;
                match c {
                    b'<' => ops.push(RuleOp::RejectLess(n)),
                    b'>' => ops.push(RuleOp::RejectGreater(n)),
                    b'_' => ops.push(RuleOp::RejectEqual(n)),
                    _ => unreachable!(),
                }
                i += 1;
            }

            // Reject — char
            b'!' | b'/' => {
                if i + 1 >= bytes.len() {
                    return Err(format!("'{c}' needs a char parameter"));
                }
                let ch = bytes[i + 1];
                match c {
                    b'!' => ops.push(RuleOp::RejectContains(ch)),
                    b'/' => ops.push(RuleOp::RejectNotContains(ch)),
                    _ => unreachable!(),
                }
                i += 1;
            }

            // Reject — first/last char
            b'(' | b')' => {
                if i + 1 >= bytes.len() {
                    return Err(format!("'{c}' needs a char parameter"));
                }
                let ch = bytes[i + 1];
                match c {
                    b'(' => ops.push(RuleOp::RejectFirst(ch)),
                    b')' => ops.push(RuleOp::RejectLast(ch)),
                    _ => unreachable!(),
                }
                i += 1;
            }

            // Reject — position + char =NX
            b'=' => {
                if i + 2 >= bytes.len() {
                    return Err("'=' needs position + char parameters".to_string());
                }
                let n = parse_pos(bytes[i + 1])?;
                let ch = bytes[i + 2];
                ops.push(RuleOp::RejectAt(n, ch));
                i += 2;
            }

            // Reject — count + char %NX
            b'%' => {
                if i + 2 >= bytes.len() {
                    return Err("'%' needs count + char parameters".to_string());
                }
                let n = parse_pos(bytes[i + 1])?;
                let ch = bytes[i + 2];
                ops.push(RuleOp::RejectContainsN(n, ch));
                i += 2;
            }

            // Reject memory
            b'Q' => ops.push(RuleOp::RejectMemory),

            // Memory ops
            b'M' => ops.push(RuleOp::Memorize),
            b'4' => ops.push(RuleOp::AppendMemory),
            b'6' => ops.push(RuleOp::PrependMemory),

            // Extract from memory — XNML
            b'X' => {
                if i + 3 >= bytes.len() {
                    return Err("'X' needs three parameters (pos, len, M)".to_string());
                }
                let start = parse_pos(bytes[i + 1])?;
                let len = parse_pos(bytes[i + 2])?;
                let m = parse_pos(bytes[i + 3])?;
                ops.push(RuleOp::ExtractMem(start, len, m));
                i += 3;
            }

            // hashcat: bare space is an explicit noop (rp_cpu.c:1205)
            b' ' => {}

            // Class-based operations — ~ prefix (hashcat RULE_OP_CLASS_BASED)
            // ~s?CY  replace all chars in class C with char Y
            // ~@?C   purge all chars in class C
            // ~e?C   title case with class-based separator
            // ~!?C   reject if contains any char in class C
            // ~/?C   reject if doesn't contain any char in class C
            // ~(?C   reject if first char NOT in class C (keep if IN)
            // ~)?C   reject if last char NOT in class C (keep if IN)
            // ~=N?C  reject if char at N NOT in class C (keep if IN)
            // ~%N?C  reject if contains fewer than N chars in class C
            b'~' => {
                if i + 1 >= bytes.len() {
                    return Err("'~' needs a sub-operation selector".to_string());
                }
                let sub = bytes[i + 1];
                match sub {
                    // ~s?CY — replace class
                    b's' => {
                        // i=~, i+1=s, i+2=?, i+3=class, i+4=Y
                        if i + 4 >= bytes.len() {
                            return Err("'~s' needs ?C and a replacement char".to_string());
                        }
                        if bytes[i + 2] != b'?' {
                            return Err("'~s' must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 3];
                        let newc = bytes[i + 4];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::ReplaceClass(class, newc));
                        i += 4;
                    }
                    // ~@?C — purge class
                    b'@' => {
                        if i + 3 >= bytes.len() {
                            return Err("'~@' needs ?C".to_string());
                        }
                        if bytes[i + 2] != b'?' {
                            return Err("'~@' must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 3];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::PurgeClass(class));
                        i += 3;
                    }
                    // ~e?C — title sep class
                    b'e' => {
                        if i + 3 >= bytes.len() {
                            return Err("'~e' needs ?C".to_string());
                        }
                        if bytes[i + 2] != b'?' {
                            return Err("'~e' must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 3];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::TitleSepClass(class));
                        i += 3;
                    }
                    // ~!?C — reject contain class
                    b'!' => {
                        if i + 3 >= bytes.len() {
                            return Err("'~!' needs ?C".to_string());
                        }
                        if bytes[i + 2] != b'?' {
                            return Err("'~!' must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 3];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::RejectContainClass(class));
                        i += 3;
                    }
                    // ~/?C — reject not contain class
                    b'/' => {
                        if i + 3 >= bytes.len() {
                            return Err("'~/' needs ?C".to_string());
                        }
                        if bytes[i + 2] != b'?' {
                            return Err("'~/' must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 3];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::RejectNotContainClass(class));
                        i += 3;
                    }
                    // ~(?C — reject equal first class
                    b'(' => {
                        if i + 3 >= bytes.len() {
                            return Err("'~(' needs ?C".to_string());
                        }
                        if bytes[i + 2] != b'?' {
                            return Err("'~(' must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 3];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::RejectEqualFirstClass(class));
                        i += 3;
                    }
                    // ~)?C — reject equal last class
                    b')' => {
                        if i + 3 >= bytes.len() {
                            return Err("'~)' needs ?C".to_string());
                        }
                        if bytes[i + 2] != b'?' {
                            return Err("'~)' must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 3];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::RejectEqualLastClass(class));
                        i += 3;
                    }
                    // ~=N?C — reject equal at class
                    b'=' => {
                        // i=~, i+1==, i+2=N, i+3=?, i+4=class
                        if i + 4 >= bytes.len() {
                            return Err("'~=' needs N?C".to_string());
                        }
                        let n = parse_pos(bytes[i + 2])?;
                        if bytes[i + 3] != b'?' {
                            return Err("'~=' position must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 4];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::RejectEqualAtClass(n, class));
                        i += 4;
                    }
                    // ~%N?C — reject contains class
                    b'%' => {
                        // i=~, i+1=%, i+2=N, i+3=?, i+4=class
                        if i + 4 >= bytes.len() {
                            return Err("'~%' needs N?C".to_string());
                        }
                        let n = parse_pos(bytes[i + 2])?;
                        if bytes[i + 3] != b'?' {
                            return Err("'~%' count must be followed by '?'".to_string());
                        }
                        let class = bytes[i + 4];
                        if !is_valid_class(class) {
                            return Err(format!("invalid class char '{class}' (0x{class:02x})"));
                        }
                        ops.push(RuleOp::RejectContainsClass(n, class));
                        i += 4;
                    }
                    _ => {
                        return Err(format!("unknown '~' sub-operation '{sub}' (0x{sub:02x})"));
                    }
                }
            }

            // Unknown operation
            _ => {
                return Err(format!("unknown rule operation '{c}' (0x{c:02x})"));
            }
        }
        i += 1;
    }

    Ok(Rule { ops })
}

/// Parse a position character (0-9, A-Z → 0-35).
/// `p` is hashcat's saved-position token, rejected by hashcat's conv_ctoi
/// (returns -1). We reject it too — it's only used in random rule generation,
/// not in user-supplied rule files.
fn parse_pos(c: u8) -> Result<usize, String> {
    match c {
        b'0'..=b'9' => Ok((c - b'0') as usize),
        b'A'..=b'Z' => Ok((c - b'A' + 10) as usize),
        b'p' => Err("position 'p' (saved-position token) is not supported — hashcat rejects it in user rules".into()), // hashcat parity: conv_ctoi returns -1 for 'p'
        _ => Err(format!("invalid position character '{c}' (0x{c:02x})")),
    }
}

/// Check if a byte is a valid class character (l/u/d/h/H/s/a).
fn is_valid_class(c: u8) -> bool {
    matches!(c, b'l' | b'u' | b'd' | b'h' | b'H' | b's' | b'a')
}

/// Load rules from a file. Each non-empty, non-comment line is a rule.
pub fn load_rules_file(path: &str) -> Result<Vec<Rule>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read rules file '{path}': {e}"))?;

    let mut rules = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        match parse_rule_line(trimmed) {
            Ok(rule) => rules.push(rule),
            Err(e) => {
                eprintln!("[!] Warning: skipping malformed rule at line {line_num}: {e}");
            }
        }
    }

    if rules.is_empty() {
        return Err(format!("No valid rules found in '{path}'"));
    }

    Ok(rules)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noop() {
        let rule = parse_rule_line(":").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"password");
    }

    #[test]
    fn test_lowercase() {
        let rule = parse_rule_line("l").unwrap();
        let result = apply_rule(&rule, b"PASSWORD", 8).unwrap();
        assert_eq!(result, b"password");
    }

    #[test]
    fn test_uppercase() {
        let rule = parse_rule_line("u").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"PASSWORD");
    }

    #[test]
    fn test_capitalize() {
        let rule = parse_rule_line("c").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"Password");
    }

    #[test]
    fn test_reverse() {
        let rule = parse_rule_line("r").unwrap();
        let result = apply_rule(&rule, b"abcdefgh", 8).unwrap();
        assert_eq!(result, b"hgfedcba");
    }

    #[test]
    fn test_append_char() {
        let rule = parse_rule_line("$1").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"password1");
    }

    #[test]
    fn test_prepend_char() {
        let rule = parse_rule_line("^1").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"1password");
    }

    #[test]
    fn test_replace() {
        let rule = parse_rule_line("sab").unwrap();
        let result = apply_rule(&rule, b"banana123", 8).unwrap();
        assert_eq!(result, b"bbnbnb123");
    }

    #[test]
    fn test_reject_length() {
        // hashcat >5 (REJECT_GREATER): reject if len < 5 → keep if len >= 5
        // "password" (8 chars) is >= 5, so it's KEPT
        let rule = parse_rule_line(">5").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_some());
        // "pass" (4 chars) is < 5, so it's REJECTED (but also fails min-8)
        assert!(apply_rule(&rule, b"pass", 8).is_none());
    }

    #[test]
    fn test_reject_less() {
        // hashcat <5 (REJECT_LESS): reject if len > 5 → keep if len <= 5
        // "password" (8) > 5 → rejected
        let rule = parse_rule_line("<5").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_none());
        // "pass" (4) <= 5 → kept, but fails min-8
        assert!(apply_rule(&rule, b"pass", 8).is_none());
    }

    #[test]
    fn test_reject_equal() {
        // hashcat _8 (REJECT_EQUAL): reject if len != 8 → keep if len == 8
        let rule = parse_rule_line("_8").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_some()); // 8 chars → kept
        assert!(apply_rule(&rule, b"pass1234", 8).is_some()); // 8 chars → kept
        assert!(apply_rule(&rule, b"pass123", 8).is_none()); // 7 chars → rejected
    }

    #[test]
    fn test_reject_first() {
        // hashcat (p (REJECT_EQUAL_FIRST): reject if first != 'p' → keep if first == 'p'
        let rule = parse_rule_line("(p").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_some()); // first is 'p' → kept
        assert!(apply_rule(&rule, b"assword1", 8).is_none()); // first is 'a' → rejected
    }

    #[test]
    fn test_reject_last() {
        // hashcat )d (REJECT_EQUAL_LAST): reject if last != 'd' → keep if last == 'd'
        let rule = parse_rule_line(")d").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_some()); // last is 'd' → kept
        assert!(apply_rule(&rule, b"passwor1", 8).is_none()); // last is '1' → rejected
    }

    #[test]
    fn test_replace_next() {
        // hashcat .1 (REPLACE_NP1): replace char at 1 with char at 2
        // "abcdefgh" → "accefgh"
        let rule = parse_rule_line(".1").unwrap();
        let result = apply_rule(&rule, b"abcdefgh", 8).unwrap();
        assert_eq!(result, b"accdefgh");
    }

    #[test]
    fn test_replace_prev() {
        // hashcat ,2 (REPLACE_NM1): replace char at 2 with char at 1
        // "abcdefgh" → "abbefgh"
        let rule = parse_rule_line(",2").unwrap();
        let result = apply_rule(&rule, b"abcdefgh", 8).unwrap();
        assert_eq!(result, b"abbdefgh");
    }

    #[test]
    fn test_extract_out_of_range() {
        // hashcat x84 on "password" (8): start=8 >= len=8 → noop, keep original
        let rule = parse_rule_line("x84").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"password");
    }

    #[test]
    fn test_extract_overflow() {
        // hashcat x38 on "abcdefghij" (10): start=3, len=8, 3+8=11 > 10 → noop
        let rule = parse_rule_line("x38").unwrap();
        let result = apply_rule(&rule, b"abcdefghij", 8).unwrap();
        assert_eq!(result, b"abcdefghij");
    }

    #[test]
    fn test_extract_in_range() {
        // hashcat x02 on "password" (8): start=0, len=2 → "pa" → fails min-8
        let rule = parse_rule_line("x02").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_none());
    }

    #[test]
    fn test_extract_in_range_positive() {
        // hashcat x08 on "passwordpassword" (16): start=0, len=8 → "password" (8 chars)
        let rule = parse_rule_line("x08").unwrap();
        let result = apply_rule(&rule, b"passwordpassword", 8).unwrap();
        assert_eq!(result, b"password");
    }

    #[test]
    fn test_omit_in_range_positive() {
        // hashcat O08 on "passwordpassword" (16): start=0, len=8 → "password" (8 chars)
        let rule = parse_rule_line("O08").unwrap();
        let result = apply_rule(&rule, b"passwordpassword", 8).unwrap();
        assert_eq!(result, b"password");
    }

    #[test]
    fn test_omit_out_of_range() {
        // hashcat O84 on "password" (8): start=8 >= len=8 → noop, keep original
        let rule = parse_rule_line("O84").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"password");
    }

    #[test]
    fn test_omit_overflow() {
        // hashcat O38 on "password" (8): start=3, len=8, 3+8=11 > 8 → noop
        let rule = parse_rule_line("O38").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"password");
    }

    #[test]
    fn test_duplicate() {
        let rule = parse_rule_line("d").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"passwordpassword");
    }

    #[test]
    fn test_dupe_first_chars() {
        // y2: prepend first 2 chars (hashcat mangle_dupeblock_prepend)
        let rule = parse_rule_line("y2").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"papassword");
    }

    #[test]
    fn test_dupe_first_n() {
        // z3: insert 3 copies of first char at position 0 (hashcat mangle_dupechar_at)
        let rule = parse_rule_line("z3").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"ppppassword");
    }

    #[test]
    fn test_delete_first() {
        let rule = parse_rule_line("[").unwrap();
        let result = apply_rule(&rule, b"password1", 8).unwrap();
        assert_eq!(result, b"assword1");
    }

    #[test]
    fn test_toggle_at() {
        let rule = parse_rule_line("T0").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"Password");
    }

    #[test]
    fn test_multiple_ops() {
        // Capitalize then append "123"
        let rule = parse_rule_line("c$1$2$3").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"Password123");
    }

    #[test]
    fn test_dedupe() {
        let rule = parse_rule_line("q").unwrap();
        let result = apply_rule(&rule, b"abcdefgh", 8).unwrap();
        assert_eq!(result, b"aabbccddeeffgghh");
    }

    #[test]
    fn test_purge() {
        let rule = parse_rule_line("@a").unwrap();
        let result = apply_rule(&rule, b"xaxaxaxbxbxbc", 8).unwrap();
        assert_eq!(result, b"xxxxbxbxbc");
    }

    #[test]
    fn test_reflect() {
        let rule = parse_rule_line("f").unwrap();
        let result = apply_rule(&rule, b"abcdef", 8).unwrap();
        assert_eq!(result, b"abcdeffedcba");
    }

    #[test]
    fn test_insert_at() {
        let rule = parse_rule_line("i3X").unwrap();
        let result = apply_rule(&rule, b"password", 8).unwrap();
        assert_eq!(result, b"pasXsword");
    }

    #[test]
    fn test_truncate_at() {
        let rule = parse_rule_line("'4").unwrap();
        // Truncated to 4 chars → fails WPA min-8 check, returns None
        let result = apply_rule(&rule, b"password", 8);
        assert!(result.is_none());
    }

    #[test]
    fn test_swap_positions() {
        let rule = parse_rule_line("*02").unwrap();
        let result = apply_rule(&rule, b"abcdefgh", 8).unwrap();
        assert_eq!(result, b"cbadefgh");
    }

    #[test]
    fn test_upper_first() {
        // hashcat C = UREST_LFIRST: uppercase all, then lowercase first char
        let rule = parse_rule_line("C").unwrap();
        let result = apply_rule(&rule, b"password1", 8).unwrap();
        assert_eq!(result, b"pASSWORD1");
    }

    #[test]
    fn test_cap_vs_upper_first() {
        // c = lower rest, upper first; C = upper rest, lower first (inverse)
        let cap = parse_rule_line("c").unwrap();
        let uf = parse_rule_line("C").unwrap();
        assert_eq!(apply_rule(&cap, b"pASSWORD1", 8).unwrap(), b"Password1");
        assert_eq!(apply_rule(&uf, b"password1", 8).unwrap(), b"pASSWORD1");
    }

    #[test]
    fn test_title_sep() {
        // e_ = title case with _ as separator only
        let rule = parse_rule_line("e_").unwrap();
        // "a.b.cdef" → "A.b.cdef" (only _ is a separator, . is not)
        let result = apply_rule(&rule, b"a.b.cdef", 8).unwrap();
        assert_eq!(result, b"A.b.cdef");
    }

    #[test]
    fn test_title_sep_alpha_at_pos0() {
        // ea = title case with 'a' as separator — word starts with 'a'
        // hashcat uppercases pos 0 after loop: "axb" → "AXb"
        let rule = parse_rule_line("ea").unwrap();
        let result = apply_rule(&rule, b"axb12345", 8).unwrap();
        assert_eq!(result, b"AXb12345");
    }

    #[test]
    fn test_title_case_uppercase_input() {
        // E on "HELLO WORLD" → "Hello World" (space is only separator)
        let rule = parse_rule_line("E").unwrap();
        let result = apply_rule(&rule, b"HELLO WORLD", 8).unwrap();
        assert_eq!(result, b"Hello World");
    }

    #[test]
    fn test_title_case_non_alpha_not_separator() {
        // E on "hello-world" → "Hello-world" (hyphen is NOT a separator)
        let rule = parse_rule_line("E").unwrap();
        let result = apply_rule(&rule, b"hello-world", 8).unwrap();
        assert_eq!(result, b"Hello-world");
    }

    #[test]
    fn test_extract_mem() {
        // X011 with empty memory → hashcat rejects (returns None)
        let extract = parse_rule_line("X011").unwrap();
        assert!(apply_rule(&extract, b"testword", 8).is_none());
    }

    #[test]
    fn test_append_memory_empty() {
        // 4 (AppendMemory) with no prior M → hashcat rejects
        let rule = parse_rule_line("4").unwrap();
        assert!(apply_rule(&rule, b"testword", 8).is_none());
    }

    #[test]
    fn test_prepend_memory_empty() {
        // 6 (PrependMemory) with no prior M → hashcat rejects
        let rule = parse_rule_line("6").unwrap();
        assert!(apply_rule(&rule, b"testword", 8).is_none());
    }

    #[test]
    fn test_space_noop() {
        // hashcat: bare space in rule is a noop (rp_cpu.c:1205)
        // "l u" = lower, space(noop), upper → should work
        let rule = parse_rule_line("l u").unwrap();
        let result = apply_rule(&rule, b"Password", 8).unwrap();
        assert_eq!(result, b"PASSWORD");
    }

    #[test]
    fn test_reject_contains_n_early_reject() {
        // hashcat %8a on "aaaaaaaa" (8): n=8, n+1=9 > 8 → reject (rp_cpu.c:1599)
        let rule = parse_rule_line("%8a").unwrap();
        assert!(apply_rule(&rule, b"aaaaaaaa", 8).is_none());
    }

    #[test]
    fn test_append_memory_overflow_reject() {
        // 4 with memory that would overflow MAX_PW → hashcat rejects (rp_cpu.c:1525)
        // M on 63-char word, then 4 → 63+63=126 > MAX_PW → reject
        let long = b"a".repeat(63);
        let rule = parse_rule_line("M4").unwrap();
        assert!(apply_rule(&rule, &long, 8).is_none()); // 63+63 > 63 → reject
    }

    #[test]
    fn test_prepend_memory_overflow_reject() {
        // 6 with memory that would overflow MAX_PW → hashcat rejects (rp_cpu.c:1532)
        // M on 63-char word, then 6 → 63+63=126 > MAX_PW → reject
        let long = b"a".repeat(63);
        let rule = parse_rule_line("M6").unwrap();
        assert!(apply_rule(&rule, &long, 8).is_none()); // 63+63 > 63 → reject
    }

    #[test]
    fn test_p_position_token() {
        // p as position token is now rejected (hashcat parity: conv_ctoi returns -1)
        // Tp = toggle at saved position — should fail to parse
        let result = parse_rule_line("Tp");
        assert!(
            result.is_err(),
            "Tp should fail to parse — 'p' position token not supported"
        );
    }

    #[test]
    fn test_extract_mem_multi_op() {
        // MX012: M stores word in memory, X012 extracts 1 char at pos 0,
        // inserts at pos 2. "testword" → "tetstword"
        let rule = parse_rule_line("MX012").unwrap();
        let result = apply_rule(&rule, b"testword", 8).unwrap();
        assert_eq!(result, b"tetstword");
    }

    #[test]
    fn test_extract_mem_over_range() {
        // X362 on "testword" (8 bytes): start=3, len=6 → 3+6=9 > 8 → hashcat rejects
        let rule = parse_rule_line("X362").unwrap();
        assert!(apply_rule(&rule, b"testword", 8).is_none());
    }

    // ── Class-based operation tests (~ prefix) ──

    #[test]
    fn test_replace_class_lower() {
        // ~s?lX: replace all lowercase letters with 'X'
        // "aBcDeFgh" → all lowercase (a,c,e,g,h) → "XBXDXFXX"
        let rule = parse_rule_line("~s?lX").unwrap();
        let result = apply_rule(&rule, b"aBcDeFgh", 8).unwrap();
        assert_eq!(result, b"XBXDXFXX");
    }

    #[test]
    fn test_replace_class_upper() {
        // ~s?uX: replace all uppercase letters with 'X'
        // "aBcDeFgh" → all uppercase (B,D,F) → "aXcXeXgh"
        let rule = parse_rule_line("~s?uX").unwrap();
        let result = apply_rule(&rule, b"aBcDeFgh", 8).unwrap();
        assert_eq!(result, b"aXcXeXgh");
    }

    #[test]
    fn test_replace_class_digit() {
        // ~s?dX: replace all digits with 'X'
        let rule = parse_rule_line("~s?dX").unwrap();
        let result = apply_rule(&rule, b"abc12345", 8).unwrap();
        assert_eq!(result, b"abcXXXXX");
    }

    #[test]
    fn test_replace_class_lower_hex() {
        // ~s?hX: replace all lower-hex chars (0-9a-f) with 'X'
        let rule = parse_rule_line("~s?hX").unwrap();
        let result = apply_rule(&rule, b"abcdefGH", 8).unwrap();
        assert_eq!(result, b"XXXXXXGH");
    }

    #[test]
    fn test_replace_class_upper_hex() {
        // ~s?HX: replace all upper-hex chars (0-9A-F) with 'X'
        let rule = parse_rule_line("~s?HX").unwrap();
        let result = apply_rule(&rule, b"ABCDEFgh", 8).unwrap();
        assert_eq!(result, b"XXXXXXgh");
    }

    #[test]
    fn test_replace_class_sym() {
        // ~s?sX: replace all symbols with 'X'
        // "a!b@c!d!e" (10 chars) → "aXbXcXdXe"
        let rule = parse_rule_line("~s?sX").unwrap();
        let result = apply_rule(&rule, b"a!b@c!d!e", 8).unwrap();
        assert_eq!(result, b"aXbXcXdXe");
    }

    #[test]
    fn test_replace_class_alpha() {
        // ~s?aX: replace all alpha chars with 'X'
        // "Ab12345cD" (9 chars) → "XX12345XX"
        // NOTE: class 'a' is a vega-crack extension — hashcat noops for 'a'
        // in class-based ops (only supports l/u/d/h/H/s).
        let rule = parse_rule_line("~s?aX").unwrap();
        let result = apply_rule(&rule, b"Ab12345cD", 8).unwrap();
        assert_eq!(result, b"XX12345XX");
    }

    #[test]
    fn test_purge_class_lower() {
        // ~@?l: purge all lowercase letters
        // "aBcDeFgHiJ" (10 chars) → "BDFHJ" (5 chars, fails min-8) → None
        let rule = parse_rule_line("~@?l").unwrap();
        assert!(apply_rule(&rule, b"aBcDeFgHiJ", 8).is_none());
        // "aBcDeFgHijklmnop" → purge lower → "BDFH" (4 chars) → None
        assert!(apply_rule(&rule, b"aBcDeFgHijklmnop", 8).is_none());
    }

    #[test]
    fn test_purge_class_digit() {
        // ~@?d: purge all digits
        // "abc12345" → "abc" (3 chars) → fails min-8 → None
        let rule = parse_rule_line("~@?d").unwrap();
        assert!(apply_rule(&rule, b"abc12345", 8).is_none());
        // "abcdefgh12345678" → purge digits → "abcdefgh" (8 chars) → kept
        let result = apply_rule(&rule, b"abcdefgh12345678", 8).unwrap();
        assert_eq!(result, b"abcdefgh");
    }

    #[test]
    fn test_purge_class_sym() {
        // ~@?s: purge all symbols
        // "a!b@c!d!e!f!g!h" (16 chars) → "abcdefgh" (8 chars) → kept
        let rule = parse_rule_line("~@?s").unwrap();
        let result = apply_rule(&rule, b"a!b@c!d!e!f!g!h", 8).unwrap();
        assert_eq!(result, b"abcdefgh");
    }

    #[test]
    fn test_purge_class_alpha() {
        // ~@?a: purge all alpha chars
        // "Ab123456cD" (10 chars) → "123456" (6 chars) → fails min-8 → None
        let rule = parse_rule_line("~@?a").unwrap();
        assert!(apply_rule(&rule, b"Ab123456cD", 8).is_none());
        // "Abcd12345678efgh" (16 chars) → purge alpha → "12345678" (8 chars) → kept
        let result = apply_rule(&rule, b"Abcd12345678efgh", 8).unwrap();
        assert_eq!(result, b"12345678");
    }

    #[test]
    fn test_title_sep_class_lower() {
        // ~e?l: title case with lowercase letters as separators
        // hashcat title_sep_class_l: upper_next=1, for each pos:
        //   upper=upper_next; upper_next=class_lower(arr[pos]); lower_at(pos); if upper, upper_at(pos)
        // "ab123456" → pos0: upper=1, upper_next=class_lower('a')=1, lower→'a', upper→'A'
        //              pos1: upper=1, upper_next=class_lower('b')=1, lower→'b', upper→'B'
        //              pos2: upper=1, upper_next=class_lower('1')=0, lower→'1', upper→'1' (noop)
        //              pos3: upper=0, upper_next=class_lower('2')=0, lower→'2'
        // Result: "AB123456"
        let rule = parse_rule_line("~e?l").unwrap();
        let result = apply_rule(&rule, b"ab123456", 8).unwrap();
        assert_eq!(result, b"AB123456");
    }

    #[test]
    fn test_title_sep_class_digit() {
        // ~e?d: title case with digits as separators
        // hashcat title_sep_class_d: digits are separators (skip+set upper_next=1)
        // "ab1cd1ef" → A b 1 C d 1 E f
        let rule = parse_rule_line("~e?d").unwrap();
        let result = apply_rule(&rule, b"ab1cd1ef", 8).unwrap();
        assert_eq!(result, b"Ab1Cd1Ef");
    }

    #[test]
    fn test_title_sep_class_sym() {
        // ~e?s: title case with symbols as separators
        // hashcat title_sep_class_s: symbols are separators (skip+set upper_next=1)
        // "ab.cd1ef": 'a'→A, 'b'→b, '.'→sep, 'c'→C, 'd'→d, '1'→1(noop), 'e'→e, 'f'→f
        // Result: "Ab.Cd1ef"
        let rule = parse_rule_line("~e?s").unwrap();
        let result = apply_rule(&rule, b"ab.cd1ef", 8).unwrap();
        assert_eq!(result, b"Ab.Cd1ef");
    }

    #[test]
    fn test_reject_contain_class_digit() {
        // ~!?d: reject if contains any digit
        // "password" has no digits → kept
        let rule = parse_rule_line("~!?d").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_some());
        // "pass1234" has digits → rejected
        assert!(apply_rule(&rule, b"pass1234", 8).is_none());
    }

    #[test]
    fn test_reject_contain_class_upper() {
        // ~!?u: reject if contains any uppercase
        // "password" → no uppercase → kept
        let rule = parse_rule_line("~!?u").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_some());
        // "Password" → has uppercase → rejected
        assert!(apply_rule(&rule, b"Password", 8).is_none());
    }

    #[test]
    fn test_reject_not_contain_class_digit() {
        // ~/?d: reject if doesn't contain any digit
        // "password" → no digits → rejected
        let rule = parse_rule_line("~/?d").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_none());
        // "pass1234" → has digits → kept
        let rule2 = parse_rule_line("~/?d").unwrap();
        assert!(apply_rule(&rule2, b"pass1234", 8).is_some());
    }

    #[test]
    fn test_reject_not_contain_class_lower() {
        // ~/?l: reject if doesn't contain any lowercase
        // "PASSWORD123" → no lowercase → rejected
        let rule = parse_rule_line("~/?l").unwrap();
        assert!(apply_rule(&rule, b"PASSWORD123", 8).is_none());
        // "Password123" → has lowercase → kept
        assert!(apply_rule(&rule, b"Password123", 8).is_some());
    }

    #[test]
    fn test_reject_equal_first_class_lower() {
        // ~(?l: hashcat rejects if first char is NOT lowercase → keep if IS lowercase
        let rule = parse_rule_line("~(?l").unwrap();
        // "password" → first is 'p' (lowercase, IN class) → kept
        assert!(apply_rule(&rule, b"password", 8).is_some());
        // "Password" → first is 'P' (uppercase, NOT in class) → rejected
        assert!(apply_rule(&rule, b"Password", 8).is_none());
        // "12345678" → first is '1' (digit, NOT in class) → rejected
        assert!(apply_rule(&rule, b"12345678", 8).is_none());
    }

    #[test]
    fn test_reject_equal_last_class_digit() {
        // ~)?d: hashcat rejects if last char is NOT digit → keep if IS digit
        let rule = parse_rule_line("~)?d").unwrap();
        // "password1" → last is '1' (digit, IN class) → kept
        assert!(apply_rule(&rule, b"password1", 8).is_some());
        // "password" → last is 'd' (NOT digit) → rejected
        assert!(apply_rule(&rule, b"password", 8).is_none());
    }

    #[test]
    fn test_reject_equal_at_class_lower() {
        // ~=0?l: hashcat rejects if char at pos 0 is NOT lowercase → keep if IS lowercase
        let rule = parse_rule_line("~=0?l").unwrap();
        // "password" → pos0='p' (lower, IN class) → kept
        assert!(apply_rule(&rule, b"password", 8).is_some());
        // "Password" → pos0='P' (upper, NOT in class) → rejected
        assert!(apply_rule(&rule, b"Password", 8).is_none());
    }

    #[test]
    fn test_reject_equal_at_class_digit() {
        // ~=3?d: hashcat rejects if char at pos 3 is NOT digit → keep if IS digit
        let rule = parse_rule_line("~=3?d").unwrap();
        // "abc12345" → pos3='1' (digit, IN class) → kept
        assert!(apply_rule(&rule, b"abc12345", 8).is_some());
        // "password" → pos3='s' (NOT digit) → rejected
        assert!(apply_rule(&rule, b"password", 8).is_none());
    }

    #[test]
    fn test_reject_equal_at_class_out_of_range() {
        // ~=8?d on "password" (8 chars): n=8, n+1=9 > 8 → reject (out of range)
        let rule = parse_rule_line("~=8?d").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_none());
    }

    #[test]
    fn test_reject_contains_class_digit() {
        // ~%3?d: reject if contains fewer than 3 digits → keep if >= 3
        let rule = parse_rule_line("~%3?d").unwrap();
        // "password12" → 2 digits < 3 → rejected
        assert!(apply_rule(&rule, b"password12", 8).is_none());
        // "password123" → 3 digits >= 3 → kept
        assert!(apply_rule(&rule, b"password123", 8).is_some());
        // "password1234" → 4 digits >= 3 → kept
        assert!(apply_rule(&rule, b"password1234", 8).is_some());
    }

    #[test]
    fn test_reject_contains_class_lower() {
        // ~%4?l: reject if contains fewer than 4 lowercase → keep if >= 4
        let rule = parse_rule_line("~%4?l").unwrap();
        // "PASS1234" → 0 lowercase < 4 → rejected
        assert!(apply_rule(&rule, b"PASS1234", 8).is_none());
        // "Password12" → 7 lowercase >= 4 → kept
        assert!(apply_rule(&rule, b"Password12", 8).is_some());
    }

    #[test]
    fn test_reject_contains_class_early_reject() {
        // ~%9?d on "password" (8 chars): n=9, n+1=10 > 8 → reject
        let rule = parse_rule_line("~%9?d").unwrap();
        assert!(apply_rule(&rule, b"password", 8).is_none());
    }

    #[test]
    fn test_class_ops_combined() {
        // ~@?d~s?lX: purge digits, then replace lowercase with 'X'
        // "abc123de12345678" (16 chars) → purge digits → "abcde" (5 chars) → fails min-8 → None
        // Use a word that stays >= 8 after purge
        // "abcdefgh12345678" → purge digits → "abcdefgh" (8) → replace lower → "XXXXXXXX"
        let rule = parse_rule_line("~@?d~s?lX").unwrap();
        let result = apply_rule(&rule, b"abcdefgh12345678", 8).unwrap();
        assert_eq!(result, b"XXXXXXXX");
    }

    #[test]
    fn test_class_parse_invalid_class() {
        // ~s?zX: 'z' is not a valid class char
        assert!(parse_rule_line("~s?zX").is_err());
    }

    #[test]
    fn test_class_parse_missing_char() {
        // ~s?l at end of rule — needs replacement char Y
        assert!(parse_rule_line("~s?l").is_err());
    }

    #[test]
    fn test_class_parse_missing_question() {
        // ~slX without '?' — should fail
        assert!(parse_rule_line("~slX").is_err());
    }

    #[test]
    fn test_class_parse_unknown_subop() {
        // ~z is not a valid sub-op
        assert!(parse_rule_line("~z?l").is_err());
    }
}
