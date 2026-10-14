#!/bin/bash
# Comprehensive test suite for vega-crack
# Tests: crypto correctness, format parsing, edge cases, performance
set -euo pipefail

BINARY="${VEGA_CRACK:-$(dirname "$0")/../target/release/vega-crack}"
RUSTICL_ENABLE=radeonsi
export RUSTICL_ENABLE

PASS=0
FAIL=0
FAILED_TESTS=()

# Test hash: password "password123" with SSID "PACE969"
# Real captured PMKID from prior testing
HASH_22000="WPA*01*2582a8281bf9d4308d6f5731d0e61c61*4604ba734d4e*89acf0e761f4*ed487162465a774bfba60eb603a39f3a***"
# hashcat example hash (password: hashcat!)
HASH_HASHCAT="WPA*01*2582a8281bf9d4308d6f5731d0e61c61*4604ba734d4e*89acf0e761f4*ed487162465a774bfba60eb603a39f3a***"
# Deprecated 16800 format
HASH_16800="2582a8281bf9d4308d6f5731d0e61c61*4604ba734d4e*89acf0e761f4*ed487162465a774bfba60eb603a39f3a"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

run_test() {
    local name="$1"
    local expected_pattern="$2"
    local actual="$3"
    if echo "$actual" | grep -q "$expected_pattern"; then
        echo "  PASS: $name"
        ((PASS++))
    else
        echo "  FAIL: $name"
        echo "    Expected: $expected_pattern"
        echo "    Got: $actual"
        ((FAIL++))
        FAILED_TESTS+=("$name")
    fi
}

run_test_exit() {
    local name="$1"
    local expected_exit="$2"
    local actual_exit="$3"
    if [ "$actual_exit" -eq "$expected_exit" ]; then
        echo "  PASS: $name"
        ((PASS++))
    else
        echo "  FAIL: $name (exit code $actual_exit, expected $expected_exit)"
        ((FAIL++))
        FAILED_TESTS+=("$name")
    fi
}

echo "=== vega-crack Test Suite ==="
echo ""

# --- Section 1: Basic functionality ---
echo "--- Section 1: Basic Functionality ---"

# Test 1: Known password in wordlist
echo "password123" > "$TMPDIR/wl_basic.txt"
echo "wrongpass1" >> "$TMPDIR/wl_basic.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_basic.txt" 2>&1 || true)
echo "$HASH_22000" > "$TMPDIR/hash_22000.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_basic.txt" 2>&1 || true)
run_test "Known password found" "PASSWORD FOUND" "$OUT"
run_test "Password value correct" "password123" "$OUT"

# Test 2: Password not in wordlist
echo "wrongpass1" > "$TMPDIR/wl_notfound.txt"
echo "wrongpass2" >> "$TMPDIR/wl_notfound.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_notfound.txt" 2>&1 || true)
run_test "Password not found" "not found" "$OUT"

# Test 3: Empty wordlist
echo -n "" > "$TMPDIR/wl_empty.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_empty.txt" 2>&1 || true)
run_test "Empty wordlist" "No valid passwords" "$OUT"

# --- Section 2: Password length edge cases ---
echo ""
echo "--- Section 2: Password Length Edge Cases ---"

# Test 4: 8-char minimum (WPA spec)
echo "12345678" > "$TMPDIR/wl_minlen.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_minlen.txt" 2>&1 || true)
# Should process (not found, but no crash)
run_test "8-char password processed" "COMPLETE" "$OUT"

# Test 5: 7-char password (below WPA minimum) — should be skipped
echo "1234567" > "$TMPDIR/wl_tooshort.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_tooshort.txt" 2>&1 || true)
run_test "7-char password skipped" "No valid passwords" "$OUT"

# Test 6: 63-char password (WPA max)
echo "$(python3 -c 'print("A"*63)')" > "$TMPDIR/wl_maxlen.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_maxlen.txt" 2>&1 || true)
run_test "63-char password processed" "COMPLETE" "$OUT"

# --- Section 3: Hash format parsing ---
echo ""
echo "--- Section 3: Hash Format Parsing ---"

# Test 7: 16800 format
echo "$HASH_16800" > "$TMPDIR/hash_16800.txt"
echo "password123" > "$TMPDIR/wl_16800.txt"
OUT=$("$BINARY" "$TMPDIR/hash_16800.txt" "$TMPDIR/wl_16800.txt" 2>&1 || true)
run_test "16800 format detected" "16800" "$OUT"

# Test 8: 16800 format with empty ESSID
echo "2582a8281bf9d4308d6f5731d0e61c61*4604ba734d4e*89acf0e761f4*" > "$TMPDIR/hash_16800_empty.txt"
OUT=$("$BINARY" "$TMPDIR/hash_16800_empty.txt" "$TMPDIR/wl_16800.txt" 2>&1 || true)
run_test "16800 empty ESSID" "16800" "$OUT"

# Test 9: 16800 format with no ESSID field (3 fields)
echo "2582a8281bf9d4308d6f5731d0e61c61*4604ba734d4e*89acf0e761f4" > "$TMPDIR/hash_16800_3field.txt"
OUT=$("$BINARY" "$TMPDIR/hash_16800_3field.txt" "$TMPDIR/wl_16800.txt" 2>&1 || true)
run_test "16800 3-field (no ESSID)" "16800" "$OUT"

# Test 10: EAPOL lines skipped
echo "WPA*02*abcdef*4604ba734d4e*89acf0e761f4*ed487162465a774bfba60eb603a39f3a***" > "$TMPDIR/hash_mixed.txt"
echo "$HASH_22000" >> "$TMPDIR/hash_mixed.txt"
OUT=$("$BINARY" "$TMPDIR/hash_mixed.txt" "$TMPDIR/wl_basic.txt" 2>&1 || true)
run_test "EAPOL skipped warning" "EAPOL" "$OUT"
run_test "PMKID still found after EAPOL" "PASSWORD FOUND" "$OUT"

# --- Section 4: Error handling ---
echo ""
echo "--- Section 4: Error Handling ---"

# Test 11: Missing hash file
OUT=$("$BINARY" "/nonexistent/hash.txt" "$TMPDIR/wl_basic.txt" 2>&1 || true)
run_test "Missing hash file error" "No such file" "$OUT"

# Test 12: Missing wordlist file
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "/nonexistent/wl.txt" 2>&1 || true)
run_test "Missing wordlist error" "not found" "$OUT"

# Test 13: Invalid hex in PMKID
echo "WPA*01*ZZZZ*4604ba734d4e*89acf0e761f4*ed487162465a774bfba60eb603a39f3a***" > "$TMPDIR/hash_badhex.txt"
OUT=$("$BINARY" "$TMPDIR/hash_badhex.txt" "$TMPDIR/wl_basic.txt" 2>&1 || true)
run_test "Invalid hex error" "hex" "$OUT"

# Test 14: No PMKID in hash file
echo "some random text" > "$TMPDIR/hash_nopmkid.txt"
OUT=$("$BINARY" "$TMPDIR/hash_nopmkid.txt" "$TMPDIR/wl_basic.txt" 2>&1 || true)
run_test "No PMKID found error" "No PMKID" "$OUT"

# Test 15: Short PMKID (not 16 bytes)
echo "WPA*01*abcd*4604ba734d4e*89acf0e761f4*ed487162465a774bfba60eb603a39f3a***" > "$TMPDIR/hash_shortpmkid.txt"
OUT=$("$BINARY" "$TMPDIR/hash_shortpmkid.txt" "$TMPDIR/wl_basic.txt" 2>&1 || true)
run_test "Short PMKID error" "16 bytes" "$OUT"

# --- Section 5: Stdin support ---
echo ""
echo "--- Section 5: Stdin Support ---"

# Test 16: stdin wordlist
OUT=$(echo "password123" | "$BINARY" "$TMPDIR/hash_22000.txt" - 2>&1 || true)
run_test "stdin wordlist" "PASSWORD FOUND" "$OUT"

# --- Section 6: Line endings ---
echo ""
echo "--- Section 6: Line Endings ---"

# Test 17: Windows CRLF line endings
printf "password123\r\nwrongpass\r\n" > "$TMPDIR/wl_crlf.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_crlf.txt" 2>&1 || true)
run_test "CRLF line endings" "password123" "$OUT"

# Test 18: No trailing newline
printf "wrongpass1\npassword123" > "$TMPDIR/wl_notrail.txt"
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_notrail.txt" 2>&1 || true)
run_test "No trailing newline" "password123" "$OUT"

# --- Section 7: Performance ---
echo ""
echo "--- Section 7: Performance ---"

# Test 19: Large wordlist speed test
python3 -c "
import random, string
words = ['password123']  # ensure target is in there
for _ in range(10000):
    w = ''.join(random.choices(string.ascii_lowercase + string.digits, k=random.randint(8, 12)))
    words.append(w)
# put target somewhere in the middle
words.insert(5000, 'password123')
print('\n'.join(words))
" > "$TMPDIR/wl_10k.txt"
START=$(date +%s%N)
OUT=$("$BINARY" "$TMPDIR/hash_22000.txt" "$TMPDIR/wl_10k.txt" 2>&1 || true)
END=$(date +%s%N)
ELAPSED=$(( (END - START) / 1000000 ))
run_test "10k wordlist finds password" "PASSWORD FOUND" "$OUT"
echo "    10k passwords in ${ELAPSED}ms"

# --- Summary ---
echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    echo "Failed tests:"
    for t in "${FAILED_TESTS[@]}"; do
        echo "  - $t"
    done
    exit 1
fi
echo "All tests passed!"