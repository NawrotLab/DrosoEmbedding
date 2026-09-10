"""
Read-only diagnostic: scans a pickle file's raw bytes for unicode-string
opcodes (SHORT_BINUNICODE, BINUNICODE, BINUNICODE8) and reports every one
that fails to decode as UTF-8, with surrounding context -- without needing
pickle.load() to successfully deserialize the whole object graph (which is
exactly what fails on some of the split pickles with a bare
UnicodeDecodeError, giving no indication of which entry is the problem).

Does not touch/modify the target file.

Usage:
    python -m scripts.analysis.diagnose_pickle_encoding /path/to/file.pickle
"""
import sys

SHORT_BINUNICODE = 0x8c   # 1-byte length prefix
BINUNICODE       = 0x58   # 'X' -- 4-byte little-endian length prefix
BINUNICODE8      = 0x8d   # 8-byte little-endian length prefix


def scan(path, max_report=10):
    with open(path, 'rb') as f:
        data = f.read()

    n = len(data)
    i = 0
    bad_count = 0
    good_count = 0
    while i < n:
        op = data[i]
        if op == SHORT_BINUNICODE:
            length = data[i + 1]
            start = i + 2
        elif op == BINUNICODE:
            length = int.from_bytes(data[i + 1:i + 5], 'little')
            start = i + 5
        elif op == BINUNICODE8:
            length = int.from_bytes(data[i + 1:i + 9], 'little')
            start = i + 9
        else:
            i += 1
            continue

        end = start + length
        if end > n or length < 0 or length > 10_000_000:
            # not actually a unicode opcode at this offset, just a byte
            # that happens to match one of the opcode values -- skip
            i += 1
            continue

        chunk = data[start:end]
        try:
            chunk.decode('utf-8')
            good_count += 1
        except UnicodeDecodeError as e:
            bad_count += 1
            if bad_count <= max_report:
                print(f"--- bad string #{bad_count} at file offset {start} (len={length}) ---")
                print(f"  error: {e}")
                print(f"  raw bytes (hex): {chunk[:120].hex()}")
                print(f"  raw bytes (repr): {chunk[:120]!r}")
                print(f"  decoded w/ latin1: {chunk.decode('latin1')[:120]!r}")
                print(f"  decoded w/ utf-8 errors=replace: {chunk.decode('utf-8', errors='replace')[:120]!r}")
                print()

        # advance past this opcode's payload; this is a heuristic scan
        # (doesn't fully parse the pickle grammar), so just move forward
        # byte-by-byte to stay robust to any misalignment
        i += 1

    print(f"Scan complete. {good_count} valid UTF-8 strings found, {bad_count} invalid.")


if __name__ == '__main__':
    scan(sys.argv[1])
