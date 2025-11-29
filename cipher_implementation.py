"""
ADFGVX cipher implementation that matches CrypTool
"""

import json
import os
import itertools

# Labels used for the rows and columns of the Polybius square
LABELS = "ADFGVX"
PADDING = True
DEBUG_OUTPUT = False


# new helper: centralize debug printing controlled by DEBUG_OUTPUT
def debug(*args, **kwargs):
    if DEBUG_OUTPUT:
        print(*args, **kwargs)


def create_polybius_square(mapping_string):
    """
    Create the Polybius square mapping dictionaries.
    """
    if len(mapping_string) != 36:
        raise ValueError("Polybius square must have exactly 36 characters.")

    polybius_map = {}
    inverse_polybius_map = {}
    index = 0
    for row in range(6):
        for column in range(6):
            label_pair = LABELS[row] + LABELS[column]
            character = mapping_string[index]
            polybius_map[character] = label_pair
            inverse_polybius_map[label_pair] = character
            index += 1
    return polybius_map, inverse_polybius_map


def fractionate_text(plaintext, polybius_map):
    """
    Replace each plaintext character with its corresponding label pair from the Polybius square.
    """
    fractionated_text = ""
    for character in plaintext.upper():
        if character in polybius_map:
            fractionated_text += polybius_map[character]
    return fractionated_text


def get_column_order(column_key):
    """
    Get the order in which columns should be read based on CrypTool's ranking system.
    """
    # Create list of (position, letter) pairs
    indexed_key = [(i, char) for i, char in enumerate(column_key)]

    # Sort by letter first, then by original position for duplicates
    sorted_indexed_key = sorted(indexed_key, key=lambda x: (x[1], x[0]))

    # Assign ranks (1, 2, 3, ...) to each position
    ranks = [0] * len(column_key)
    for rank, (original_pos, char) in enumerate(sorted_indexed_key, start=1):
        ranks[original_pos] = rank

    # Create list of (rank, position) pairs and sort by rank
    rank_pos_pairs = [(ranks[i], i) for i in range(len(column_key))]
    rank_pos_pairs.sort()

    # Extract positions in rank order
    column_order = [pos for rank, pos in rank_pos_pairs]

    return column_order


def apply_columnar_transposition(fractionated_text, column_key, padding=PADDING):
    """
    Apply columnar transposition.
    - Rectangle cells hold single letters
    - Write row-wise
    - Read columns top-to-bottom in key order
    - Padding applied only if padding=True
    """
    debug(f"\n=== TRANSPOSITION DEBUG (PADDING={padding}) ===")
    debug(f"Input fractionated text: {fractionated_text}")
    debug(f"Length: {len(fractionated_text)}")

    num_cols = len(column_key)
    text_len = len(fractionated_text)

    # Calculate complete rows and remainder
    complete_rows = text_len // num_cols
    remainder_chars = text_len % num_cols

    grid = []

    # Apply padding if enabled
    if padding and remainder_chars > 0:
        padding_length = num_cols - remainder_chars
        fractionated_text += "X" * padding_length
        complete_rows += 1
        remainder_chars = 0

    # Build rows
    for i in range(complete_rows):
        start_idx = i * num_cols
        end_idx = start_idx + num_cols
        row = list(fractionated_text[start_idx:end_idx])
        grid.append(row)
        debug(f"Row {i}: {' '.join(row)}")

    if not padding and remainder_chars > 0:
        # Handle incomplete last row without padding
        start_idx = complete_rows * num_cols
        incomplete_row = list(fractionated_text[start_idx:])
        while len(incomplete_row) < num_cols:
            incomplete_row.append(None)
        grid.append(incomplete_row)
        row_display = [c if c is not None else "_" for c in incomplete_row]
        debug(f"Row {complete_rows} (incomplete): {' '.join(row_display)}")

    total_rows = len(grid)

    # Get column order
    column_order = get_column_order(column_key)
    debug(f"Column order: {column_order}")

    # Read columns top-to-bottom
    ciphertext = ""
    for i, col_idx in enumerate(column_order):
        column_chars = []
        for row_idx in range(total_rows):
            if grid[row_idx][col_idx] is not None:
                column_chars.append(grid[row_idx][col_idx])
        column_content = "".join(column_chars)
        debug(
            f"Column {col_idx} (rank {i+1}, letter '{column_key[col_idx]}'): {column_content}"
        )
        ciphertext += column_content

    debug(f"Final ciphertext: {ciphertext}")
    debug(
        f"Final length: {len(ciphertext)} (should equal input length {len(fractionated_text)})"
    )
    debug("=== END DEBUG ===\n")

    return ciphertext


def reverse_columnar_transposition(ciphertext, column_key, padding=PADDING):
    """
    Reverse the columnar transposition.
    - Handles padding if padding=True
    - Preserves debug output
    """
    debug(f"\n=== DECRYPTION DEBUG (PADDING={padding}) ===")
    debug(f"Ciphertext to reverse: {ciphertext}")
    debug(f"Length: {len(ciphertext)}")

    num_cols = len(column_key)
    text_length = len(ciphertext)

    complete_rows = text_length // num_cols
    remainder_chars = text_length % num_cols

    if padding and remainder_chars > 0:
        # Ignore remainder when padding was applied
        remainder_chars = 0

    debug(f"Complete rows: {complete_rows}")
    debug(f"Remainder chars: {remainder_chars}")

    # Get column order
    column_order = get_column_order(column_key)
    debug(f"Column order used in encryption: {column_order}")

    # Determine column lengths
    col_lengths = [complete_rows] * num_cols
    if not padding and remainder_chars > 0:
        for i in range(remainder_chars):
            col_lengths[i] += 1  # first few columns get extra char

    debug("Column lengths:", [(i, col_lengths[i]) for i in range(num_cols)])

    # Split ciphertext into columns according to column order
    columns = [""] * num_cols
    position = 0
    for col_idx in column_order:
        col_len = col_lengths[col_idx]
        columns[col_idx] = ciphertext[position : position + col_len]
        debug(f"Column {col_idx}: '{columns[col_idx]}' (length {col_len})")
        position += col_len

    # Reconstruct fractionated text row-wise
    fractionated_text = ""
    max_col_len = max(len(col) for col in columns)
    debug(f"\nReconstructing row by row (max column length: {max_col_len}):")
    for row_idx in range(max_col_len):
        row_chars = ""
        for col_idx in range(num_cols):
            if row_idx < len(columns[col_idx]):
                char = columns[col_idx][row_idx]
                fractionated_text += char
                row_chars += char
            else:
                row_chars += "_"  # missing chars
        debug(f"Row {row_idx}: {row_chars}")

    debug(f"Reconstructed fractionated text: {fractionated_text}")
    debug("=== END DECRYPTION DEBUG ===\n")

    return fractionated_text


def encrypt_message(plaintext, polybius_map, column_key, padding=PADDING):
    """
    Encrypt the plaintext using the ADFGVX cipher.
    """
    fractionated = fractionate_text(plaintext, polybius_map)
    ciphertext = apply_columnar_transposition(fractionated, column_key, padding=padding)

    # Format output into pairs of two symbols separated by spaces
    return " ".join(ciphertext[i : i + 2] for i in range(0, len(ciphertext), 2))


def decrypt_message(ciphertext, inverse_polybius_map, column_key, padding=PADDING):
    """
    Decrypt an ADFGVX ciphertext back to plaintext.
    - padding: if True, ciphertext was produced with padding and trailing 'X' should be removed;
               if False, no automatic stripping of trailing 'X' is performed.
    """
    # Remove spaces
    ciphertext = ciphertext.replace(" ", "")

    # Heuristic debug warning if caller says no padding but ciphertext length is multiple of num_cols
    if not padding and len(ciphertext) % len(column_key) == 0:
        debug(
            "Warning: ciphertext length is divisible by number of columns; it may have been produced with padding=True."
        )

    # Reverse the transposition (pass padding flag so reconstruction is consistent)
    fractionated_text = reverse_columnar_transposition(
        ciphertext, column_key, padding=padding
    )

    # Remove padding X's from the end only if padding was actually used
    if padding:
        fractionated_text = fractionated_text.rstrip("X")
        debug(
            f"After removing padding: {fractionated_text} (length: {len(fractionated_text)})"
        )
    else:
        debug(
            f"Padding disabled: leaving fractionated text as-is: {fractionated_text} (length: {len(fractionated_text)})"
        )

    # Convert label pairs back to plaintext
    plaintext = ""
    for i in range(0, len(fractionated_text), 2):
        if i + 1 < len(fractionated_text):
            label_pair = fractionated_text[i : i + 2]
            if label_pair in inverse_polybius_map:
                plaintext += inverse_polybius_map[label_pair]

    return plaintext


def fill_msg_dict(entry_id, messages_json_path=None, overwrite=True):
    """
    Fill the 'ciphertexts' field for the given integer entry_id inside messages.json.
    - entry_id: integer key (the JSON uses string keys like "1", "2", ...)
    - messages_json_path: optional path to messages.json (defaults to same directory as this file)
    - overwrite: if True (default) existing 'ciphertexts' for the entry will be replaced;
                 if False and ciphertexts already exist the function will skip writing.

    The function reads messages.json, uses the entry's "plaintexts", "key" and "alphabet"
    to create a Polybius square and produce ciphertexts via encrypt_message().
    Intermediate conversion details are printed for each plaintext. The updated JSON is
    written back and the updated entry dict is returned.
    """
    if messages_json_path is None:
        messages_json_path = os.path.join(
            os.path.dirname(__file__), "auxiliary", "messages.json"
        )

    if not os.path.isfile(messages_json_path):
        raise FileNotFoundError(f"messages.json not found at: {messages_json_path}")

    with open(messages_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    key_str = str(entry_id)
    if key_str not in data:
        raise KeyError(f"Entry '{key_str}' not found in messages.json")

    entry = data[key_str]
    plaintexts = entry.get("plaintexts", [])
    column_key = entry.get("key")
    alphabet = entry.get("alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    if column_key is None:
        raise ValueError(f"Entry '{key_str}' has no 'key' field")

    if not isinstance(plaintexts, list):
        raise ValueError(
            f"Entry '{key_str}' has invalid 'plaintexts' field (expected list)"
        )

    if not isinstance(alphabet, str) or len(alphabet) != 36:
        raise ValueError(
            f"Entry '{key_str}' has invalid 'alphabet' (must be 36-character string)"
        )

    # If ciphertexts already exist, handle according to overwrite flag
    existing_ciphertexts = entry.get("ciphertexts")
    if existing_ciphertexts:
        if not overwrite:
            print(
                f"Entry '{key_str}' already has {len(existing_ciphertexts)} ciphertext(s) and overwrite=False; skipping."
            )
            return entry
        else:
            print(
                f"Overwriting {len(existing_ciphertexts)} existing ciphertext(s) for entry '{key_str}'."
            )

    # Build Polybius square
    polybius_map, _ = create_polybius_square(alphabet)

    # Print header about processing
    print(
        f"Processing messages.json entry '{key_str}' with key '{column_key}' and alphabet of length {len(alphabet)}"
    )
    print(f"Found {len(plaintexts)} plaintext(s). Converting to ciphertexts now...\n")

    ciphertexts = []
    for idx, pt in enumerate(plaintexts, start=1):
        # fractionate then encrypt (encrypt_message also fractionates internally, but show both steps)
        fractionated = fractionate_text(pt, polybius_map)
        ct = encrypt_message(pt, polybius_map, column_key)
        ciphertexts.append(ct)

        # Print intermediate results
        print(f"[{idx}] Plaintext: {pt}")
        print(f"    Fractionated: {fractionated}")
        print(f"    Ciphertext:   {ct}\n")

    # Update entry and write back (this will overwrite the 'ciphertexts' field)
    entry["ciphertexts"] = ciphertexts
    data[key_str] = entry

    with open(messages_json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    # Final success message
    print(
        f"Successfully wrote {len(ciphertexts)} ciphertext(s) for entry '{key_str}' to {messages_json_path}"
    )

    return entry


def run_example(plaintext=None, key=None, alphabet=None):
    """
    Run a small example. If arguments are omitted or invalid, sensible defaults are used:
    - default alphabet: "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    - default key: "MATH"
    - default plaintext: "I LOVE CRYPTOGRAPHY"
    """
    # choose defaults if needed
    default_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    if not isinstance(alphabet, str) or len(alphabet) != 36:
        alphabet = default_alphabet

    try:
        polybius_map, inverse_polybius_map = create_polybius_square(alphabet)
    except Exception as e:
        # fall back to default alphabet if creation fails
        debug(f"create_polybius_square failed for given alphabet, falling back: {e}")
        polybius_map, inverse_polybius_map = create_polybius_square(default_alphabet)

    column_key = key if (isinstance(key, str) and len(key) > 0) else "MATH"
    plaintext = (
        plaintext
        if (isinstance(plaintext, str) and len(plaintext) > 0)
        else "I LOVE CRYPTOGRAPHY"
    )

    # Display the Polybius square for verification (debug only)
    debug("Polybius Square:")
    debug("   ", " ".join(LABELS))
    for i, row_label in enumerate(LABELS):
        row_chars = [alphabet[i * 6 + j] for j in range(6)]
        debug(f"{row_label}:  {' '.join(row_chars)}")
    debug()

    # Show the fractionation step with detailed character mapping
    debug("=== FRACTIONATION DEBUG ===")
    debug(f"Plaintext: {plaintext}")
    fractionated = ""
    for i, char in enumerate(plaintext.upper()):
        if char in polybius_map:
            pair = polybius_map[char]
            fractionated += pair
            debug(f"  {char} -> {pair}")
        else:
            debug(f"  {char} -> SKIPPED (not in alphabet)")

    debug(f"Complete fractionated: {fractionated}")
    debug(f"Length: {len(fractionated)}")
    debug("=== END FRACTIONATION DEBUG ===\n")

    # Show column order
    column_order = get_column_order(column_key)
    debug(f"Column key: {column_key}")
    debug(f"Column order: {column_order}")
    debug(f"Key positions: {[(i, column_key[i]) for i in column_order]}")

    # Show the ranking like CrypTool
    indexed_key = [(i, char) for i, char in enumerate(column_key)]
    sorted_indexed_key = sorted(indexed_key, key=lambda x: (x[1], x[0]))
    ranks = [0] * len(column_key)
    for rank, (original_pos, char) in enumerate(sorted_indexed_key, start=1):
        ranks[original_pos] = rank

    sorted_chars = "".join([char for pos, char in sorted_indexed_key])
    rank_string = "-".join(map(str, ranks))
    debug(f"Transposition key (CrypTool format): {rank_string} ({sorted_chars})")

    # Encrypt
    ciphertext = encrypt_message(plaintext, polybius_map, column_key)

    # Decrypt to verify
    decrypted = decrypt_message(ciphertext, inverse_polybius_map, column_key)

    print(f"Final Ciphertext: {ciphertext}")
    print(f"Decrypted Plaintext: {decrypted}")


def get_all_possible_key_orders(key, print_output=False, file_output=False):
    """
    Return all possible column-order lists (lists of original indices) that are
    consistent with CrypTool-style ranking of the provided key string.
    The ordering of character-groups (by letter) is fixed; indices belonging to
    the same letter may be permuted arbitrarily.
    Example: duplicate-letter groups with sizes 3,2,2 produce 3!*2!*2! variants.

    Added parameter:
    - file_output: if True, write the textual output that would usually be printed
      to auxiliary/key_order_output.txt (overwriting the file).
    """
    if not isinstance(key, str):
        raise TypeError("key must be a string")

    # Build list of (original_pos, char) and sort by (char, original_pos)
    indexed = [(i, ch) for i, ch in enumerate(key)]
    sorted_indexed = sorted(indexed, key=lambda t: (t[1], t[0]))

    # Group the sorted entries by character; each group yields a block in final order
    groups = []
    for ch, group_iter in itertools.groupby(sorted_indexed, key=lambda t: t[1]):
        group_positions = [pos for pos, _ in group_iter]
        groups.append(group_positions)

    # For each group, enumerate all permutations (if size 1 -> single permutation)
    group_perms = [list(itertools.permutations(g)) for g in groups]

    # Cartesian product of group permutations, concatenated into full orders
    results = []
    for combo in itertools.product(*group_perms):
        order = []
        for perm in combo:
            order.extend(list(perm))
        results.append(order)

    # Deduplicate while preserving order (just in case)
    seen = set()
    unique_results = []
    for r in results:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            unique_results.append(list(r))

    # Prepare optional output (console and/or file)
    if print_output or file_output:
        header = f"Found {len(unique_results)} unique key order(s) for key '{key}':"
        if print_output:
            print(header)
        if file_output:
            output_lines = [header]

        for order in unique_results:
            # Print numeric order
            if print_output:
                print(order)
            if file_output:
                output_lines.append(str(order))

            # Additionally print the canonical encoded key letters derived from this numeric order.
            # Convert order (list of original indices in reading order) -> rank at each original position -> letters A..
            try:
                m = len(order)
                rank = [0] * m
                for r, colpos in enumerate(order):
                    rank[colpos] = r
                encoded_letters = "".join(chr(ord("A") + r) for r in rank)
                encoded_line = f"Encoded key letters: '{encoded_letters}'"
                if print_output:
                    print(encoded_line)
                if file_output:
                    output_lines.append(encoded_line)
            except Exception:
                # Be tolerant: if something goes wrong, don't interrupt other prints.
                pass

        # If requested, write output lines to auxiliary/key_order_output.txt (overwrite)
        if file_output:
            aux_dir = os.path.join(os.path.dirname(__file__), "auxiliary")
            os.makedirs(aux_dir, exist_ok=True)
            out_path = os.path.join(aux_dir, "key_order_output.txt")
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(output_lines))
                fh.write("\n")

    return unique_results


if __name__ == "__main__":

    # run_example(
    #     plaintext="I LOVE CRYPTOGRAPHY",
    #     key="MATH",
    #     alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    # )

    fill_msg_dict(1)
    fill_msg_dict(2)
    fill_msg_dict(3)
    fill_msg_dict(4)
    fill_msg_dict(5)
    fill_msg_dict(6)
    fill_msg_dict(7)
    fill_msg_dict(8)

    get_all_possible_key_orders("AVENIMANAILUYI", file_output=True)
    # get_all_possible_key_orders("THISPASSWORDISVERY", file_output=True)
