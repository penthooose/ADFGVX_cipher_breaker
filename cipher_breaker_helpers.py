from typing import List, Tuple, Dict, Optional, Iterable
from collections import Counter
import math
from cipher_implementation import (
    create_polybius_square,
    reverse_columnar_transposition,
    get_column_order,
)


def infer_fragment_seeding_variables(key_length: int) -> Dict[str, int]:
    """
    Return a dictionary of fragment seeding and multi-phase search parameters
    based on key length. This provides length-specific tuning for:
    - Phase 1 (fragment seeding): total_seeds, restart_per_seed, max_iterations_per_seed
    - Phase 2 (deeper search): best_candidates_phase_2, restarts_phase_2, max_iterations_phase_2
    - Phase 3 (final refinement): best_candidates_phase_3, restarts, max_iterations
    - ic_threshold: key-length specific IC threshold for early stopping

    Args:
        key_length: The columnar transposition key length

    Returns:
        Dictionary containing all phase-specific parameters including ic_threshold
    """
    # Default baseline for lengths <= 17
    if key_length <= 14:
        return {
            # Phase 1: Fragment seeding
            "total_seeds": 100,
            "restart_per_seed": 1,
            "max_iterations_per_seed": 500,
            # Phase 2: Deeper search on best Phase 1 candidates
            "best_candidates_phase_2": 10,
            "restarts_phase_2": 1000,
            "max_iterations_phase_2": 5000,
            # Phase 3: Final refinement on best Phase 2 candidates
            "best_candidates_phase_3": 2,
            "restarts": 1000,
            "max_iterations": 20000,
            # IC threshold
            "ic_threshold": 0.0397,
            "tetragram_even_key_additive_value": 0.2,
            "tetragram_ic_scale_factor": 7.0,
        }
    elif key_length <= 16:
        return {
            # Phase 1: Fragment seeding
            "total_seeds": 100,
            "restart_per_seed": 1,
            "max_iterations_per_seed": 1000,
            # Phase 2: Deeper search on best Phase 1 candidates
            "best_candidates_phase_2": 10,
            "restarts_phase_2": 1000,
            "max_iterations_phase_2": 5000,
            # Phase 3: Final refinement on best Phase 2 candidates
            "best_candidates_phase_3": 2,
            "restarts": 1000,
            "max_iterations": 20000,
            # IC threshold
            "ic_threshold": 0.059,
            "tetragram_even_key_additive_value": 0.3,
            "tetragram_ic_scale_factor": 15.0,
        }
    elif key_length <= 17:
        return {
            # Phase 1: Fragment seeding
            "total_seeds": 100,
            "restart_per_seed": 1,
            "max_iterations_per_seed": 1000,
            # Phase 2: Deeper search on best Phase 1 candidates
            "best_candidates_phase_2": 10,
            "restarts_phase_2": 1000,
            "max_iterations_phase_2": 5000,
            # Phase 3: Final refinement on best Phase 2 candidates
            "best_candidates_phase_3": 2,
            "restarts": 1000,
            "max_iterations": 20000,
            # IC threshold
            "ic_threshold": 0.0526,
            "tetragram_even_key_additive_value": 0.1,
            "tetragram_ic_scale_factor": 15.0,
        }
    elif key_length <= 18:
        return {
            "total_seeds": 100,
            "restart_per_seed": 1,
            "max_iterations_per_seed": 1000,
            "best_candidates_phase_2": 10,
            "restarts_phase_2": 1500,
            "max_iterations_phase_2": 7000,
            "best_candidates_phase_3": 3,
            "restarts": 1500,
            "max_iterations": 25000,
            "ic_threshold": 0.0526,
            "tetragram_even_key_additive_value": 0.4,
            "tetragram_ic_scale_factor": 15.0,
        }

    elif key_length <= 20:
        return {
            "total_seeds": 150,
            "restart_per_seed": 1,
            "max_iterations_per_seed": 1500,
            "best_candidates_phase_2": 10,
            "restarts_phase_2": 2000,
            "max_iterations_phase_2": 8000,
            "best_candidates_phase_3": 2,
            "restarts": 200,
            "max_iterations": 25000,
            "ic_threshold": 0.07,
            "tetragram_even_key_additive_value": 0.4,
            "tetragram_ic_scale_factor": 15.0,
        }

    # Long keys (21-25): significantly increase all phases
    elif key_length <= 25:
        return {
            "total_seeds": 200,
            "restart_per_seed": 4,
            "max_iterations_per_seed": 2000,
            "best_candidates_phase_2": 20,
            "restarts_phase_2": 2000,
            "max_iterations_phase_2": 10000,
            "best_candidates_phase_3": 4,
            "restarts": 2000,
            "max_iterations": 30000,
            "ic_threshold": 0.043,
            "tetragram_ic_scale_factor": 15.0,
        }

    # Very long keys (>30): aggressive search
    elif key_length <= 30:
        return {
            "total_seeds": 300,
            "restart_per_seed": 6,
            "max_iterations_per_seed": 3000,
            "best_candidates_phase_2": 30,
            "restarts_phase_2": 3000,
            "max_iterations_phase_2": 15000,
            "best_candidates_phase_3": 6,
            "restarts": 3000,
            "max_iterations": 40000,
            "ic_threshold": 0.040,
            "tetragram_ic_scale_factor": 15.0,
        }

    # Extreme keys (>30): aggressive search
    else:
        return {
            "total_seeds": 300,
            "restart_per_seed": 6,
            "max_iterations_per_seed": 3000,
            "best_candidates_phase_2": 30,
            "restarts_phase_2": 3000,
            "max_iterations_phase_2": 15000,
            "best_candidates_phase_3": 6,
            "restarts": 3000,
            "max_iterations": 40000,
            "ic_threshold": 0.040,
            "tetragram_ic_scale_factor": 15.0,
        }


def set_config_helpers(cfg: Dict):
    """Initialize module-level CONFIG (copy) so helpers use the same settings as caller."""
    global CONFIG
    if isinstance(cfg, dict):
        CONFIG = cfg.copy()
    else:
        CONFIG = dict(cfg)


def compute_token_ic(tokens: List[str]) -> float:
    """Compute Index of Coincidence treating each token as an atomic symbol."""
    N = len(tokens)
    if N <= 1:
        return 0.0
    cnt = Counter(tokens)
    return sum(v * (v - 1) for v in cnt.values()) / (N * (N - 1))


def get_bigrams(text: str) -> List[str]:
    """Return overlapping consecutive bigrams (length-2 substrings)."""
    return [text[i : i + 2] for i in range(len(text) - 1)]


def get_tetragrams(text: str) -> List[str]:
    """Return overlapping consecutive tetragrams (length-4 substrings)."""
    return [text[i : i + 4] for i in range(len(text) - 3)]


def string_key_from_column_order(column_order: List[int]) -> str:
    """Convert a column order (list of column indices in reading order) into a string key."""
    m = len(column_order)
    rank = [0] * m
    for r, colpos in enumerate(column_order):
        rank[colpos] = r
    key_chars = [chr(ord("A") + r) for r in rank]
    return "".join(key_chars)


def adfgvx_pair_regularity(fractionated: str, adfgvx_chars: List[str]) -> float:
    """Score how well the fractionated stream exhibits expected ADFGVX pair structure."""
    if len(fractionated) < 10:
        return 0.0

    even_chars = [fractionated[i] for i in range(0, len(fractionated), 2)]
    odd_chars = [
        fractionated[i] for i in range(1, len(fractionated), 2) if i < len(fractionated)
    ]

    even_dist = Counter(even_chars)
    odd_dist = Counter(odd_chars)

    total_even = len(even_chars) or 1
    total_odd = len(odd_chars) or 1

    divergence = 0.0
    for char in adfgvx_chars:
        even_freq = even_dist.get(char, 0) / total_even
        odd_freq = odd_dist.get(char, 0) / total_odd
        avg_freq = (even_freq + odd_freq) / 2.0
        if avg_freq > 0:
            divergence += (even_freq - avg_freq) ** 2 + (odd_freq - avg_freq) ** 2

    regularity_bonus = math.exp(-5.0 * divergence)
    return regularity_bonus


def position_entropy_score(fractionated: str) -> float:
    """Measure entropy variance across positions in fractionated pairs."""
    if len(fractionated) < 20:
        return 0.0

    even_chars = [fractionated[i] for i in range(0, len(fractionated), 2)]
    odd_chars = [
        fractionated[i] for i in range(1, len(fractionated), 2) if i < len(fractionated)
    ]

    def compute_entropy(chars):
        if not chars:
            return 0.0
        freq = Counter(chars)
        total = len(chars)
        return -sum(
            (count / total) * math.log(count / total)
            for count in freq.values()
            if count > 0
        )

    even_entropy = compute_entropy(even_chars)
    odd_entropy = compute_entropy(odd_chars)

    entropy_variance = abs(even_entropy - odd_entropy)
    entropy_bonus = math.exp(-2.0 * entropy_variance)
    return entropy_bonus


def generate_transformations(key_order: List[int]) -> List[Tuple[str, List[int]]]:
    """Produce Lasry et al.'s transformation set given a key order list."""
    m = len(key_order)
    transforms: List[Tuple[str, List[int]]] = []

    # 1) Swap any two elements
    for i in range(m):
        for j in range(i + 1, m):
            nk = key_order.copy()
            nk[i], nk[j] = nk[j], nk[i]
            transforms.append((f"SwapElements({i},{j})", nk))

    # 2) Swap two consecutive segments of equal length
    for i in range(m):
        for L in range(1, (m - i) // 2 + 1):
            j = i + L
            if j + L <= m:
                nk = key_order.copy()
                seg1 = nk[i : i + L]
                seg2 = nk[j : j + L]
                nk[i : i + L] = seg2
                nk[j : j + L] = seg1
                transforms.append((f"SwapSegments({i},{L},{j},{L})", nk))

    # 3) Rotate a segment
    for i in range(m):
        for L in range(2, m - i + 1):
            for k in range(1, L):
                nk = key_order.copy()
                seg = nk[i : i + L]
                nk[i : i + L] = seg[k:] + seg[:k]
                transforms.append((f"RotateSegment({i},{L},{k})", nk))

    # 4) Reverse the whole key
    nk = key_order.copy()
    nk.reverse()
    transforms.append(("ReverseKey", nk))

    # 5) Swap pairs
    if m >= 2:
        nk = key_order.copy()
        for i in range(0, m - 1, 2):
            nk[i], nk[i + 1] = nk[i + 1], nk[i]
        transforms.append(("SwapPairs", nk))

    return transforms


def score_fragment_adjacency(left_frag: str, right_frag: str) -> float:
    """
    Score how well two column fragments fit adjacent to each other.
    Uses overlapping bigrams across the boundary (interleaved pairs).
    Higher score = better adjacency.
    """
    if not left_frag or not right_frag:
        return 0.0

    # Interleave characters from left and right fragments to form bigrams
    min_len = min(len(left_frag), len(right_frag))
    bigrams = []
    for i in range(min_len):
        bigrams.append(left_frag[i] + right_frag[i])

    if len(bigrams) < 2:
        return 0.0

    # Score using bigram IC
    return compute_token_ic(bigrams)


def build_adjacency_matrix(fragments: List[str]) -> List[List[float]]:
    """
    Build an m×m adjacency weight matrix where entry [i][j]
    represents the score for placing fragment j immediately after fragment i.
    """
    m = len(fragments)
    matrix = [[0.0] * m for _ in range(m)]

    for i in range(m):
        for j in range(m):
            if i != j:
                matrix[i][j] = score_fragment_adjacency(fragments[i], fragments[j])

    return matrix


def greedy_fragment_ordering(adjacency_matrix: List[List[float]]) -> List[int]:
    """
    Greedily construct a column ordering from adjacency weights.
    Start with the pair having highest weight, then extend by adding
    fragments that score best at either end.
    Returns a column-order list (indices into fragments array).
    """
    m = len(adjacency_matrix)
    if m <= 1:
        return list(range(m))

    # Find the best initial pair
    best_score = float("-inf")
    best_i, best_j = 0, 1
    for i in range(m):
        for j in range(m):
            if i != j and adjacency_matrix[i][j] > best_score:
                best_score = adjacency_matrix[i][j]
                best_i, best_j = i, j

    # Start with the best pair
    ordering = [best_i, best_j]
    used = {best_i, best_j}

    # Greedily extend at either end
    while len(ordering) < m:
        best_score = float("-inf")
        best_pos = None  # 'left' or 'right'
        best_frag = None

        left_idx = ordering[0]
        right_idx = ordering[-1]

        for frag in range(m):
            if frag in used:
                continue

            # Try adding at left (frag -> left_idx)
            score_left = adjacency_matrix[frag][left_idx]
            if score_left > best_score:
                best_score = score_left
                best_pos = "left"
                best_frag = frag

            # Try adding at right (right_idx -> frag)
            score_right = adjacency_matrix[right_idx][frag]
            if score_right > best_score:
                best_score = score_right
                best_pos = "right"
                best_frag = frag
        if best_frag is None:
            # No more fragments to add (shouldn't happen)
            break

        if best_pos == "left":
            ordering.insert(0, best_frag)
        else:
            ordering.append(best_frag)

        used.add(best_frag)

    return ordering


def beam_fragment_ordering(
    adjacency_matrix: List[List[float]], beam_width: int = 5
) -> List[List[int]]:
    """
    Use beam search to generate multiple plausible column orderings.
    Returns up to beam_width complete orderings, scored by cumulative adjacency.
    """
    m = len(adjacency_matrix)
    if m <= 1:
        return [list(range(m))]

    # State: (ordering, used_set, cumulative_score)
    # Start with all possible single fragments
    beam = [([i], {i}, 0.0) for i in range(m)]

    for step in range(m - 1):
        candidates = []

        for ordering, used, cum_score in beam:
            left_idx = ordering[0]
            right_idx = ordering[-1]

            for frag in range(m):
                if frag in used:
                    continue

                # Extend left
                score_left = adjacency_matrix[frag][left_idx]
                new_ordering_left = [frag] + ordering
                new_used_left = used | {frag}
                new_score_left = cum_score + score_left
                candidates.append((new_ordering_left, new_used_left, new_score_left))

                # Extend right
                score_right = adjacency_matrix[right_idx][frag]
                new_ordering_right = ordering + [frag]
                new_used_right = used | {frag}
                new_score_right = cum_score + score_right
                candidates.append((new_ordering_right, new_used_right, new_score_right))

        # Keep top beam_width candidates
        candidates.sort(key=lambda x: x[2], reverse=True)
        beam = candidates[:beam_width]

    # Return completed orderings sorted by score
    return [
        ordering for ordering, _, _ in sorted(beam, key=lambda x: x[2], reverse=True)
    ]


def extract_column_fragments(ciphertext: str, key_length: int) -> List[str]:
    """
    Extract column fragments from a columnar transposition ciphertext.
    The ciphertext is written column-by-column, so we slice contiguous blocks.

    For a ciphertext of length n and key_length m:
    - The first (n % m) columns have length ceil(n/m)
    - The remaining columns have length floor(n/m)

    Returns a list of m strings (one per column in reading order).
    """
    if not ciphertext or key_length <= 0:
        return [""] * max(1, key_length)

    n = len(ciphertext)
    m = key_length

    # Calculate column lengths
    base_len = n // m  # floor(n/m)
    remainder = n % m  # number of columns with one extra character

    fragments = []
    pos = 0

    for col_idx in range(m):
        # First 'remainder' columns have base_len + 1 chars
        if col_idx < remainder:
            col_len = base_len + 1
        else:
            col_len = base_len

        # Slice the contiguous block for this column
        fragments.append(ciphertext[pos : pos + col_len])
        pos += col_len

    return fragments


def debug_enabled(cfg: Dict) -> bool:
    return bool(cfg.get("debug_output", False))


def infer_key_search_variables(key_length: int) -> Tuple[int, int, float]:
    """Return (restarts, max_iterations, ic_threshold) based on key length.

    This is a heuristic piecewise schedule intended to allocate more
    search effort to mid/large key lengths.
    """
    # sensible defaults
    if key_length <= 6:
        return 50, 2000, 0.06
    elif key_length <= 8:
        return 100, 4000, 0.055
    elif key_length <= 10:
        return 150, 6000, 0.052
    elif key_length == 11:
        return 200, 8000, 0.051
    elif key_length == 12:
        return 250, 9000, 0.05
    elif key_length == 13:
        return 300, 10000, 0.049
    elif key_length == 14:
        return 350, 11000, 0.048
    elif key_length <= 16:
        return 400, 12000, 0.047
    elif key_length <= 18:
        return 500, 14000, 0.046
    elif key_length <= 20:
        return 700, 16000, 0.045
    elif key_length <= 22:
        return 900, 18000, 0.044
    elif key_length <= 24:
        return 1200, 20000, 0.043
    elif key_length <= 30:
        return 1500, 25000, 0.042
    else:
        return 2000, 30000, 0.04


def is_better(a: float, b: float, eps: float = 1e-8) -> bool:
    """Return True when a is meaningfully greater than b."""
    return a > b + eps


def is_tie(a: float, b: float, eps: float = 1e-8) -> bool:
    """Return True when a and b are effectively equal within eps."""
    return abs(a - b) <= eps


def fractionated_to_plain_with_map(fractionated: str, inv_map: Dict[str, str]) -> str:
    """Convert a fractionated ADFGVX stream to plaintext using an inverse map.

    Trailing padding 'X' characters are stripped before conversion. Unknown
    digrams are replaced with '?'.
    """
    if not fractionated:
        return ""
    fractionated = fractionated.rstrip("X")
    out = []
    for i in range(0, len(fractionated), 2):
        pair = fractionated[i : i + 2]
        if len(pair) < 2:
            # stray single symbol at end
            out.append("?")
            continue
        out.append(inv_map.get(pair, "?"))
    return "".join(out)


# keep a small, focused English letter-frequency based scorer
def english_score_texts(texts: List[str]) -> float:
    LETTER_LOG_FREQ = {
        "E": math.log(0.12702),
        "T": math.log(0.09056),
        "A": math.log(0.08167),
        "O": math.log(0.07507),
        "I": math.log(0.06966),
        "N": math.log(0.06749),
        "S": math.log(0.06327),
        "H": math.log(0.06094),
        "R": math.log(0.05987),
        "D": math.log(0.04253),
        "L": math.log(0.04025),
        "C": math.log(0.02782),
        "U": math.log(0.02758),
        "M": math.log(0.02406),
        "W": math.log(0.02360),
        "F": math.log(0.02228),
        "G": math.log(0.02015),
        "Y": math.log(0.01974),
        "P": math.log(0.01929),
        "B": math.log(0.01492),
        "V": math.log(0.00978),
        "K": math.log(0.00772),
        "J": math.log(0.00153),
        "X": math.log(0.00150),
        "Q": math.log(0.00095),
        "Z": math.log(0.00074),
    }
    LOG_FLOOR = math.log(1e-5)

    COMMON_WORDS_EN = {
        "THE",
        "AND",
        "TO",
        "OF",
        "IN",
        "IS",
        "THIS",
        "THAT",
        "WITH",
        "FOR",
        "AS",
    }

    # Only enable the common-words bonus when explicitly configured.
    use_word_bonus = bool(CONFIG.get("use_common_words_bonus_in_key_search", False))

    total_log = 0.0
    letter_count = 0
    word_bonus = 0.0
    total_len = 0
    alpha_count = 0
    for t in texts:
        total_len += len(t)
        u = t.upper()
        words = u.split()
        # apply word bonus only when configured
        if use_word_bonus:
            for w in words:
                if w in COMMON_WORDS_EN:
                    word_bonus += 0.25
        for ch in u:
            if "A" <= ch <= "Z":
                alpha_count += 1
                letter_count += 1
                total_log += LETTER_LOG_FREQ.get(ch, LOG_FLOOR)
            else:
                # small penalty for non-letters
                total_log += LOG_FLOOR

    avg_log = total_log / max(1, letter_count)
    alpha_ratio = alpha_count / max(1, total_len)
    # include word_bonus only if it was enabled
    return avg_log + (word_bonus if use_word_bonus else 0.0) + alpha_ratio


def german_score_texts(texts: List[str]) -> float:
    LETTER_LOG_FREQ_DE = {
        "E": math.log(0.1659),
        "N": math.log(0.0978),
        "I": math.log(0.0755),
        "S": math.log(0.0727),
        "R": math.log(0.0700),
        "A": math.log(0.0651),
        "T": math.log(0.0615),
        "D": math.log(0.0508),
        "H": math.log(0.0476),
        "U": math.log(0.0435),
        "L": math.log(0.0344),
        "O": math.log(0.0283),
        "M": math.log(0.0251),
        "G": math.log(0.0162),
        "B": math.log(0.0162),
        "W": math.log(0.0161),
        "F": math.log(0.0106),
        "K": math.log(0.0121),
        "Z": math.log(0.0123),
        "P": math.log(0.0079),
        "V": math.log(0.0068),
        "Y": math.log(0.0004),
        "X": math.log(0.0003),
        "J": math.log(0.0027),
        "Q": math.log(0.0005),
        "C": math.log(0.003),
    }
    LOG_FLOOR = math.log(1e-6)

    GERMAN_COMMON_CANDIDATES = {
        "DER",
        "DIE",
        "DAS",
        "UND",
        "ZU",
        "IM",
        "IN",
        "IST",
        "EIN",
        "MIT",
        "VON",
        "AUF",
        "AN",
        "ALS",
    }
    GERMAN_COMMON = {w for w in GERMAN_COMMON_CANDIDATES if 2 <= len(w) <= 3}

    # Only enable the common-words bonus when explicitly configured.
    use_word_bonus = bool(CONFIG.get("use_common_words_bonus_in_key_search", False))

    total_log = 0.0
    letter_count = 0
    total_len = 0
    alpha_count = 0
    for t in texts:
        total_len += len(t)
        u = t.upper()
        words = u.split()
        # apply word bonus only when configured
        if use_word_bonus:
            for w in words:
                if w in GERMAN_COMMON:
                    total_log += 0.2
        for ch in u:
            if "A" <= ch <= "Z":
                alpha_count += 1
                letter_count += 1
                total_log += LETTER_LOG_FREQ_DE.get(ch, LOG_FLOOR)
            else:
                total_log += LOG_FLOOR

    avg_log = total_log / max(1, letter_count)
    alpha_ratio = alpha_count / max(1, total_len)
    return avg_log + alpha_ratio


def score_texts(texts: List[str], lang: str = "EN") -> float:
    if (lang or "").upper() == "DE":
        return german_score_texts(texts)
    return english_score_texts(texts)


def reconstruct_long_key_seeds(
    ciphertexts: List[str], key_length: int, num_seeds: int = 5
) -> List[List[int]]:
    """
    Generate seed column orderings for long keys using fragment adjacency analysis.
    Produces many raw seeds, expands them with small randomized local moves,
    and returns up to `num_seeds` candidates (may contain similar/duplicate orders).
    """
    if not ciphertexts:
        return [list(range(key_length))]

    # --- NEW: Check if fragment voting seeds are requested ---
    if CONFIG.get("fragment_voting_seeds", False):
        import random as _rnd
        from collections import Counter

        if CONFIG.get("intermediate_output", True):
            print(
                "[FRAGMENT-VOTING] Generating seeds based on fragment successor frequencies..."
            )

        # Extract and concatenate fragments from all ciphertexts
        all_fragments = [[] for _ in range(key_length)]
        for ct in ciphertexts:
            ct_clean = ct.replace(" ", "")
            fragments = extract_column_fragments(ct_clean, key_length)
            for i, frag in enumerate(fragments):
                all_fragments[i].append(frag)
        combined_fragments = ["".join(frags) for frags in all_fragments]

        # Build successor frequency maps for each fragment
        # successor_counts[i][j] = how often fragment j follows fragment i in interleaved reading
        successor_counts = [Counter() for _ in range(key_length)]

        # Analyze interleaved bigrams across all fragments
        for frag_idx, frag in enumerate(combined_fragments):
            for pos in range(len(frag) - 1):
                # For each position in this fragment, find which fragments appear at next position
                # when reading the fractionated stream in interleaved fashion
                # This simulates reading bigrams across fragment boundaries
                for other_idx in range(key_length):
                    if other_idx != frag_idx and pos < len(
                        combined_fragments[other_idx]
                    ):
                        # Count this as a potential successor relationship
                        successor_counts[frag_idx][other_idx] += 1

        # Generate orderings using greedy "most frequent successor" strategy
        voting_seeds = []

        # Try starting from each fragment as a potential first column
        for start_frag in range(key_length):
            ordering = [start_frag]
            used = {start_frag}
            current = start_frag

            # Greedily add most frequent successor at each step
            while len(ordering) < key_length:
                # Get successor counts for current fragment
                successors = successor_counts[current]

                # Filter to unused fragments
                candidates = [
                    (count, frag)
                    for frag, count in successors.items()
                    if frag not in used
                ]

                if not candidates:
                    # No successor data - pick random unused fragment
                    remaining = [f for f in range(key_length) if f not in used]
                    if remaining:
                        next_frag = _rnd.choice(remaining)
                    else:
                        break
                else:
                    # Pick fragment with highest successor count
                    candidates.sort(reverse=True)
                    next_frag = candidates[0][1]

                ordering.append(next_frag)
                used.add(next_frag)
                current = next_frag

            if len(ordering) == key_length:
                voting_seeds.append(ordering)

        if CONFIG.get("intermediate_output", True):
            print(
                f"[FRAGMENT-VOTING] Generated {len(voting_seeds)} base orderings from successor voting"
            )

        # Diversify the voting seeds with variations
        diversified_seeds = []
        for seed in voting_seeds:
            diversified_seeds.append(seed.copy())

            # Add variations: reversed
            diversified_seeds.append(seed[::-1])

            # Add variations: swap adjacent pairs
            if key_length >= 2:
                swapped = seed.copy()
                for i in range(0, key_length - 1, 2):
                    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
                diversified_seeds.append(swapped)

            # Add variations: rotate by small amounts
            for k in [1, 2, key_length // 3, key_length // 2]:
                if k > 0 and k < key_length:
                    rotated = seed[k:] + seed[:k]
                    diversified_seeds.append(rotated)

            # Add variations: small random perturbations
            for _ in range(3):
                perturbed = seed.copy()
                # Swap 2-3 random pairs
                num_swaps = _rnd.randint(2, 3)
                for _ in range(num_swaps):
                    i, j = _rnd.sample(range(key_length), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                diversified_seeds.append(perturbed)

        # Deduplicate while preserving order
        seen = set()
        unique_seeds = []
        for seed in diversified_seeds:
            seed_tuple = tuple(seed)
            if seed_tuple not in seen:
                seen.add(seed_tuple)
                unique_seeds.append(seed)

        if CONFIG.get("intermediate_output", True):
            print(
                f"[FRAGMENT-VOTING] Total unique seeds after diversification: {len(unique_seeds)}"
            )

        # --- NEW: Score seeds with combined bigram + tetragram IC ---
        if len(unique_seeds) > num_seeds:
            if CONFIG.get("intermediate_output", True):
                print(
                    f"[FRAGMENT-VOTING] Scoring {len(unique_seeds)} seeds with combined IC (bigram + tetragram)..."
                )

            # Helper to compute tetragram weight for this key length
            def compute_tetra_weight_for_length(m: int) -> float:
                base_weight = float(CONFIG.get("tetragram_base_weight", 0.1))
                boost_start = int(CONFIG.get("tetragram_boost_start_length", 12))
                boost_per_col = float(CONFIG.get("tetragram_boost_per_column", 0.02))
                max_weight = float(CONFIG.get("tetragram_max_weight", 0.85))

                w = base_weight
                if m >= boost_start:
                    w += max(0, m - boost_start) * boost_per_col

                # add per-length even-key additive if present from infer_fragment_seeding_variables
                try:
                    phase_params = infer_fragment_seeding_variables(m)
                    even_add = float(
                        phase_params.get("tetragram_even_key_additive_value", 0.0)
                    )
                except Exception:
                    even_add = float(
                        CONFIG.get("tetragram_even_key_additive_value", 0.0)
                    )
                if m % 2 == 0:
                    w += even_add

                return min(max(w, 0.0), max_weight)

            # Score each seed

            scored_seeds = []
            tetragram_weight = compute_tetra_weight_for_length(key_length)
            tetragram_scale = CONFIG.get("tetragram_ic_scale_factor", 15.0)

            for seed_order in unique_seeds:
                seed_key = string_key_from_column_order(seed_order)

                # Decrypt with this seed
                decrypted_parts = []
                for ct in ciphertexts:
                    ct_clean = ct.replace(" ", "")
                    frac = reverse_columnar_transposition(
                        ct_clean, seed_key, padding=CONFIG.get("padding", True)
                    )
                    # Strip padding if enabled
                    if CONFIG.get("padding", True):
                        frac = frac.rstrip("X")
                    decrypted_parts.append(frac)

                combined = "".join(decrypted_parts)

                if len(combined) < 2:
                    scored_seeds.append((0.0, seed_order))
                    continue

                # Compute bigram IC
                bigrams = get_bigrams(combined)
                bigram_ic = compute_token_ic(bigrams)

                # Compute tetragram IC if text is long enough
                if len(combined) >= 4:
                    tetragrams = get_tetragrams(combined)
                    tetragram_ic = compute_token_ic(tetragrams)
                    # Scale tetragram IC to bigram range
                    tetragram_ic_scaled = tetragram_ic * tetragram_scale
                    # Combined IC with tetragram weight
                    combined_ic = (
                        1.0 - tetragram_weight
                    ) * bigram_ic + tetragram_weight * tetragram_ic_scaled
                else:
                    combined_ic = bigram_ic

                scored_seeds.append((combined_ic, seed_order))

            # Sort by combined IC (descending) and take top num_seeds
            scored_seeds.sort(key=lambda x: x[0], reverse=True)
            unique_seeds = [seed for _, seed in scored_seeds[:num_seeds]]

            if CONFIG.get("intermediate_output", True):
                print(
                    f"[FRAGMENT-VOTING] Selected top {len(unique_seeds)} seeds by combined IC "
                    f"(tetragram_weight={tetragram_weight:.3f})"
                )

        # If we don't have enough seeds, supplement with random ones
        while len(unique_seeds) < num_seeds:
            random_seed = list(range(key_length))
            _rnd.shuffle(random_seed)
            seed_tuple = tuple(random_seed)
            if seed_tuple not in seen:
                seen.add(seed_tuple)
                unique_seeds.append(random_seed)

        # Return requested number of seeds
        return unique_seeds[:num_seeds]

    # --- NEW: Check if random seeds are requested ---
    if CONFIG.get("random_seeds", False):
        import random as _rnd

        # Generate completely random shuffled orderings
        raw_seeds = max(1000, key_length * 100)
        random_orderings = []
        seen_tuples = set()

        # Generate random orderings until we have enough unique ones
        attempts = 0
        max_attempts = raw_seeds * 10  # Prevent infinite loop

        while len(random_orderings) < raw_seeds and attempts < max_attempts:
            attempts += 1
            ordering = list(range(key_length))
            _rnd.shuffle(ordering)

            # Deduplicate using tuple representation
            ordering_tuple = tuple(ordering)
            if ordering_tuple not in seen_tuples:
                seen_tuples.add(ordering_tuple)
                random_orderings.append(ordering)

        # Return requested number of seeds (or all generated if fewer)
        desired = max(num_seeds, min(len(random_orderings), num_seeds))
        return random_orderings[:desired]

    # --- EXISTING: Fragment adjacency analysis approach ---
    # Produce many raw seeds, expands them with small randomized local moves,
    # and returns up to `num_seeds` candidates (may contain similar/duplicate orders).
    if not ciphertexts:
        return [list(range(key_length))]

    # --- CHANGED: more aggressive raw seeds based on key length (key_length * 10) ---
    # Aim for a large diverse pool: base on key_length * 10, with a reasonable lower bound.
    raw_seeds = max(1000, key_length * 100)
    # if key_length >= 50:
    #     raw_seeds = max(raw_seeds, 200)

    # Extract and concatenate fragments from all ciphertexts
    all_fragments = [[] for _ in range(key_length)]
    for ct in ciphertexts:
        ct_clean = ct.replace(" ", "")
        fragments = extract_column_fragments(ct_clean, key_length)
        for i, frag in enumerate(fragments):
            all_fragments[i].append(frag)
    combined_fragments = ["".join(frags) for frags in all_fragments]

    # adjacency matrix
    adjacency_matrix = build_adjacency_matrix(combined_fragments)

    # --- CHANGED: beam width clamped into [8,12] for diversity ---
    beam_w = min(12, max(8, raw_seeds // 8))

    # Generate beam candidates and greedy
    beam_orderings = beam_fragment_ordering(adjacency_matrix, beam_width=beam_w)
    greedy_ordering = greedy_fragment_ordering(adjacency_matrix)

    # Start with greedy + beam, then diversify
    seeds: List[List[int]] = [greedy_ordering] + beam_orderings

    # If insufficient, add randomized orderings derived from adjacency scores
    # Also apply stronger diversification (more tries and broader moves) to each seed
    def diversify(ordering: List[int], tries: int = 6) -> Iterable[List[int]]:
        import random

        out = []
        m = len(ordering)
        for t in range(tries):
            o = ordering.copy()
            # multiple small random swaps
            swaps = 1 + (t % 3)
            for _ in range(swaps):
                i = random.randrange(0, m)
                j = random.randrange(0, m)
                o[i], o[j] = o[j], o[i]

            # occasional small rotation of a short segment
            if m >= 4 and random.random() < 0.6:
                a = random.randrange(0, max(1, m - 2))
                L = min(6, m - a)
                k = random.randrange(1, L)
                seg = o[a : a + L]
                o[a : a + L] = seg[k:] + seg[:k]

            # occasional short reversal
            if m >= 3 and random.random() < 0.4:
                a = random.randrange(0, m - 2)
                b = min(m, a + 3 + random.randrange(0, min(4, m - a)))
                o[a:b] = list(reversed(o[a:b]))

            # jitter: move a small element to another position
            if m >= 3 and random.random() < 0.5:
                i = random.randrange(0, m)
                val = o.pop(i)
                j = random.randrange(0, m - 1)
                o.insert(j, val)

            out.append(o)
        return out

    # Build expanded seed list (allow duplicates)
    expanded: List[List[int]] = []
    # ensure we include the greedy + beam candidates first
    for s in seeds:
        expanded.append(s)
        for v in diversify(s, tries=6):
            expanded.append(v)
        if len(expanded) >= raw_seeds:
            break

    # If still short, add some completely random orderings and structured variants
    import random as _rnd

    while len(expanded) < raw_seeds:
        o = list(range(key_length))
        _rnd.shuffle(o)
        # add a deterministic variant (reverse/interleave) occasionally to increase spread
        if len(expanded) % 5 == 0:
            expanded.append(o[::-1])
        elif len(expanded) % 7 == 0:
            # interleave halves
            a = o[: key_length // 2]
            b = o[key_length // 2 :]
            inter = [
                val
                for pair in zip(a, b + [None] * (len(a) - len(b)))
                for val in pair
                if val is not None
            ]
            if len(inter) == key_length:
                expanded.append(inter)
            else:
                expanded.append(o)
        else:
            expanded.append(o)

    # Reduce to requested num_seeds but if caller asked for more keep at least that many unique-ish seeds.
    # Respect the caller's `num_seeds` as a minimum; if caller passed a small value, still return varied pool.
    desired = max(num_seeds, min(len(expanded), num_seeds))
    # return the first `desired` entries (they are already diversified)
    final = expanded[:desired]
    return final


#
# Runners for larger algorithms (accept breaker instance)
#


def simulated_annealing_runner(
    breaker,
    ciphertexts: List[str],
    key_length: int,
    start_key=None,
    T_init: float = 1.0,
    T_min: float = 1e-4,
    alpha: float = 0.95,
    iterations_per_temp: int = 200,
    score_cache: Optional[Dict] = None,
    debug: bool = False,
):
    """Run simulated annealing using the provided breaker instance."""
    import math

    # initialize current order
    if start_key is None:
        current_order = breaker.random_order(key_length)
    else:
        if isinstance(start_key, str):
            current_order = breaker.key_string_to_column_order(start_key)
        elif isinstance(start_key, list):
            current_order = start_key.copy()
        else:
            current_order = breaker.random_order(key_length)

    current_key_string = breaker.string_key_from_column_order(current_order)
    current_score = breaker.score_key_transposition(
        ciphertexts, current_key_string, cache=score_cache
    )

    best_order = current_order.copy()
    best_score = current_score

    T = T_init
    while T > T_min:
        for _ in range(iterations_per_temp):
            transforms = breaker.generate_transformations(current_order)
            if not transforms:
                continue
            _, new_order = breaker.rng.choice(transforms)
            new_key_string = breaker.string_key_from_column_order(new_order)
            new_score = breaker.score_key_transposition(
                ciphertexts, new_key_string, cache=score_cache
            )
            delta = new_score - current_score
            if delta > 0 or breaker.rng.random() < math.exp(delta / T):
                current_order = new_order.copy()
                current_score = new_score
            if new_score > best_score:
                best_order = current_order.copy()
                best_score = current_score
                if debug:
                    try:
                        best_key_str = breaker.string_key_from_column_order(best_order)
                        best_key_digits = get_column_order(best_key_str)
                    except Exception:
                        best_key_str = str(best_order)
                        best_key_digits = best_order
                    print(
                        f"[SIM-ANNEAL] New best at T={T:.5f}: key='{best_key_str}' key_digits={best_key_digits} score={best_score:.6f}"
                    )
        T *= alpha

    best_key_string = breaker.string_key_from_column_order(best_order)
    if debug:
        try:
            final_key_digits = get_column_order(best_key_string)
        except Exception:
            final_key_digits = best_order
        print(
            f"[SIM-ANNEAL] Finished SA -> best_key='{best_key_string}' key_digits={final_key_digits} score={best_score:.6f}"
        )
    return best_key_string, best_score


def polish_key_runner(
    breaker,
    ciphertexts: List[str],
    start_key: str,
    max_iterations: int = 1000,
    score_cache: Optional[Dict] = None,
    debug: bool = False,
):
    """Best-improving hill-climb starting from an explicit key (no restarts)."""
    if score_cache is None:
        score_cache = {}
    batch = [ct.replace(" ", "") for ct in ciphertexts]
    order = (
        breaker.key_string_to_column_order(start_key)
        if isinstance(start_key, str)
        else start_key.copy()
    )
    key_string = breaker.string_key_from_column_order(order)
    cur_score = breaker.score_key_transposition(batch, key_string, cache=score_cache)

    iteration = 0
    improved = True
    while improved and iteration < max_iterations:
        iteration += 1
        improved = False
        best_iter_score = cur_score
        best_iter_order = order
        for _, cand_order in breaker.generate_transformations(order):
            cand_key = breaker.string_key_from_column_order(cand_order)
            cand_score = breaker.score_key_transposition(
                batch, cand_key, cache=score_cache
            )
            if cand_score > best_iter_score:
                best_iter_score = cand_score
                best_iter_order = cand_order
            elif abs(cand_score - best_iter_score) <= 1e-8:
                if breaker.rng.choice([True, False]):
                    best_iter_order = cand_order
        if best_iter_score > cur_score:
            order = best_iter_order
            key_string = breaker.string_key_from_column_order(order)
            cur_score = best_iter_score
            improved = True
    return breaker.string_key_from_column_order(order), cur_score


def polish_key_with_alphabet_runner(
    breaker,
    ciphertexts: List[str],
    start_key: str,
    alphabet: str,
    max_iterations: int = 500,
    debug: bool = False,
):
    """Hill-climb scoring candidates by decoding with the provided Polybius alphabet."""
    try:
        _, inv_map = create_polybius_square(alphabet)
    except Exception:
        _, inv_map = create_polybius_square("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    def score_order_by_alphabet(key_string: str) -> float:
        parts = []
        for ct in ciphertexts:
            ct_clean = ct.replace(" ", "")
            frac = reverse_columnar_transposition(
                ct_clean, key_string, padding=breaker.padding
            )
            frac = frac.rstrip("X") if breaker.padding else frac
            out = []
            for i in range(0, len(frac), 2):
                if i + 1 >= len(frac):
                    break
                pair = frac[i : i + 2]
                out.append(inv_map.get(pair, "?"))
            parts.append("".join(out))
        return score_texts(parts)

    order = (
        breaker.key_string_to_column_order(start_key)
        if isinstance(start_key, str)
        else start_key.copy()
    )
    key_string = breaker.string_key_from_column_order(order)
    cur_score = score_order_by_alphabet(key_string)

    iteration = 0
    improved = True
    while improved and iteration < max_iterations:
        iteration += 1
        improved = False
        best_iter_score = cur_score
        best_iter_order = order
        for _, cand_order in breaker.generate_transformations(order):
            cand_key = breaker.string_key_from_column_order(cand_order)
            cand_score = score_order_by_alphabet(cand_key)
            if cand_score > best_iter_score:
                best_iter_score = cand_score
                best_iter_order = cand_order
            elif abs(cand_score - best_iter_score) <= 1e-8:
                if breaker.rng.choice([True, False]):
                    best_iter_order = cand_order
        if best_iter_score > cur_score:
            order = best_iter_order
            key_string = breaker.string_key_from_column_order(order)
            cur_score = best_iter_score
            improved = True

    return breaker.string_key_from_column_order(order), cur_score


def hill_climb_sequential_runner(
    breaker,
    ciphertexts: List[str],
    key_length: int,
    restarts: int = 100,
    max_iterations: int = 1000,
    ic_threshold: float = 0.05,
    use_hybrid: bool = False,
    intermediate_output: bool = False,
    debug: bool = False,
    ic_earlystop_min_restarts: int = 10,
    score_eps: float = 1e-8,
):
    """Sequential hill-climb implementation extracted from ADFGVXBreaker.hill_climb."""
    best_global_key = None
    best_global_score = float("-inf")
    batch = [ct.replace(" ", "") for ct in ciphertexts]
    per_restart_results = []

    # dynamic restart addition percentage (read from CONFIG)
    total_restarts = restarts
    dynamic_pct = int(CONFIG.get("dynamic_restart_addition", 0) or 0)
    # number of restarts to add when triggered (derived from original total_restarts)
    add_restarts_base = (
        max(1, int(total_restarts * dynamic_pct / 100)) if dynamic_pct > 0 else 0
    )
    # threshold: how many restarts must have passed since previous best for an addition
    threshold_count = (
        max(1, int(total_restarts * dynamic_pct / 100)) if dynamic_pct > 0 else 0
    )

    # track when the previous global-best was last updated (restart index)
    prev_best_update = -1

    # NEW: Check if we have injected seed orderings and should diversify them
    injected = getattr(breaker, "_injected_seed_orderings", []) or []
    use_nondeterministic_rng = CONFIG.get(
        "non_deterministic_RNG_seed_per_restart", False
    )

    # NEW: Create perturbed variants if we have injected seeds
    restart_orderings = []
    if injected:
        variants_per_seed = min(6, max(3, key_length // 4))
        if intermediate_output:
            print(
                f"[DIVERSIFY-SEQ] Creating {variants_per_seed} variants per seed "
                f"({'non-deterministic' if use_nondeterministic_rng else 'deterministic'} RNG)"
            )

        for seed_order in injected:
            # Base variant
            restart_orderings.append(seed_order.copy())

            # Perturbed variants
            for variant_idx in range(1, variants_per_seed):
                perturbed = seed_order.copy()

                # Apply same perturbation logic as parallel version
                if variant_idx == 1 and key_length >= 2:
                    i, j = breaker.rng.sample(range(key_length), 2)
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
                elif variant_idx == 2 and key_length >= 4:
                    pos = breaker.rng.randrange(0, key_length - 3)
                    perturbed[pos], perturbed[pos + 1] = (
                        perturbed[pos + 1],
                        perturbed[pos],
                    )
                    perturbed[pos + 2], perturbed[pos + 3] = (
                        perturbed[pos + 3],
                        perturbed[pos + 2],
                    )
                elif variant_idx == 3 and key_length >= 3:
                    seg_len = min(4, key_length // 2)
                    start = breaker.rng.randrange(0, key_length - seg_len + 1)
                    perturbed[start : start + seg_len] = perturbed[
                        start : start + seg_len
                    ][::-1]
                elif variant_idx == 4 and key_length >= 4:
                    seg_len = min(5, key_length // 2)
                    start = breaker.rng.randrange(0, key_length - seg_len + 1)
                    k = breaker.rng.randrange(1, seg_len)
                    seg = perturbed[start : start + seg_len]
                    perturbed[start : start + seg_len] = seg[k:] + seg[:k]
                else:
                    for _ in range(2):
                        if key_length >= 2:
                            i, j = breaker.rng.sample(range(key_length), 2)
                            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

                restart_orderings.append(perturbed)

                if len(restart_orderings) >= restarts:
                    break

            if len(restart_orderings) >= restarts:
                break

    r = 0
    current_restarts = restarts

    while r < current_restarts:
        eval_cache: Dict[str, float] = {}
        if intermediate_output and r > 0 and (r % 500) == 0:
            print(f"[STATUS] restart={r}")

        if r % 10 == 0 and r > 0 and debug:
            print(
                f"Hill-climb restart {r}/{current_restarts}, best_global_score={best_global_score:.6f}"
            )

        # NEW: Use perturbed ordering if available, else generate random
        if r < len(restart_orderings):
            order = restart_orderings[r].copy()
        else:
            order = list(range(key_length))
            if r % 5 == 0:
                pass
            elif r % 5 == 1:
                order.reverse()
            else:
                breaker.rng.shuffle(order)

        key_string = breaker.string_key_from_column_order(order)
        cur_score = breaker.score_key_transposition(batch, key_string, cache=eval_cache)

        # preliminary pass
        current_improved = True
        while current_improved:
            current_improved = False
            transforms = breaker.generate_transformations(order)
            for name, cand_order in transforms:
                cand_key = breaker.string_key_from_column_order(cand_order)
                cand_score = breaker.score_key_transposition(
                    batch, cand_key, cache=eval_cache
                )
                if is_better(cand_score, cur_score):
                    order = cand_order
                    key_string = cand_key
                    cur_score = cand_score
                    current_improved = True
                    if intermediate_output:
                        col_order = get_column_order(key_string)
                        # Only emit the verbose PRE_PASS line when debug mode is enabled.
                        if debug:
                            print(
                                f"[PRE_PASS] restart={r} accepted {name} -> score={cur_score:.6f} key='{key_string}' key_digits={col_order}"
                            )
                    break

        # main HC loop
        iteration = 0
        improved = True
        while improved and iteration < max_iterations:
            iteration += 1
            improved = False
            transforms = breaker.generate_transformations(order)
            best_iter_score = cur_score
            best_candidates = []
            for name, cand_order in transforms:
                cand_key = breaker.string_key_from_column_order(cand_order)
                cand_score = breaker.score_key_transposition(
                    batch, cand_key, cache=eval_cache
                )
                if is_better(cand_score, best_iter_score + score_eps):
                    best_iter_score = cand_score
                    best_candidates = [cand_order]
                elif is_tie(cand_score, best_iter_score):
                    best_candidates.append(cand_order)

            if is_better(best_iter_score, cur_score) and best_candidates:
                best_iter_order = breaker.rng.choice(best_candidates)
                order = best_iter_order
                key_string = breaker.string_key_from_column_order(order)
                cur_score = best_iter_score
                improved = True

        # short SA
        if use_hybrid:
            try:
                sa_key, sa_score = simulated_annealing_runner(
                    breaker,
                    batch,
                    key_length,
                    start_key=key_string,
                    T_init=0.25,
                    T_min=0.01,
                    alpha=0.90,
                    iterations_per_temp=60,
                )
                polish_key_str, polish_score = polish_key_runner(
                    breaker, batch, sa_key, max_iterations=200
                )
                if polish_score > cur_score:
                    order = breaker.key_string_to_column_order(polish_key_str)
                    key_string = polish_key_str
                    cur_score = polish_score
                    if intermediate_output:
                        print(
                            f"[SHORT-SA] restart={r} accepted SA+polish -> key='{key_string}' score={cur_score:.6f}"
                        )
            except Exception as e:
                if debug:
                    print(f"Short SA failed on restart {r}: {e}")

        per_restart_results.append((cur_score, key_string))

        # Publish incremental per-restart candidates after each restart so an
        # external KeyboardInterrupt handler can read partial results.
        try:
            if not hasattr(breaker, "_per_length_candidates"):
                breaker._per_length_candidates = {}
            breaker._per_length_candidates[key_length] = per_restart_results.copy()
        except Exception:
            # be tolerant: do not let bookkeeping errors stop the search
            pass

        # Check for new global best and apply dynamic restart logic
        if cur_score > best_global_score:
            # If we have a previous best, check its age relative to threshold
            if prev_best_update != -1 and dynamic_pct > 0:
                age = r - prev_best_update
                if age >= threshold_count:
                    add_restarts = add_restarts_base
                    current_restarts += add_restarts
                    if intermediate_output:
                        remaining = max(0, current_restarts - r - 1)
                        print(
                            f"[DYNAMIC] Found new global-best at restart={r}; previous best was {age} restarts ago >= threshold {threshold_count}. "
                            f"Extending total restarts by {add_restarts}. Total remaining restarts now: {remaining}"
                        )
            # update bookkeeping for best found
            best_global_score = cur_score
            best_global_key = key_string
            prev_best_update = r

            if intermediate_output:
                print(
                    f"[UPDATE] restart={r} new best score={best_global_score:.6f} key='{best_global_key}'"
                )

        # EARLY STOP: allow immediate early stop when configured (do not require min_restarts)
        if (
            CONFIG.get("early_stop_if_ic_threshold_reached", False)
            and best_global_score >= ic_threshold - score_eps
        ):
            if intermediate_output:
                print(
                    f"Early stop at restart {r}, score {best_global_score:.6f} >= threshold {ic_threshold} (early_stop_if_ic_threshold_reached enabled)"
                )
            break

        # existing conditional early stop that depends on minimum re


#
# CLASSES
#


class GlobalLeaderboard:
    """
    Maintains a sorted list of best key candidates across all search phases.
    Prevents duplicate work and enables dynamic prioritization.
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.candidates = []  # List of (score, key_string) tuples
        self.seen_keys = set()  # Track unique keys to prevent duplicates

    def add_candidate(self, score: float, key_string: str) -> bool:
        """
        Add a candidate to the leaderboard. Returns True if added/updated.
        Maintains sorted order (descending by score) and prevents duplicates.
        """
        if not key_string:
            return False

        # Check if key already exists
        if key_string in self.seen_keys:
            # Update score if better
            for i, (old_score, old_key) in enumerate(self.candidates):
                if old_key == key_string:
                    if score > old_score:
                        self.candidates[i] = (score, key_string)
                        self._resort()
                        return True
                    return False

        # Add new candidate
        self.candidates.append((score, key_string))
        self.seen_keys.add(key_string)

        # Sort and trim to max_size
        self._resort()
        if len(self.candidates) > self.max_size:
            removed = self.candidates[self.max_size :]
            self.candidates = self.candidates[: self.max_size]
            # Remove keys that are no longer in leaderboard
            for _, removed_key in removed:
                self.seen_keys.discard(removed_key)

        return True

    def _resort(self):
        """Sort candidates by score (descending)."""
        self.candidates.sort(key=lambda x: x[0], reverse=True)

    def get_top_n(self, n: int) -> List[Tuple[float, str]]:
        """Return top N candidates."""
        return self.candidates[:n]

    def get_best(self) -> Optional[Tuple[float, str]]:
        """Return the best candidate."""
        return self.candidates[0] if self.candidates else None

    def size(self) -> int:
        """Return current number of candidates."""
        return len(self.candidates)

    def clear(self):
        """Clear all candidates."""
        self.candidates.clear()
        self.seen_keys.clear()
