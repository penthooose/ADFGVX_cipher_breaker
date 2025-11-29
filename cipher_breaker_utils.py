"""Utility helpers for cipher_breaker.

This module contains stateless helper functions moved out of
`cipher_breaker.py` to keep that file focused on core classes and
orchestration logic.
"""

import random
import concurrent.futures
from typing import List, Tuple, Dict, Optional, Iterable
from collections import Counter
import math
from cipher_implementation import (
    get_all_possible_key_orders,
    create_polybius_square,
    reverse_columnar_transposition,
    get_column_order,
)
import multiprocessing

from cipher_breaker_helpers import (
    GlobalLeaderboard,
    set_config_helpers,
    infer_fragment_seeding_variables,
    compute_token_ic,
    get_bigrams,
    get_tetragrams,
    string_key_from_column_order,
    adfgvx_pair_regularity,
    position_entropy_score,
    generate_transformations,
    reconstruct_long_key_seeds,
    fractionated_to_plain_with_map,
    score_texts,
    simulated_annealing_runner,
    polish_key_runner,
    polish_key_with_alphabet_runner,
    hill_climb_sequential_runner,
    infer_key_search_variables,
    is_better,
    is_tie,
)


CONFIG: Dict = {}


# Use a plain assignment with a type comment to avoid referencing a runtime
# symbol in the annotation which some type checkers reject.
_STOP_EVENT = None  # type: Optional[object]
_KEYBOARD_INTERRUPT_FLAG = False  # Track keyboard interrupts globally


def set_keyboard_interrupt() -> None:
    """Mark that a KeyboardInterrupt was seen (process-global flag)."""
    global _KEYBOARD_INTERRUPT_FLAG
    _KEYBOARD_INTERRUPT_FLAG = True


def clear_keyboard_interrupt() -> None:
    """Clear the KeyboardInterrupt flag (useful for tests or controlled runs)."""
    global _KEYBOARD_INTERRUPT_FLAG
    _KEYBOARD_INTERRUPT_FLAG = False


def was_keyboard_interrupted() -> bool:
    """Query whether a KeyboardInterrupt has been signalled."""
    return bool(_KEYBOARD_INTERRUPT_FLAG)


def _worker_initializer(stop_event, config_dict):
    """Initialize worker process with shared stop event and CONFIG."""
    global _STOP_EVENT
    _STOP_EVENT = stop_event
    # Initialize worker's CONFIG with parent's settings
    set_config(config_dict)
    set_config_helpers(config_dict)


def _should_stop() -> bool:
    """Check if worker should stop (cooperative cancellation)."""
    return _STOP_EVENT is not None and _STOP_EVENT.is_set()


def set_config(cfg: Dict):
    """Initialize module-level CONFIG (copy) so utils use the same settings as caller."""
    global CONFIG
    if isinstance(cfg, dict):
        CONFIG = cfg.copy()
    else:
        # accept any mapping-like object
        CONFIG = dict(cfg)


def debug(*args, **kwargs):
    """Print only when CONFIG['debug_output'] is True."""
    if CONFIG.get("debug_output", False):
        print(*args, **kwargs)


def get_tetragram_weight(key_length: int) -> float:
    """
    Compute tetragram weight based on key length with continuous scaling.
    Applies even-key additive boost when applicable.

    Parameters from CONFIG:
    - tetragram_base_weight: base weight (default 0.5)
    - tetragram_boost_start_length: start boosting at this length (default 12)
    - tetragram_boost_per_column: boost per extra column (default 0.02)
    - tetragram_max_weight: maximum weight cap (default 0.85)

    Per-length parameters (from infer_fragment_seeding_variables):
    - tetragram_even_key_additive_value: absolute value added for even keys (if set)

    Returns:
        float: tetragram weight in range [base_weight, max_weight]
    """
    base_weight = CONFIG.get("tetragram_base_weight", 0.5)
    boost_start = CONFIG.get("tetragram_boost_start_length", 12)
    boost_per_col = CONFIG.get("tetragram_boost_per_column", 0.02)
    max_weight = CONFIG.get("tetragram_max_weight", 0.85)

    # Start with base weight
    weight = base_weight

    # Add continuous scaling based on length
    if key_length >= boost_start:
        extra_columns = key_length - boost_start
        weight += extra_columns * boost_per_col

    # Apply even-key additive boost if configured for this key length
    if key_length % 2 == 0:
        # Check if there's a per-length additive value from infer_fragment_seeding_variables
        phase_params = infer_fragment_seeding_variables(key_length)
        even_additive = phase_params.get("tetragram_even_key_additive_value", 0.0)
        weight += even_additive

    # Cap at max weight
    weight = min(weight, max_weight)

    return weight


PROVIDED_KEY_HINT: Optional[str] = None


def set_provided_key_hint(hint: Optional[str]) -> Optional[str]:
    """Set the current provided key hint used by utils. Returns previous value."""
    global PROVIDED_KEY_HINT
    prev = PROVIDED_KEY_HINT
    PROVIDED_KEY_HINT = hint
    return prev


def get_provided_key_hint() -> Optional[str]:
    """Return the currently set provided key hint (or None)."""
    return PROVIDED_KEY_HINT


class ADFGVXBreaker:
    """
    Lasry-style ADFGVX breaker (transposition recovery phase).
    - Uses IC of overlapping bigrams and (for even key lengths) tetragrams.
    - Uses Lasry's five transformation types.
    - Best-improving-move hill-climb with optional simulated annealing restarts.
    - If `self.padding` is True, trailing 'X' characters (common padding) are stripped
      from decrypted fractionated streams before scoring.
    """

    def __init__(self, config: Optional[Dict] = None, padding: Optional[bool] = None):
        # If a caller supplies a config dict, initialise module CONFIG so
        # utility functions use the same settings. This keeps backward
        # compatibility with calls that don't pass a config.
        if config is not None:
            set_config(config)
            set_config_helpers(config)
        self.adfgvx_chars = ["A", "D", "F", "G", "V", "X"]
        # default padding is taken from CONFIG when not explicitly provided
        self.padding = padding if padding is not None else CONFIG.get("padding", True)
        self.rng = random.Random()  # Add RNG instance attribute

        # Bind module-level helper functions to the instance so calls like
        # self.get_bigrams(...) and self.generate_transformations(...) work.
        # These functions are defined later in this module; binding them here
        # makes the instance self-contained and avoids AttributeError.
        self.get_bigrams = get_bigrams
        self.get_tetragrams = get_tetragrams
        self.compute_token_ic = compute_token_ic
        self.generate_transformations = generate_transformations
        # wrap adfgvx_pair_regularity to supply this instance's char set
        self._adfgvx_pair_regularity = lambda frac: adfgvx_pair_regularity(
            frac, self.adfgvx_chars
        )
        # position entropy scoring function
        self._position_entropy_score = position_entropy_score

        # Initialize global leaderboard if dynamic updates enabled
        if CONFIG.get("update_global_best_candidates_dynamically", False):
            leaderboard_size = max(
                CONFIG.get("best_candidates_phase_2", 20) * 3,
                CONFIG.get("best_candidates_phase_3", 2) * 10,
                100,
            )
            self._global_leaderboard = GlobalLeaderboard(max_size=leaderboard_size)
        else:
            self._global_leaderboard = None

        debug(f"ADFGVXBreaker(padding={'ENABLED' if self.padding else 'DISABLED'})")

    def string_key_from_column_order(self, column_order: List[int]) -> str:
        """
        Convert a column order (list of column indices in reading order)
        into a string key such that get_column_order(key) == column_order.
        """
        key = string_key_from_column_order(column_order)
        debug(f"string_key_from_column_order: {column_order} -> '{key}'")
        return key

    def key_string_to_column_order(self, key_string: str) -> List[int]:
        """Convert a string key (e.g. 'MATH') to a column-order list using get_column_order."""
        return get_column_order(key_string)

    def random_order(self, key_length: int) -> List[int]:
        """Return a random column-order list of given length."""
        order = list(range(key_length))
        # use instance RNG for deterministic behavior
        self.rng.shuffle(order)
        return order

    # ---------------------------
    # Scoring
    # ---------------------------
    def _normalize_decrypted(self, fractionated: str) -> str:
        """
        Optionally strip padding. If padding enabled, remove trailing 'X' characters
        that likely correspond to rectangle padding. This is deliberately conservative:
        only trailing Xs are removed (so interior X's remain).
        """
        if not self.padding:
            return fractionated
        return fractionated.rstrip("X")

    def score_key_transposition(
        self,
        batch: List[str],
        key_string: str,
        cache: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        IC-only scorer for the transposition stage with length-dependent tetragram weighting.
        - batch: list of ciphertext strings with spaces removed (cleaned)
        - key_string: transposition key in letter form
        - cache: optional dict mapping (key_string) -> score to avoid duplicate work
        Returns combined bigram + tetragram IC (higher is better).
        """
        if cache is None:
            cache = {}
        # use key_string as cache key (caller ensures batch consistency)
        if key_string in cache:
            return cache[key_string]

        decrypted_parts = []
        for ct in batch:
            # reverse_columnar_transposition expects cleaned ciphertext
            frac = reverse_columnar_transposition(ct, key_string, padding=self.padding)
            frac = self._normalize_decrypted(frac)
            decrypted_parts.append(frac)
        combined = "".join(decrypted_parts)
        if len(combined) < 2:
            cache[key_string] = 0.0
            return 0.0

        # Compute bigram IC
        bigrams = self.get_bigrams(combined)
        bigram_ic = self.compute_token_ic(bigrams)

        # Compute tetragram IC if text is long enough
        if len(combined) >= 4:
            tetragrams = self.get_tetragrams(combined)
            tetragram_ic = self.compute_token_ic(tetragrams)

            # Get length-dependent tetragram weight
            key_length = len(key_string)
            tetragram_weight = get_tetragram_weight(key_length)
            bigram_weight = 1.0 - tetragram_weight

            # --- CONTINUOUS PAIR-REGULARITY SCALING (always active) ---
            # Scale tetragram weight by pair-regularity strength to reward true ADFGVX structure.
            # This ensures keys with stronger pair-regularity get higher tetragram contribution.
            pair_reg = self._adfgvx_pair_regularity(combined)

            # Use pair_reg directly as a continuous multiplier (range 0..1)
            # Add small floor to prevent complete zeroing when pair_reg is very low
            pair_reg_floor = (
                0.1  # minimum 10% tetragram weight even with terrible pair_reg
            )
            pair_reg_factor = max(pair_reg_floor, pair_reg)

            # Apply continuous scaling (tetragram gets full weight only when pair_reg ≈ 1.0)
            tetragram_weight = tetragram_weight * pair_reg_factor
            bigram_weight = 1.0 - tetragram_weight

            # Ensure weights are sane
            tetragram_weight = min(max(tetragram_weight, 0.0), 1.0)
            bigram_weight = 1.0 - tetragram_weight

            # Tetragram IC values are typically ~10-20x smaller than bigram IC
            # for the same text, so we scale up to compensate
            # Use per-length scale factor with fallback to CONFIG
            try:
                phase_params = infer_fragment_seeding_variables(key_length)
                tetragram_scale = phase_params.get(
                    "tetragram_ic_scale_factor",
                    CONFIG.get("tetragram_ic_scale_factor", 15.0),
                )
            except Exception:
                tetragram_scale = CONFIG.get("tetragram_ic_scale_factor", 15.0)

            scaled_tetragram_ic = tetragram_ic * tetragram_scale

            # Combined score with scaled tetragram IC and length-dependent weighting
            score = bigram_weight * bigram_ic + tetragram_weight * scaled_tetragram_ic

            if CONFIG.get("debug_output", False):
                debug(
                    f"[IC-SCORE] key_len={key_length} bigram_ic={bigram_ic:.6f} "
                    f"tetragram_ic={tetragram_ic:.6f} scaled_tetragram_ic={scaled_tetragram_ic:.6f} "
                    f"weights=({bigram_weight:.3f}, {tetragram_weight:.3f}) "
                    f"score={score:.6f} even_key={key_length % 2 == 0}"
                )
        else:
            score = bigram_ic

        cache[key_string] = score
        return score

    def score_key(self, ciphertexts: List[str], key_string: str) -> float:
        """
        Score a transposition key using multiple complementary metrics:
        - Overlapping bigram IC and tetragram IC (primary) - with length-dependent weighting
        - ADFGVX pair regularity (secondary)
        - Entropy variance across positions (tertiary)
        Returns combined score (higher = better).
        """
        # Decrypt and concat (remove spaces first)
        decrypted_parts = []
        for ct in ciphertexts:
            ct_clean = ct.replace(" ", "")
            frac = reverse_columnar_transposition(
                ct_clean, key_string, padding=self.padding
            )
            frac = self._normalize_decrypted(frac)
            decrypted_parts.append(frac)
        combined = "".join(decrypted_parts)

        if len(combined) < 2:
            return 0.0

        # Primary scoring: bigram and tetragram IC (with length-dependent weighting)
        bigrams = self.get_bigrams(combined)
        bigram_ic = self.compute_token_ic(bigrams)

        if len(combined) >= 4:
            tetragrams = self.get_tetragrams(combined)
            tetragram_ic = self.compute_token_ic(tetragrams)

            # Get length-dependent tetragram weight (same as score_key_transposition)
            key_length = len(key_string)
            tetragram_weight = get_tetragram_weight(key_length)
            bigram_weight = 1.0 - tetragram_weight

            # --- CONTINUOUS PAIR-REGULARITY SCALING (always active) ---
            # Scale tetragram weight by pair-regularity strength to reward true ADFGVX structure.
            # This ensures keys with stronger pair-regularity get higher tetragram contribution.
            pair_reg = self._adfgvx_pair_regularity(combined)

            # Use pair_reg directly as a continuous multiplier (range 0..1)
            # Add small floor to prevent complete zeroing when pair_reg is very low
            pair_reg_floor = (
                0.1  # minimum 10% tetragram weight even with terrible pair_reg
            )
            pair_reg_factor = max(pair_reg_floor, pair_reg)

            # Apply continuous scaling (tetragram gets full weight only when pair_reg ≈ 1.0)
            tetragram_weight = tetragram_weight * pair_reg_factor
            bigram_weight = 1.0 - tetragram_weight

            # Ensure weights are sane
            tetragram_weight = min(max(tetragram_weight, 0.0), 1.0)
            bigram_weight = 1.0 - tetragram_weight

            # Scale tetragram IC to bigram range (same as score_key_transposition)
            # Use per-length scale factor with fallback to CONFIG
            try:
                phase_params = infer_fragment_seeding_variables(key_length)
                tetragram_scale = phase_params.get(
                    "tetragram_ic_scale_factor",
                    CONFIG.get("tetragram_ic_scale_factor", 15.0),
                )
            except Exception:
                tetragram_scale = CONFIG.get("tetragram_ic_scale_factor", 15.0)

            scaled_tetragram_ic = tetragram_ic * tetragram_scale

            # Combined score with length-dependent weighting
            ic_score = (
                bigram_weight * bigram_ic + tetragram_weight * scaled_tetragram_ic
            )
        else:
            ic_score = bigram_ic
            # Compute pair_reg even for short texts (needed below)
            pair_reg = self._adfgvx_pair_regularity(combined)

        # Secondary: ADFGVX pair regularity bonus
        # Correct decryption should have valid ADFGVX pairs (even positions match odd positions in distribution)
        pair_regularity = self._adfgvx_pair_regularity(combined)

        # Tertiary: Position entropy variance
        # Correct decryption should have lower variance in symbol entropy across even/odd positions
        entropy_score = self._position_entropy_score(combined)

        # Combine with weights (IC is primary). Scale secondary bonuses relative to ic_score
        # so they do not dominate raw IC magnitude. If ic_score is tiny, keep small additive floor.
        bonus_scale = max(ic_score, 1e-6)
        total_score = ic_score * (1.0 + 0.5 * pair_regularity + 0.35 * entropy_score)
        # ensure tiny positive floor to avoid zero scores
        total_score = max(total_score, ic_score + 1e-9)

        debug(
            f"score_key: key={key_string} ic={ic_score:.6f} pair_reg={pair_regularity:.6f} "
            f"entropy={entropy_score:.6f} total={total_score:.6f}"
        )

        return total_score

    # ---------------------------
    # Simulated annealing (optional)
    # ---------------------------
    def simulated_annealing(
        self,
        ciphertexts: List[str],
        key_length: int,
        start_key=None,
        T_init: float = 1.0,
        T_min: float = 1e-4,
        alpha: float = 0.95,
        iterations_per_temp: int = 200,
        score_cache: Optional[Dict] = None,
    ):
        """
        Simulated annealing starting from either a random key (None) or a provided key string.
        Operates on column-order lists internally; returns (key_string, score).
        Uses self.rng for randomness to keep behavior consistent with hill-climb.
        """
        return simulated_annealing_runner(
            breaker=self,
            ciphertexts=ciphertexts,
            key_length=key_length,
            start_key=start_key,
            T_init=T_init,
            T_min=T_min,
            alpha=alpha,
            iterations_per_temp=iterations_per_temp,
            score_cache=score_cache,
            debug=CONFIG.get("debug_output", False),
        )

    def hybrid_search(
        self,
        ciphertexts: List[str],
        key_length: int,
        restarts: int = 100,
        max_iterations: int = 1000,
        ic_threshold: float = 0.05,
    ):
        """
        1. Run hill-climb
        2. Run simulated annealing to escape local optimum
        3. Run hill-climb again to polish
        """
        hc_key, hc_score = self.hill_climb(
            ciphertexts,
            key_length,
            restarts=restarts,
            max_iterations=max_iterations,
            ic_threshold=ic_threshold,
        )

        sa_key, sa_score = self.simulated_annealing(
            ciphertexts, key_length, start_key=hc_key
        )

        polish_key, polish_score = self.polish_key(
            ciphertexts, sa_key, max_iterations=max_iterations
        )

        best_key, best_score = max(
            [(hc_key, hc_score), (sa_key, sa_score), (polish_key, polish_score)],
            key=lambda item: item[1],
        )
        return best_key, best_score

    def polish_key(
        self,
        ciphertexts: List[str],
        start_key: str,
        max_iterations: int = 1000,
        score_cache: Optional[Dict] = None,
    ) -> Tuple[str, float]:
        """
        Best-improving hill-climb starting from an explicit key (no random restarts).
        Uses IC-only scorer and optional shared cache.
        """
        return polish_key_runner(
            breaker=self,
            ciphertexts=ciphertexts,
            start_key=start_key,
            max_iterations=max_iterations,
            score_cache=score_cache,
            debug=CONFIG.get("debug_output", False),
        )

    def polish_key_with_alphabet(
        self,
        ciphertexts: List[str],
        start_key: str,
        alphabet: str,
        max_iterations: int = 500,
    ) -> Tuple[str, float]:
        """
        Hill-climb the column-order starting from start_key but score candidates by
        decoding with the provided Polybius `alphabet` and using english_score_texts.
        Returns (best_key_string, best_score).
        """
        return polish_key_with_alphabet_runner(
            breaker=self,
            ciphertexts=ciphertexts,
            start_key=start_key,
            alphabet=alphabet,
            max_iterations=max_iterations,
            debug=CONFIG.get("debug_output", False),
        )

    # ---------------------------
    # Hill-climb (best-improving move)
    # ---------------------------
    def hill_climb(
        self,
        ciphertexts: List[str],
        key_length: int,
        restarts: int = 100,
        max_iterations: int = 1000,
        ic_threshold: float = 0.05,
        use_hybrid: bool = False,
    ) -> Tuple[Optional[str], float]:
        """
        Best-improving hill-climb over Lasry neighborhood with parallel processing.
        Returns (best_key_string, best_score).
        """
        best_global_key: Optional[str] = None
        best_global_score: float = float("-inf")

        # prepare ciphertext batch: cleaned (spaces removed)
        batch = [ct.replace(" ", "") for ct in ciphertexts]

        # collect per-restart final results for intermediate reporting
        per_restart_results: List[Tuple[float, Optional[str]]] = []

        # Check if parallel processing is enabled
        key_search_workers = CONFIG.get("key_search_workers", 1)
        use_parallel = key_search_workers > 1 and restarts > 1

        # Check if hybrid mode is enabled (determines whether SA passes run)
        use_hybrid = CONFIG.get("use_hybrid", False)

        if use_parallel:
            if CONFIG.get("intermediate_output", True):
                print(
                    f"[KEY-SEARCH] Using parallel processing with {key_search_workers} workers for {restarts} restarts and {max_iterations} max. iterations"
                )

            # Prepare worker arguments (now include use_hybrid flag and optional start_order)
            worker_args = []

            # If the breaker has injected seed orderings, place them first (up to restarts)
            injected = getattr(self, "_injected_seed_orderings", []) or []
            num_injected = min(len(injected), restarts)

            use_nondeterministic_rng = CONFIG.get(
                "non_deterministic_RNG_seed_per_restart", False
            )

            # For each injected seed, create multiple perturbed variants
            if num_injected > 0:
                # FIXED: Create enough variants to fill all restarts
                variants_per_seed = (
                    restarts // num_injected if num_injected > 0 else restarts
                )
                variants_per_seed = max(variants_per_seed, 10)  # At least 10 per seed

                # Determine perturbation intensity based on key length
                # For key_length=18, this gives ~4 operations per variant
                num_ops = max(2, min(key_length // 4, 8))

                if CONFIG.get("intermediate_output", True):
                    print(
                        f"[DIVERSIFY] Creating {variants_per_seed} perturbed variants per injected seed "
                        f"(num_ops={num_ops}, target {restarts} workers) "
                        f"({'non-deterministic' if use_nondeterministic_rng else 'deterministic'} RNG)"
                    )

                for seed_idx, seed_order in enumerate(injected):
                    # Base variant: use the seed as-is
                    if use_nondeterministic_rng:
                        # Salt with seed index and variant number for unique RNG behavior
                        seed_int = (
                            sum((v + 1) * (i + 31) for i, v in enumerate(seed_order))
                            ^ (seed_idx * 997)
                        ) & ((1 << 30) - 1)
                    else:
                        seed_int = sum(
                            (v + 1) * (i + 31) for i, v in enumerate(seed_order)
                        ) & ((1 << 30) - 1)

                    worker_args.append(
                        (
                            batch,
                            key_length,
                            seed_int,
                            max_iterations,
                            ic_threshold,
                            self.padding,
                            CONFIG.get("debug_output", False),
                            use_hybrid,
                            seed_order,  # pass start_order into worker
                        )
                    )

                    # Track unique variants with deduplication
                    seen_variants = {tuple(seed_order)}

                    # Create perturbed variants using controlled randomness
                    for variant_idx in range(1, variants_per_seed):
                        perturbed = seed_order.copy()

                        # Apply random number of perturbation operations
                        # Randomize: how many ops (within range)?
                        actual_ops = self.rng.randint(max(1, num_ops - 2), num_ops + 2)

                        for op_num in range(actual_ops):
                            # Randomize: which operation type?
                            op_type = self.rng.randint(0, 4)  # 5 operation types

                            if op_type == 0 and key_length >= 2:
                                # Random swap of two elements
                                i, j = self.rng.sample(range(key_length), 2)
                                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]

                            elif op_type == 1 and key_length >= 3:
                                # Reverse a random segment (randomize segment length)
                                seg_len = self.rng.randint(2, min(6, key_length // 2))
                                start = self.rng.randrange(0, key_length - seg_len + 1)
                                perturbed[start : start + seg_len] = perturbed[
                                    start : start + seg_len
                                ][::-1]

                            elif op_type == 2 and key_length >= 4:
                                # Rotate a random segment (randomize segment length & rotation amount)
                                seg_len = self.rng.randint(3, min(7, key_length // 2))
                                start = self.rng.randrange(0, key_length - seg_len + 1)
                                k = self.rng.randrange(1, seg_len)
                                seg = perturbed[start : start + seg_len]
                                perturbed[start : start + seg_len] = seg[k:] + seg[:k]

                            elif op_type == 3 and key_length >= 4:
                                # Swap adjacent pairs in a random region
                                start = self.rng.randrange(0, key_length - 3)
                                perturbed[start], perturbed[start + 1] = (
                                    perturbed[start + 1],
                                    perturbed[start],
                                )

                            else:
                                # Move random element to random position
                                if key_length >= 3:
                                    i = self.rng.randrange(0, key_length)
                                    j = self.rng.randrange(0, key_length)
                                    val = perturbed.pop(i)
                                    perturbed.insert(j, val)

                        # Deduplicate: only add if unique
                        variant_tuple = tuple(perturbed)
                        if variant_tuple not in seen_variants:
                            seen_variants.add(variant_tuple)

                            # Generate unique seed_int for this variant
                            if use_nondeterministic_rng:
                                # Salt with seed index, variant number, and ordering for unique RNG
                                seed_int = (
                                    sum(
                                        (v + 1) * (i + 31)
                                        for i, v in enumerate(perturbed)
                                    )
                                    ^ (seed_idx * 997)
                                    ^ (variant_idx * 7919)
                                ) & ((1 << 30) - 1)
                            else:
                                seed_int = sum(
                                    (v + 1) * (i + 31) for i, v in enumerate(perturbed)
                                ) & ((1 << 30) - 1)

                            worker_args.append(
                                (
                                    batch,
                                    key_length,
                                    seed_int,
                                    max_iterations,
                                    ic_threshold,
                                    self.padding,
                                    CONFIG.get("debug_output", False),
                                    use_hybrid,
                                    perturbed,
                                )
                            )

                        # Stop if we've generated enough unique variants
                        if len(worker_args) >= restarts:
                            break

                    if len(worker_args) >= restarts:
                        break

                if (
                    CONFIG.get("intermediate_output", True)
                    and len(worker_args) < restarts
                ):
                    print(
                        f"[DIVERSIFY] Generated {len(worker_args)} unique variants "
                        f"({len(seen_variants)} unique orderings from {variants_per_seed} attempts)"
                    )
            # Fill remaining worker args up to `restarts`
            for r in range(len(worker_args), restarts):
                if use_nondeterministic_rng:
                    # Use system randomness + counter for true non-determinism
                    import time

                    seed_int = (
                        self.rng.randint(0, (1 << 30) - 1) ^ int(time.time() * 1000) ^ r
                    ) & ((1 << 30) - 1)
                else:
                    seed_int = self.rng.randint(0, (1 << 30) - 1)

                worker_args.append(
                    (
                        batch,
                        key_length,
                        seed_int,
                        max_iterations,
                        ic_threshold,
                        self.padding,
                        CONFIG.get("debug_output", False),
                        use_hybrid,
                        None,  # explicit None start_order for random initialization
                    )
                )

            # Run workers in parallel
            max_workers = min(key_search_workers, len(worker_args))

            # Create shared stop event for cooperative cancellation
            manager = multiprocessing.Manager()
            stop_event = manager.Event()

            # Ensure we have a set to track already-processed keys in this parallel run
            # so the KeyboardInterrupt recovery path can deduplicate safely.
            phase1_tested_keys = set()

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_worker_initializer,
                initargs=(stop_event, CONFIG),
            ) as executor:
                futures = [
                    executor.submit(key_search_worker, args) for args in worker_args
                ]

                if CONFIG.get("intermediate_output", True):
                    print(f"[KEY-SEARCH] Submitted {len(futures)} worker tasks")

                # monitor progress as futures complete and print periodic STATUS messages
                completed = 0
                total = len(futures)
                early_stop_triggered = False
                eps = CONFIG.get("score_eps", 1e-8)

                try:
                    for idx, future in enumerate(
                        concurrent.futures.as_completed(futures)
                    ):
                        completed += 1
                        # Print periodic status updates during the actual run
                        if CONFIG.get("intermediate_output", True):
                            if (completed % 500) == 0 or completed == total:
                                print(f"[STATUS] restart={completed}")

                        try:
                            key_string, cur_score, worker_log = future.result()

                            if CONFIG.get("debug_output", False):
                                print(
                                    f"\n[KEYSEARCH] Worker {idx+1}/{restarts} completed: score={cur_score:.6f} key='{key_string}'"
                                )
                                if worker_log:
                                    print(f"[WORKER-LOG]\n{worker_log}")

                            if key_string and cur_score > float("-inf"):
                                per_restart_results.append((cur_score, key_string))

                                if not hasattr(self, "_per_length_candidates"):
                                    self._per_length_candidates = {}
                                self._per_length_candidates[key_length] = (
                                    per_restart_results.copy()
                                )

                                if (
                                    CONFIG.get(
                                        "update_global_best_candidates_dynamically",
                                        False,
                                    )
                                    and self._global_leaderboard
                                ):
                                    added = self._global_leaderboard.add_candidate(
                                        cur_score, key_string
                                    )
                                    if added and CONFIG.get("debug_output", True):
                                        leaderboard_size = (
                                            self._global_leaderboard.size()
                                        )
                                        print(
                                            f"[LEADERBOARD] Updated (size={leaderboard_size})"
                                        )

                                if is_better(cur_score, best_global_score):
                                    best_global_score = cur_score
                                    best_global_key = key_string
                                    if CONFIG.get("intermediate_output", True):
                                        print(
                                            f"[KEY-SEARCH] New global best: key='{best_global_key}' score={best_global_score:.6f}"
                                        )

                                    # EARLY-STOP: if configured and threshold reached, break early
                                    if CONFIG.get(
                                        "early_stop_if_ic_threshold_reached", False
                                    ):
                                        try:
                                            # compare against provided ic_threshold param (use eps tolerance)
                                            if best_global_score >= ic_threshold - eps:
                                                early_stop_triggered = True
                                                if CONFIG.get(
                                                    "intermediate_output", True
                                                ):
                                                    print(
                                                        f"[EARLY-STOP] Stopping key-search early because score {best_global_score:.6f} >= ic_threshold {ic_threshold:.6f}"
                                                    )

                                                stop_event.set()
                                                break
                                        except Exception:
                                            # be tolerant: do not crash on comparison errors
                                            pass

                        except Exception as e:
                            if CONFIG.get("intermediate_output", True):
                                import traceback

                                print(
                                    f"[KEY-SEARCH] Worker {idx+1} failed:\n{traceback.format_exc()}"
                                )

                except KeyboardInterrupt:
                    if CONFIG.get("intermediate_output", True):
                        print(
                            f"\n[KEYBOARD-INTERRUPT] Ctrl+C detected in Phase 1, stopping workers..."
                        )
                    set_keyboard_interrupt()
                    stop_event.set()

                    # Collect results from already-completed futures before canceling
                    for future in futures:
                        if future.done():
                            try:
                                polished_key, polished_score, worker_log = (
                                    future.result(timeout=0)
                                )

                                if CONFIG.get("debug_output", False) and worker_log:
                                    print(
                                        f"\n[FRAGMENT-WORKER-LOG] (interrupt recovery):\n{worker_log}"
                                    )

                                # Skip duplicate keys
                                if polished_key and polished_key in phase1_tested_keys:
                                    continue

                                if polished_key and polished_score > float("-inf"):
                                    phase1_tested_keys.add(polished_key)
                                    # record into the per-restart results used elsewhere in this function
                                    per_restart_results.append(
                                        (polished_score, polished_key)
                                    )

                                    # Add to global leaderboard
                                    if (
                                        CONFIG.get(
                                            "update_global_best_candidates_dynamically",
                                            False,
                                        )
                                        and self._global_leaderboard
                                    ):
                                        self._global_leaderboard.add_candidate(
                                            polished_score, polished_key
                                        )

                                    # update the function-level global best trackers
                                    if polished_score > best_global_score:
                                        best_global_score = polished_score
                                        best_global_key = polished_key

                                        if CONFIG.get("intermediate_output", True):
                                            print(
                                                f"[PHASE-1-RECOVERY] Found completed result: "
                                                f"key='{best_global_key}' score={best_global_score:.6f}"
                                            )
                            except Exception:
                                pass  # Skip futures that error during result collection

                    # Don't re-raise, let cleanup happen below
                    # Don't print top candidates here - let outer handler do it

                # Cleanup: signal stop and cancel remaining futures if early stop or interrupt
                if early_stop_triggered or was_keyboard_interrupted():
                    stop_event.set()
                    early_stop_triggered = True  # Treat interrupt as early stop
                    try:
                        # Collect any already-completed future results so we don't lose work
                        recovered = 0
                        for f in futures:
                            if f.done():
                                try:
                                    polished_key, polished_score, worker_log = f.result(
                                        timeout=0
                                    )
                                except Exception:
                                    continue
                                # skip duplicates
                                if polished_key and polished_key in phase1_tested_keys:
                                    continue
                                if polished_key and polished_score > float("-inf"):
                                    phase1_tested_keys.add(polished_key)
                                    # record into the per-restart results used elsewhere in this function
                                    per_restart_results.append(
                                        (polished_score, polished_key)
                                    )
                                    # update the function-level global best trackers
                                    if polished_score > best_global_score:
                                        best_global_score = polished_score
                                        best_global_key = polished_key
                                        if CONFIG.get("intermediate_output", True):
                                            print(
                                                f"[PHASE-1-RECOVERY] Recovered result: key='{best_global_key}' score={best_global_score:.6f}"
                                            )
                                    recovered += 1
                        if CONFIG.get("debug_output", False):
                            debug(
                                f"[PHASE-1] Recovered {recovered} completed worker results before cancellation"
                            )
                        # cancel individual futures
                        for f in futures:
                            if not f.done():
                                try:
                                    f.cancel()
                                except Exception:
                                    pass
                        # attempt to shutdown executor and cancel pending tasks (if supported)
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            # older Python may not support cancel_futures; ignore
                            pass
                    except Exception:
                        pass

        else:
            # Use sequential runner from utils
            best_global_key, best_global_score, per_restart_results = (
                hill_climb_sequential_runner(
                    breaker=self,
                    ciphertexts=ciphertexts,
                    key_length=key_length,
                    restarts=restarts,
                    max_iterations=max_iterations,
                    ic_threshold=ic_threshold,
                    use_hybrid=use_hybrid,
                    intermediate_output=CONFIG.get("intermediate_output", True),
                    debug=CONFIG.get("debug_output", False),
                    ic_earlystop_min_restarts=CONFIG.get(
                        "ic_earlystop_min_restarts", 10
                    ),
                    score_eps=CONFIG.get("score_eps", 1e-8),
                )
            )

        # ---------- GLOBAL SA PASS (only if hybrid enabled) ----------
        # Skip global SA if we triggered early stop above -- proceed to final reporting.
        if not (
            CONFIG.get("early_stop_if_ic_threshold_reached", False)
            and best_global_score >= ic_threshold - CONFIG.get("score_eps", 1e-8)
        ):
            if (
                use_hybrid
                and best_global_key is not None
                and best_global_score > float("-inf")
            ):
                if CONFIG.get("intermediate_output", True):
                    print(
                        f"\n[GLOBAL-SA] Running global SA pass seeded from best key '{best_global_key}' (score={best_global_score:.6f})"
                    )

                try:
                    # Run SA with moderate parameters for global exploration
                    global_sa_key, global_sa_score = self.simulated_annealing(
                        batch,
                        key_length,
                        start_key=best_global_key,
                        T_init=0.5,  # Higher initial temp for global exploration
                        T_min=0.005,  # Lower final temp for thorough refinement
                        alpha=0.92,  # Slower cooling for more thorough search
                        iterations_per_temp=100,
                    )

                    # Polish the SA result
                    polished_global_key, polished_global_score = self.polish_key(
                        batch, global_sa_key, max_iterations=300
                    )

                    if polished_global_score > best_global_score:
                        if CONFIG.get("intermediate_output", True):
                            print(
                                f"[GLOBAL-SA] Improved! New best: key='{polished_global_key}' score={polished_global_score:.6f} (was {best_global_score:.6f})"
                            )
                        best_global_key = polished_global_key
                        best_global_score = polished_global_score
                        per_restart_results.append(
                            (polished_global_score, polished_global_key)
                        )
                    else:
                        if CONFIG.get("intermediate_output", True):
                            print(
                                f"[GLOBAL-SA] No improvement: {polished_global_score:.6f} <= {best_global_score:.6f}"
                            )
                except Exception as e:
                    if CONFIG.get("intermediate_output", True):
                        print(f"[GLOBAL-SA] Global SA pass failed: {e}")
        else:
            # note: skipping heavy global-SA due to early-stop condition
            if CONFIG.get("intermediate_output", True):
                print(
                    "[SKIP] Skipping global-SA / additional hybrid passes due to early stop.\n"
                )

        # store per-restart candidates so external callers (run_demo) can inspect top-N
        if not hasattr(self, "_per_length_candidates"):
            self._per_length_candidates = {}
        self._per_length_candidates[key_length] = per_restart_results.copy()

        # Print top-20 intermediate candidates for this key length (if requested)
        if CONFIG.get("intermediate_output", True) and per_restart_results:
            # sort descending by score
            sorted_results = sorted(
                per_restart_results, key=lambda x: x[0], reverse=True
            )
            top_n = min(8, len(sorted_results))
            print(f"\nTop {top_n} candidates for key length {key_length}:")

            # If a provided key hint exists, build the set of all equivalent numeric orders to match against.
            provided_hint = get_provided_key_hint()
            possible_orders_set = None
            if provided_hint:
                try:
                    possible_orders = get_all_possible_key_orders(provided_hint)
                    # convert to tuple form for quick membership testing
                    possible_orders_set = {tuple(o) for o in possible_orders}
                except Exception:
                    possible_orders_set = None

            # collect matches for summary printing
            matches_for_hint = []

            for idx in range(top_n):
                score, key = sorted_results[idx]
                col_order = get_column_order(key) if key else None

                # check whether this candidate's numeric order matches any equivalent order of the provided hint
                if possible_orders_set is not None and col_order is not None:
                    if tuple(col_order) in possible_orders_set:
                        matches_for_hint.append((idx + 1, col_order))

                # Show additional diagnostic info for top candidates
                if idx < 5 and key:  # Show detail for top 5
                    # Decrypt with this key and compute sub-scores
                    batch = [ct.replace(" ", "") for ct in ciphertexts]
                    decrypted_parts = []
                    for ct in batch:
                        frac = reverse_columnar_transposition(
                            ct, key, padding=self.padding
                        )
                        frac = self._normalize_decrypted(frac)
                        decrypted_parts.append(frac)
                    combined = "".join(decrypted_parts)

                    # Compute sub-scores
                    pair_reg = self._adfgvx_pair_regularity(combined)
                    entropy_sc = self._position_entropy_score(combined)

                    print(
                        f"  [{idx+1}] key_letters='{key}' key_digits={col_order} score={score:.6f} "
                        f"(pair_reg={pair_reg:.4f} entropy={entropy_sc:.4f})"
                    )
                else:
                    print(
                        f"  [{idx+1}] key_letters='{key}' key_digits={col_order} score={score:.6f}"
                    )
            print("")

            # If we had a provided hint, print which top candidates matched its equivalent orders
            if possible_orders_set is not None:
                if matches_for_hint:
                    print("Matches with provided key hint (order_number, key_digits):")
                    for rank_no, digits in matches_for_hint:
                        print(f"  [{rank_no}] {digits}")
                else:
                    print(
                        "No top candidates matched any equivalent orders of the provided key hint."
                    )

        return best_global_key, best_global_score

    # ---------------------------
    # Top-level orchestration
    # ---------------------------
    def break_transposition(
        self,
        ciphertexts: List[str],
        candidate_lengths: Optional[List[int]] = None,
        restarts: int = 100,
        max_iterations: int = 1000,
        ic_threshold: float = 0.05,
        use_hybrid: Optional[bool] = None,
    ) -> Dict[int, Tuple[Optional[str], float]]:
        """
        Try a range of candidate key lengths and return best key per length.
        """
        if candidate_lengths is None:
            candidate_lengths = list(range(16, 24))  # default from Lasry

        cleaned = [ct.replace(" ", "") for ct in ciphertexts]
        results: Dict[int, Tuple[Optional[str], float]] = {}

        debug(f"break_transposition: trying lengths {candidate_lengths}")

        # If a non-default language is selected for scoring, inform the user
        # before starting key search (affects scoring during transposition search).
        lang = CONFIG.get("language", "EN").upper()
        if lang != "EN" and CONFIG.get("intermediate_output", True):
            print(
                f"[KEY-SEARCH] Using language='{lang}' for key-search scoring (affects scoring during transposition search)."
            )

        # Decide whether to auto-infer variables per-length from global CONFIG
        auto_infer = CONFIG.get("infer_key_search_variables_automatically", False)

        # Check if long-key strategy is enabled
        enable_fragment_seeding = CONFIG.get("enable_fragment_seeding", False)
        enable_three_phase = CONFIG.get("enable_three_phase_keysearch", False)

        for m in candidate_lengths:
            # Determine if this is a "long key" requiring special treatment
            is_long_key = enable_fragment_seeding

            if is_long_key and CONFIG.get("intermediate_output", True):
                print(
                    f"\n[LONG-KEY] Using fragment reconstruction strategy for key length {m}"
                )

            if CONFIG.get("intermediate_output", True):
                weight = get_tetragram_weight(m)
                bigram_weight = max(0.0, 1.0 - weight)
                print(f"[BIGRAM WEIGHT] Key length {m}: weight={bigram_weight:.3f}")
                print(f"[TETRAGRAM WEIGHT] Key length {m}: weight={weight:.3f}")

            # If configured, infer per-length search variables and print them
            if auto_infer:
                local_restarts, local_max_iters, local_ic = infer_key_search_variables(
                    m
                )
                print(
                    f"[INFER] key_length={m} -> restarts={local_restarts} max_iterations={local_max_iters} ic_threshold={local_ic}\n"
                )
            else:
                local_restarts, local_max_iters, local_ic = (
                    restarts,
                    max_iterations,
                    ic_threshold,
                )

            # Choose search strategy: if caller passed use_hybrid=None use CONFIG value
            selected_hybrid = (
                CONFIG.get("use_hybrid", False) if use_hybrid is None else use_hybrid
            )

            # --- PHASE 1: Initial seed-based search (if enabled) ---
            if is_long_key:

                infer_fragment_vars = CONFIG.get(
                    "infer_fragment_seeding_variables_automatically", False
                )

                if infer_fragment_vars:
                    # Get all phase-specific parameters for this key length
                    phase_params = infer_fragment_seeding_variables(m)

                    num_seeds = phase_params["total_seeds"]
                    restarts_per_seed = phase_params["restart_per_seed"]
                    max_iterations_per_seed = phase_params["max_iterations_per_seed"]

                    # Store phase 2/3 params for later use
                    num_phase2_candidates = phase_params["best_candidates_phase_2"]
                    restarts_phase2 = phase_params["restarts_phase_2"]
                    max_iters_phase2 = phase_params["max_iterations_phase_2"]

                    num_phase3_candidates = phase_params["best_candidates_phase_3"]
                    local_restarts = phase_params["restarts"]
                    local_max_iters = phase_params["max_iterations"]

                    local_ic = phase_params["ic_threshold"]

                    if CONFIG.get("intermediate_output", True):
                        print(
                            f"[INFER-FRAGMENT] key_length={m} -> "
                            f"Phase1(seeds={num_seeds}, restarts/seed={restarts_per_seed}, iters={max_iterations_per_seed}) "
                            f"Phase2(top={num_phase2_candidates}, restarts={restarts_phase2}, iters={max_iters_phase2}) "
                            f"Phase3(top={num_phase3_candidates}, restarts={local_restarts}, iters={local_max_iters}) "
                            f"ic_threshold={local_ic}"
                        )
                else:
                    # Use CONFIG values for fragment seeding parameters
                    num_seeds = CONFIG.get("total_seeds", max(1000, m * 3))
                    restarts_per_seed = CONFIG.get("restart_per_seed", 50)
                    max_iterations_per_seed = CONFIG.get(
                        "max_iterations_per_seed", 1000
                    )

                if CONFIG.get("intermediate_output", True):
                    phase_label = "[PHASE-1]" if enable_three_phase else "[LONG-KEY]"
                    print(
                        f"{phase_label} Generating {num_seeds} fragment-based seed orderings..."
                    )

                seed_orderings = reconstruct_long_key_seeds(
                    cleaned, m, num_seeds=num_seeds
                )

                if CONFIG.get("intermediate_output", True):
                    phase_label = "[PHASE-1]" if enable_three_phase else "[LONG-KEY]"
                    print(
                        f"{phase_label} Generated {len(seed_orderings)} seed orderings, testing each with "
                        f"{restarts_per_seed} restarts and max_iterations={max_iterations_per_seed}..."
                    )

                # Track all seed results for Phase 2
                all_seed_results = []
                best_key_overall = None
                best_score_overall = float("-inf")

                phase1_tested_keys = set()

                # Check if parallel processing is enabled
                key_search_workers = CONFIG.get("key_search_workers", 1)
                use_parallel = key_search_workers > 1 and len(seed_orderings) > 1

                manager = multiprocessing.Manager()
                stop_event = manager.Event()

                if use_parallel:
                    if CONFIG.get("intermediate_output", True):
                        phase_label = (
                            "[PHASE-1]" if enable_three_phase else "[LONG-KEY]"
                        )
                        print(
                            f"{phase_label} Using parallel processing with {key_search_workers} workers for {len(seed_orderings)} fragment seeds"
                        )

                    # Prepare worker arguments using CONFIG values
                    worker_args = [
                        (
                            cleaned,
                            seed_ordering,
                            m,
                            restarts_per_seed,
                            max_iterations_per_seed,
                            self.padding,
                            CONFIG.get("debug_output", False),
                        )
                        for seed_ordering in seed_orderings
                    ]

                    max_workers = min(key_search_workers, len(worker_args))

                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=_worker_initializer,
                        initargs=(stop_event, CONFIG),
                    ) as executor:
                        futures = [
                            executor.submit(fragment_seed_worker, args)
                            for args in worker_args
                        ]

                        if CONFIG.get("intermediate_output", True):
                            phase_label = (
                                "[PHASE-1]" if enable_three_phase else "[LONG-KEY]"
                            )
                            print(
                                f"{phase_label} Submitted {len(futures)} fragment seed worker tasks"
                            )

                        completed = 0
                        total = len(futures)
                        eps = CONFIG.get("score_eps", 1e-8)
                        early_stop_triggered = False

                        # Track tested keys in Phase 1 during parallel execution
                        phase1_tested_keys = set()

                        try:
                            for idx, future in enumerate(
                                concurrent.futures.as_completed(futures)
                            ):
                                completed += 1

                                if CONFIG.get("intermediate_output", True):
                                    if (completed % 10) == 0 or completed == total:
                                        phase_label = (
                                            "[PHASE-1-STATUS]"
                                            if enable_three_phase
                                            else "[LONG-KEY-STATUS]"
                                        )
                                        print(
                                            f"{phase_label} Processed {completed}/{total} seeds"
                                        )

                                try:
                                    polished_key, polished_score, worker_log = (
                                        future.result()
                                    )

                                    if CONFIG.get("debug_output", False) and worker_log:
                                        print(
                                            f"\n[FRAGMENT-WORKER-LOG] Worker {idx+1}/{total}:\n{worker_log}"
                                        )

                                    # Skip duplicate keys in Phase 1
                                    if (
                                        polished_key
                                        and polished_key in phase1_tested_keys
                                    ):
                                        if CONFIG.get("debug_output", True):
                                            print(
                                                f"[PHASE-1] Skipping duplicate result from worker {idx+1}: key='{polished_key}'"
                                            )
                                        continue

                                    if polished_key and polished_score > float("-inf"):

                                        phase1_tested_keys.add(polished_key)

                                        all_seed_results.append(
                                            (polished_score, polished_key)
                                        )

                                        if (
                                            CONFIG.get(
                                                "update_global_best_candidates_dynamically",
                                                False,
                                            )
                                            and self._global_leaderboard
                                        ):
                                            added = (
                                                self._global_leaderboard.add_candidate(
                                                    polished_score, polished_key
                                                )
                                            )
                                            if added and CONFIG.get(
                                                "intermediate_output", True
                                            ):
                                                leaderboard_size = (
                                                    self._global_leaderboard.size()
                                                )
                                                print(
                                                    f"[LEADERBOARD] Updated (size={leaderboard_size})"
                                                )

                                        if polished_score > best_score_overall:
                                            best_score_overall = polished_score
                                            best_key_overall = polished_key

                                            if CONFIG.get("intermediate_output", True):
                                                phase_label = (
                                                    "[PHASE-1]"
                                                    if enable_three_phase
                                                    else "[LONG-KEY]"
                                                )
                                                print(
                                                    f"{phase_label} New best from seed #{idx+1}/{total}: "
                                                    f"key='{best_key_overall}' score={best_score_overall:.6f}"
                                                )

                                            # Check early stop for Phase 1
                                            if CONFIG.get(
                                                "early_stop_if_ic_threshold_reached",
                                                False,
                                            ):
                                                if best_score_overall >= local_ic - eps:
                                                    early_stop_triggered = True
                                                    if CONFIG.get(
                                                        "intermediate_output", True
                                                    ):
                                                        print(
                                                            f"[PHASE-1-EARLY-STOP] Stopping Phase 1 early because score {best_score_overall:.6f} >= ic_threshold {local_ic:.6f}"
                                                        )

                                                    stop_event.set()
                                                    break

                                except Exception as e:
                                    if CONFIG.get("intermediate_output", True):
                                        import traceback

                                        print(
                                            f"[PHASE-1] Worker {idx+1} failed:\n{traceback.format_exc()}"
                                        )

                        except KeyboardInterrupt:
                            if CONFIG.get("intermediate_output", True):
                                print(
                                    f"\n[KEYBOARD-INTERRUPT] Ctrl+C detected, stopping workers gracefully..."
                                )
                            set_keyboard_interrupt()
                            stop_event.set()
                            # Don't re-raise, let cleanup happen
                            # Don't print top candidates here - let outer handler do it

                else:
                    # Sequential processing
                    eps = CONFIG.get("score_eps", 1e-8)
                    early_stop_triggered = False

                    try:
                        for seed_idx, seed_ordering in enumerate(
                            seed_orderings, start=1
                        ):
                            seed_key = self.string_key_from_column_order(seed_ordering)

                            if seed_key in phase1_tested_keys:
                                if CONFIG.get("intermediate_output", True):
                                    print(
                                        f"[PHASE-1] Skipping duplicate seed #{seed_idx}: key='{seed_key}'"
                                    )
                                continue

                            phase1_tested_keys.add(seed_key)

                            if CONFIG.get("intermediate_output", True):
                                phase_label = (
                                    "[PHASE-1]" if enable_three_phase else "[LONG-KEY]"
                                )
                                print(
                                    f"{phase_label} Testing seed #{seed_idx}/{len(seed_orderings)}: key='{seed_key}' "
                                    f"with {restarts_per_seed} restarts and max_iterations={max_iterations_per_seed}"
                                )

                            # Run multiple restarts for this seed
                            seed_best_key = seed_key
                            seed_best_score = float("-inf")

                            for restart_idx in range(restarts_per_seed):
                                polished_key, polished_score = self.polish_key(
                                    cleaned,
                                    seed_key if restart_idx == 0 else seed_best_key,
                                    max_iterations=max_iterations_per_seed,
                                )

                                if polished_score > seed_best_score:
                                    seed_best_score = polished_score
                                    seed_best_key = polished_key

                            if seed_best_key and seed_best_score > float("-inf"):
                                all_seed_results.append(
                                    (seed_best_score, seed_best_key)
                                )

                                if (
                                    CONFIG.get(
                                        "update_global_best_candidates_dynamically",
                                        False,
                                    )
                                    and self._global_leaderboard
                                ):
                                    added = self._global_leaderboard.add_candidate(
                                        seed_best_score, seed_best_key
                                    )
                                    if added and CONFIG.get(
                                        "intermediate_output", True
                                    ):
                                        leaderboard_size = (
                                            self._global_leaderboard.size()
                                        )
                                        print(
                                            f"[LEADERBOARD] Updated (size={leaderboard_size})"
                                        )

                                if seed_best_score > best_score_overall:
                                    best_score_overall = seed_best_score
                                    best_key_overall = seed_best_key

                                    if CONFIG.get("intermediate_output", True):
                                        phase_label = (
                                            "[PHASE-1]"
                                            if enable_three_phase
                                            else "[LONG-KEY]"
                                        )
                                        print(
                                            f"{phase_label} New best from seed #{seed_idx}: "
                                            f"key='{best_key_overall}' score={best_score_overall:.6f}"
                                        )

                                    # Check early stop for Phase 1
                                    if CONFIG.get(
                                        "early_stop_if_ic_threshold_reached", False
                                    ):
                                        if best_score_overall >= local_ic - eps:
                                            early_stop_triggered = True
                                            if CONFIG.get("intermediate_output", True):
                                                print(
                                                    f"[PHASE-1-EARLY-STOP] Stopping Phase 1 early because score {best_score_overall:.6f} >= ic_threshold {local_ic:.6f}"
                                                )

                                            stop_event.set()
                                            break

                    except KeyboardInterrupt:
                        if CONFIG.get("intermediate_output", True):
                            print(
                                f"\n[KEYBOARD-INTERRUPT] Ctrl+C detected in Phase 1 (sequential), stopping..."
                            )
                        set_keyboard_interrupt()
                        stop_event.set()
                        # Results already collected in all_seed_results, continue to Phase 2/3 checks

                # --- PHASE 2: Deeper search on best Phase 1 candidates (if three-phase enabled) ---
                if (
                    enable_three_phase
                    and not early_stop_triggered
                    and not was_keyboard_interrupted()
                ):

                    if infer_fragment_vars:
                        # num_phase2_candidates, restarts_phase2, max_iters_phase2 already set above
                        pass
                    else:
                        num_phase2_candidates = CONFIG.get(
                            "best_candidates_phase_2", 20
                        )
                        restarts_phase2 = CONFIG.get("restarts_phase_2", 500)
                        max_iters_phase2 = CONFIG.get("max_iterations_phase_2", 5000)

                    if (
                        CONFIG.get("update_global_best_candidates_dynamically", False)
                        and self._global_leaderboard
                    ):
                        phase2_candidates_raw = self._global_leaderboard.get_top_n(
                            num_phase2_candidates
                        )
                        # Deduplicate by converting to dict
                        seen_keys_p2 = set()
                        phase2_candidates = []
                        for score, key in phase2_candidates_raw:
                            if key not in seen_keys_p2:
                                seen_keys_p2.add(key)
                                phase2_candidates.append((score, key))

                        if CONFIG.get("intermediate_output", True):
                            print(
                                f"\n[PHASE-2] Using top {len(phase2_candidates)} unique candidates from global leaderboard "
                                f"(leaderboard size={self._global_leaderboard.size()}, "
                                f"duplicates removed={len(phase2_candidates_raw) - len(phase2_candidates)})"
                            )
                    else:
                        # Fallback to previous behavior with deduplication
                        all_seed_results_sorted = sorted(
                            all_seed_results, key=lambda x: x[0], reverse=True
                        )
                        # Deduplicate while preserving score order
                        seen_keys_p2 = set()
                        phase2_candidates = []
                        for score, key in all_seed_results_sorted:
                            if key not in seen_keys_p2:
                                seen_keys_p2.add(key)
                                phase2_candidates.append((score, key))
                                if len(phase2_candidates) >= num_phase2_candidates:
                                    break

                        if CONFIG.get("intermediate_output", True):
                            duplicates_removed = len(all_seed_results_sorted) - len(
                                phase2_candidates
                            )
                            print(
                                f"\n[PHASE-2] Running deeper search on top {len(phase2_candidates)} unique Phase 1 candidates "
                                f"with {restarts_phase2} restarts and {max_iters_phase2} max_iterations "
                                f"(duplicates removed={duplicates_removed})"
                            )

                    phase2_results = []

                    phase2_tested_keys = set()

                    for cand_idx, (cand_score, cand_key) in enumerate(
                        phase2_candidates, start=1
                    ):

                        if was_keyboard_interrupted():
                            if CONFIG.get("intermediate_output", True):
                                print(
                                    f"[PHASE-2-INTERRUPT] Keyboard interrupt detected, skipping remaining Phase 2 candidates"
                                )
                            early_stop_triggered = True
                            break

                        if cand_key in phase2_tested_keys:
                            if CONFIG.get("intermediate_output", True):
                                print(
                                    f"[PHASE-2] Skipping duplicate candidate #{cand_idx}: key='{cand_key}'"
                                )
                            continue

                        phase2_tested_keys.add(cand_key)

                        if CONFIG.get("intermediate_output", True):
                            print(
                                f"[PHASE-2] Processing candidate #{cand_idx}/{len(phase2_candidates)}: "
                                f"key='{cand_key}' score={cand_score:.6f}"
                            )

                        # Inject this candidate as seed ordering for hill_climb
                        try:
                            cand_order = self.key_string_to_column_order(cand_key)
                        except Exception:
                            cand_order = list(range(m))

                        self._injected_seed_orderings = [cand_order]

                        # Run hill_climb with Phase 2 parameters
                        phase2_key, phase2_score = self.hill_climb(
                            cleaned,
                            m,
                            restarts=restarts_phase2,
                            max_iterations=max_iters_phase2,
                            ic_threshold=local_ic,  # CHANGED: use local_ic instead of ic_threshold
                            use_hybrid=selected_hybrid,
                        )

                        if phase2_key and phase2_score > float("-inf"):
                            phase2_results.append((phase2_score, phase2_key))

                            if (
                                CONFIG.get(
                                    "update_global_best_candidates_dynamically", False
                                )
                                and self._global_leaderboard
                            ):
                                added = self._global_leaderboard.add_candidate(
                                    phase2_score, phase2_key
                                )
                                if added and CONFIG.get("intermediate_output", True):
                                    leaderboard_size = self._global_leaderboard.size()
                                    print(
                                        f"[LEADERBOARD] Updated from Phase 2 (size={leaderboard_size})"
                                    )

                            if phase2_score > best_score_overall:
                                best_score_overall = phase2_score
                                best_key_overall = phase2_key

                                if CONFIG.get("intermediate_output", True):
                                    print(
                                        f"[PHASE-2] New best from candidate #{cand_idx}: "
                                        f"key='{best_key_overall}' score={best_score_overall:.6f}"
                                    )

                        # Check early stop for Phase 2
                        if CONFIG.get("early_stop_if_ic_threshold_reached", False):
                            eps = CONFIG.get("score_eps", 1e-8)
                            if (
                                best_score_overall >= local_ic - eps
                            ):  # CHANGED: use local_ic instead of ic_threshold
                                if CONFIG.get("intermediate_output", True):
                                    print(
                                        f"[PHASE-2-EARLY-STOP] Stopping Phase 2 early because score {best_score_overall:.6f} >= ic_threshold {local_ic:.6f}"
                                    )
                                early_stop_triggered = True
                                break

                    # Clear injected seed orderings
                    self._injected_seed_orderings = []

                    # --- PHASE 3: Final refinement on best Phase 2 candidates ---
                    if not early_stop_triggered and not was_keyboard_interrupted():

                        if not infer_fragment_vars:
                            num_phase3_candidates = CONFIG.get(
                                "best_candidates_phase_3", 1
                            )

                        if (
                            CONFIG.get(
                                "update_global_best_candidates_dynamically", False
                            )
                            and self._global_leaderboard
                        ):
                            phase3_candidates_raw = self._global_leaderboard.get_top_n(
                                num_phase3_candidates
                            )

                            seen_keys_p3 = set()
                            phase3_candidates = []
                            for score, key in phase3_candidates_raw:
                                if key not in seen_keys_p3:
                                    seen_keys_p3.add(key)
                                    phase3_candidates.append((score, key))

                            if CONFIG.get("intermediate_output", True):
                                print(
                                    f"\n[PHASE-3] Using top {len(phase3_candidates)} unique candidates from global leaderboard "
                                    f"(leaderboard size={self._global_leaderboard.size()}, "
                                    f"duplicates removed={len(phase3_candidates_raw) - len(phase3_candidates)})"
                                )
                        else:
                            # Fallback to previous behavior with deduplication
                            phase2_results_sorted = sorted(
                                phase2_results, key=lambda x: x[0], reverse=True
                            )

                            seen_keys_p3 = set()
                            phase3_candidates = []
                            for score, key in phase2_results_sorted:
                                if key not in seen_keys_p3:
                                    seen_keys_p3.add(key)
                                    phase3_candidates.append((score, key))
                                    if len(phase3_candidates) >= num_phase3_candidates:
                                        break

                            if CONFIG.get("intermediate_output", True):
                                duplicates_removed = len(phase2_results_sorted) - len(
                                    phase3_candidates
                                )
                                print(
                                    f"\n[PHASE-3] Running final refinement on top {len(phase3_candidates)} unique Phase 2 candidates "
                                    f"with {local_restarts} restarts and {local_max_iters} max_iterations "
                                    f"(duplicates removed={duplicates_removed})"
                                )

                        phase3_tested_keys = set()

                        for cand_idx, (cand_score, cand_key) in enumerate(
                            phase3_candidates, start=1
                        ):

                            if was_keyboard_interrupted():
                                if CONFIG.get("intermediate_output", True):
                                    print(
                                        f"[PHASE-3-INTERRUPT] Keyboard interrupt detected, skipping remaining Phase 3 candidates"
                                    )
                                early_stop_triggered = True
                                break

                            if cand_key in phase3_tested_keys:
                                if CONFIG.get("intermediate_output", True):
                                    print(
                                        f"[PHASE-3] Skipping duplicate candidate #{cand_idx}: key='{cand_key}'"
                                    )
                                continue

                            phase3_tested_keys.add(cand_key)

                            if CONFIG.get("intermediate_output", True):
                                print(
                                    f"[PHASE-3] Processing candidate #{cand_idx}/{len(phase3_candidates)}: "
                                    f"key='{cand_key}' score={cand_score:.6f}"
                                )

                            # Inject this candidate as seed ordering
                            try:
                                cand_order = self.key_string_to_column_order(cand_key)
                            except Exception:
                                cand_order = list(range(m))

                            self._injected_seed_orderings = [cand_order]

                            # Run final hill_climb with Phase 3 parameters
                            if selected_hybrid:
                                final_key, final_score = self.hybrid_search(
                                    cleaned,
                                    m,
                                    local_restarts,
                                    local_max_iters,
                                    local_ic,
                                )
                            else:
                                final_key, final_score = self.hill_climb(
                                    cleaned,
                                    m,
                                    local_restarts,
                                    local_max_iters,
                                    local_ic,
                                )

                            if final_score > best_score_overall:

                                if (
                                    CONFIG.get(
                                        "update_global_best_candidates_dynamically",
                                        False,
                                    )
                                    and self._global_leaderboard
                                ):
                                    added = self._global_leaderboard.add_candidate(
                                        final_score, final_key
                                    )
                                    if added and CONFIG.get(
                                        "intermediate_output", True
                                    ):
                                        leaderboard_size = (
                                            self._global_leaderboard.size()
                                        )
                                        print(
                                            f"[LEADERBOARD] Updated from Phase 3 (size={leaderboard_size})"
                                        )

                                best_score_overall = final_score
                                best_key_overall = final_key

                                if CONFIG.get("intermediate_output", True):
                                    print(
                                        f"[PHASE-3] New best from candidate #{cand_idx}: "
                                        f"key='{best_key_overall}' score={best_score_overall:.6f}"
                                    )

                            # Check early stop for Phase 3
                            if CONFIG.get("early_stop_if_ic_threshold_reached", False):
                                eps = CONFIG.get("score_eps", 1e-8)
                                if (
                                    best_score_overall >= local_ic - eps
                                ):  # CHANGED: use local_ic instead of ic_threshold
                                    if CONFIG.get("intermediate_output", True):
                                        print(
                                            f"[PHASE-3-EARLY-STOP] Stopping Phase 3 early because score {best_score_overall:.6f} >= ic_threshold {local_ic:.6f}"
                                        )
                                    break

                        # Clear injected seed orderings
                        self._injected_seed_orderings = []

                    if (
                        CONFIG.get("update_global_best_candidates_dynamically", False)
                        and self._global_leaderboard
                        and CONFIG.get("intermediate_output", True)
                    ):
                        top_5 = self._global_leaderboard.get_top_n(5)
                        print(f"\n[LEADERBOARD] Final top-5 candidates:")
                        for rank, (score, key) in enumerate(top_5, start=1):
                            try:
                                key_digits = get_column_order(key)
                            except Exception:
                                key_digits = None
                            print(
                                f"  [{rank}] key='{key}' key_digits={key_digits} score={score:.6f}"
                            )

                    key, score = best_key_overall, best_score_overall

                else:
                    # Three-phase disabled or early stopped: run standard refinement
                    if (
                        best_key_overall
                        and not early_stop_triggered
                        and not was_keyboard_interrupted()
                    ):
                        if CONFIG.get("intermediate_output", True):
                            phase_label = (
                                "[PHASE-3]" if enable_three_phase else "[LONG-KEY]"
                            )
                            print(
                                f"\n{phase_label} Running global refinement from best seed: "
                                f"key='{best_key_overall}' score={best_score_overall:.6f}"
                            )

                        # Run standard hill-climb/hybrid starting from best seed
                        if selected_hybrid:
                            final_key, final_score = self.hybrid_search(
                                cleaned, m, local_restarts, local_max_iters, local_ic
                            )
                        else:
                            final_key, final_score = self.hill_climb(
                                cleaned, m, local_restarts, local_max_iters, local_ic
                            )

                        # Keep the better of seed-based or hill-climb result
                        if final_score > best_score_overall:
                            key, score = final_key, final_score
                        else:
                            key, score = best_key_overall, best_score_overall
                    else:
                        # Fallback if no seeds produced valid results or early stopped
                        key, score = best_key_overall, best_score_overall
                        if not key:
                            if selected_hybrid:
                                key, score = self.hybrid_search(
                                    cleaned,
                                    m,
                                    local_restarts,
                                    local_max_iters,
                                    local_ic,
                                )
                            else:
                                key, score = self.hill_climb(
                                    cleaned,
                                    m,
                                    local_restarts,
                                    local_max_iters,
                                    local_ic,
                                )
            else:
                # Standard search for short keys
                if selected_hybrid:
                    key, score = self.hybrid_search(
                        cleaned, m, local_restarts, local_max_iters, local_ic
                    )
                else:
                    key, score = self.hill_climb(
                        cleaned, m, local_restarts, local_max_iters, local_ic
                    )

            results[m] = (key, score)
            debug(f"Result length {m}: key={key} score={score:.6f}")

        return results

    def decrypt_with_key(self, ciphertext: str, key_string: str) -> str:
        """Apply inverse columnar transposition (returns fractionated stream)."""
        return reverse_columnar_transposition(
            ciphertext.replace(" ", ""), key_string, padding=self.padding
        )


class BaseDecoderForProcesses:
    """
    Top-level picklable decoder holding fractionated streams for process workers.
    Call with an alphabet string and return (base_score, plaintexts).
    """

    def __init__(self, fractionated_streams: Iterable[str]):
        self.streams = list(fractionated_streams)

    def __call__(self, alpha: str) -> Tuple[float, List[str]]:
        # Build inverse polybius for the candidate alphabet and decode all streams
        try:
            _, inv = create_polybius_square(alpha)
        except Exception:
            # fallback to canonical alphabet if given alphabet is invalid
            _, inv = create_polybius_square(
                CONFIG.get("initial_alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            )
        texts = [fractionated_to_plain_with_map(frac, inv) for frac in self.streams]
        # use language-aware scorer (no common-words bonus)
        base_score = score_texts(texts, CONFIG.get("language", "EN").upper())
        return base_score, texts


def triage_seed_worker(args):
    """
    Early triage worker: runs a quick 40-move micro-hill-climb on the seed.
    If the best score never exceeds ic_threshold - 0.01, the seed is dropped.
    Otherwise, the seed proceeds to full key_search_worker.

    Args: same as key_search_worker (8 or 9-tuple)
    Returns: (final_key_string, final_score, worker_log) or (None, float('-inf'), '') if triaged out
    """
    # Extract ic_threshold from args (position depends on tuple length)
    if len(args) == 8:
        (
            ciphertexts,
            key_length,
            seed_int,
            max_iterations,
            ic_threshold,
            padding,
            debug,
            use_hybrid,
        ) = args
        start_order = None
    else:
        (
            ciphertexts,
            key_length,
            seed_int,
            max_iterations,
            ic_threshold,
            padding,
            debug,
            use_hybrid,
            start_order,
        ) = args

    triage_threshold = ic_threshold - 0.01
    triage_moves = 40

    try:
        # Initialize worker components
        worker_rng = random.Random(seed_int)
        breaker = ADFGVXBreaker(padding=padding)
        breaker.rng = worker_rng
        eval_cache = {}

        # Initialize order from provided start_order if present
        if start_order:
            if isinstance(start_order, str):
                try:
                    order = breaker.key_string_to_column_order(start_order)
                except Exception:
                    order = list(range(key_length))
            else:
                order = start_order.copy()
        else:
            order = list(range(key_length))
            if seed_int % 5 == 0:
                pass
            elif seed_int % 5 == 1:
                order.reverse()
            else:
                worker_rng.shuffle(order)

        # Micro-hill-climb triage (40 moves)
        key_string = breaker.string_key_from_column_order(order)
        cur_score = breaker.score_key_transposition(
            ciphertexts, key_string, cache=eval_cache
        )
        best_triage_score = cur_score

        for move_idx in range(triage_moves):
            improved = False
            transforms = breaker.generate_transformations(order)

            for name, cand_order in transforms:
                cand_key = breaker.string_key_from_column_order(cand_order)
                cand_score = breaker.score_key_transposition(
                    ciphertexts, cand_key, cache=eval_cache
                )

                if is_better(cand_score, cur_score):
                    order = cand_order
                    key_string = cand_key
                    cur_score = cand_score
                    improved = True

                    if cur_score > best_triage_score:
                        best_triage_score = cur_score

                    break

            if not improved:
                break

        # Check triage threshold
        if best_triage_score < triage_threshold:
            if debug:
                print(
                    f"[TRIAGE] Seed dropped: best_score={best_triage_score:.6f} < "
                    f"threshold={triage_threshold:.6f} (seed_int={seed_int})"
                )
            return None, float("-inf"), ""

        # Seed survived triage - proceed to full search
        if debug:
            print(
                f"[TRIAGE] Seed passed: best_score={best_triage_score:.6f} >= "
                f"threshold={triage_threshold:.6f} (seed_int={seed_int})"
            )

        # Reconstruct args with the improved order as start_order
        improved_args = (
            ciphertexts,
            key_length,
            seed_int,
            max_iterations,
            ic_threshold,
            padding,
            debug,
            use_hybrid,
            order,  # Pass the improved order from triage
        )

        # Delegate to full key_search_worker
        return key_search_worker(improved_args)

    except Exception as e:
        if debug:
            import traceback

            return None, float("-inf"), f"[TRIAGE-ERROR] {traceback.format_exc()}"
        return None, float("-inf"), ""


def key_search_worker(args):
    """
    Module-level worker function for parallel key search.
    Args: (ciphertexts, key_length, seed_int, max_iterations, ic_threshold, padding, debug, use_hybrid[, start_order])
    Returns: (final_key_string, final_score, worker_log)
    """

    # Accept either 8-tuple (legacy) or 9-tuple with start_order
    if len(args) == 8:
        (
            ciphertexts,
            key_length,
            seed_int,
            max_iterations,
            ic_threshold,
            padding,
            debug,
            use_hybrid,
        ) = args
        start_order = None
    else:
        (
            ciphertexts,
            key_length,
            seed_int,
            max_iterations,
            ic_threshold,
            padding,
            debug,
            use_hybrid,
            start_order,
        ) = args

    try:
        # Check for early stop signal
        if _should_stop():
            return None, float("-inf"), "[STOPPED]"

        if debug:
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                worker_rng = random.Random(seed_int)
                breaker = ADFGVXBreaker(padding=padding)
                breaker.rng = worker_rng

                eval_cache = {}

                # Initialize order from provided start_order if present, else previous logic
                if start_order:
                    if isinstance(start_order, str):
                        try:
                            order = breaker.key_string_to_column_order(start_order)

                        except Exception:
                            order = list(range(key_length))
                    else:
                        order = start_order.copy()
                else:
                    order = list(range(key_length))
                    if seed_int % 5 == 0:
                        pass
                    elif seed_int % 5 == 1:
                        order.reverse()
                    else:
                        worker_rng.shuffle(order)

                key_string = breaker.string_key_from_column_order(order)
                cur_score = breaker.score_key_transposition(
                    ciphertexts, key_string, cache=eval_cache
                )

                # preliminary / HC / optional SA (same code as before) ...
                # For brevity reuse existing code but ensure it operates on `order` / `key_string`.
                improved = True
                while improved:
                    # Check stop signal during preliminary pass
                    if _should_stop():
                        return None, float("-inf"), "[STOPPED]"

                    improved = False
                    transforms = breaker.generate_transformations(order)
                    for name, cand_order in transforms:
                        cand_key = breaker.string_key_from_column_order(cand_order)
                        cand_score = breaker.score_key_transposition(
                            ciphertexts, cand_key, cache=eval_cache
                        )
                        if is_better(cand_score, cur_score):
                            order = cand_order
                            key_string = cand_key
                            cur_score = cand_score
                            improved = True
                            if debug:
                                try:
                                    col_order = get_column_order(key_string)
                                except Exception:
                                    col_order = None
                                print(
                                    f"[WORKER-PRE_PASS] accepted {name} -> score={cur_score:.6f} key='{key_string}' \nkey_digits={col_order}"
                                )
                            break

                iteration = 0
                improved = True
                while improved and iteration < max_iterations:
                    # Check stop signal every 10 iterations
                    if iteration % 10 == 0 and _should_stop():
                        return None, float("-inf"), "[STOPPED]"

                    iteration += 1
                    improved = False
                    transforms = breaker.generate_transformations(order)
                    best_iter_score = cur_score
                    best_candidates = []

                    for name, cand_order in transforms:
                        cand_key = breaker.string_key_from_column_order(cand_order)
                        cand_score = breaker.score_key_transposition(
                            ciphertexts, cand_key, cache=eval_cache
                        )
                        if is_better(
                            cand_score, best_iter_score + CONFIG.get("score_eps", 1e-8)
                        ):
                            best_iter_score = cand_score
                            best_candidates = [cand_order]
                        elif is_tie(cand_score, best_iter_score):
                            best_candidates.append(cand_order)

                    if is_better(best_iter_score, cur_score) and best_candidates:
                        best_iter_order = worker_rng.choice(best_candidates)
                        order = best_iter_order
                        key_string = breaker.string_key_from_column_order(order)
                        cur_score = best_iter_score
                        improved = True
                        if debug:
                            print(
                                f"[WORKER-HC] iter={iteration} improved score={cur_score:.6f}"
                            )

                # Check before SA
                if _should_stop():
                    return None, float("-inf"), "[STOPPED]"

                if use_hybrid:
                    try:
                        sa_key, sa_score = breaker.simulated_annealing(
                            ciphertexts,
                            key_length,
                            start_key=key_string,
                            T_init=0.25,
                            T_min=0.01,
                            alpha=0.90,
                            iterations_per_temp=60,
                            score_cache=eval_cache,
                        )
                        polish_key_str, polish_score = breaker.polish_key(
                            ciphertexts,
                            sa_key,
                            max_iterations=200,
                            score_cache=eval_cache,
                        )
                        if is_better(polish_score, cur_score):
                            order = breaker.key_string_to_column_order(polish_key_str)
                            key_string = polish_key_str
                            cur_score = polish_score
                            if debug:
                                print(
                                    f"[WORKER-SA] accepted SA+polish -> key='{key_string}' score={cur_score:.6f}"
                                )
                    except Exception as e:
                        if debug:
                            print(f"[WORKER-SA] SA failed: {e}")

            worker_log = buf.getvalue()
        else:
            # Non-debug branch mirrors above initialization and HC/SA logic without output capture

            if _should_stop():
                return None, float("-inf"), ""

            worker_rng = random.Random(seed_int)
            breaker = ADFGVXBreaker(padding=padding)
            breaker.rng = worker_rng
            eval_cache = {}

            if start_order:
                if isinstance(start_order, str):
                    try:
                        order = breaker.key_string_to_column_order(start_order)
                    except Exception:
                        order = list(range(key_length))
                else:
                    order = start_order.copy()
            else:
                order = list(range(key_length))
                if seed_int % 5 == 0:
                    pass
                elif seed_int % 5 == 1:
                    order.reverse()
                else:
                    worker_rng.shuffle(order)

            key_string = breaker.string_key_from_column_order(order)
            cur_score = breaker.score_key_transposition(
                ciphertexts, key_string, cache=eval_cache
            )

            # preliminary pass
            current_improved = True
            while current_improved:

                if _should_stop():
                    return None, float("-inf"), ""

                current_improved = False
                transforms = breaker.generate_transformations(order)
                for name, cand_order in transforms:
                    cand_key = breaker.string_key_from_column_order(cand_order)
                    cand_score = breaker.score_key_transposition(
                        ciphertexts, cand_key, cache=eval_cache
                    )
                    if is_better(cand_score, cur_score):
                        order = cand_order
                        key_string = cand_key
                        cur_score = cand_score
                        current_improved = True
                        break

            # main HC loop
            iteration = 0
            improved = True
            while improved and iteration < max_iterations:

                if iteration % 10 == 0 and _should_stop():
                    return None, float("-inf"), ""

                iteration += 1
                improved = False
                transforms = breaker.generate_transformations(order)
                best_iter_score = cur_score
                best_candidates = []
                for name, cand_order in transforms:
                    cand_key = breaker.string_key_from_column_order(cand_order)
                    cand_score = breaker.score_key_transposition(
                        ciphertexts, cand_key, cache=eval_cache
                    )
                    if is_better(
                        cand_score, best_iter_score + CONFIG.get("score_eps", 1e-8)
                    ):
                        best_iter_score = cand_score
                        best_candidates = [cand_order]
                    elif is_tie(cand_score, best_iter_score):
                        best_candidates.append(cand_order)

                if is_better(best_iter_score, cur_score) and best_candidates:
                    best_iter_order = worker_rng.choice(best_candidates)
                    order = best_iter_order
                    key_string = breaker.string_key_from_column_order(order)
                    cur_score = best_iter_score
                    improved = True

            if _should_stop():
                return None, float("-inf"), ""

            if use_hybrid:
                try:
                    sa_key, sa_score = breaker.simulated_annealing(
                        ciphertexts,
                        key_length,
                        start_key=key_string,
                        T_init=0.25,
                        T_min=0.01,
                        alpha=0.90,
                        iterations_per_temp=60,
                        score_cache=eval_cache,
                    )
                    polish_key_str, polish_score = breaker.polish_key(
                        ciphertexts, sa_key, max_iterations=200, score_cache=eval_cache
                    )
                    if is_better(polish_score, cur_score):
                        key_string = polish_key_str
                        cur_score = polish_score
                except Exception:
                    pass

            worker_log = ""

        return key_string, cur_score, worker_log

    except Exception as e:
        if debug:
            import traceback

            return None, float("-inf"), traceback.format_exc()
        return None, float("-inf"), ""


# ---------------------------
# Workers
# ---------------------------


def fragment_seed_worker(args):
    """
    Worker function for parallel fragment seed evaluation.
    Args: (ciphertexts, seed_ordering, key_length, restarts_per_seed, max_iterations, padding, debug)
    Returns: (best_key_string, best_score, worker_log)
    """
    (
        ciphertexts,
        seed_ordering,
        key_length,
        restarts_per_seed,
        max_iterations,
        padding,
        debug,
    ) = args

    try:

        if _should_stop():
            return None, float("-inf"), "[STOPPED]"

        breaker = ADFGVXBreaker(padding=padding)
        seed_key = breaker.string_key_from_column_order(seed_ordering)

        # Run multiple restarts starting from the seed key
        best_key = seed_key
        best_score = float("-inf")

        for restart_idx in range(restarts_per_seed):
            # Check stop signal between restarts
            if _should_stop():
                return best_key if best_key else None, best_score, "[STOPPED]"

            # Polish the seed with hill-climb
            polished_key, polished_score = breaker.polish_key(
                ciphertexts,
                seed_key if restart_idx == 0 else best_key,
                max_iterations=max_iterations,
            )

            if polished_score > best_score:
                best_score = polished_score
                best_key = polished_key

        return best_key, best_score, ""

    except Exception as e:
        if debug:
            import traceback

            return None, float("-inf"), traceback.format_exc()
        return None, float("-inf"), ""


def multi_seed_worker(args):
    """
    Process several seeds sequentially inside one worker process and return
    list of (best_key, best_score, log) for each seed in the chunk.
    Args tuple layout matches fragment_seed_worker but seed_ordering is a list.
    """
    (
        ciphertexts,
        seed_orderings,
        key_length,
        restarts_per_seed,
        max_iterations,
        padding,
        debug,
    ) = args

    results = []
    for seed_ordering in seed_orderings:
        try:
            breaker = ADFGVXBreaker(padding=padding)
            seed_key = breaker.string_key_from_column_order(seed_ordering)

            # Run multiple restarts starting from the seed key
            best_key = seed_key
            best_score = float("-inf")

            for restart_idx in range(restarts_per_seed):
                # Polish the seed with hill-climb
                polished_key, polished_score = breaker.polish_key(
                    ciphertexts,
                    seed_key if restart_idx == 0 else best_key,
                    max_iterations=max_iterations,
                )

                if polished_score > best_score:
                    best_score = polished_score
                    best_key = polished_key

            results.append((best_key, best_score, ""))

        except Exception as e:
            if debug:
                import traceback

                results.append((None, float("-inf"), traceback.format_exc()))
            else:
                results.append((None, float("-inf"), ""))

    return results
