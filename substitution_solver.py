from __future__ import annotations

import math
import random
import itertools
from typing import Callable, Iterable, List, Tuple, Optional, Dict, Set
from collections import deque, Counter, defaultdict
import os
import concurrent.futures

Alphabet = str
ScoreFn = Callable[[Alphabet], float]

LANGUAGE = "english"

DICT_ALPHA = os.path.join(
    os.path.dirname(__file__), "auxiliary", LANGUAGE + "_dictionary_alpha.txt"
)
COMMON_WORDS = os.path.join(
    os.path.dirname(__file__), "auxiliary", "top_" + LANGUAGE + "_words_mixed.txt"
)
QUADGRAM_FILE = os.path.join(
    os.path.dirname(__file__), "auxiliary", LANGUAGE + "_quadgrams.txt"
)
TRIGRAM_FILE = os.path.join(
    os.path.dirname(__file__), "auxiliary", LANGUAGE + "_trigrams.txt"
)
BIGRAM_FILE = os.path.join(
    os.path.dirname(__file__), "auxiliary", LANGUAGE + "_bigrams.txt"
)
MONOGRAM_FILE = os.path.join(
    os.path.dirname(__file__), "auxiliary", LANGUAGE + "_monograms.txt"
)


def _process_worker_run_v(args):
    """Worker that does local search + polish without recursion."""
    (
        symbols,
        initial_alpha,
        evaluate,
        restarts,
        max_iterations,
        debug,
        fix_digits,
        seed_int,
    ) = args
    try:
        if debug:
            import io
            import contextlib

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                opt = AlphabetOptimizer(
                    symbols=symbols,
                    rng=random.Random(seed_int),
                    debug=debug,
                    workers=0,
                    fix_digits=fix_digits,
                )
                current = opt._normalize_alpha(initial_alpha)
                current = opt._local_search(
                    current, evaluate, iterations=max_iterations
                )
                current = opt._final_polish(current, evaluate)
                score, sample = opt._score_alphabet(current, evaluate)
                alpha = current
            worker_log = buf.getvalue()
        else:
            opt = AlphabetOptimizer(
                symbols=symbols,
                rng=random.Random(seed_int),
                debug=debug,
                workers=0,
                fix_digits=fix_digits,
            )
            current = opt._normalize_alpha(initial_alpha)
            current = opt._local_search(current, evaluate, iterations=max_iterations)
            current = opt._final_polish(current, evaluate)
            score, sample = opt._score_alphabet(current, evaluate)
            alpha = current
            worker_log = ""

        return alpha, score, sample, worker_log
    except Exception as e:
        if debug:
            import traceback

            return symbols, float("-inf"), "", traceback.format_exc()
        return symbols, float("-inf"), "", ""


class AlphabetOptimizer:
    """
    Constraint-driven alphabet optimizer with:
    - Quadgram language model scoring
    - Pattern-based partial mapping extraction
    - CSP backtracking solver with forward checking
    - Deterministic swap and permutation polishing
    - ProcessPoolExecutor for parallel evaluation
    """

    def __init__(
        self,
        symbols: Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        rng: Optional[random.Random] = None,
        debug: bool = False,
        workers: int = 4,
        fix_digits: bool = True,
        language: str = "EN",
    ):
        if len(symbols) != 36 or len(set(symbols)) != 36:
            raise ValueError("Alphabet must be a 36-symbol permutation.")

        self.symbols = symbols
        self.canonical_alphabet = symbols
        self.rng = rng or random.Random()
        self.debug = debug
        self.workers = workers
        self.fix_digits = bool(fix_digits)

        self.language = language.upper() if language.upper() in ("EN", "DE") else "EN"
        lang = "german" if self.language == "DE" else "english"

        base = os.path.dirname(__file__)
        self.DICT_ALPHA = os.path.join(
            base, "auxiliary", f"{lang}_dictionary_alpha.txt"
        )
        self.COMMON_WORDS_PATH = os.path.join(
            base, "auxiliary", f"top_{lang}_words_mixed.txt"
        )
        self.QUADGRAM_FILE = os.path.join(base, "auxiliary", f"{lang}_quadgrams.txt")
        self.TRIGRAM_FILE = os.path.join(base, "auxiliary", f"{lang}_trigrams.txt")
        self.BIGRAM_FILE = os.path.join(base, "auxiliary", f"{lang}_bigrams.txt")
        self.MONOGRAM_FILE = os.path.join(base, "auxiliary", f"{lang}_monograms.txt")

        self._letters = [c for c in symbols if c.isalpha()]
        self._digits = [c for c in symbols if c.isdigit()]
        self.letter_count = len(self._letters)
        self.symbol_count = len(symbols)
        self._digits_str = "".join(self._digits)

        self.optimize_n = self.letter_count if self.fix_digits else self.symbol_count

        self._eval_cache: Dict[Alphabet, float] = {}
        self._decode_cache: Dict[Alphabet, Tuple[float, List[str]]] = {}
        self._score_cache: Dict[Alphabet, Tuple[float, str]] = {}

        self._quadgram_log, self._quadgram_floor = self._load_ngrams(
            getattr(self, "QUADGRAM_FILE", QUADGRAM_FILE), 4
        )
        self._trigram_log, self._trigram_floor = self._load_ngrams(
            getattr(self, "TRIGRAM_FILE", TRIGRAM_FILE), 3
        )
        self._bigram_log, self._bigram_floor = self._load_ngrams(
            getattr(self, "BIGRAM_FILE", BIGRAM_FILE), 2
        )
        self._monogram_log, self._monogram_floor = self._load_ngrams(
            getattr(self, "MONOGRAM_FILE", MONOGRAM_FILE), 1
        )

        self._common_words = self._load_common_words()
        self._dictionary_patterns = self._build_dictionary_patterns()

        freq_rank = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
        self._symbol_rank = {c: i for i, c in enumerate(freq_rank)}
        for c in self._letters:
            self._symbol_rank.setdefault(c, len(self._symbol_rank))

        self._global_best_history: List[Tuple[float, Alphabet, str]] = []

    def search(
        self,
        evaluate: ScoreFn,
        initial: Optional[Alphabet] = None,
        restarts: int = 8,
        max_iterations: int = 2000,
        **kwargs,
    ) -> Tuple[Alphabet, float]:
        """Main search using constraint-driven approach with parallel workers."""
        if self.workers == 0:
            if self.debug:
                print("[DIRECT] Running single-threaded search (workers=0)")
            current = self._normalize_alpha(initial or self.canonical_alphabet)
            current = self._local_search(current, evaluate, iterations=max_iterations)
            current = self._final_polish(current, evaluate)
            score, sample = self._score_alphabet(current, evaluate)
            self._global_best_history.append((score, current, sample))
            return current, score

        self._eval_cache.clear()
        self._decode_cache.clear()
        self._score_cache.clear()
        self._global_best_history.clear()

        seeds = self._generate_seed_alphabets(initial, evaluate)

        initials: List[Alphabet] = []
        if restarts <= 0:
            restarts = max(1, len(seeds))
        for s in seeds:
            if len(initials) >= restarts:
                break
            initials.append(self._normalize_alpha(s))
        while len(initials) < restarts:
            initials.append(self._random_alpha())

        if self.debug:
            print(f"[SEARCH] Starting search with {restarts} restarts")

        worker_args = []
        for i, init_alpha in enumerate(initials):
            seed_int = self.rng.randint(0, (1 << 30) - 1)
            worker_args.append(
                (
                    self.symbols,
                    init_alpha,
                    evaluate,
                    1,
                    max_iterations,
                    self.debug,
                    self.fix_digits,
                    seed_int,
                )
            )

        results: List[Tuple[float, Alphabet, str]] = []
        max_workers = min(self.workers or 1, len(worker_args) or 1)

        if self.debug:
            print(
                f"[SEARCH] Prepared {len(worker_args)} worker tasks, using max_workers={max_workers}"
            )

        if max_workers <= 1:
            if self.debug:
                print(f"[SEARCH] Running inline (max_workers={max_workers})")
            for a in worker_args:
                try:
                    if self.debug:
                        print(f"[SEARCH] Starting inline worker task")
                    alpha, score, sample, worker_log = _process_worker_run_v(a)
                    if self.debug:
                        print(f"[SEARCH] Inline worker completed: score={score:.4f}")
                except Exception:
                    import traceback

                    if self.debug:
                        print(
                            f"[SEARCH] Inline worker failed:\n{traceback.format_exc()}"
                        )
                    alpha, score, sample, worker_log = (
                        self.symbols,
                        float("-inf"),
                        "",
                        "",
                    )
                try:
                    alpha_norm = self._normalize_alpha(alpha)
                except Exception:
                    alpha_norm = alpha
                results.append((score, alpha_norm, sample))
                self._global_best_history.append((score, alpha_norm, sample))
                if self.debug and worker_log:
                    print(f"[WORKER-LOG]\n{worker_log}", flush=True)
        else:
            if self.debug:
                print(f"[SEARCH] Using ProcessPoolExecutor with {max_workers} workers")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_process_worker_run_v, a) for a in worker_args]
                if self.debug:
                    print(f"[SEARCH] Submitted {len(futures)} futures")
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        alpha, score, sample, worker_log = fut.result()
                        if self.debug:
                            print(f"[SEARCH] Worker completed: score={score:.4f}")
                    except Exception:
                        import traceback

                        if self.debug:
                            print(f"[SEARCH] Worker failed:\n{traceback.format_exc()}")
                        alpha, score, sample, worker_log = (
                            self.symbols,
                            float("-inf"),
                            "",
                            "",
                        )
                    try:
                        alpha_norm = self._normalize_alpha(alpha)
                    except Exception:
                        alpha_norm = alpha
                    results.append((score, alpha_norm, sample))
                    self._global_best_history.append((score, alpha_norm, sample))
                    if self.debug and worker_log:
                        print(f"[WORKER-LOG]\n{worker_log}", flush=True)

        if self.debug:
            print(f"[SEARCH] Collected {len(results)} results")

        if not results:
            best_alpha = self.canonical_alphabet
            best_score = float("-inf")
        else:
            best_score, best_alpha, best_sample = max(results, key=lambda t: t[0])

        if self.debug:
            print(
                f"\n[GLOBAL-BEST] Best across {len(results)} workers: {best_score:.4f} alpha='{best_alpha}'"
            )
            if best_sample:
                print(f"[GLOBAL-BEST] Sample: {best_sample[:150]}...")

        if not self.fix_digits:
            try:
                if self.debug:
                    print(
                        "\n[DIGIT-ASSIGNMENT] Re-assigning all 10 digit positions greedily..."
                    )
                assigned_alpha = self._assign_all_digits_greedy(best_alpha, evaluate)
                assigned_score, assigned_sample = self._score_alphabet(
                    assigned_alpha, evaluate
                )
                best_alpha, best_score = assigned_alpha, assigned_score
                self._global_best_history.append(
                    (best_score, best_alpha, assigned_sample)
                )
                if self.debug:
                    print(
                        f"\n[POST-PROCESS] After digit assignment: {best_score:.4f} alpha='{best_alpha}'"
                    )
                    print(f"[POST-PROCESS] Sample: {assigned_sample[:150]}...")
            except Exception:
                if self.debug:
                    print("Digit assignment failed; returning best worker alphabet")

        return best_alpha, best_score

    def get_top_alphabets(self, n: int = 5) -> List[Tuple[float, Alphabet, str]]:
        """Return top N alphabets from search history."""
        seen_alphas = set()
        unique_results = []

        for score, alpha, sample in sorted(
            self._global_best_history, key=lambda x: x[0], reverse=True
        ):
            if alpha not in seen_alphas:
                seen_alphas.add(alpha)
                unique_results.append((score, alpha, sample))
                if len(unique_results) >= n:
                    break

        return unique_results

    def _generate_seed_alphabets(
        self, initial: Optional[Alphabet], evaluate: ScoreFn
    ) -> List[Alphabet]:
        """Generate diverse high-quality seed alphabets."""
        seeds = [self.canonical_alphabet]

        if initial:
            seeds.append(self._normalize_alpha(initial))

        for shift in [1, 3, 5, 7]:
            letters = self.canonical_alphabet[: self.letter_count]
            rotated = letters[shift:] + letters[:shift]
            seeds.append(rotated + self._digits_str)

        freq_seed = self._frequency_seed(evaluate)
        if freq_seed:
            seeds.append(freq_seed)

        pattern_seed = self._pattern_seed(evaluate)
        if pattern_seed:
            seeds.append(pattern_seed)

        return list(dict.fromkeys(self._normalize_alpha(s) for s in seeds))

    def _frequency_seed(self, evaluate: ScoreFn) -> Optional[Alphabet]:
        """Create alphabet by mapping ETAOIN to observed frequency order."""
        _, texts = self._fetch_plaintexts(self.canonical_alphabet, evaluate)
        if not texts:
            return None

        corpus = "".join(texts).upper()
        letters_only = [c for c in corpus if "A" <= c <= "Z"]
        if not letters_only:
            return None

        freq_counts = Counter(letters_only)
        freq_order = [c for c, _ in freq_counts.most_common()]

        for c in self._letters:
            if c not in freq_order:
                freq_order.append(c)

        eta = "ETAOINSHRDLCUMWFGYPBVKJXQZ"
        mapping = {}
        for src, tgt in zip(eta, freq_order):
            if src in self._letters and tgt in self._letters:
                mapping[src] = tgt

        result = []
        for ch in self._letters:
            result.append(mapping.get(ch, ch))

        return "".join(result[: self.letter_count]) + self._digits_str

    def _pattern_seed(self, evaluate: ScoreFn) -> Optional[Alphabet]:
        """Create alphabet using pattern matching on long words."""
        mapping = self._build_partial_mapping(self.canonical_alphabet, evaluate)
        if not mapping or len(mapping) < 10:
            return None

        candidate = self._solve_with_csp(mapping)
        return candidate if candidate else None

    def _refine_with_constraints(
        self, seed: Alphabet, evaluate: ScoreFn, restarts: int
    ) -> Alphabet:
        """Refine seed using constraint satisfaction and local search."""
        alpha = self._normalize_alpha(seed)
        mapping = self._build_partial_mapping(alpha, evaluate)

        candidate = None
        if mapping and len(mapping) >= 8:
            if self.debug:
                print(f"[CSP] Found {len(mapping)} letter constraints from patterns")
            candidate = self._solve_with_csp(mapping)

        if candidate:
            if self.debug:
                score, _ = self._score_alphabet(candidate, evaluate)
                print(f"[CSP] Solution found with score: {score:.4f}")
            polished = self._final_polish(candidate, evaluate)
            return polished

        if self.debug:
            print(f"[CSP] No solution found, using local search")

        fallback = self._local_search(
            alpha, evaluate, iterations=min(1000, restarts * 200)
        )
        polished = self._final_polish(fallback, evaluate)
        return polished

    def _final_polish(self, alpha: Alphabet, evaluate: ScoreFn) -> Alphabet:
        """Apply deterministic polishing passes."""
        polished = self._deterministic_swap_polish(alpha, evaluate, max_passes=3)
        polished = self._focused_permutation_polish(polished, evaluate)
        return polished

    def _local_search(
        self, seed: Alphabet, evaluate: ScoreFn, iterations: int = 1000
    ) -> Alphabet:
        """Hill-climbing local search over the mutable prefix."""
        current = self._normalize_alpha(seed)
        best_score, _ = self._score_alphabet(current, evaluate)

        for _ in range(iterations):
            if self.rng.random() < 0.3:
                if self.optimize_n <= 2:
                    continue
                i = self.rng.randrange(self.optimize_n - 1)
                j = i + self.rng.randrange(1, min(4, self.optimize_n - i))
            else:
                if self.optimize_n <= 1:
                    continue
                i, j = self.rng.sample(range(self.optimize_n), 2)

            candidate = self._swap_letters(current, i, j)
            score, _ = self._score_alphabet(candidate, evaluate)

            if score > best_score:
                current, best_score = candidate, score

        return current

    def _deterministic_swap_polish(
        self, alpha: Alphabet, evaluate: ScoreFn, max_passes: int = 3
    ) -> Alphabet:
        """Exhaustive pairwise swap polishing over the mutable prefix."""
        best_alpha = self._normalize_alpha(alpha)
        best_score, _ = self._score_alphabet(best_alpha, evaluate)

        for _ in range(max_passes):
            improved = False

            for i in range(self.optimize_n - 1):
                for j in range(i + 1, self.optimize_n):
                    candidate = self._swap_letters(best_alpha, i, j)
                    score, _ = self._score_alphabet(candidate, evaluate)

                    if score > best_score:
                        best_alpha, best_score = candidate, score
                        improved = True
                        if self.debug:
                            print(
                                f"[SWAP] Improved by swapping positions {i},{j}: {best_score:.4f}"
                            )
                        break

                if improved:
                    break

            if not improved:
                break

        return best_alpha

    def _focused_permutation_polish(
        self, alpha: Alphabet, evaluate: ScoreFn
    ) -> Alphabet:
        """Try small permutations on problematic positions inside the mutable prefix."""
        baseline_score, _ = self._score_alphabet(alpha, evaluate)
        prefix = self._prefix(alpha)
        n = len(prefix)

        problematic: Set[int] = set()
        for i in range(max(0, n - 1)):
            swapped = self._swap_letters(alpha, i, i + 1)
            score, _ = self._score_alphabet(swapped, evaluate)
            if score >= baseline_score * 0.999:
                problematic.update({i, i + 1})

        if not problematic:
            problematic.update(range(min(4, n)))

        idx_list = sorted(problematic)[:6]
        if not idx_list:
            return alpha

        best_alpha = alpha
        best_score = baseline_score

        remaining_symbols = [prefix[i] for i in idx_list]

        max_perms = 5000 if len(idx_list) <= 6 else 2000
        checked = 0

        for perm in itertools.permutations(remaining_symbols):
            checked += 1
            if checked > max_perms:
                break

            candidate_prefix = prefix.copy()
            for idx, sym in zip(idx_list, perm):
                candidate_prefix[idx] = sym

            candidate_alpha = self._rebuild_alpha_from_prefix(alpha, candidate_prefix)
            score, _ = self._score_alphabet(candidate_alpha, evaluate)

            if score > best_score:
                best_alpha, best_score = candidate_alpha, score
                if self.debug:
                    print(f"[PERM] Improved: {best_score:.4f}")

        return best_alpha

    def _build_partial_mapping(
        self, alpha: Alphabet, evaluate: ScoreFn, min_len: int = 6
    ) -> Dict[str, str]:
        """Extract partial letter mappings from long word patterns."""
        _, texts = self._fetch_plaintexts(alpha, evaluate)
        if not texts:
            return {}

        mapping: Dict[str, str] = {}
        reverse: Dict[str, str] = {}

        for raw_word in "".join(texts).upper().split():
            word = "".join(ch for ch in raw_word if "A" <= ch <= "Z")
            if len(word) < min_len:
                continue

            pattern = self._pattern(word)
            candidates = self._dictionary_patterns.get(pattern, [])

            if len(candidates) != 1:
                continue

            target = candidates[0]
            pairs = list(zip(target, word))

            consistent = True
            for t_char, obs_char in pairs:
                if obs_char not in self._letters:
                    consistent = False
                    break
                if t_char in mapping and mapping[t_char] != obs_char:
                    consistent = False
                    break
                if obs_char in reverse and reverse[obs_char] != t_char:
                    consistent = False
                    break

            if consistent:
                for t_char, obs_char in pairs:
                    mapping[t_char] = obs_char
                    reverse[obs_char] = t_char

        return mapping

    def _solve_with_csp(self, partial_map: Dict[str, str]) -> Optional[Alphabet]:
        """Solve remaining assignments using CSP backtracking."""
        assignment = dict(partial_map)
        used = set(assignment.values())

        remaining_vars = [c for c in self._letters if c not in assignment]
        remaining_symbols = [c for c in self._letters if c not in used]

        if not remaining_vars:
            return self._alphabet_from_assignment(assignment)

        if len(remaining_vars) <= 8:
            for perm in itertools.permutations(remaining_symbols):
                candidate_map = assignment.copy()
                candidate_map.update(zip(remaining_vars, perm))
                candidate = self._alphabet_from_assignment(candidate_map)
                if candidate:
                    return candidate
            return None

        domains = {var: set(remaining_symbols) for var in remaining_vars}
        result = self._csp_backtrack(assignment, domains, used)

        return self._alphabet_from_assignment(result) if result else None

    def _csp_backtrack(
        self,
        assignment: Dict[str, str],
        domains: Dict[str, Set[str]],
        used: Set[str],
    ) -> Optional[Dict[str, str]]:
        """Backtracking search with forward checking."""
        if len(assignment) == self.letter_count:
            return assignment

        unassigned = [c for c in self._letters if c not in assignment]
        var = min(unassigned, key=lambda v: len(domains.get(v, set())))

        domain = sorted(
            domains.get(var, set(self._letters) - used),
            key=lambda v: self._symbol_rank.get(v, 999),
        )

        for value in domain:
            if value in used:
                continue

            new_assignment = assignment.copy()
            new_assignment[var] = value
            new_used = used | {value}

            new_domains = {
                k: (v - {value}) if k != var else set() for k, v in domains.items()
            }

            if any(
                not vals and k not in new_assignment for k, vals in new_domains.items()
            ):
                continue

            result = self._csp_backtrack(new_assignment, new_domains, new_used)
            if result:
                return result

        return None

    def _alphabet_from_assignment(
        self, assignment: Dict[str, str]
    ) -> Optional[Alphabet]:
        """Convert letter assignment to full alphabet string."""
        if len(set(assignment.values())) != len(assignment):
            return None

        ordered = []
        for ch in self._letters:
            if ch not in assignment:
                return None
            ordered.append(assignment[ch])

        if len(set(ordered)) != len(ordered):
            return None

        return "".join(ordered) + self._digits_str

    def _score_alphabet(self, alpha: Alphabet, evaluate: ScoreFn) -> Tuple[float, str]:
        """Score alphabet using multi-gram LM + word bonus + base evaluator."""
        normalized = self._normalize_alpha(alpha)

        if normalized in self._score_cache:
            return self._score_cache[normalized]

        eval_score = self._cached_evaluate(normalized, evaluate)
        lm_score = 0.0
        word_bonus = 0.0
        sample = ""

        if hasattr(evaluate, "base_decoder"):
            _, texts = self._fetch_plaintexts(normalized, evaluate)
            if texts:
                lm_score = self._language_model_score(texts)
                word_bonus = self._word_bonus(texts)
                sample = texts[0][:200] if texts[0] else ""

        total = eval_score + 1.5 * lm_score + 1.0 * word_bonus

        self._score_cache[normalized] = (total, sample)
        return total, sample

    def _fetch_plaintexts(
        self, alpha: Alphabet, evaluate: ScoreFn
    ) -> Tuple[float, List[str]]:
        """Get plaintexts from base decoder."""
        normalized = self._normalize_alpha(alpha)

        if normalized in self._decode_cache:
            return self._decode_cache[normalized]

        if hasattr(evaluate, "base_decoder"):
            base_score, texts = evaluate.base_decoder(normalized)
            self._decode_cache[normalized] = (base_score, texts)
            return base_score, texts

        return 0.0, []

    def _language_model_score(self, texts: List[str]) -> float:
        """Compute weighted multi-gram log-likelihood score."""
        corpus = "".join(texts).upper()
        letters = "".join(ch for ch in corpus if "A" <= ch <= "Z")

        if len(letters) < 4:
            return 0.0

        scores = []

        if len(letters) >= 4:
            quad_total = 0.0
            quad_count = 0
            for i in range(len(letters) - 3):
                gram = letters[i : i + 4]
                quad_total += self._quadgram_log.get(gram, self._quadgram_floor)
                quad_count += 1
            if quad_count > 0:
                scores.append((4.0, quad_total / quad_count))

        if len(letters) >= 3:
            tri_total = 0.0
            tri_count = 0
            for i in range(len(letters) - 2):
                gram = letters[i : i + 3]
                tri_total += self._trigram_log.get(gram, self._trigram_floor)
                tri_count += 1
            if tri_count > 0:
                scores.append((2.0, tri_total / tri_count))

        if len(letters) >= 2:
            bi_total = 0.0
            bi_count = 0
            for i in range(len(letters) - 1):
                gram = letters[i : i + 2]
                bi_total += self._bigram_log.get(gram, self._bigram_floor)
                bi_count += 1
            if bi_count > 0:
                scores.append((1.0, bi_total / bi_count))

        if scores:
            weighted_sum = sum(w * s for w, s in scores)
            weight_total = sum(w for w, _ in scores)
            return weighted_sum / weight_total

        return 0.0

    def _word_bonus(self, texts: List[str]) -> float:
        """Bonus for common words."""
        bonus = 0.0
        for raw in "".join(texts).upper().split():
            word = "".join(ch for ch in raw if "A" <= ch <= "Z")
            if word in self._common_words:
                bonus += 1.0 + 0.15 * max(0, len(word) - 5)
        return bonus

    def _load_ngrams(self, filepath: str, n: int) -> Tuple[Dict[str, float], float]:
        """Load n-gram frequencies from file."""
        ngrams: Dict[str, int] = {}
        total = 0

        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        gram, count = parts[0].upper(), int(parts[1])
                        if len(gram) == n:
                            ngrams[gram] = ngrams.get(gram, 0) + count
                            total += count

        if not ngrams:
            if n == 4:
                ngrams = {"TION": 10587, "THER": 10010, "THAT": 8755}
            elif n == 3:
                ngrams = {"THE": 50000, "AND": 40000, "ING": 35000}
            elif n == 2:
                ngrams = {"TH": 100000, "HE": 90000, "IN": 80000}
            elif n == 1:
                ngrams = {"E": 120000, "T": 90000, "A": 80000}
            total = sum(ngrams.values())

        log_ngrams = {g: math.log10(c / total) for g, c in ngrams.items()}
        floor = math.log10(0.01 / total)

        return log_ngrams, floor

    def _load_common_words(self) -> Set[str]:
        """Load common words for bonus scoring."""

        def _loader(filepath: Optional[str]) -> Set[str]:
            words = set()
            if filepath and os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        if i >= 5000:
                            break
                        w = line.strip().upper()
                        if w and w.isalpha():
                            words.add(w)
            if not words:
                words = {
                    "THE",
                    "AND",
                    "THAT",
                    "WITH",
                    "HAVE",
                    "THERE",
                    "ABOUT",
                    "WHICH",
                    "BECAUSE",
                    "BETWEEN",
                    "INCLUDE",
                    "WITHOUT",
                    "THROUGH",
                    "ANOTHER",
                    "HOWEVER",
                }
            return words

        return _loader(getattr(self, "COMMON_WORDS_PATH", COMMON_WORDS))

    def _build_dictionary_patterns(self) -> Dict[str, List[str]]:
        """Build pattern dictionary for long word matching."""
        patterns: Dict[str, List[str]] = defaultdict(list)
        filepath = getattr(self, "DICT_ALPHA", DICT_ALPHA)
        if os.path.isfile(filepath):
            with open(filepath, "r", encoding="utf-8") as fh:
                for line in fh:
                    word = line.strip().upper()
                    if len(word) >= 5 and word.isalpha():
                        patterns[self._pattern(word)].append(word)
        else:
            fallback = [
                "BECAUSE",
                "WITHOUT",
                "ANOTHER",
                "BETWEEN",
                "THROUGH",
                "TOWARDS",
                "HIMSELF",
                "YOURSELF",
                "COUNTER",
                "ABILITY",
                "QUESTION",
                "REMEMBER",
            ]
            for word in fallback:
                patterns[self._pattern(word)].append(word)

        return patterns

    def _pattern(self, word: str) -> str:
        """Convert word to pattern string."""
        mapping: Dict[str, int] = {}
        pattern = []
        next_id = 0

        for ch in word:
            if ch not in mapping:
                mapping[ch] = next_id
                next_id += 1
            pattern.append(str(mapping[ch]))

        return "-".join(pattern)

    def _assign_digits_greedy(self, alpha: Alphabet, evaluate: ScoreFn) -> Alphabet:
        """Fill digit slots greedily when fix_digits=False."""
        base = self._normalize_alpha(alpha)
        chars = list(base)
        slots = [i for i, ch in enumerate(chars) if not ch.isalpha()]
        if not slots:
            return base

        for i in slots:
            chars[i] = "?"

        remaining_digits = list("0123456789")
        for pos in slots:
            best_d = None
            best_sc = float("-inf")
            for d in remaining_digits:
                cand = chars.copy()
                cand[pos] = d
                cand_alpha = "".join(cand)
                cand_alpha = self._normalize_alpha(cand_alpha)
                sc, _ = self._score_alphabet(cand_alpha, evaluate)
                if sc > best_sc:
                    best_sc = sc
                    best_d = d
            if best_d is None:
                best_d = remaining_digits[0]
            chars[pos] = best_d
            remaining_digits.remove(best_d)

        final = "".join(chars)
        return self._normalize_alpha(final)

    def _assign_all_digits_greedy(self, alpha: Alphabet, evaluate: ScoreFn) -> Alphabet:
        """Assign all 10 digits greedily when fix_digits=False."""
        base = self._normalize_alpha(alpha)
        letters_part = base[:26]

        current = letters_part + "0123456789"
        current_score, _ = self._score_alphabet(current, evaluate)

        if self.debug:
            print(f"[DIGIT-ASSIGN] Starting: {current} score={current_score:.4f}")

        for pos in range(26, 36):
            best_digit = current[pos]
            best_score = current_score

            for digit in "0123456789":
                if digit == current[pos]:
                    continue
                if digit in current[26:pos] or digit in current[pos + 1 : 36]:
                    continue

                candidate = current[:pos] + digit + current[pos + 1 :]
                score, _ = self._score_alphabet(candidate, evaluate)

                if score > best_score:
                    best_score = score
                    best_digit = digit

            if best_digit != current[pos]:
                old_digit = current[pos]
                swap_pos = current.index(best_digit, 26)
                current = (
                    current[:pos]
                    + best_digit
                    + current[pos + 1 : swap_pos]
                    + old_digit
                    + current[swap_pos + 1 :]
                )
                current_score = best_score
                if self.debug:
                    print(
                        f"[DIGIT-ASSIGN] Position {pos}: chose '{best_digit}' (was '{old_digit}') score={current_score:.4f}"
                    )

        if self.debug:
            print(f"[DIGIT-ASSIGN] Final: {current} score={current_score:.4f}")

        return self._normalize_alpha(current)

    def _prefix(self, alpha: Alphabet) -> List[str]:
        """Get the mutable prefix of an alphabet."""
        return list(alpha[: self.optimize_n])

    def _rebuild_alpha_from_prefix(
        self, alpha: Alphabet, prefix_list: List[str]
    ) -> Alphabet:
        """Rebuild alphabet after changing the prefix."""
        prefix_set = set(prefix_list)
        remaining = [c for c in self.symbols if c not in prefix_set]
        return "".join(prefix_list) + "".join(remaining)

    def _swap_letters(self, alpha: Alphabet, i: int, j: int) -> Alphabet:
        """Swap two mutable positions in the prefix."""
        if i >= self.optimize_n or j >= self.optimize_n:
            return alpha
        p = self._prefix(alpha)
        p[i], p[j] = p[j], p[i]
        return self._rebuild_alpha_from_prefix(alpha, p)

    def _micro_neighbors(
        self, alpha: Alphabet, sample_size: Optional[int] = None
    ) -> Iterable[Alphabet]:
        """Generate micro-neighborhood of small swaps."""
        letters_part = self._prefix(alpha)
        neighbors = []

        for i in range(len(letters_part) - 1):
            p = letters_part.copy()
            p[i], p[i + 1] = p[i + 1], p[i]
            neighbors.append(self._rebuild_alpha_from_prefix(alpha, p))

        for i in range(min(15, len(letters_part))):
            for j in range(i + 2, min(i + 6, len(letters_part))):
                p = letters_part.copy()
                p[i], p[j] = p[j], p[i]
                neighbors.append(self._rebuild_alpha_from_prefix(alpha, p))

        for i in range(min(12, len(letters_part) - 2)):
            for j in range(i + 1, min(i + 4, len(letters_part) - 1)):
                for k in range(j + 1, min(j + 3, len(letters_part))):
                    p = letters_part.copy()
                    p[i], p[j], p[k] = p[k], p[i], p[j]
                    neighbors.append(self._rebuild_alpha_from_prefix(alpha, p))

        if sample_size and sample_size < len(neighbors):
            return self.rng.sample(neighbors, sample_size)
        return neighbors

    def _macro_neighbors(
        self, alpha: Alphabet, sample_size: Optional[int] = None
    ) -> Iterable[Alphabet]:
        """Generate macro-neighborhood of larger swaps."""
        letters_part = self._prefix(alpha)
        neighbors = []

        for i in range(len(letters_part) - 1):
            for j in range(i + 1, len(letters_part)):
                p = letters_part.copy()
                p[i], p[j] = p[j], p[i]
                neighbors.append(self._rebuild_alpha_from_prefix(alpha, p))

        for i in range(min(12, len(letters_part) - 2)):
            for j in range(i + 1, min(i + 4, len(letters_part) - 1)):
                for k in range(j + 1, min(j + 3, len(letters_part))):
                    p = letters_part.copy()
                    p[i], p[j], p[k] = p[k], p[i], p[j]
                    neighbors.append(self._rebuild_alpha_from_prefix(alpha, p))

        if sample_size and sample_size < len(neighbors):
            return self.rng.sample(neighbors, sample_size)
        return neighbors

    def _normalize_alpha(self, alpha: Alphabet) -> Alphabet:
        """Normalize alphabet to ensure valid permutation."""
        seen = set()
        out = []
        for ch in alpha:
            if ch in self.symbols and ch not in seen:
                out.append(ch)
                seen.add(ch)

        for ch in self.symbols:
            if ch not in seen:
                out.append(ch)
                seen.add(ch)

        if self.fix_digits:
            letters_out = [ch for ch in out if ch.isalpha()]
            letters_out = letters_out[: self.letter_count]
            return "".join(letters_out) + self._digits_str
        else:
            return "".join(out[: self.symbol_count])

    def _random_alpha(self) -> Alphabet:
        """Generate a random alphabet."""
        if self.fix_digits:
            r = self.rng.random()

            if r < 0.25:
                letters = [
                    c for c in "ETAOINSHRDLCUMWFGYPBVKJXQZ" if c in self._letters
                ]
                missing = [c for c in self._letters if c not in letters]
                letters.extend(missing)

            elif r < 0.6:
                ranked = sorted(
                    self._letters, key=lambda c: self._symbol_rank.get(c, 999)
                )
                top, mid, low = ranked[:10], ranked[10:20], ranked[20:]
                self.rng.shuffle(top)
                self.rng.shuffle(mid)
                self.rng.shuffle(low)
                letters = top + mid + low

            else:
                letters = self._letters.copy()
                self.rng.shuffle(letters)

            return "".join(letters[: self.letter_count]) + self._digits_str
        else:
            letters = [c for c in "ETAOINSHRDLUCMFWYPBGVKJXQZ" if c in self._letters]
            missing = [c for c in self._letters if c not in letters]
            letters.extend(missing)
            self.rng.shuffle(letters)
            symbols = letters + self._digits.copy()
            self.rng.shuffle(symbols)
            return "".join(symbols[: self.symbol_count])

    def _cached_evaluate(
        self, alpha: Alphabet, evaluate: ScoreFn, executor=None
    ) -> float:
        """Cached evaluation."""
        alpha = self._normalize_alpha(alpha)

        if alpha in self._eval_cache:
            return self._eval_cache[alpha]

        if hasattr(evaluate, "evaluate"):
            score = evaluate.evaluate(alpha)
        else:
            score = evaluate(alpha)

        self._eval_cache[alpha] = score
        return score


class NGramScoringEvaluator:
    """Pickleable evaluator wrapper with n-gram scoring."""

    def __init__(
        self,
        base_decoder,
        trigram_weight=3.0,
        word_weight=2.0,
        digit_penalty=1.2,
        common_words_path: Optional[str] = None,
    ):
        self.base_decoder = base_decoder
        self.trigram_weight = trigram_weight
        self.word_weight = word_weight
        self.digit_penalty = digit_penalty

        self._common_words_path = common_words_path
        self.common_words = self._load_common_words()

        def _load_ngrams_local(filepath: Optional[str], n: int):
            ngrams: Dict[str, int] = {}
            total = 0
            if filepath and os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) == 2:
                            gram, cnt = parts[0].upper(), int(parts[1])
                            if len(gram) == n:
                                ngrams[gram] = ngrams.get(gram, 0) + cnt
                                total += cnt

            if not ngrams:
                if n == 4:
                    ngrams = {"TION": 10587, "THER": 10010, "THAT": 8755}
                elif n == 3:
                    ngrams = {"THE": 50000, "AND": 40000, "ING": 35000}
                elif n == 2:
                    ngrams = {"TH": 100000, "HE": 90000, "IN": 80000}
                elif n == 1:
                    ngrams = {"E": 120000, "T": 90000, "A": 80000}
                total = sum(ngrams.values())

            log_ngrams = {g: math.log10(c / total) for g, c in ngrams.items()}
            floor = math.log10(0.01 / total)
            return log_ngrams, floor

        self.trigram_freqs, self.trigram_floor = _load_ngrams_local(
            getattr(self, "_trigram_path", TRIGRAM_FILE), 3
        )

    def _load_common_words(self) -> Set[str]:
        def _loader(filepath: Optional[str]) -> Set[str]:
            words = set()
            if filepath and os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 2000:
                            break
                        w = line.strip().upper()
                        if w and w.isalpha():
                            words.add(w)
            if not words:
                return {"THE", "AND", "TO", "OF", "A", "IN", "IS", "IT", "YOU"}
            return words

        return _loader(getattr(self, "_common_words_path", COMMON_WORDS))

    def __call__(self, alpha: str) -> float:
        return self.evaluate(alpha)

    def evaluate(self, alpha: str) -> float:
        base_score, plaintexts = self.base_decoder(alpha)

        if not plaintexts:
            return base_score

        combined = "".join(plaintexts).upper()
        letters_only = "".join(ch for ch in combined if ch.isalpha())

        # Trigram scoring
        trigram_score = 0.0
        count = 0
        for i in range(len(letters_only) - 2):
            tri = letters_only[i : i + 3]
            trigram_score += self.trigram_freqs.get(tri, -9.0)
            count += 1
        if count > 0:
            trigram_score /= count

        # Word scoring
        words = []
        for w in combined.split():
            clean = "".join(ch for ch in w if ch.isalpha())
            if clean:
                words.append(clean)
        word_hits = sum(1.0 for w in words if w in self.common_words)
        word_score = word_hits / max(1, len(words))

        # Intra-word digit penalty
        intra_word_digits = 0
        for i, ch in enumerate(combined):
            if ch.isdigit():
                has_left_letter = i > 0 and combined[i - 1].isalpha()
                has_right_letter = i < len(combined) - 1 and combined[i + 1].isalpha()
                if has_left_letter or has_right_letter:
                    intra_word_digits += 1

        total_chars = max(1, len(combined))
        digit_count = sum(1 for c in combined if c.isdigit())
        digit_ratio = digit_count / total_chars

        intra_word_penalty = -50.0 * intra_word_digits
        overall_digit_penalty = -self.digit_penalty * 2.0 * digit_ratio

        if digit_ratio > 0.05:
            overall_digit_penalty -= (digit_ratio - 0.05) * 10.0

        digit_score = intra_word_penalty + overall_digit_penalty

        return (
            base_score
            + self.trigram_weight * trigram_score
            + self.word_weight * word_score
            + digit_score
        )


def wrap_score_with_ngram_scoring(
    base_decoder: Callable[[Alphabet], Tuple[float, List[str]]],
    trigram_weight: float = 3.0,
    word_weight: float = 2.0,
    digit_penalty: float = 1.2,
    common_words_path: Optional[str] = None,
) -> NGramScoringEvaluator:
    """Wrap decoder with n-gram scoring."""
    return NGramScoringEvaluator(
        base_decoder=base_decoder,
        trigram_weight=trigram_weight,
        word_weight=word_weight,
        digit_penalty=digit_penalty,
        common_words_path=common_words_path,
    )
