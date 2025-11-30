from typing import List, Tuple, Dict, Optional, Iterable
import json
import os

from cipher_implementation import (
    create_polybius_square,
    encrypt_message,
    get_column_order,
    reverse_columnar_transposition,
    fractionate_text,
    get_all_possible_key_orders,
    decrypt_message,
)

from substitution_solver import AlphabetOptimizer, wrap_score_with_ngram_scoring
from cipher_breaker_utils import (
    score_texts,
    set_config,
    set_provided_key_hint,
    get_provided_key_hint,
    clear_keyboard_interrupt,
    was_keyboard_interrupted,
)

from cipher_breaker_utils import ADFGVXBreaker, BaseDecoderForProcesses

from cipher_breaker_helpers import set_config_helpers


CONFIG = {
    # GENERAL SETTINGS
    "debug_output": False,  # if True, print detailed debug info during search
    "intermediate_output": True,  # if True, print intermediate results during search
    "messages_json_path": None,  # defaults to messages.json next to this module
    "overwrite_json_entries": False,  # if True, overwrite existing entries (key, alphabet, plaintexts) in messages.json (results are always overwritten)
    "score_eps": 1e-8,  # for floating-point comparisons
    "language": "EN",  # language for alphabet optimization ('EN' or 'DE')
    "padding": True,  # whether to strip trailing 'X' padding in decrypted streams
    "initial_alphabet": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    #
    # TETRAGRAM IC SCORING PARAMETERS
    "tetragram_base_weight": 0.1,  # base weight for tetragram IC
    "tetragram_boost_start_length": 12,  # start boosting tetragram weight at this key length
    "tetragram_boost_per_column": 0.02,  # boost per extra column beyond start length
    "tetragram_max_weight": 0.85,  # maximum tetragram weight cap
    "tetragram_ic_scale_factor": 6.0,  # scale factor to normalize tetragram IC to bigram IC rangeale_factor": 15.0,  # scale factor to normalize tetragram IC to bigram IC range
    # LNEHJAIBKFMDGC
    # KEY SEARCH SETTINGS
    "enable_fragment_seeding": True,  # if True, use fragment seeding strategy (recommended for key length >13)
    "enable_three_phase_keysearch": True,  # if True, use three-phase key search (recommended for key length >16)
    "update_global_best_candidates_dynamically": False,  # if True, dynamically update best candidates during all key searches in phases 2 and 3
    "enable_early_triage": True,  # if True, use early triage to filter unpromising seeds before full search
    #
    # FRAGMENT SEEDING SETTINGS AND INITIAL KEY SEARCH (Phase 1)
    "seeds_per_worker": 1,  # how many seeds each worker process should handle in one chunk
    "infer_fragment_seeding_variables_automatically": True,  # if True, automatically set total_seeds, restart_per_seed, max_iterations_per_seed
    "random_seeds": True,  # if True, seeds are created completely randomly (overridden by fragment_voting_seeds)
    "fragment_voting_seeds": True,  # if True, use fragment voting strategy based on successor frequencies (overrides random_seeds)
    "total_seeds": 100,  # how many seeds to try in fragment seeding
    "restart_per_seed": 2,  # how many restarts per seed
    "max_iterations_per_seed": 1000,  # max iterations per seed
    #
    # PHASE 2 KEY SEARCH SETTINGS
    "best_candidates_phase_2": 10,  # how many best candidates from phase 1 to use in phase 2
    "restarts_phase_2": 1000,
    "max_iterations_phase_2": 5000,
    #
    # PHASE 3 KEY SEARCH SETTINGS
    "best_candidates_phase_3": 2,  # how many best candidates from phase 2 to use in phase 3
    "restarts": 1000,  # for hill-climb search
    "max_iterations": 20000,  # per restart
    #
    # GENERAL KEY SEARCH SETTINGS
    "use_hybrid": False,  # uses additionally simulated annealing after hill-climb for potential improvements (takes >3x time)
    "use_common_words_bonus_in_key_search": False,  # if True, add bonus for common words in key search scoring (can bias if alphabet completely scrambled)
    "infer_key_search_variables_automatically": False,  # DEPRECATED, use infer_fragment_seeding_variables_automatically. if True, automatically set restarts, max_iterations, and ic_threshold
    "dynamic_restart_addition": 10,  # adds additional restarts in percent of restarts amount in case a new global best was found
    "non_deterministic_RNG_seed_per_restart": True,  # if True, use non-deterministic RNG seed for each restart (more diverse search)
    "early_stop_if_ic_threshold_reached": True,
    "ic_threshold": 0.046,
    "ic_earlystop_min_restarts": 10,
    "key_search_only": True,  # if True, skip alphabet optimization
    "key_search_workers": 12,
    "top_key_candidates_runs": 3,  # how many top candidates to keep for all given key lengths
    #
    # SUBSTITUTION SOLVER SETTINGS
    "top_alphabet_candidates_per_key_length": 5,  # how many top alphabet candidates to keep per key length
    "top_alphabet_candidates_runs": 5,  # how many top candidates to keep per key order candidate
    "final_manual_alphabet_testing": True,
    "save_fully_rejected_runs": True,  # writes results for one key length into JSON file, when all were manually rejected
    "fix_alphabet_digits": False,
    "alphabet_restarts": 20,
    "alphabet_max_iterations": 20000,
    "alphabet_workers": 10,
}

set_config(CONFIG)
set_config_helpers(CONFIG)


def debug(*args, **kwargs):
    if CONFIG.get("debug_output", False):
        print(*args, **kwargs)


def process_entry_logic(
    entry_id: str,
    ciphertexts: List[str],
    plaintexts: List[str],
    column_key: str,
    alphabet: str,
    cfg: Dict,
    messages_json_path: str,
    test_alphabet: Optional[str] = None,
):
    """Run the breaker + alphabet-optimizer flow for a single entry."""
    entry: Dict = {}
    entry["plaintexts"] = plaintexts
    entry["ciphertexts"] = ciphertexts
    entry["key"] = column_key
    entry["alphabet"] = alphabet

    if CONFIG.get("intermediate_output", True):
        key_digits = None
        key_len = None
        if column_key:
            key_len = len(column_key)
            try:
                key_digits = get_column_order(column_key)
            except Exception:
                key_digits = None
        lengths_info = cfg.get("lengths", "auto")
        print(
            f"Breaking entry '{entry_id}' with key hint '{column_key}'"
            + (f" (length={key_len})" if key_len is not None else "")
            + (
                f" key_digits={key_digits}"
                if key_digits is not None
                else " (no digit order available)"
            )
            + f" alphabet='{alphabet}' Trying lengths: {lengths_info}\n"
        )

        if key_digits is not None:
            try:
                m = len(key_digits)
                rank = [0] * m
                for r, colpos in enumerate(key_digits):
                    rank[colpos] = r
                encoded_key = "".join(chr(ord("A") + r) for r in rank)
                print(f"Encoded key string: '{encoded_key}'\n\n")
            except Exception:
                pass

    if isinstance(alphabet, str) and len(alphabet) == 36 and len(set(alphabet)) == 36:
        working_alphabet = alphabet
    else:
        working_alphabet = CONFIG.get(
            "initial_alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        if CONFIG.get("intermediate_output", True):
            print(
                "[NOTICE] Provided alphabet missing or invalid -> using fallback canonical alphabet for optimization."
            )

    candidate_lengths = cfg.get("lengths")
    if not isinstance(candidate_lengths, list) or len(candidate_lengths) == 0:
        candidate_lengths = list(range(3, 9))

    breaker = ADFGVXBreaker(config=CONFIG, padding=cfg.get("padding", None))

    prev_provided = get_provided_key_hint()
    set_provided_key_hint(column_key if column_key else None)
    clear_keyboard_interrupt()

    try:
        results = breaker.break_transposition(
            ciphertexts,
            candidate_lengths=candidate_lengths,
            restarts=cfg.get("restarts", 3),
            max_iterations=cfg.get("max_iterations", 500),
            ic_threshold=cfg.get("ic_threshold", 0.12),
            use_hybrid=cfg.get("use_hybrid", None),
        )
    except KeyboardInterrupt:
        if CONFIG.get("intermediate_output", True):
            print("\n[KEYBOARD-INTERRUPT] Received during break_transposition...")
        pass
    finally:
        set_provided_key_hint(prev_provided)

    if was_keyboard_interrupted():
        if CONFIG.get("intermediate_output", True):
            print(
                "\n[INTERRUPT] Keyboard interrupt detected — displaying best candidates found so far...\n"
            )

        clear_keyboard_interrupt()
        per_length = getattr(breaker, "_per_length_candidates", {}) or {}

        results = {}
        for m in candidate_lengths:
            cands = per_length.get(m, [])
            if cands:
                best_score, best_key = max(cands, key=lambda t: t[0])
                results[m] = (best_key, best_score)
            else:
                results[m] = (None, float("-inf"))

        if CONFIG.get("intermediate_output", True):
            provided_hint = get_provided_key_hint()
            possible_orders_set = None
            if provided_hint:
                try:
                    possible_orders = get_all_possible_key_orders(provided_hint)
                    possible_orders_set = {tuple(o) for o in possible_orders}
                except Exception:
                    possible_orders_set = None

            TOP_SHOW = 8
            print("\n" + "=" * 80)
            print("BEST CANDIDATES FOUND (interrupted by Ctrl+C):")
            print("=" * 80 + "\n")

            for m in candidate_lengths:
                sorted_results = sorted(
                    per_length.get(m, []), key=lambda t: t[0], reverse=True
                )
                if not sorted_results:
                    print(f"Key length {m}: No candidates found yet")
                    continue

                top_n = min(TOP_SHOW, len(sorted_results))
                print(f"\nTop {top_n} candidates for key length {m}:")
                matches_for_hint = []

                for idx in range(top_n):
                    score, key_letters = sorted_results[idx]
                    try:
                        key_digits = (
                            get_column_order(key_letters) if key_letters else None
                        )
                    except Exception:
                        key_digits = None

                    if possible_orders_set is not None and key_digits is not None:
                        if tuple(key_digits) in possible_orders_set:
                            matches_for_hint.append((idx + 1, key_digits))

                    if idx < 5 and key_letters:
                        try:
                            decrypted_parts = []
                            for ct in ciphertexts:
                                try:
                                    frac = reverse_columnar_transposition(
                                        ct.replace(" ", ""),
                                        key_letters,
                                        padding=getattr(breaker, "padding", None),
                                    )
                                    frac = breaker._normalize_decrypted(frac)
                                except Exception:
                                    frac = ""
                                decrypted_parts.append(frac)
                            combined = "".join(decrypted_parts)

                            pair_reg = breaker._adfgvx_pair_regularity(combined)
                            entropy_sc = breaker._position_entropy_score(combined)
                            print(
                                f"  [{idx+1}] key='{key_letters}' digits={key_digits} score={score:.6f} "
                                f"(pair_reg={pair_reg:.4f} entropy={entropy_sc:.4f})"
                            )
                        except Exception:
                            print(
                                f"  [{idx+1}] key='{key_letters}' digits={key_digits} score={score:.6f}"
                            )
                    else:
                        print(
                            f"  [{idx+1}] key='{key_letters}' digits={key_digits} score={score:.6f}"
                        )

                if possible_orders_set is not None:
                    if matches_for_hint:
                        print("\n  Matches with provided key hint:")
                        for rank_no, digits in matches_for_hint:
                            print(f"    [{rank_no}] {digits}")
                    else:
                        print(
                            "\n  (No matches with provided key hint in top candidates)"
                        )

            print("\n" + "=" * 80)
            print("Search interrupted - results above are partial")
            print("=" * 80 + "\n")

        entry["_keyboard_interrupt"] = True
        entry["results"] = [
            {
                "length": m,
                "key": (results[m][0] if m in results else None),
                "score": (results[m][1] if m in results else float("-inf")),
            }
            for m in sorted(results.keys())
        ]

    per_length = getattr(breaker, "_per_length_candidates", {})

    if cfg.get("key_search_only", False):
        results_list = []
        try:
            _, inverse_poly_map = create_polybius_square(alphabet)
        except Exception:
            _, inverse_poly_map = create_polybius_square(
                CONFIG.get("initial_alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            )

        for length, (key_string, score) in results.items():
            entry_result = {
                "length": length,
                "key": key_string,
                "score": score,
                "column_order": get_column_order(key_string) if key_string else None,
                "recovered_plaintexts": [],
                "matches": [],
            }
            if key_string:
                for ct in ciphertexts:
                    try:
                        plain = decrypt_message(ct, inverse_poly_map, key_string)
                    except Exception as e:
                        plain = f"<decryption_error:{e}>"
                    entry_result["recovered_plaintexts"].append(plain)
            results_list.append(entry_result)

        if results_list:
            results_list_sorted = sorted(
                results_list, key=lambda x: x.get("score", 0), reverse=True
            )
            top3 = results_list_sorted[:3]
            entry["results"] = [
                {
                    "length": r["length"],
                    "key": r["key"],
                    "score": r["score"],
                    "recovered_plaintexts": r["recovered_plaintexts"],
                }
                for r in top3
            ]
        else:
            entry["results"] = []

        breaker._optimizer_outcomes = {}
        return entry

    # Alphabet optimization
    optimizer_outcomes: Dict[int, List[Dict]] = {}
    TOP_CANDIDATES_PER_LENGTH = cfg.get("top_alphabet_candidates_per_key_length", 10)
    TOP_LENGTH_RUNS = cfg.get("top_key_candidates_runs", 3)
    TOP_ALPHABET_CANDIDATES = cfg.get("top_alphabet_candidates_runs", 5)
    all_alphabet_candidates: List[Tuple[float, str, List[str], int, str, float]] = []

    lengths_ranked = sorted(
        results.keys(),
        key=lambda L: (results.get(L, (None, float("-inf")))[1] or float("-inf")),
        reverse=True,
    )
    lengths_to_optimize = lengths_ranked[:TOP_LENGTH_RUNS]
    if CONFIG.get("intermediate_output", True):
        print(
            f"Optimizing alphabets for top {len(lengths_to_optimize)} lengths: {lengths_to_optimize}"
        )

    for length in lengths_to_optimize:
        best_key_string, best_score = results.get(length, (None, float("-inf")))
        cands = per_length.get(length, [])
        if not cands:
            continue
        cands_sorted = sorted(cands, key=lambda x: x[0], reverse=True)

        try:
            _, canonical_inv = create_polybius_square(
                CONFIG.get("initial_alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            )
        except Exception:
            _, canonical_inv = create_polybius_square(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

        eng_score_cache = {}

        def candidate_english_score(key_letters: str) -> float:
            if not key_letters:
                return float("-inf")
            if key_letters in eng_score_cache:
                return eng_score_cache[key_letters]
            texts = [
                decrypt_message(ct, canonical_inv, key_letters) for ct in ciphertexts
            ]
            sc = score_texts(texts, CONFIG.get("language", "EN").upper())
            eng_score_cache[key_letters] = sc
            return sc

        seen_keys = set()
        enriched = []
        for key_score, key_letters in cands_sorted:
            if not key_letters or key_letters in seen_keys:
                continue
            seen_keys.add(key_letters)
            eng_sc = candidate_english_score(key_letters)
            enriched.append((key_score, key_letters, eng_sc))

        enriched_sorted = sorted(enriched, key=lambda t: (t[0], t[2]), reverse=True)
        cands_sorted = [(s, k) for s, k, _ in enriched_sorted]
        k_take = min(TOP_CANDIDATES_PER_LENGTH, len(cands_sorted))
        optimizer_outcomes[length] = []
        print(
            f"\nRunning alphabet optimization for top {k_take} candidates (key length={length}):"
        )

        for rank, (key_score, key_string) in enumerate(cands_sorted[:k_take], start=1):
            print(
                f"\n  Candidate #{rank}: key='{key_string}' key_score={key_score:.6f} column_order={get_column_order(key_string)}"
            )

            fractionated_streams = [
                breaker.decrypt_with_key(ct.replace(" ", ""), key_string)
                for ct in ciphertexts
            ]
            base_decoder = BaseDecoderForProcesses(fractionated_streams)
            evaluate = wrap_score_with_ngram_scoring(
                base_decoder, trigram_weight=2.0, word_weight=1.0, digit_penalty=0.8
            )

            symbols_domain = CONFIG.get(
                "initial_alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
            optimizer = AlphabetOptimizer(
                symbols=symbols_domain,
                debug=CONFIG.get("intermediate_output", True),
                workers=cfg.get("alphabet_workers", 1),
                fix_digits=cfg.get("fix_alphabet_digits", True),
                language=cfg.get("language", "EN"),
            )

            ALTERNATING_ROUNDS = cfg.get("alternating_rounds", 2)
            current_key = key_string
            try:
                current_eval_score = evaluate(symbols_domain)
            except Exception:
                current_eval_score = float("-inf")
            try:
                current_english_score, _ = base_decoder(symbols_domain)
            except Exception:
                current_english_score = float("-inf")
            current_key_score = current_eval_score

            alpha_best_overall = symbols_domain
            alpha_score_overall = float("-inf")

            for rnd in range(ALTERNATING_ROUNDS):
                try:
                    alpha_candidate, alpha_candidate_score = optimizer.search(
                        evaluate=evaluate,
                        initial=(
                            alpha_best_overall
                            if alpha_score_overall > float("-inf")
                            else symbols_domain
                        ),
                        restarts=cfg.get("alphabet_restarts", 5),
                        max_iterations=cfg.get("alphabet_max_iterations", 5000),
                        temp_start=cfg.get("temp_start", 0.3),
                        temp_end=cfg.get("temp_end", 0.02),
                        adaptive_neighbors=True,
                        macro_move_prob=0.08,
                    )
                except Exception as e:
                    print(f"    Optimizer failed on round {rnd+1}: {e}")
                    break

                if alpha_candidate_score > alpha_score_overall:
                    alpha_best_overall = alpha_candidate
                    alpha_score_overall = alpha_candidate_score

                base_score_val, recovered_texts = base_decoder(alpha_candidate)

                try:
                    polished_key, polished_score = breaker.polish_key_with_alphabet(
                        ciphertexts, current_key, alpha_candidate, max_iterations=200
                    )
                except Exception as e:
                    polished_key, polished_score = current_key, current_key_score
                    if CONFIG.get("intermediate_output", True):
                        print(f"    [POLISH] polishing failed on round {rnd+1}: {e}")

                if polished_score > current_english_score:
                    if CONFIG.get("intermediate_output", True):
                        print(
                            f"    [ROUND {rnd+1}] Polished key accepted: '{polished_key}' english_score={polished_score:.4f} (was {current_english_score:.4f})"
                        )
                    current_key = polished_key
                    fractionated_streams = [
                        breaker.decrypt_with_key(ct.replace(" ", ""), current_key)
                        for ct in ciphertexts
                    ]
                    base_decoder = BaseDecoderForProcesses(fractionated_streams)
                    evaluate = wrap_score_with_ngram_scoring(
                        base_decoder,
                        trigram_weight=2.0,
                        word_weight=1.0,
                        digit_penalty=0.8,
                    )
                    try:
                        current_eval_score = evaluate(symbols_domain)
                    except Exception:
                        current_eval_score = float("-inf")
                    try:
                        current_english_score, _ = base_decoder(symbols_domain)
                    except Exception:
                        current_english_score = float("-inf")
                    current_key_score = current_eval_score
                else:
                    if CONFIG.get("intermediate_output", True):
                        print(
                            f"    [ROUND {rnd+1}] Polished key not better: polished_english={polished_score:.4f} <= current_english={current_english_score:.4f}"
                        )

            try:
                _, recovered_texts = base_decoder(alpha_best_overall)
            except Exception:
                recovered_texts = []

            print(
                f"    Best alphabet: '{alpha_best_overall}' score={alpha_score_overall:.4f}"
            )
            print("    Sample recovered plaintexts (first two):")
            for ti, rec in enumerate(recovered_texts[:2], start=1):
                print(f"      #{ti}: {rec[:200]}{'...' if len(rec)>200 else ''}")

            optimizer_outcomes[length].append(
                {
                    "candidate_rank": rank,
                    "key_letters": current_key,
                    "key_score": current_key_score,
                    "alphabet": alpha_best_overall,
                    "alpha_score": alpha_score_overall,
                    "recovered_texts": recovered_texts,
                }
            )

            if cfg.get("final_manual_alphabet_testing", False):
                TOP_PROMPT_PER_LENGTH = min(
                    5, cfg.get("top_alphabet_candidates_runs", 5)
                )
                top_k = sorted(
                    optimizer_outcomes.get(length, []),
                    key=lambda e: e.get("alpha_score", float("-inf")),
                    reverse=True,
                )[:TOP_PROMPT_PER_LENGTH]

                if top_k:
                    print(
                        f"\nManual review for key length={length} (top {len(top_k)} candidates):"
                    )
                    for rank, cand in enumerate(top_k, start=1):
                        alpha = cand.get("alphabet")
                        a_score = cand.get("alpha_score")
                        recovered = cand.get("recovered_texts", [])
                        key_letters = cand.get("key_letters")

                        print(
                            f"\n  Candidate #{rank} (alpha_score={a_score:.4f}) key='{key_letters}'\n  Alphabet: '{alpha}'"
                        )
                        if recovered:
                            for mi, rec in enumerate(recovered[:2], start=1):
                                print(
                                    f"    Message {mi}: {rec[:200]}{'...' if len(rec)>200 else ''}"
                                )
                        else:
                            print("    (no recovered plaintexts available for preview)")

                        resp = (
                            input(
                                "\nIs this the correct alphabet? (y/1 = yes, n/0 = no): "
                            )
                            .strip()
                            .lower()
                        )
                        if resp in ("y", "1", "yes"):
                            entry_result = {
                                "length": length,
                                "key": key_letters,
                                "alphabet": alpha,
                                "score": a_score,
                                "recovered_plaintexts": recovered,
                            }
                            entry["results"] = [entry_result]
                            breaker._optimizer_outcomes = optimizer_outcomes
                            return entry

    breaker._optimizer_outcomes = optimizer_outcomes

    # Prepare final results
    manual_verification = cfg.get("final_manual_alphabet_testing", False)
    save_fully_rejected = cfg.get("save_fully_rejected_runs", False)

    length_best_alpha_score: Dict[int, float] = {}
    for length, entries in optimizer_outcomes.items():
        if entries:
            best = max(entries, key=lambda e: e.get("alpha_score", float("-inf")))
            length_best_alpha_score[length] = best.get("alpha_score", float("-inf"))
    for length, (kstr, kscore) in results.items():
        length_best_alpha_score.setdefault(
            length, kscore if kscore is not None else float("-inf")
        )

    lengths_sorted = sorted(
        length_best_alpha_score.keys(),
        key=lambda L: length_best_alpha_score[L],
        reverse=True,
    )

    if manual_verification:
        for length in lengths_sorted:
            cands = optimizer_outcomes.get(length, [])
            if not cands:
                continue
            cands_sorted = sorted(
                cands, key=lambda e: e.get("alpha_score", float("-inf")), reverse=True
            )
            for rank, cand in enumerate(cands_sorted, start=1):
                alpha = cand.get("alphabet")
                a_score = cand.get("alpha_score")
                recovered = cand.get("recovered_texts", [])
                key_letters = cand.get(
                    "key_letters", results.get(length, (None, None))[0]
                )
                print(
                    f"\nLength {length} — Candidate #{rank} (alpha_score={a_score:.4f})\nAlphabet: '{alpha}'\nSample: {recovered[0][:200] if recovered else ''}\n"
                )
                for mi, rec in enumerate(recovered, start=1):
                    print(f"  Message {mi}: {rec[:150]}{'...' if len(rec)>150 else ''}")
                resp = (
                    input("\nIs this the correct alphabet? (y/1 = yes, n/0 = no): ")
                    .strip()
                    .lower()
                )
                if resp in ("y", "1", "yes"):
                    entry_result = {
                        "length": length,
                        "key": key_letters,
                        "alphabet": alpha,
                        "score": a_score,
                        "recovered_plaintexts": recovered,
                    }
                    entry["results"] = [entry_result]
                    return entry
        if not save_fully_rejected:
            entry["results"] = []
            return entry

    results_list = []
    provided_plaintexts_norm = None
    try:
        _, inverse_poly_map = create_polybius_square(alphabet)
        valid_chars = set(inverse_poly_map.values())
    except Exception:
        valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    if plaintexts:
        provided_plaintexts_norm = []
        for pt in plaintexts:
            norm = "".join(ch for ch in pt.upper() if ch in valid_chars)
            provided_plaintexts_norm.append(norm)

    for length, (key_string, score) in results.items():
        entry_result = {
            "length": length,
            "key": key_string,
            "score": score,
            "column_order": get_column_order(key_string) if key_string else None,
            "recovered_plaintexts": [],
            "matches": [],
        }
        if key_string:
            for ct in ciphertexts:
                try:
                    plain = decrypt_message(ct, inverse_poly_map, key_string)
                except Exception as e:
                    plain = f"<decryption_error:{e}>"
                entry_result["recovered_plaintexts"].append(plain)
            if provided_plaintexts_norm:
                for rec, expected in zip(
                    entry_result["recovered_plaintexts"], provided_plaintexts_norm
                ):
                    entry_result["matches"].append(rec == expected)

        if entry_result["recovered_plaintexts"] or save_fully_rejected:
            results_list.append(entry_result)

    if results_list:
        results_list_sorted = sorted(
            results_list, key=lambda x: x.get("score", 0), reverse=True
        )
        top3 = results_list_sorted[:3]
        entry["results"] = [
            {
                "length": r["length"],
                "key": r["key"],
                "score": r["score"],
                "recovered_plaintexts": r["recovered_plaintexts"],
            }
            for r in top3
        ]
    else:
        entry["results"] = []

    return entry


def break_cipher_from_file(entry_id, messages_json_path_override: Optional[str] = None):
    """Load entry from messages.json, run the breaker, and write results back."""
    set_config(CONFIG)

    cfg = CONFIG.copy()
    if messages_json_path_override is not None:
        cfg["messages_json_path"] = messages_json_path_override

    messages_json_path = cfg["messages_json_path"]
    if messages_json_path is None:
        messages_json_path = os.path.join(
            os.path.dirname(__file__), "auxiliary", "messages.json"
        )

    restarts = cfg["restarts"]
    max_iterations = cfg["max_iterations"]
    ic_threshold = cfg["ic_threshold"]
    use_hybrid = cfg["use_hybrid"]

    if not os.path.isfile(messages_json_path):
        raise FileNotFoundError(f"messages.json not found at: {messages_json_path}")

    with open(messages_json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    key_str = str(entry_id)
    if key_str not in data:
        raise KeyError(f"Entry '{key_str}' not found in messages.json")

    entry = data[key_str]
    plaintexts = entry.get("plaintexts", [])
    ciphertexts = entry.get("ciphertexts", []) or []
    column_key = entry.get("key")
    alphabet = entry.get("alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

    if column_key is None:
        raise ValueError(f"Entry '{key_str}' has no 'key' field")

    prev_lang = CONFIG.get("language", "EN")
    entry_lang = entry.get("language")
    if isinstance(entry_lang, str) and entry_lang.strip():
        entry_lang_up = entry_lang.strip().upper()
        if entry_lang_up in ("EN", "DE"):
            CONFIG["language"] = entry_lang_up
            if CONFIG.get("intermediate_output", True):
                print(
                    f"[LANG] Using language='{CONFIG['language']}' from messages.json entry '{key_str}'"
                )

    entry_lengths = entry.get("lengths")
    if isinstance(entry_lengths, list) and entry_lengths:
        cfg["lengths"] = entry_lengths
        print(
            f"Using candidate lengths from messages.json for entry '{key_str}': {cfg['lengths']}"
        )

    if (not ciphertexts) and plaintexts:
        print(
            f"Entry '{key_str}': no ciphertexts found — generating from plaintexts using key '{column_key}'"
        )
        try:
            polybius_map, _ = create_polybius_square(alphabet)
        except Exception:
            polybius_map, _ = create_polybius_square(
                CONFIG.get("initial_alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
            )
            if CONFIG.get("intermediate_output", True):
                print(
                    "[NOTICE] Provided alphabet missing/invalid while generating ciphertexts; used fallback canonical alphabet."
                )
        generated = []
        for i, pt in enumerate(plaintexts, start=1):
            frac = fractionate_text(pt, polybius_map)
            ct = encrypt_message(pt, polybius_map, column_key)
            generated.append(ct)
            print(
                f"  [{i}] Plaintext -> fractionated: {frac[:80]}{'...' if len(frac)>80 else ''}"
            )
            print(f"       Ciphertext: {ct[:80]}{'...' if len(ct)>80 else ''}")
        ciphertexts = generated
        entry["ciphertexts"] = ciphertexts
        data[key_str] = entry
        with open(messages_json_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        print(
            f"  Wrote {len(ciphertexts)} generated ciphertext(s) back to {messages_json_path}\n"
        )

    if not ciphertexts:
        raise ValueError(f"Entry '{key_str}' contains no ciphertexts to break")

    updated_entry = process_entry_logic(
        entry_id=key_str,
        ciphertexts=ciphertexts,
        plaintexts=plaintexts,
        column_key=column_key,
        alphabet=alphabet,
        cfg=cfg,
        messages_json_path=messages_json_path,
        test_alphabet=None,
    )

    if updated_entry.get("_keyboard_interrupt"):
        print(
            f"Processing for entry '{key_str}' was interrupted by user; not writing results to {messages_json_path}"
        )
        CONFIG["language"] = prev_lang
        return updated_entry

    if cfg.get("key_search_only", False):
        print(
            f"Key-search-only mode enabled: not writing results back to {messages_json_path}"
        )
        return updated_entry

    existing_entry = data.get(key_str, {})
    existing_entry["results"] = updated_entry.get("results", [])

    write_key = None
    write_alphabet = None
    top_results = existing_entry.get("results") or []
    if top_results:
        top0 = top_results[0]
        write_key = top0.get("key") or column_key
        write_alphabet = top0.get("alphabet") or alphabet
    else:
        write_key = column_key
        write_alphabet = alphabet

    recovered_plaintexts: List[str] = []
    try:
        _, inv_map = create_polybius_square(write_alphabet)
        for ct in ciphertexts:
            try:
                recovered_plaintexts.append(decrypt_message(ct, inv_map, write_key))
            except Exception:
                recovered_plaintexts.append("")
    except Exception:
        recovered_plaintexts = []

    overwrite_all = cfg.get("overwrite_json_entries", False)

    if overwrite_all:
        existing_entry["key"] = write_key
        existing_entry["alphabet"] = write_alphabet
        existing_entry["plaintexts"] = recovered_plaintexts
    else:
        if not existing_entry.get("key"):
            existing_entry["key"] = write_key
        if not existing_entry.get("alphabet"):
            existing_entry["alphabet"] = write_alphabet
        if not existing_entry.get("plaintexts"):
            existing_entry["plaintexts"] = recovered_plaintexts

    data[key_str] = existing_entry
    with open(messages_json_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)

    print(
        f"Finished processing entry '{key_str}', results written to {messages_json_path}"
    )
    CONFIG["language"] = prev_lang

    return updated_entry


def break_cipher(
    alphabet: str,
    column_key: str,
    plaintexts: Iterable[str],
    key_lengths: Optional[List[int]] = None,
    test_alphabet: Optional[str] = None,
):
    """Run the breaker + optimizer pipeline for provided alphabet, key, and plaintexts."""
    set_config(CONFIG)

    if isinstance(plaintexts, str):
        pts = [plaintexts]
    else:
        pts = list(plaintexts)

    try:
        polybius_map, _ = create_polybius_square(alphabet)
    except Exception:
        polybius_map, _ = create_polybius_square(
            CONFIG.get("initial_alphabet", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        )

    ciphertexts = [encrypt_message(pt, polybius_map, column_key) for pt in pts]

    cfg = CONFIG.copy()
    if key_lengths is not None:
        cfg["lengths"] = list(key_lengths)

    updated_entry = process_entry_logic(
        entry_id="break_cipher",
        ciphertexts=ciphertexts,
        plaintexts=pts,
        column_key=column_key,
        alphabet=alphabet,
        cfg=cfg,
        messages_json_path=os.path.join(os.path.dirname(__file__), "messages.json"),
        test_alphabet=test_alphabet,
    )

    print("\nBreak-cipher finished.")
    for r in updated_entry.get("results", []):
        print(f"  length={r.get('length')} key={r.get('key')} score={r.get('score')}")
    return updated_entry


if __name__ == "__main__":

    # Example invocation: provide alphabet, key, plaintexts and optional key lengths

    # plaintexts = [
    #     "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOGS AND DARTS THROUGH THE MEADOW WHILE BIRDS SING SWEETLY IN THE TREES AND THE SUN SHINES BRIGHTLY ABOVE THE FIELD.",
    #     "EVERY GOOD BOY DESERVES FUN AND EVERY GIRL SHOULD LEARN TO READ BOOKS DAILY TO EXPAND HER KNOWLEDGE, ENJOY LITERATURE, AND IMPROVE HER THINKING AND WISDOM.",
    #     "PACK MY BOX WITH FIVE DOZEN LIQUOR BOTTLES CAREFULLY SO NONE BREAKS DURING TRANSPORTATION, AND MAKE SURE ALL LABELS ARE FACING THE CORRECT DIRECTION FOR EASY INSPECTION.",
    #     "JUMPING FOXES AVOID THE HUNTER BY RUNNING THROUGH DENSE FORESTS, LEAPING OVER FENCES, AND HIDING IN SHRUBS WHILE THE SUN SETS AND SHADOWS STRETCH ACROSS THE HILLS AND VALLEYS.",
    #     "IN THE SILENT NIGHT, STARS SPARKLE BRIGHTLY ABOVE THE QUIET VILLAGE, WHILE PEOPLE SLEEP PEACEFULLY, WINDS SWAY THE TREES, AND THE MOON CASTS SOFT LIGHT ACROSS THE FIELDS.",
    # ]

    # break_cipher(
    #     alphabet="0ZDEFABCGHIJKLMNOPQRSTUVWXY123456789",
    #     column_key="MATHES",
    #     plaintexts=plaintexts,
    #     key_lengths=[5, 6],
    #     test_alphabet="0ZDEFABCGHIJKLMNOPQRSTUVWXY123456789",
    # )

    break_cipher_from_file(5)
