High-level pseudocode

```pseudocode
# Input: ciphertexts (list), optional key_hint, optional alphabet_hint, CONFIG
# Output: ranked candidate plaintexts, keys and alphabets; persisted results (messages.json)

procedure BREAK_ENTRY(ciphertexts, key_hint=None, alphabet_hint=None, CONFIG):
    normalize ciphertexts (remove spaces)
    set module CONFIG for utilities

    # 1. Transposition (column order) search — Multi-Phase Approach
    candidate_lengths := choose_lengths(CONFIG or heuristics)
    for each length in candidate_lengths:
        if enable_fragment_seeding:
            # PHASE 1: Fragment-based seed generation
            seeds := generate_seeds(ciphertexts, length, method=fragment_voting|random)
            for seed in seeds:
                polished := quick_polish(seed, restarts_per_seed, max_iterations_per_seed)
                record candidate polished

            if enable_three_phase_keysearch:
                # PHASE 2: Deeper search on best Phase 1 candidates
                top_candidates := select_top_n(phase1_results, best_candidates_phase_2)
                for candidate in top_candidates:
                    refined := hill_climb(candidate, restarts_phase_2, max_iterations_phase_2)
                    record refined

                # PHASE 3: Final refinement on best Phase 2 candidates
                top_candidates := select_top_n(phase2_results, best_candidates_phase_3)
                for candidate in top_candidates:
                    final := hill_climb(candidate, restarts, max_iterations)
                    record final

            best := best_from_all_phases
        else:
            run hill_climb (many restarts, parallel if enabled)
            optionally run simulated_annealing and polish
            best := best found for this length
        collect per-length candidates and scores

    # 2. Shortlist keys and lightweight re-rank by simple English/German scorer
    shortlist := select_top_keys_per_length(per-length candidates)

    # 3. Alphabet optimization for each shortlisted key (if not key_search_only)
    for each key in shortlist:
        fractionated_streams := decrypt_fractionated_streams(ciphertexts, key)
        base_decoder := BaseDecoderForProcesses(fractionated_streams)
        evaluator := wrap_score_with_ngram_scoring(base_decoder)
        optimizer := AlphabetOptimizer(symbols=initial_alphabet, fix_digits=cfg)
        best_alphabet := optimizer.search(evaluator)
        polished_key := polish_key_with_alphabet(ciphertexts, key, best_alphabet)
        record (key, best_alphabet, recovered_plaintexts, scores)

    # 4. Manual verification (optional) and persistence
    if final_manual_alphabet_testing:
        prompt user over top candidates; accept/reject
    persist results to messages.json according to overwrite policy
    return results

end procedure
```

Cipher Breaker — High-level design and techniques

Overview
This project combines a transposition-key recovery stage (ADFGVX-style fractionation + columnar transposition) with a substitution/alphabet optimization stage (Polybius-square / substitution alphabet). The pipeline is implemented across three cooperating modules:

- cipher_breaker.py — orchestration, I/O, run entrypoints and user interaction.
- cipher_breaker_utils.py — core search primitives for columnar transposition recovery (scoring, transformations, hill-climb, simulated annealing, fragment reconstruction, runners).
- cipher_breaker_helpers.py — helper functions for fragment seeding, adjacency computation, transformations, and phase-specific parameter inference.
- substitution_solver.py — alphabet optimization engine: evaluators, local search, CSP-based pattern constraints, and polishing passes.

Key techniques (transposition stage)

Multi-Phase Key Search (Default: 3 Phases)

The system now uses a **three-phase optimization approach** by default for longer keys:

- **Phase 1 (Seed Generation & Quick Polish)**: Generate many seed orderings using fragment-based methods (fragment voting or random), then quickly polish each with limited restarts.
- **Phase 2 (Deeper Search)**: Take the top candidates from Phase 1 and run deeper hill-climb with more restarts and iterations.
- **Phase 3 (Final Refinement)**: Take the best candidates from Phase 2 and run final intensive hill-climb/hybrid search.

This progressive narrowing allows broad exploration early while focusing compute on promising candidates later.

Combined Bigram/Tetragram IC Scoring

The scoring system uses a **weighted combination of bigram and tetragram Index of Coincidence (IC)**:

- **Bigram IC**: Measures frequency patterns in overlapping 2-character sequences.
- **Tetragram IC**: Measures frequency patterns in overlapping 4-character sequences (more discriminative for English structure).

**Adaptive Weighting Based on Key Length**:

- Base tetragram weight starts at `tetragram_base_weight` (default 0.1)
- Weight increases by `tetragram_boost_per_column` for each column beyond `tetragram_boost_start_length`
- **Even key lengths receive additional tetragram boost** via `tetragram_even_key_additive_value` (configured per key length in `infer_fragment_seeding_variables`)
- Weight is capped at `tetragram_max_weight` (default 0.85)

The even-key boost reflects the observation that even key lengths tend to produce more predictable tetragram patterns in ADFGVX ciphers.

**Tetragram IC Scaling**: Since tetragram IC values are typically 10-20x smaller than bigram IC, a `tetragram_ic_scale_factor` (default ~6-15 depending on key length) normalizes them to comparable ranges.

**Pair-Regularity Scaling**: The tetragram weight is further modulated by ADFGVX pair regularity — keys producing better pair structure get higher tetragram contribution.

Fragment Seeding Strategies

Two main strategies for generating initial seed orderings:

1. **Fragment Voting Seeds** (`fragment_voting_seeds=True`): Analyzes fragment successor frequencies to build orderings based on which columns most frequently follow each other in the interleaved fractionated stream.

2. **Random Seeds** (`random_seeds=True`): Generates completely random shuffled orderings as a baseline approach.

Both strategies produce diversified variants (reversed, rotated, swapped, perturbed) to maximize exploration.

- Fragment extraction and adjacency: ciphertext is split into column fragments (based on assumed key length) and adjacency scores are computed via overlapping bigrams/tetragrams to infer column order.
- Greedy, beam and diversification: greedy ordering gives a baseline; beam search produces multiple plausible orderings; randomized diversification creates many seeds for long keys.
- Lasry-style transformations: the search neighborhood includes swaps, segment swaps, rotations, reversals and pair-swaps to explore column-order permutations.
- Hill-climb (best-improving move): iteratively accept best neighbor; optional short simulated annealing (SA) passes escape local optima.
- Parallelism and dynamic restarts: worker processes run independent restarts; the system can extend restart count dynamically on finding improvements.

Key techniques (alphabet optimization stage, substitution_solver.py)

- Alphabet representation: 36-symbol alphabet (26 letters + 10 digits). The optimizer may fix digits to canonical suffix or allow them to be permuted.
- Evaluator wrapping: BaseDecoderForProcesses decodes fractionated streams to plaintexts; wrap_score_with_ngram_scoring adds trigram/word/digit penalties on top of decoder base score.
- Search strategies:
  - Seed generation: canonical rotations, frequency-based seed (map ETAOIN to observed frequency), pattern-based seed via dictionary pattern matching.
  - Local search / hill-climb: randomized swaps inside a mutable prefix (letters-only or full alphabet), cached evaluations, micro & macro neighbor generators.
  - CSP + pattern extraction: extract consistent partial mappings from long matched words, then solve remaining letters with backtracking and forward checking (CSP).
  - Deterministic polishing: exhaustive pairwise swaps in prefix, focused small-permutation polishing, final greedy digit assignment (when digits movable).
- Scoring: multi-gram log-likelihood (quad/tri/bi/mono), common-word bonuses, and base-decoder score. The NGramScoringEvaluator imposes strong penalties for digits occurring inside words to discourage interspersed digits.

Where to find the workflow and runbook

- The runtime workflow, configuration guidance, examples and troubleshooting tips have been moved to workflow_doc.md. Consult that file for step-by-step operational guidance, inputs/outputs, and recommended parameters.

This document is intended as a compact guide to the techniques and the interactions between modules. For implementation details, inspect the functions/classes named in the three modules described above.

Integration and practical notes (cipher_breaker.py)

- Orchestrator responsibilities: configure run parameters (CONFIG), call ADFGVXBreaker.break_transposition, select top transposition candidates, run alphabet optimizer per candidate, optionally perform manual verification and persist results.
- BaseDecoderForProcesses and worker-friendly evaluators are used to allow parallel alphabet search (ProcessPoolExecutor) while keeping evaluation deterministic and cacheable.
- The system supports Language selection ("EN"/"DE"), debug/intermediate output toggles, long-key heuristics, hybrid hill-climb+SA modes, and saving policies for rejected runs.
- Important configuration keys: restarts, max_iterations, key_search_workers, use_long_key_strategy, use_hybrid, fix_alphabet_digits, alphabet_restarts, alphabet_max_iterations, final_manual_alphabet_testing.

Running tips & diagnostics

- For quick runs: set key_search_workers=1 and small restarts for debugging.
- For long keys enable enable_fragment_seeding and enable_three_phase_keysearch to leverage multi-phase optimization.
- To allow digits to move inside the alphabet search set fix_alphabet_digits=False — but expect strong digit-penalties in evaluator; the optimizer contains greedy digit-assignment helpers to finalize digits.
- Use the provided break_cipher API or break_cipher_from_file to run experiments; the program writes JSON results and can generate ciphertexts from plaintext entries.

Where to look for the core algorithms

- Column ordering, adjacency, transformations, SA & hill-climb: cipher_breaker_utils.py.
- Fragment seeding, phase parameter inference, helper functions: cipher_breaker_helpers.py.
- Alphabet search, CSP mapping, n-gram scoring, polishing passes: substitution_solver.py.
- Orchestration, result merging and user prompts: cipher_breaker.py.

This document is intended as a compact guide to the techniques and the interactions between modules. For implementation details, inspect the functions/classes named in the three modules described above.

Classes and functions — detailed reference

This section lists the main classes and functions across cipher_breaker.py, cipher_breaker_utils.py, cipher_breaker_helpers.py and substitution_solver.py with concise signatures and responsibilities. Use these as a reference when describing the implementation in a paper.

1. cipher_breaker.py — orchestration and I/O

- CONFIG: Dict
  - Central run-time configuration (workers, restarts, thresholds, language, verbosity).
- process_entry_logic(entry_id: str, ciphertexts: List[str], plaintexts: List[str], column_key: str, alphabet: str, cfg: Dict, messages_json_path: str, test_alphabet: Optional[str] = None) -> Dict
  - End-to-end per-entry coordinator: runs transposition recovery, optional alphabet optimization, handles manual verification and returns a serialisable entry dict.
- break_cipher_from_file(entry_id, messages_json_path_override: Optional[str] = None) -> Dict
  - Loads a JSON entry, prepares ciphertexts, delegates to process_entry_logic and persists merged results back to messages.json.
- break_cipher(alphabet: str, column_key: str, plaintexts: Iterable[str], key_lengths: Optional[List[int]] = None, test_alphabet: Optional[str] = None) -> Dict
  - API for programmatic runs: generates ciphertexts and invokes the pipeline.
- debug(\*args, \*\*kwargs)
  - Helper printing controlled by CONFIG['debug_output'].

2. cipher_breaker_utils.py — transposition search and scoring

- set_config(cfg: Dict)
  - Initialize module-level CONFIG copy used by utils for consistent settings.
- set_provided_key_hint(hint: Optional[str]) / get_provided_key_hint()
  - Store and retrieve a user-provided key-hint for diagnostics.
- get_tetragram_weight(key_length: int) -> float
  - Compute tetragram weight based on key length with continuous scaling and even-key boost.
  - Parameters from CONFIG: tetragram_base_weight, tetragram_boost_start_length, tetragram_boost_per_column, tetragram_max_weight.
  - Per-length even-key additive from infer_fragment_seeding_variables.
- ADFGVXBreaker(config: Optional[Dict] = None, padding: Optional[bool] = None)
  - Methods:
    - break_transposition(ciphertexts: List[str], candidate_lengths: Optional[List[int]] = None, restarts: int = 100, max_iterations: int = 1000, ic_threshold: float = 0.05, use_hybrid: Optional[bool] = None) -> Dict[int, Tuple[Optional[str], float]]
      - High-level driver implementing **3-phase key search** when enable_three_phase_keysearch is True.
      - Phase 1: Fragment seeding with quick polish per seed.
      - Phase 2: Deeper search on top Phase 1 candidates.
      - Phase 3: Final refinement on top Phase 2 candidates.
    - score_key_transposition(batch, key_string, cache) -> float
      - **Combined bigram + tetragram IC scorer** with length-dependent weighting.
      - Applies pair-regularity scaling to tetragram weight.
      - Uses tetragram_ic_scale_factor to normalize tetragram IC to bigram range.
    - score_key(ciphertexts: List[str], key_string: str) -> float
      - Composite transposition score (bigram/tetragram IC with adaptive weighting, ADFGVX pair regularity, position-entropy).
    - hill_climb(...) -> Tuple[Optional[str], float]
      - Best-improving Lasry neighborhood hill-climb with optional parallel restarts.
      - Supports injected seed orderings for multi-phase workflow.
    - simulated_annealing(...)-> (key_string, score)
    - polish_key_with_alphabet(ciphertexts, start_key, alphabet, max_iterations=500) -> (key_string, score)
    - decrypt_with_key(ciphertext: str, key_string: str) -> str
- Workers:
  - key_search_worker(args) — Worker for parallel key search with perturbed seed variants.
  - fragment_seed_worker(args) — Worker for parallel fragment seed evaluation.

3. cipher_breaker_helpers.py — fragment seeding and phase parameters

- infer_fragment_seeding_variables(key_length: int) -> Dict[str, int]
  - Returns phase-specific parameters based on key length:
    - Phase 1: total_seeds, restart_per_seed, max_iterations_per_seed
    - Phase 2: best_candidates_phase_2, restarts_phase_2, max_iterations_phase_2
    - Phase 3: best_candidates_phase_3, restarts, max_iterations
    - ic_threshold: key-length specific threshold
    - **tetragram_even_key_additive_value**: extra tetragram weight for even keys
    - **tetragram_ic_scale_factor**: per-length scale factor
- reconstruct_long_key_seeds(ciphertexts: List[str], key_length: int, num_seeds: int = 5) -> List[List[int]]
  - Generates seed orderings using fragment voting or random strategies.
  - When fragment_voting_seeds=True: builds orderings based on successor frequencies.
  - Scores seeds with combined bigram + tetragram IC before selecting top candidates.
- extract_column_fragments(ciphertext: str, key_length: int) -> List[str]
- score_fragment_adjacency(left_frag: str, right_frag: str) -> float
- build_adjacency_matrix(fragments: List[str]) -> List[List[float]]
- greedy_fragment_ordering(adjacency_matrix: List[List[float]]) -> List[int]
- beam_fragment_ordering(adjacency_matrix: List[List[float]], beam_width: int = 5) -> List[List[int]]
- generate_transformations(key_order: List[int]) -> List[Tuple[str, List[int]]]
  - Lasry-style neighborhood moves (swaps, segment swaps, rotations, reverse, pair swaps).
- GlobalLeaderboard class
  - Maintains sorted list of best key candidates across all search phases.
  - Prevents duplicate work and enables dynamic prioritization.

4. substitution_solver.py — alphabet optimization and scoring

- AlphabetOptimizer(symbols: Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", rng: Optional[random.Random] = None, debug: bool = False, workers: int = 4, fix_digits: bool = True, language: str = "EN")
  - High-level responsibilities:
    - Seed generation: \_generate_seed_alphabets(initial, evaluate)
    - Local search: \_local_search(seed, evaluate, iterations)
    - Deterministic polishing: \_deterministic_swap_polish, \_focused_permutation_polish
    - CSP-based partial mapping and backtracking: \_build_partial_mapping, \_solve_with_csp, \_csp_backtrack
    - Digit assignment helpers: \_assign_digits_greedy, \_assign_all_digits_greedy
    - Search orchestration: search(evaluate: ScoreFn, initial: Optional[Alphabet] = None, restarts: int = 8, max_iterations: int = 2000, \*\*kwargs) -> Tuple[Alphabet, float]
      - Parallel worker orchestration with safe single-thread fallback, caching and history tracking.
    - Utilities:
      - \_normalize_alpha(alpha), \_random_alpha(), \_score_alphabet(alpha, evaluate) (wraps base decoder + LM scoring)
      - get_top_alphabets(n=5) -> List[Tuple[float, Alphabet, str]]
  - Caching: evaluation and decode caches to avoid repeated expensive decodes.
  - Language-awareness: per-instance resource paths and n-gram loads; supports EN/DE.
- BaseDecoderForProcesses(fractionated_streams: Iterable[str])
  - Pickleable callable used by worker processes: given an alphabet returns (base_score, List[str]) where base_score is a lightweight language-aware score and the list is decoded plaintexts.
- NGramScoringEvaluator(base_decoder, trigram_weight=3.0, word_weight=2.0, digit_penalty=1.2, common_words_path: Optional[str] = None)
  - Callable wrapper combining:
    - base_decoder(alpha) -> (base_score, plaintexts)
    - trigram averaged log-likelihood on letters-only text
    - common-word proportion bonus
    - strong intra-word digit penalties (catastrophic penalty to avoid digits inside words)
  - Implements evaluate(alpha) and **call** for optimizer compatibility.
- wrap_score_with_ngram_scoring(base_decoder, ...) -> NGramScoringEvaluator
  - Convenience wrapper producing a picklable evaluator for AlphabetOptimizer.

Notes for paper writing (useful phrasing)

- "The transposition stage uses a **three-phase optimization approach**: Phase 1 generates diverse seed orderings via fragment voting (analyzing successor frequencies) or random sampling, with quick polishing; Phase 2 performs deeper hill-climb on the most promising Phase 1 candidates; Phase 3 applies final intensive refinement."
- "Scoring combines **bigram and tetragram Index of Coincidence** with adaptive weighting: tetragram weight increases with key length and receives an additional boost for even key lengths, reflecting the observation that even-length keys produce more predictable tetragram patterns in ADFGVX ciphers."
- "The tetragram weight is further modulated by pair-regularity strength, ensuring keys with stronger ADFGVX pair structure receive higher tetragram contribution."
- "Fragment extraction and adjacency: ciphertext is split into column fragments (based on assumed key length) and adjacency scores are computed via overlapping bigrams to infer column order."
- "Local search operates on column-order permutations using a best-improving move (hill-climb) with optional simulated annealing for diversification."
- "Alphabet optimization treats the 36-symbol alphabet as a permutation optimization problem, combining seed generation (frequency and pattern-based), constrained CSP extraction from long words, hill-climb over a mutable prefix and exhaustive polishing."

This appended reference is designed to be directly cited or expanded into the Methods section of a paper; each bullet gives a concise mapping from code artefact to algorithmic role.

Configuration variables

This section explains the main CONFIG keys used to control behavior across cipher_breaker.py, cipher_breaker_utils.py, cipher_breaker_helpers.py and substitution_solver.py.
Only the most relevant flags and tuning knobs are listed; see the code for exact defaults.

General

- debug_output (bool): Enable verbose debug traces (very chatty). Use when developing or diagnosing algorithm internals.
- intermediate_output (bool): Print status and progress summaries during long runs (recommended for interactive sessions).
- messages_json_path (str | None): Path to messages.json; if None a default auxiliary/messages.json next to the module is used.
- overwrite_json_entries (bool): When True, overwrite top-level entry fields (key, alphabet, plaintexts) with recovered values; otherwise only populate missing fields.
- score_eps (float): Small epsilon for floating-point score comparisons / tie logic.
- language (str): 'EN' or 'DE' — selects language model and common-word lists for scoring.
- padding (bool): If True, trailing 'X' padding is stripped from fractionated streams before scoring.
- initial_alphabet (str): 36-character fallback alphabet (letters then digits) used when an explicit alphabet is missing or invalid.

Tetragram IC Scoring Parameters (NEW)

- tetragram_base_weight (float): Base weight for tetragram IC in combined scoring (default 0.1).
- tetragram_boost_start_length (int): Start boosting tetragram weight at this key length (default 12).
- tetragram_boost_per_column (float): Weight increase per column beyond start length (default 0.02).
- tetragram_max_weight (float): Maximum tetragram weight cap (default 0.85).
- tetragram_ic_scale_factor (float): Scale factor to normalize tetragram IC to bigram IC range (default ~6-15 depending on key length).

Multi-Phase Key Search (NEW)

- enable_fragment_seeding (bool): Enable fragment-based seeding strategy (recommended for all key lengths).
- enable_three_phase_keysearch (bool): Enable three-phase progressive refinement.
- update_global_best_candidates_dynamically (bool): Dynamically update best candidates across phases using GlobalLeaderboard.
- infer_fragment_seeding_variables_automatically (bool): Auto-set phase parameters based on key length.

Phase 1 (Fragment Seeding)

- fragment_voting_seeds (bool): Use fragment voting strategy based on successor frequencies.
- random_seeds (bool): Use random seed generation (overridden by fragment_voting_seeds).
- total_seeds (int): Number of seeds to generate in Phase 1.
- restart_per_seed (int): Restarts per seed during Phase 1 polish.
- max_iterations_per_seed (int): Max iterations per seed in Phase 1.

Phase 2 (Deeper Search)

- best_candidates_phase_2 (int): How many top Phase 1 candidates to process.
- restarts_phase_2 (int): Restarts for Phase 2 hill-climb.
- max_iterations_phase_2 (int): Max iterations for Phase 2.

Phase 3 (Final Refinement)

- best_candidates_phase_3 (int): How many top Phase 2 candidates to process.
- restarts (int): Restarts for Phase 3 (and non-phased) hill-climb.
- max_iterations (int): Max iterations for Phase 3.

Transposition (key search) — existing options

- use_long_key_strategy (bool): When True, use fragment-based seeding & diversification for long keys (>13) instead of pure random restarts.
- use_hybrid (bool): When True, run simulated annealing passes in addition to hill-climb to escape local optima (slower).
- use_common_words_bonus_in_key_search (bool): Enable a small word-presence bonus during lightweight English/German re-ranking of transposition candidates.
- dynamic_restart_addition (int): Percentage used to extend total restart budget when new global bests are discovered.
- non_deterministic_RNG_seed_per_restart (bool): Use non-deterministic RNG seed for each restart (more diverse search).
- early_stop_if_ic_threshold_reached (bool): Stop early when IC threshold is reached.
- ic_threshold (float): IC threshold for early stopping.
- key_search_only (bool): When True, skip alphabet optimization stage.
- key_search_workers (int): Number of processes for parallel key-search.
- top_key_candidates_runs (int): How many top key-lengths to keep for downstream alphabet optimization.

Alphabet optimization

- top_alphabet_candidates_per_key_length (int): How many transposition key candidates to attempt alphabet optimization for per length.
- top_alphabet_candidates_runs (int): How many alphabet candidates to keep per key-order candidate.
- final_manual_alphabet_testing (bool): If True, prompt the user interactively to accept/reject top alphabets (manual verification).
- save_fully_rejected_runs (bool): When True, persist results for lengths even if manual verification rejects all candidates.
- fix_alphabet_digits (bool): When True, keep digits as canonical suffix (optimize only letters); when False optimizer may permute digits (slower, requires special digit assignment passes).
- alphabet_restarts (int): Number of restarts used inside the alphabet optimizer.
- alphabet_max_iterations (int): Max iterations per alphabet optimizer run.
- alphabet_workers (int): Number of parallel worker processes used by AlphabetOptimizer (set to 0/1 to avoid pickling).
- alternating_rounds (int) [optional]: Number of alternating alphabet/key polishing rounds performed in process_entry_logic (if present in cfg).
- temp_start, temp_end (float) [optional]: Temperature schedule endpoints passed to some optimizer routines (if provided in cfg).
- lengths (List[int]) [optional per-entry override]: Candidate key lengths to test for a specific messages.json entry.

Advanced / operational notes

- When pickling problems occur (common when evaluator objects reference local closures), run with alphabet_workers=0 or key_search_workers=1 to force inline execution.
- Increasing restarts and alphabet_restarts increases robustness at the cost of runtime; match restarts/workers to available CPU and patience.
- Use fix_alphabet_digits=True when digits are known to be non-informative (e.g., always appended) to speed search and improve stability.
- The dynamic_restart_addition mechanism can extend the search budget automatically when promising improvements are found; tune the percentage to control aggressiveness.

Example tuning recipe

- Quick debug: {"key_search_workers": 1, "restarts": 50, "alphabet_workers": 0, "alphabet_restarts": 5, "intermediate_output": True, "enable_fragment_seeding": False}
- Serious run (multi-core): {"key_search_workers": N, "enable_fragment_seeding": True, "enable_three_phase_keysearch": True, "infer_fragment_seeding_variables_automatically": True}
- Long-key specialist: enable enable_fragment_seeding, enable_three_phase_keysearch, and set fragment_voting_seeds=True for best results on keys >14 characters.
