Cipher Breaker — Workflow and runtime guide

Purpose
This document explains the runtime workflow of the cipher_breaker system: expected inputs, the sequential processing stages, how modules interact at runtime, recommended configuration knobs and practical troubleshooting tips.

Quick overview (one-line)
Given ciphertext(s) and an optional key hint, the system attempts to recover a columnar transposition key using a **multi-phase optimization** with **combined bigram/tetragram scoring**, then optionally recovers the underlying Polybius/substitution alphabet, producing best candidate plaintexts and a persisting result entry.

Inputs and primary data structures

- messages.json entry (recommended): fields used are "plaintexts", "ciphertexts", "key" (column key hint), "alphabet" and optional "lengths" and "language".
- In-memory pipelines: lists of cleaned ciphertext strings (spaces removed), fractionated streams (pairs for Polybius decoding), candidate alphabets (36-char permutations).
- Config object (CONFIG): controls search depth, parallelism, language, output verbosity and behavior when saving results.

High-level pipeline (step-by-step)

1. Pre-flight and normalization

   - Read messages.json entry or accept direct inputs via break_cipher or break_cipher_from_file.
   - Ensure an initial alphabet exists (fallback: canonical "A..Z0..9"), normalize ciphertexts (strip spaces).
   - Optionally generate ciphertexts from plaintexts when ciphertexts are missing.

2. Transposition-recovery (ADFGVX/columnar stage) — **Multi-Phase Approach**

   ADFGVXBreaker.break_transposition tries a set of candidate key lengths. For each length, the system now uses a **three-phase optimization** by default:

   **Phase 1: Seed Generation & Quick Polish**

   - Generate seed orderings using one of two strategies:
     - **Fragment Voting** (`fragment_voting_seeds=True`): Analyzes which columns frequently follow each other based on successor frequencies in the fractionated stream.
     - **Random Seeds** (`random_seeds=True`): Generates completely random orderings.
   - Seeds are diversified with variations (reversed, rotated, swapped, perturbed).
   - Each seed is quickly polished with limited restarts (`restart_per_seed`) and iterations (`max_iterations_per_seed`).
   - Seeds are scored using **combined bigram + tetragram IC** to select top candidates.

   **Phase 2: Deeper Search**

   - Take the top `best_candidates_phase_2` candidates from Phase 1.
   - Run deeper hill-climb with `restarts_phase_2` restarts and `max_iterations_phase_2` iterations.
   - Update global leaderboard with improved candidates.

   **Phase 3: Final Refinement**

   - Take the top `best_candidates_phase_3` candidates from Phase 2.
   - Run final intensive hill-climb (or hybrid search) with full `restarts` and `max_iterations`.
   - Produce the best key for this length.

   **Scoring: Combined Bigram + Tetragram IC**

   The scoring function combines bigram and tetragram Index of Coincidence with adaptive weighting:

   ```
   score = bigram_weight × bigram_IC + tetragram_weight × (tetragram_IC × scale_factor)
   ```

   Where:

   - `tetragram_weight` increases with key length (starts at `tetragram_base_weight`, increases by `tetragram_boost_per_column` per column beyond `tetragram_boost_start_length`)
   - **Even key lengths get additional tetragram boost** via `tetragram_even_key_additive_value`
   - `scale_factor` normalizes tetragram IC to bigram IC range (tetragram IC is typically 10-20x smaller)
   - Pair-regularity further modulates tetragram weight (better ADFGVX structure → higher tetragram contribution)
   - `bigram_weight = 1 - tetragram_weight`

   Output: for each candidate length a best key string and a ranked set of per-restart candidates kept on the breaker instance for later re-ranking.

3. Shortlisting and lightweight re-ranking

   - Top transposition candidates per length are re-scored using a light English/German text scorer by decrypting with the canonical alphabet. This helps promote keys that already yield plausible letter distributions.

4. Alphabet (substitution) optimization

   - For selected promising keys, fractionated streams are reconstructed (reverse columnar transposition) and passed to an alphabet optimizer (AlphabetOptimizer).
   - AlphabetOptimizer workflow:
     - Seed generation (canonical rotations, frequency mapping, pattern-based CSP seeds).
     - Local search (hill-climb over a mutable prefix; letters-only by default, digits optionally movable).
     - CSP extraction from long matched words to create deterministic partial mappings and backtracking solving.
     - Deterministic polishing (pairwise swaps, focused permutations) and greedy digit assignment if digits are not fixed.
   - Evaluators:
     - BaseDecoderForProcesses: decodes fractionated streams with a candidate alphabet and returns decoded plaintexts + a base score (language-aware letter-frequency).
     - NGramScoringEvaluator: wraps base decoder with trigram / common-word scoring and strong penalties for digits appearing inside words.
   - Output: top alphabet candidates per key candidate, accompanied by recovered plaintexts and scores.

5. Polishing and cross-checks

   - For accepted alphabets, the system runs polish passes that attempt to re-score column-order candidates by actually decoding with the alphabet and re-running hill-climb under the plaintext scoring objective.
   - This cross-check reduces false positives where a transposition key looked good under fractionated-only metrics but fails to produce readable plaintext when decoded.

6. Manual verification and persistence
   - If final_manual_alphabet_testing is enabled, a short interactive review loop presents top recovered plaintexts for the operator to accept or reject candidate alphabets.
   - Results are merged into messages.json under the entry's "results". Behavior (overwrite vs merge) is controlled by CONFIG["overwrite_json_entries"] and related flags.

Key configuration knobs (most impactful)

**Multi-Phase Key Search (NEW)**

- enable_fragment_seeding: Enable fragment-based seeding (recommended for all key lengths).
- enable_three_phase_keysearch: Enable three-phase progressive refinement (default True).
- fragment_voting_seeds: Use fragment voting strategy based on successor frequencies.
- infer_fragment_seeding_variables_automatically: Auto-set phase parameters based on key length.

**Tetragram Scoring (NEW)**

- tetragram_base_weight: Base weight for tetragram IC (default 0.1).
- tetragram_boost_start_length: Start boosting tetragram weight at this key length (default 12).
- tetragram_even_key_additive_value: Extra tetragram weight for even key lengths (per-length in infer_fragment_seeding_variables).
- tetragram_ic_scale_factor: Scale factor to normalize tetragram IC (default ~6-15).

**Existing options**

- key_search_workers: number of processes used for the transposition search.
- restarts, max_iterations: depth of the transposition hill-climb restarts and iterations per restart.
- use_hybrid: allow short simulated annealing passes during/after hill-climb (improves escape from local optima, slower).
- fix_alphabet_digits: when True digits remain a canonical suffix and only letters are permuted; when False the optimizer can move digits but stronger digit-penalties are applied.
- alphabet_restarts, alphabet_max_iterations, alphabet_workers: controls alphabet optimizer workload/parallelism.
- intermediate_output and debug_output: verbosity and implementation debugging traces.

Runtime outputs and where to inspect them

- process returns an in-memory entry structure with entry["results"] (reduced top-N candidates).
- messages.json is updated/merged per configuration and will contain recovered plaintexts when set to persist them.
- The breaker instance attaches helper structures:
  - `._per_length_candidates`: per-restart candidates for each key length
  - `._optimizer_outcomes`: alphabet optimization results
  - `._global_leaderboard`: GlobalLeaderboard instance tracking best candidates across phases (when `update_global_best_candidates_dynamically=True`)

Recommended defaults and tips

- For quick experimentation: key_search_workers=1, enable_fragment_seeding=False, small restarts (e.g., 50), alphabet_workers=1.
- For serious runs: increase key_search_workers and alphabet_workers to match CPU cores, enable enable_fragment_seeding=True and enable_three_phase_keysearch=True.
- For long keys (>13): enable all three-phase options with infer_fragment_seeding_variables_automatically=True for auto-tuned parameters.
- For even key lengths: the system automatically applies higher tetragram weighting; no manual adjustment needed.
- If digits are believed to be static (e.g., digits always appended), set fix_alphabet_digits=True for faster and more robust search.
- Use intermediate_output=True during development to get progressive status and candidate lists.

Common failure modes and debugging hints

- No plausible candidates returned:
  - Check that ciphertexts were normalized and not corrupted; inspect fractionated streams directly (breaker.decrypt_with_key).
  - Lower ic_threshold or increase restarts/max_iterations to allow more exploration.
  - Try enabling fragment_voting_seeds=True for better seed quality.
  - If fractionation alphabet mismatch suspected, run alphabet optimization with a larger alphabet_restarts and allow digits to move.
- Phase 1 produces poor seeds:
  - Increase total_seeds to generate more diverse starting points.
  - Try switching between fragment_voting_seeds and random_seeds strategies.
  - Check if ciphertext length is sufficient (longer texts give better IC discrimination).
- Optimizer picks alphabets with digits inside words:
  - NGramScoringEvaluator applies catastrophic penalties for intra-word digits. If your dataset legitimately contains digits, relax digit_penalty or set fix_alphabet_digits=False and tune digit_penalty lower.
- Pickling failures when using ProcessPoolExecutor:
  - Run with workers=0 (single-threaded) to avoid pickling/serialization of local evaluator objects.

Example end-to-end usage

1. Provide an entry in messages.json with "ciphertexts" and "key" (or provide plaintexts and an alphabet to generate ciphertexts).
2. Call break_cipher_from_file(entry_id) or use break_cipher(...) for programmatic runs.
3. Observe Phase 1/2/3 progress in console output (with intermediate_output=True).
4. Inspect printed top candidates, optionally perform manual verification, and check messages.json for persisted results.

Where to find the implementation pieces

- Orchestration, I/O and entry processing: cipher_breaker.py (process_entry_logic, break_cipher_from_file).
- Transposition recovery, scoring, hill-climb: cipher_breaker_utils.py (ADFGVXBreaker, get_tetragram_weight, key_search_worker).
- Fragment seeding, phase parameters, transformations: cipher_breaker_helpers.py (infer_fragment_seeding_variables, reconstruct_long_key_seeds, generate_transformations, GlobalLeaderboard).
- Alphabet optimization and scoring: substitution_solver.py (AlphabetOptimizer, NGramScoringEvaluator, BaseDecoderForProcesses).

This workflow document complements the module-level design notes in cipher_breaker_doc.md; use it as a practical runbook for experiments and debugging.
