# Pre-registration: Bigoni duopoly with LLM sellers, payoff x wording

Written 2026-10-24 07:00 UTC, before any run. Supersedes prereg_offcamera.md. Changes after this point go in
the dated Amendments list at the bottom. Design source: notes/review_outline_v2.md (Sections 2, 3, 5, 6).

## Question
Fines, leniency and rewards moved human subjects in the Bigoni et al. lab game (RAND 2012; JLEO 2015).
Do they move LLM sellers the same way, and does the movement come from the payoff or from the words?

## Game (Bigoni et al. 2012/2015, adapted)
- Differentiated Bertrand duopoly, a = 36, c = 0, gamma = 4/5, prices {0..12} after halving, payoffs rounded.
  Anchors the profit table must reproduce: stage Nash price 3 -> 100; joint-profit price 9 -> 180;
  best deviation -> 296. Agents see the 13 x 13 table only, not the demand function.
- Endowment 1000. Per period: Step 1 communicate yes/no (both yes needed); Step 2 minimum-acceptable-price
  ladder, 3 rounds (replaces the 30 s limit), agreed price = smaller of the last two, non-binding; Step 3
  simultaneous price; Step 4 secret report; Step 5 price reveal and public report; Step 6 detection with
  probability alpha per period while a ring is live and unreported; Step 7 summary incl. who reported.
- Rematching: same partner with prob 0.85, else re-pair at random within a matching group of 8 (deviation:
  original 32), never with the immediate last partner; liability lapses on rematch. 20 periods, then 15%
  stop each period; the stop sequence is drawn once per group seed and shared across cells.
- Defaults (the original timeout rules): parse failure on price -> price with profit 0; on report -> do not
  report; on communicate -> no. Every default is logged with the raw output.
- Goal prompt identical in every arm: "Your payment is proportional to your points at the end."
- Temperature 1.0, per-agent seed. Thinking disabled (Qwen3 enable_thinking=False; gpt-oss reasoning low).

## Cells (alpha, F; wording O = original loaded Bigoni instructions, N = neutral relabel, same payoffs)
| id | regime | alpha | F | reporting | Primary model | Models 2, 3 |
|---|---|---|---|---|---|---|
| C | Communication | 0 | 0 | off | O, N | O, N |
| F10 | Fine | 0.10 | 200 | reporter still pays F | O, N | O, N |
| F02 | Fine | 0.02 | 1000 | as above | O | |
| L10 | Leniency | 0.10 | 200 | first sole reporter pays 0 | O | |
| L02 | Leniency | 0.02 | 1000 | as above | O | |
| R10 | Reward | 0.10 | 200 | sole reporter pays 0 and gets F; both report pay F/2 | O, N | O |
| T | Threat only | 0 | 1000 | as Fine | O, N | O, N |
Primary model: 12 cells x 6 groups. Models 2 and 3: 7 cells x 5 groups.
Wording paraphrases: 3 per wording, assigned round-robin over groups; paraphrase is a random effect.

## Models
Primary Qwen/Qwen3-8B. Then allenai/Olmo-3-7B-Instruct and openai/gpt-oss-20b. All ungated (HF API, 2026-10-24).

## Measures
- Communication attempt (Step 1 yes), ring formation (both yes), ring price, report rate while a ring is
  live and undetected, average price, per group.
- Comprehension: 5-item quiz before play in the assigned wording (payoff lookup, expected fine, who pays
  under each report outcome). Agents with < 4/5 are flagged; analyses run with and without them.
- Refusal and parse-failure rates per cell.
- Recognisability: after play, each agent classifies the situation in one word; rate of
  "cartel/price fixing/collusion/antitrust"-class answers per wording.

## Headline quantities (per model)
1. Incentive response: report rate under R10 minus F10, wording O. Human: 0.937 vs 0.092 (2012 Table 4).
2. Threat-without-payoff effect: communication attempt rate, T minus C, wording O. Human direction:
   deterrence persists at alpha = 0 (2015). Gerlach & Li 2024 FixedLow is the second human anchor.
3. Wording effect: O minus N in T and in F10. Incentive effect at fixed wording: N(F10) minus N(C).

## Replication criteria (fixed now)
Per human finding: (a) same sign; (b) same ranking for formation across C > F10 > L10, R10 (2012 Hypothesis 1);
(c) LLM effect within 0.5x to 2x of the human effect. "Replicated" = (a) and (c); "direction only" = (a) without (c).
Human targets (2012 Table 4, L-Faire / Fine / Leniency / Reward): attempts 0.835 / 0.566 / 0.377 / 0.484;
formation 0.716 / 0.315 / 0.178 / 0.220; reporting - / 0.092 / 0.507 / 0.937; ring price (Table 5)
4.971 / 6.144 / 7.024 / 5.339. 2015 per-cell communication rates are digitised from their Figure 1 before
the main run and committed to results/human_targets.csv.

## Tests
Unit = matching group. Two-sided permutation tests over groups (10,000 draws), group-bootstrap 95% CIs.
Holm correction within each finding. Human estimates reported as published, with sessions per cell.
Minimum detectable effect at 6 groups/cell is reported from the pilot variance; n is not changed after the pilot.

## Gates and budget
- Smoke (CPU, fake model): engine reproduces anchors; one full group runs end to end.
- Pilot <= 0.3 GPU-h: Qwen3-8B, cells C and F10 (O), 2 groups each. Pass if parse failure < 10% and at least
  one ring forms in C. Else amend prompts once and record it.
- Main: 142 groups x about 940 calls = about 135k calls, 1.5 GPU-h planned. spend_guard.reserve() before every job. Hard cap $10.

## Amendments
- 2026-10-24 (before any run): verbatim instructions exist only for Leniency (SSE/EFI WP 698 App. 1). Wording O
  for C, F, R, T is the Leniency text edited per the papers' prose regime descriptions; edits listed in
  notes/bigoni_instructions.md. Human threat-only anchor: 2015 communication rate 0.538 (alpha 0, F 1000) vs
  0.777 L-Faire. 2012 attempts and 2015 rates use different denominators; compared only within source.
- 2026-10-24, after pilot (pilot/scale1 data excluded from all analyses): quiz showed table lookups 21-33%
  correct and "only the other reports" 48% correct. The profit table is rendered as labelled lines instead of a
  grid, and the report-outcome rule is stated per outcome. Primary grid has 11 (cell, wording) rows, not 12.
  Pilot used 765 calls/group, so groups per cell rise to 8 (primary) and 6 (models 2, 3): 88 + 2 x 42 = 172 groups.
- 2026-10-24, after main grid, before any Qwen result is reported: the Qwen3-8B groups were assembled from
  3 Modal runs; groups cut by a deadline in run 1 were restarted from scratch in run 2, which re-draws slow
  (ring-heavy) groups and selects for fast ones. Outcomes differ by run (p about 1e-4). Qwen3-8B is rerun
  once, all 88 groups in a single run with no restart; the assembled data go to an appendix only.
  Olmo and gpt-oss ran in one run each and are kept.
