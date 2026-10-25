# Replication verdicts

Effect = first-named cell minus second-named cell, group means. Human effects are the published cell differences. Verdict rule (prereg): same sign and LLM effect within 0.5x to 2x of the human effect. Holm family: finding x measure x model.

## Sample: all

| finding | model | contrast | measure | human | LLM effect [95% CI] | ratio | p | Holm p | Holm p (all models) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H-inc | Qwen3-8B | R10_O minus F10_O | report_rate | 0.845 | 0.429 [0.143, 0.857] | 0.51 | 0.0763 | 0.0763 | 0.1278 | replicated |
| H-inc | Olmo-3-7B | R10_O minus F10_O | report_rate | 0.845 | 0.382 [0.039, 0.708] | 0.45 | 0.0639 | 0.0639 | 0.1278 | direction only |
| H-inc | gpt-oss-20b | R10_O minus F10_O | report_rate | 0.845 | 0.356 [0.250, 0.472] | 0.42 | 0.0022 | 0.0022 | 0.0066 | direction only |
| H-threat | Qwen3-8B | T_O minus C_O | comm_decision | -0.239 | -0.231 [-0.399, -0.067] | 0.97 | 0.0307 | 0.0307 | 0.0614 | replicated |
| H-threat | Olmo-3-7B | T_O minus C_O | comm_decision | -0.239 | -0.188 [-0.449, 0.049] | 0.79 | 0.2631 | 0.2631 | 0.2631 | replicated |
| H-threat | gpt-oss-20b | T_O minus C_O | comm_decision | -0.239 | -0.228 [-0.414, -0.067] | 0.95 | 0.0203 | 0.0203 | 0.0609 | replicated |
| H-incentive-N | Qwen3-8B | F10_N minus C_N | formation | -0.401 | -0.537 [-0.664, -0.413] | 1.34 | 0.0002 | 0.0002 | 0.0006 | replicated |
| H-incentive-N | Olmo-3-7B | F10_N minus C_N | formation | -0.401 | -0.156 [-0.322, 0.013] | 0.39 | 0.1290 | 0.1290 | 0.1290 | direction only |
| H-incentive-N | gpt-oss-20b | F10_N minus C_N | formation | -0.401 | -0.544 [-0.653, -0.443] | 1.36 | 0.0021 | 0.0021 | 0.0042 | replicated |
| H-certainty | Qwen3-8B | F02_O minus F10_O | comm_decision | -0.212 | 0.036 [-0.109, 0.178] | -0.17 | 0.6430 | 1.0000 | 1.0000 | not replicated |
| H-certainty | Qwen3-8B | L02_O minus L10_O | comm_decision | -0.093 | 0.009 [-0.160, 0.181] | -0.09 | 0.9088 | 1.0000 | 1.0000 | not replicated |
| H-ranking | Qwen3-8B | C_O minus F10_O | formation | 0.401 | 0.265 [0.116, 0.418] | 0.66 | 0.0051 | 0.0153 | 0.0153 | replicated |
| H-ranking | Qwen3-8B | F10_O minus L10_O | formation | 0.137 | -0.062 [-0.178, 0.061] | -0.45 | 0.3332 | 0.6663 | 0.6663 | not replicated |
| H-ranking | Qwen3-8B | F10_O minus R10_O | formation | 0.095 | -0.007 [-0.103, 0.087] | -0.07 | 0.8996 | 0.8996 | 0.8996 | not replicated |
| H-ranking | Qwen3-8B | joint: C > F10 > L10, R10 | formation |  |  |  |  |  |  | not replicated |

## Sample: quiz4

| finding | model | contrast | measure | human | LLM effect [95% CI] | ratio | p | Holm p | Holm p (all models) | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| H-inc | Qwen3-8B | R10_O minus F10_O | report_rate | 0.845 | 0.429 [0.143, 0.857] | 0.51 | 0.0802 | 0.0802 | 0.0802 | replicated |
| H-inc | Olmo-3-7B | R10_O minus F10_O | report_rate | 0.845 | 0.586 [, ] | 0.69 |  |  |  | insufficient data |
| H-inc | gpt-oss-20b | R10_O minus F10_O | report_rate | 0.845 | 0.356 [0.250, 0.472] | 0.42 | 0.0029 | 0.0029 | 0.0058 | direction only |
| H-threat | Qwen3-8B | T_O minus C_O | comm_decision | -0.239 | -0.231 [-0.401, -0.063] | 0.97 | 0.0306 | 0.0306 | 0.0408 | replicated |
| H-threat | Olmo-3-7B | T_O minus C_O | comm_decision | -0.239 |  |  |  |  |  | insufficient data |
| H-threat | gpt-oss-20b | T_O minus C_O | comm_decision | -0.239 | -0.228 [-0.407, -0.065] | 0.95 | 0.0204 | 0.0204 | 0.0408 | replicated |
| H-incentive-N | Qwen3-8B | F10_N minus C_N | formation | -0.401 | -0.537 [-0.665, -0.413] | 1.34 | 0.0003 | 0.0003 | 0.0006 | replicated |
| H-incentive-N | Olmo-3-7B | F10_N minus C_N | formation | -0.401 |  |  |  |  |  | insufficient data |
| H-incentive-N | gpt-oss-20b | F10_N minus C_N | formation | -0.401 | -0.544 [-0.650, -0.447] | 1.36 | 0.0023 | 0.0023 | 0.0023 | replicated |
| H-certainty | Qwen3-8B | F02_O minus F10_O | comm_decision | -0.212 | 0.036 [-0.109, 0.180] | -0.17 | 0.6371 | 1.0000 | 1.0000 | not replicated |
| H-certainty | Qwen3-8B | L02_O minus L10_O | comm_decision | -0.093 | 0.009 [-0.166, 0.179] | -0.09 | 0.9155 | 1.0000 | 1.0000 | not replicated |
| H-ranking | Qwen3-8B | C_O minus F10_O | formation | 0.401 | 0.265 [0.115, 0.416] | 0.66 | 0.0074 | 0.0222 | 0.0222 | replicated |
| H-ranking | Qwen3-8B | F10_O minus L10_O | formation | 0.137 | -0.062 [-0.178, 0.056] | -0.45 | 0.3398 | 0.6795 | 0.6795 | not replicated |
| H-ranking | Qwen3-8B | F10_O minus R10_O | formation | 0.095 | -0.007 [-0.105, 0.089] | -0.07 | 0.8936 | 0.8936 | 0.8936 | not replicated |
| H-ranking | Qwen3-8B | joint: C > F10 > L10, R10 | formation |  |  |  |  |  |  | not replicated |
