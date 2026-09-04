# Phase 3 — Alignment A/B Test

- Records: **2500** on **30** shots (named chars, q≥0.65)
- per_char: `{'SMY': 891, 'DOCTOR': 881, 'mbq': 219, 'lw': 509}`
- time: 791.1s
- thresholds frozen: 0.40 / 0.50 / 0.30

## Same-person

| ver | n | mean | median | std | p10 | p90 |
|-----|--:|-----:|-------:|----:|----:|----:|
| v1 | 3003 | 0.6328 | 1.0000 | 0.4153 | 0.0336 | 1.0000 |
| v2 | 3003 | 0.6710 | 1.0000 | 0.3666 | 0.2208 | 1.0000 |

## Different-person

| ver | n | mean | median | std | p10 | p90 |
|-----|--:|-----:|-------:|----:|----:|----:|
| v1 | 3000 | 0.1816 | 0.1290 | 0.1616 | 0.0305 | 0.3802 |
| v2 | 3000 | 0.1518 | 0.1420 | 0.1389 | 0.0208 | 0.2504 |

## Separation: v1=0.4512  v2=0.5191  delta=0.0679

## FAR/FRR

| metric | v1 | v2 |
|--------|---:|---:|
| FAR@0.40 | 0.0610 | 0.0313 |
| FAR@0.50 | 0.0240 | 0.0143 |
| FRR@0.40 | 0.4496 | 0.4496 |
| FRR@0.50 | 0.4496 | 0.4496 |

## PASS

- separation_improved: **True**
- false_merge_not_worse: **True**
- same_person_mean_ok: **True**
- **OVERALL: PASS**

Next: Phase 4 full re-embed only if PASS.
