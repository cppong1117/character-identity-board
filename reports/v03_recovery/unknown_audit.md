# Unknown Root Cause Audit (Phase 1)

- Population Unknown tracklets (film): **1431**
- Sample: **300** (seed=42)
- Embeddings: **arcface_v1 unaligned** (current DB)
- Method: evidence signals (crop geometry, quality/blur, intra-tracklet embedding consistency, similarity to named-character centroids). **Not** threshold lowering.
- Named centroids built from top quality active obs per character (SMY/ZY/DOCTOR/lw/mbq/…).

## Distribution

| UNKNOWN_REASON | count | % |
|----------------|------:|--:|
| KNOWN_PERSON_RECOGNITION_FAIL | 129 | 43.0% |
| BAD_ALIGNMENT | 77 | 25.7% |
| TRUE_UNKNOWN | 66 | 22.0% |
| LOW_QUALITY | 18 | 6.0% |
| MIXED_TRACKLET | 10 | 3.3% |

## Interpretation (what to fix next)

Dominant drivers (sample):
1. **KNOWN_PERSON_RECOGNITION_FAIL** — 43.0%
1. **BAD_ALIGNMENT** — 25.7%
1. **TRUE_UNKNOWN** — 22.0%
1. **LOW_QUALITY** — 6.0%
1. **MIXED_TRACKLET** — 3.3%

### Mapping to V0.3 phases

| Reason | Next phase |
|--------|------------|
| BAD_CROP / FALSE_POSITIVE | Phase 9 geometry FP + Phase 7 quality evidence split |
| BAD_ALIGNMENT / KNOWN_PERSON_RECOGNITION_FAIL | **Phase 2–4 alignment + re-embed** (P0) |
| MIXED_TRACKLET | Phase 6 contamination / split |
| LOW_QUALITY | Phase 7 identity-evidence gate (keep obs, block identity) |
| TRUE_UNKNOWN | Accept after pipeline recovery; reference bank gaps (Phase 10) |
| WRONG_PREVIOUS_ASSIGNMENT | Not auto-labeled in this pass (needs manual labels) |

## Signal notes

- `KNOWN_PERSON_RECOGNITION_FAIL`: tracklet prototype sim to a named centroid ≥0.35–0.50 with optional margin — still parked on Unknown.
- `MIXED_TRACKLET`: ≥3 embeddings with mean pairwise sim <0.35 and min <0.15.
- `BAD_CROP` / `FALSE_POSITIVE`: missing files, tiny area/bytes, weak quality.
- `BAD_ALIGNMENT`: large/decent crops but confused top-2 margin or mid sim band (proxy until landmarks stored).
- This audit is **diagnostic**; labels are heuristic. Human spot-check recommended on 30–50 rows in `unknown_audit_sample300.json`.

## Examples (per class, up to 3)

### MIXED_TRACKLET
- tracklet `1573` best=mbq sim=0.426 margin=0.03373423218727112 q=0.726 area=588225 intra=0.3179984390735626
- tracklet `52` best=mbq sim=0.494 margin=0.014132946729660034 q=0.957 area=378068 intra=0.32956618070602417
- tracklet `706` best=mbq sim=0.401 margin=0.03902918100357056 q=0.629 area=138217 intra=0.3339745104312897

### BAD_ALIGNMENT
- tracklet `1917` best=mbq sim=0.282 margin=0.010973036289215088 q=0.662 area=406603 intra=None
- tracklet `269` best=mbq sim=0.426 margin=0.013797342777252197 q=0.678 area=322343 intra=0.5668231248855591
- tracklet `2013` best=mbq sim=0.409 margin=0.018419355154037476 q=0.683 area=91120 intra=None

### LOW_QUALITY
- tracklet `578` best=mbq sim=0.433 margin=0.03912582993507385 q=0.649 area=330198 intra=0.7994400858879089
- tracklet `1781` best=mbq sim=0.395 margin=0.021482795476913452 q=0.633 area=24482 intra=0.5221524834632874
- tracklet `398` best=mbq sim=0.408 margin=0.021027177572250366 q=0.596 area=20810 intra=0.8227902054786682

### KNOWN_PERSON_RECOGNITION_FAIL
- tracklet `2084` best=DOCTOR sim=0.383 margin=0.0006019473075866699 q=0.770 area=250056 intra=0.9835323691368103
- tracklet `742` best=mbq sim=0.447 margin=0.02537032961845398 q=0.825 area=45651 intra=0.7807766795158386
- tracklet `402` best=mbq sim=0.433 margin=0.03485468029975891 q=0.840 area=302324 intra=0.9873031973838806

### TRUE_UNKNOWN
- tracklet `1375` best=mbq sim=0.228 margin=0.059729382395744324 q=0.776 area=456864 intra=0.9019022583961487
- tracklet `906` best=DOCTOR sim=0.145 margin=0.006337210536003113 q=0.781 area=576078 intra=0.46141913533210754
- tracklet `1206` best=mbq sim=0.114 margin=0.01625111699104309 q=0.670 area=493116 intra=None
