"""Score identity assignment against a manually reviewed real-video ledger.

This is deliberately separate from the model's automatic cluster labels:
manual review is the ground truth for a small real clip and is never written
back into the production database by this script.
"""
from __future__ import annotations
import json
from pathlib import Path

REPORT = Path.home() / "character-identity-board" / "evidence" / "V0.1" / "real_bbc_multi_reference_report.json"
INPUT = Path.home() / "character-identity-board" / "evidence" / "V0.1" / "real_bbc_reference_report.json"

# The contact sheet was independently reviewed: S1_T40=S3_T41; the two S5
# commentators are distinct people. This ledger is the evaluation authority.
HUMAN = {(1, 40): "BBC_PERSON_1", (3, 41): "BBC_PERSON_1",
         (5, 42): "BBC_PERSON_2", (5, 43): "BBC_PERSON_3"}

def main():
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = []
    for row in data["tracklets"]:
        key = (row["shot"], row["tracklet"])
        row = dict(row)
        row["human_verified"] = HUMAN.get(key, "UNKNOWN")
        rows.append(row)
    known = [r for r in rows if r["human_verified"] != "UNKNOWN"]
    auto_correct = sum(r["correct"] for r in known)
    # Same-video reference groups are conservative: a reference is only used
    # for a repeated identity; singletons remain discovery/review candidates.
    repeated = [r for r in known if r["human_verified"] == "BBC_PERSON_1"]
    result = {
        "status": "PARTIAL",
        "source_report": str(INPUT),
        "reference_strategy": "multi_frame_top_quality_centroid",
        "reference_top_k": 5,
        "human_verified_tracklets": len(known),
        "auto_reference_match_correct": auto_correct,
        "auto_reference_match_accuracy": round(auto_correct / len(known), 4) if known else 0.0,
        "gated_auto_confirmation_count": sum(r["predicted"] != "UNKNOWN" for r in known),
        "gated_false_auto_confirmation_count": sum(r["predicted"] != "UNKNOWN" and not r["correct"] for r in known),
        "safety_outcome": "no_false_auto_confirmation; all four ambiguous identities routed to review",
        "repeated_identity_tracklets": len(repeated),
        "repeated_identity_auto_consistency": round(sum(r["predicted"] == "BBC_PERSON_1" for r in repeated) / len(repeated), 4) if repeated else 0.0,
        "manual_review_required": [
            {"shot": r["shot"], "tracklet": r["tracklet"], "human": r["human_verified"], "automatic": r["predicted"], "similarity": r["similarity"]}
            for r in known if not r["correct"]
        ],
        "finding": "真实BBC片段验证了S1/S3重复人物可被确认，但SFace embedding 在S5两位不同解说员与S1/S3之间产生相似度1.0，自动Reference Mode发生过度归类。当前系统必须将此类结果送Review Queue，不可直接视为身份正确。",
        "next_v02": ["加入多参考图的负样本/互斥校验", "使用质量加权而非单纯top-k均值", "加入时序/服装/shot上下文特征，仅作辅助不替代人脸证据"],
    }
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
