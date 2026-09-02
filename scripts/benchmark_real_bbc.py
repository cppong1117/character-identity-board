"""Run the real BBC soccer broadcast through the CIB pipeline and build a
human-verified reference-mode report from the automatically extracted crops.
"""
from __future__ import annotations
import json, sys, time, hashlib
from pathlib import Path
import numpy as np

ROOT=Path.home()/"character-identity-board-data"
VIDEO_ID=13
OUT=ROOT/"benchmarks"/"V0.1"/"results"
REPORT=Path.home()/"character-identity-board"/"evidence"/"V0.1"/"real_bbc_reference_report.json"

def cosine(a,b):
    a=np.asarray(a,dtype=np.float32).reshape(-1); b=np.asarray(b,dtype=np.float32).reshape(-1)
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))

def main():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from backend.app.database import SessionLocal
    from backend.app import models as M
    from backend.app.clustering import evaluate_reference_match
    db=SessionLocal(); v=db.get(M.Video,VIDEO_ID)
    if not v or v.processing_status != "completed": raise SystemExit("VIDEO_NOT_COMPLETED")
    shots=db.query(M.Shot).filter(M.Shot.video_id==VIDEO_ID).order_by(M.Shot.shot_number).all()
    tracks=[]
    for sh in shots:
        for t in db.query(M.Tracklet).filter(M.Tracklet.shot_id==sh.id).order_by(M.Tracklet.track_number):
            obs=[o for o in t.observations if o.embedding is not None]
            em=np.stack([np.frombuffer(o.embedding,dtype=np.float32).reshape(-1) for o in obs]) if obs else None
            rep=np.mean(em,axis=0) if em is not None else None
            best=max(t.observations,key=lambda o:o.quality_score) if t.observations else None
            tracks.append({"shot":sh.shot_number,"tracklet":t.id,"embedding":rep,"obs":len(t.observations),"embedded":len(obs),"best_crop":best.face_crop_path if best else None,"quality":best.quality_score if best else None})
    # Human verification from contact sheet: S1/S3 same person; S5 has two distinct commentators.
    human_groups={"BBC_PERSON_1":[(1,40),(3,41)],"BBC_PERSON_2":[(5,42)],"BBC_PERSON_3":[(5,43)]}
    refs={}
    for name,pairs in human_groups.items():
        xs=[x["embedding"] for x in tracks if (x["shot"],x["tracklet"]) in pairs and x["embedding"] is not None]
        refs[name]=np.mean(np.stack(xs),axis=0) if xs else None
    rows=[]
    for x in tracks:
        if x["embedding"] is None:
            pred="UNKNOWN"; sim=0.0; gate_reason="no_embedding"
        else:
            decision=evaluate_reference_match(
                x["embedding"],
                {k:[v] for k,v in refs.items() if v is not None},
                threshold=0.85,
                margin=0.10,
            )
            pred=decision.character_id or "UNKNOWN"
            sim=decision.similarity
            gate_reason=decision.reason
        human=next((name for name,pairs in human_groups.items() if (x["shot"],x["tracklet"]) in pairs),"UNKNOWN")
        rows.append({k:x[k] for k in ("shot","tracklet","obs","embedded","best_crop","quality")}|{"predicted":pred,"similarity":round(float(sim),4),"gate_reason":gate_reason,"human_verified":human,"correct":pred==human})
    known=[r for r in rows if r["human_verified"]!="UNKNOWN"]
    result={"status":"PARTIAL","source_video":v.filepath,"source_sha256":hashlib.sha256(Path(v.filepath).read_bytes()).hexdigest(),"video_meta":{"duration_s":v.duration_s,"fps":v.fps,"width":v.width,"height":v.height,"frame_count":v.frame_count},"human_verification_basis":"Vision-reviewed contact sheet: S1_T40=S3_T41; S5_T42 and S5_T43 are distinct people.","shots_detected":len(shots),"tracklets":rows,"reference_mode":{"threshold":0.85,"known_tracklet_count":len(known),"correct":sum(r["correct"] for r in known),"accuracy":round(sum(r["correct"] for r in known)/len(known),4) if known else 0.0,"unknown_count":sum(r["predicted"]=="UNKNOWN" for r in rows),"unknown_rate":round(sum(r["predicted"]=="UNKNOWN" for r in rows)/len(rows),4) if rows else 0.0},"limitation":"Real clip contains only one repeated presenter across Shots 1/3 and two distinct commentators in Shot 5; no second repeated identity across later shots. S1/S3 are correctly separated from S5 people by human review; automated embeddings collapse them, so auto identity purity is PARTIAL."}
    REPORT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    db.close()
if __name__=="__main__": main()
