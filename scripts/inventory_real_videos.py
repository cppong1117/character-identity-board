import json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone

VIDEO = Path('/mnt/c/Users/ponky_re6000/Videos/2026-08-13 17-07-40.mp4')
OUT = Path.home() / 'character-identity-board' / 'evidence' / 'V0.1' / 'real_video_candidates.json'

def probe(p):
    r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration,size:stream=index,codec_type,codec_name,width,height,r_frame_rate,nb_frames','-of','json',str(p)], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)

candidates=[]
for p in [
 Path('/mnt/c/Users/ponky_re6000/Videos/2026-04-28 11-29-00.mp4'),
 Path('/mnt/c/Users/ponky_re6000/Videos/2026-04-28 11-53-45.mp4'),
 Path('/mnt/c/Users/ponky_re6000/Videos/2026-04-28 11-55-40.mp4'),
 Path('/mnt/c/Users/ponky_re6000/Videos/2026-05-30 17-18-44.mp4'),
 Path('/mnt/c/Users/ponky_re6000/Videos/2026-08-11 10-59-18.mp4'),
 Path('/mnt/c/Users/ponky_re6000/Videos/2026-08-13 17-07-40.mp4'),
]:
    if not p.exists(): continue
    d=probe(p); f=d.get('format',{}); streams=d.get('streams',[]); v=next((x for x in streams if x.get('codec_type')=='video'),{})
    candidates.append({'path':str(p),'duration_s':float(f.get('duration') or 0),'size_bytes':int(f.get('size') or 0),'width':v.get('width'),'height':v.get('height'),'fps':v.get('r_frame_rate'),'codec':v.get('codec_name'),'has_audio':any(x.get('codec_type')=='audio' for x in streams)})
OUT.write_text(json.dumps({'captured_at':datetime.now(timezone.utc).isoformat(),'candidates':candidates},indent=2),encoding='utf-8')
print(OUT)
print(json.dumps(candidates,indent=2))
