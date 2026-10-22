from pathlib import Path
import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

class DirectFullFrameDetector:
    """Send one complete pan/tilt photo to knife and face models."""
    def __init__(self, person_path, knife_path, conf=0.35):
        self.conf=float(conf)
        self.person=RKNNLite(); self.knife=RKNNLite()
        self.runtimes=[self.person, self.knife]
        if self.person.load_rknn(str(person_path)) != 0 or self.person.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            self.release(); raise RuntimeError(f'model load failed: {person_path}')
        if self.knife.load_rknn(str(knife_path)) != 0 or self.knife.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            self.release(); raise RuntimeError(f'model load failed: {knife_path}')
        print(f'[Vision] direct full-frame models loaded: {Path(person_path).name}, {Path(knife_path).name}')

    def _input(self, frame):
        h,w=frame.shape[:2]
        image=cv2.resize(frame,(640,384),interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)[None],dtype=np.uint8), (w/640.0,h/384.0)

    def _decode(self, output, frame_shape, scale, id_map):
        a=np.squeeze(np.asarray(output,dtype=np.float32))
        if a.ndim!=2: return None
        if a.shape[0] < a.shape[1]: a=a.T
        if a.shape[1] < 5: return None
        boxes=a[:,:4]; scores=a[:,4:]
        if float(scores.min()) < 0.0 or float(scores.max()) > 1.0:
            scores=1.0/(1.0+np.exp(-np.clip(scores,-60,60)))
        cls=scores.argmax(1); conf=scores.max(1); keep=conf>=self.conf
        if not np.any(keep): return None
        boxes,cls,conf=boxes[keep],cls[keep],conf[keep]
        xywh=[[float(x-w/2),float(y-h/2),float(w),float(h)] for x,y,w,h in boxes]
        inds=cv2.dnn.NMSBoxes(xywh,conf.astype(float).tolist(),self.conf,0.45)
        if len(inds)==0: return None
        j=int(np.asarray(inds).reshape(-1)[int(np.argmax(conf[np.asarray(inds).reshape(-1)]))])
        x,y,w,h=xywh[j]; fh,fw=frame_shape[:2]; sx,sy=scale
        box=(max(0,int(x*sx)),max(0,int(y*sy)),min(fw,int((x+w)*sx)),min(fh,int((y+h)*sy)))
        return int(id_map[int(cls[j])]), float(conf[j]), box

    def detect(self, frame):
        if frame is None or frame.size==0: return -1, frame
        inp,scale=self._input(frame)
        person_out=self.person.inference(inputs=[inp])
        knife_out=self.knife.inference(inputs=[inp])
        person=self._decode(person_out[0],frame.shape,scale,(1,10,2,3,4,5,6,7,8,9))
        knife=self._decode(knife_out[0],frame.shape,scale,tuple(range(11,21)))
        candidates=[]
        if person is not None: candidates.append(('person',person[1],person[0],person[2]))
        if knife is not None: candidates.append(('knife',knife[1],knife[0],knife[2]))
        annotated=frame.copy()
        for kind,score,bid,box in candidates:
            x1,y1,x2,y2=box; color=(255,0,0) if kind=='person' else (0,165,255)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
            cv2.putText(annotated,f'{kind} ID={bid} conf={score:.2f}',(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
        if not candidates: return -1,annotated
        kind,score,bid,_=max(candidates,key=lambda item:item[1])
        print(f'[Vision] full-frame compare: person={person[1] if person else None}, knife={knife[1] if knife else None} -> {kind} ID={bid} score={score:.3f}')
        return bid,annotated

    def release(self):
        for runtime in getattr(self,'runtimes',[]):
            try: runtime.release()
            except Exception: pass
