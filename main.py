import os
# EXTREME MEMORY OPTIMIZATIONS FOR REPLIT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import base64
import json
import asyncio
import gc
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

app = FastAPI()

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AILog(Base):
    __tablename__ = "ai_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    feature_type = Column(String)
    result_value = Column(String)
    confidence = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models (loaded lazily)
_hands_model = None
_deepface_loaded = False

def get_hands_model():
    global _hands_model
    if _hands_model is None:
        import mediapipe as mp
        _hands_model = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    return _hands_model

@app.get("/api")
async def root():
    return {"message": "MHS AI Engine Online (Lite Mode)"}

@app.get("/api/logs")
async def get_logs():
    try:
        db = SessionLocal()
        logs = db.query(AILog).order_by(AILog.timestamp.desc()).limit(20).all()
        db.close()
        return [
            {
                "id": log.id,
                "timestamp": log.timestamp.isoformat(),
                "feature_type": log.feature_type,
                "result_value": log.result_value,
                "confidence": log.confidence
            } for log in logs
        ]
    except:
        return []

@app.websocket("/api/ws/air-writing")
async def air_writing_websocket(websocket: WebSocket):
    await websocket.accept()
    canvas = None
    points = []
    lock = asyncio.Lock()
    
    try:
        while True:
            data = await websocket.receive_text()
            if lock.locked(): continue
            
            async with lock:
                msg = json.loads(data)
                if msg['type'] == 'reset':
                    canvas = None
                    points = []
                    continue
                
                img_bytes = base64.b64decode(msg['image'].split(',')[1])
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)
                
                if canvas is None:
                    canvas = np.zeros_like(frame)

                # Process hand tracking
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = await asyncio.to_thread(get_hands_model().process, rgb_frame)
                
                fingertip = None
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        lm8 = hand_lms.landmark[8]
                        lm6 = hand_lms.landmark[6]
                        cx, cy = int(lm8.x * w), int(lm8.y * h)
                        fingertip = (cx, cy)
                        if lm8.y < lm6.y:
                            points.append(fingertip)
                        else:
                            points.append(None)

                # Draw
                for i in range(1, len(points)):
                    if points[i-1] and points[i]:
                        cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 7)
                
                combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
                _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
                encoded = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_text(json.dumps({
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "fingertip": fingertip
                }))
                
                # Cleanup
                del frame, rgb_frame, combined
                gc.collect()
                
    except:
        pass

@app.websocket("/api/ws/age-detection")
async def age_detection_websocket(websocket: WebSocket):
    await websocket.accept()
    last_age = None
    lock = asyncio.Lock()
    
    try:
        while True:
            data = await websocket.receive_text()
            if lock.locked(): continue
            
            async with lock:
                msg = json.loads(data)
                img_bytes = base64.b64decode(msg['image'].split(',')[1])
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                try:
                    from deepface import DeepFace
                    results = await asyncio.to_thread(DeepFace.analyze, frame, actions=['age'], enforce_detection=False)
                    if results:
                        age = results[0]['dominant_age']
                        conf = results[0].get('face_confidence', 0.9)
                        
                        is_new = False
                        if age != last_age:
                            db = SessionLocal()
                            log = AILog(feature_type='age', result_value=str(age), confidence=float(conf))
                            db.add(log)
                            db.commit()
                            db.close()
                            last_age = age
                            is_new = True

                        await websocket.send_text(json.dumps({
                            "age": age,
                            "confidence": conf,
                            "new_log": is_new
                        }))
                except:
                    pass
                
                del frame
                gc.collect()
                
    except:
        pass

# Static File Serving
current_dir = Path(__file__).parent
dist_path = current_dir.parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="frontend")

@app.exception_handler(404)
async def catch_all(request, exc):
    if dist_path and dist_path.exists():
        return FileResponse(dist_path / "index.html")
    return JSONResponse({"error": "Not Found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)



