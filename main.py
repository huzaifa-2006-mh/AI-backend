import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import base64
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from deepface import DeepFace
from sqlalchemy import create_all_metadata, create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

import asyncio
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=4)

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class AILog(Base):
    __tablename__ = "ai_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    feature_type = Column(String)  # 'age' or 'writing'
    result_value = Column(String)
    confidence = Column(Float, nullable=True)

Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

@app.get("/api")
async def root():
    return {"message": "MHS AI Engine Online"}

@app.get("/api/logs")
async def get_logs():
    try:
        db = SessionLocal()
        logs = db.query(AILog).order_by(AILog.timestamp.desc()).limit(30).all()
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
    except Exception as e:
        print(f"DB Error: {e}")
        return []

async def process_air_writing(frame, hands_model, canvas, points):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run Mediapipe in a separate thread to keep event loop free
    results = await asyncio.to_thread(hands_model.process, rgb_frame)
    
    fingertip = None
    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            lm8 = hand_lms.landmark[8] # Index Tip
            lm6 = hand_lms.landmark[6] # Index PIP
            lm5 = hand_lms.landmark[5] # Index MCP
            
            cx, cy = int(lm8.x * w), int(lm8.y * h)
            fingertip = (cx, cy)
            
            # Smart gesture: Tip must be above PIP and MCP (index extended)
            # and other fingers should ideally be folded (simplified here)
            if lm8.y < lm6.y and lm8.y < lm5.y:
                points.append(fingertip)
            else:
                points.append(None)

    # Drawing logic on canvas
    for i in range(1, len(points)):
        if points[i-1] is not None and points[i] is not None:
            # Draw with a slight glow effect
            cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 7)
    
    return cv2.addWeighted(frame, 0.7, canvas, 0.3, 0), fingertip

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
                    await websocket.send_text(json.dumps({"status": "reset"}))
                    continue
                
                img_bytes = base64.b64decode(msg['image'].split(',')[1])
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1) # Mirror for user
                
                if canvas is None:
                    canvas = np.zeros_like(frame)

                combined, tip = await process_air_writing(frame, hands, canvas, points)
                
                _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
                encoded = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_text(json.dumps({
                    "image": f"data:image/jpeg;base64,{encoded}",
                    "fingertip": tip
                }))
                
    except Exception as e:
        print(f"Air Writing WS Closed: {e}")

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
                    # OFF-LOAD HEAVY AI TO THREAD
                    results = await asyncio.to_thread(DeepFace.analyze, frame, actions=['age'], enforce_detection=False)
                    if results:
                        res = results[0]
                        age = res['dominant_age']
                        conf = res.get('face_confidence', 0.95)
                        
                        is_new = False
                        if age != last_age:
                            # ASYNC DB SAVE
                            def save_log():
                                db = SessionLocal()
                                log = AILog(feature_type='age', result_value=str(age), confidence=float(conf))
                                db.add(log)
                                db.commit()
                                db.close()
                            
                            await asyncio.to_thread(save_log)
                            last_age = age
                            is_new = True

                        await websocket.send_text(json.dumps({
                            "age": age,
                            "confidence": conf,
                            "new_log": is_new
                        }))
                except Exception as ai_e:
                    print(f"AI Error: {ai_e}")
                    
    except Exception as e:
        print(f"Age WS Closed: {e}")

# Static File Serving
current_dir = Path(__file__).parent
dist_path = current_dir.parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="frontend")

@app.exception_handler(404)
async def catch_all(request, exc):
    if dist_path.exists():
        return FileResponse(dist_path / "index.html")
    return JSONResponse({"error": "Not Found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



