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
    return {"message": "MHS AI Backend Running"}

@app.get("/api/logs")
async def get_logs():
    try:
        db = SessionLocal()
        logs = db.query(AILog).order_by(AILog.timestamp.desc()).limit(50).all()
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
        print(f"Database Error: {e}")
        return []

@app.websocket("/api/ws/air-writing")
async def air_writing_websocket(websocket: WebSocket):
    await websocket.accept()
    canvas = None
    points = []
    processing_lock = asyncio.Lock()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # If we are already processing a frame, skip this one to avoid lag
            if processing_lock.locked():
                continue
                
            async with processing_lock:
                message = json.loads(data)
                
                if message['type'] == 'reset':
                    canvas = None
                    points = []
                    await websocket.send_text(json.dumps({"status": "reset"}))
                    continue
                    
                img_data = base64.b64decode(message['image'].split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                
                if canvas is None:
                    canvas = np.zeros((h, w, 3), np.uint8)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                fingertip = None
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        # Index fingertip (8) and index PIP joint (6)
                        lm8 = hand_lms.landmark[8]
                        lm6 = hand_lms.landmark[6]
                        lm0 = hand_lms.landmark[0] # Wrist
                        
                        cx, cy = int(lm8.x * w), int(lm8.y * h)
                        fingertip = (cx, cy)
                        
                        # Robust Gesture: Index finger is extended if tip is higher than PIP
                        # AND tip is significantly higher than wrist (relative to hand size)
                        if lm8.y < lm6.y and (lm0.y - lm8.y) > 0.1:
                            points.append(fingertip)
                        else:
                            points.append(None)

                # Draw smooth lines
                for i in range(1, len(points)):
                    if points[i-1] is not None and points[i] is not None:
                        cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 6)
                
                combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
                _, buffer = cv2.imencode('.jpg', combined)
                encoded_image = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_text(json.dumps({
                    "image": f"data:image/jpeg;base64,{encoded_image}",
                    "fingertip": fingertip
                }))
            
    except WebSocketDisconnect:
        print("Air Writing client disconnected")
    except Exception as e:
        print(f"Air Writing Error: {e}")

@app.websocket("/api/ws/age-detection")
async def age_detection_websocket(websocket: WebSocket):
    await websocket.accept()
    last_logged_age = None
    processing_lock = asyncio.Lock()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if processing_lock.locked():
                continue
                
            async with processing_lock:
                message = json.loads(data)
                img_data = base64.b64decode(message['image'].split(',')[1])
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                try:
                    # Run DeepFace analysis (heavy task)
                    results = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                    if results:
                        res = results[0]
                        age = res['dominant_age']
                        # Use face detection confidence if available, or default
                        conf = res.get('face_confidence', 0.9)
                        
                        is_new = False
                        if age != last_logged_age:
                            try:
                                db = SessionLocal()
                                new_log = AILog(feature_type='age', result_value=str(age), confidence=float(conf))
                                db.add(new_log)
                                db.commit()
                                db.close()
                                last_logged_age = age
                                is_new = True
                            except Exception as db_err:
                                print(f"Logging Error: {db_err}")

                        await websocket.send_text(json.dumps({
                            "age": age,
                            "confidence": conf,
                            "new_log": is_new
                        }))
                except Exception as e:
                    print(f"DeepFace Analysis Error: {e}")
                    await websocket.send_text(json.dumps({"error": "Model busy"}))
                
    except WebSocketDisconnect:
        print("Age Detection client disconnected")
    except Exception as e:
        print(f"Age Detection WS Error: {e}")

# Serve Frontend Static Files with robust path checking
current_dir = Path(__file__).parent
possible_paths = [
    current_dir.parent / "frontend" / "dist", # Local structure
    current_dir / "frontend" / "dist",        # Flat structure
    current_dir / "dist"                       # Build in same folder
]

frontend_path = None
for p in possible_paths:
    if p.exists() and (p / "index.html").exists():
        frontend_path = p
        break

if frontend_path:
    print(f"Serving frontend from: {frontend_path}")
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
else:
    print("Warning: Frontend dist folder not found. API only mode.")

@app.exception_handler(404)
async def not_found_handler(request, exc):
    if frontend_path:
        return FileResponse(frontend_path / "index.html")
    return JSONResponse(status_code=404, content={"message": "Not Found"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


