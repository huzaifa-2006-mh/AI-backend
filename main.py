import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
mp_draw = mp.solutions.drawing_utils

# Canvas for Air Writing
canvas = None

@app.get("/")
async def root():
    return {"message": "MHS AI Backend Running"}

@app.get("/logs")
async def get_logs():
    db = SessionLocal()
    logs = db.query(AILog).order_by(AILog.timestamp.desc()).limit(50).all()
    db.close()
    return logs

@app.websocket("/ws/air-writing")
async def air_writing_websocket(websocket: WebSocket):
    await websocket.accept()
    global canvas
    points = []
    
    try:
        while True:
            data = await websocket.receive_text()
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
                    lm8 = hand_lms.landmark[8]
                    cx, cy = int(lm8.x * w), int(lm8.y * h)
                    fingertip = (cx, cy)
                    
                    lm6 = hand_lms.landmark[6]
                    if lm8.y < lm6.y:
                        points.append(fingertip)
                    else:
                        points.append(None)

            for i in range(1, len(points)):
                if points[i-1] is not None and points[i] is not None:
                    cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 5)
            
            combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
            _, buffer = cv2.imencode('.jpg', combined)
            encoded_image = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_text(json.dumps({
                "image": f"data:image/jpeg;base64,{encoded_image}",
                "fingertip": fingertip
            }))
            
    except WebSocketDisconnect:
        print("Client disconnected")

@app.websocket("/ws/age-detection")
async def age_detection_websocket(websocket: WebSocket):
    await websocket.accept()
    last_logged_age = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            img_data = base64.b64decode(message['image'].split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            try:
                results = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                if results:
                    age = results[0]['dominant_age']
                    
                    # Log to DB if age changed or enough time passed
                    if age != last_logged_age:
                        db = SessionLocal()
                        new_log = AILog(feature_type='age', result_value=str(age))
                        db.add(new_log)
                        db.commit()
                        db.close()
                        last_logged_age = age

                    await websocket.send_text(json.dumps({
                        "age": age
                    }))
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "error": str(e)
                }))
                
    except WebSocketDisconnect:
        print("Client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
