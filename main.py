import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import base64
import json
from deepface import DeepFace

app = FastAPI()

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
                
            # Decode frame
            img_data = base64.b64decode(message['image'].split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            if canvas is None:
                canvas = np.zeros((h, w, 3), np.uint8)

            # Process with Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            fingertip = None
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Index finger tip is landmark 8
                    lm8 = hand_lms.landmark[8]
                    cx, cy = int(lm8.x * w), int(lm8.y * h)
                    fingertip = (cx, cy)
                    
                    # Check if finger is up (simplified)
                    # We can use landmark 6 (index pip) to check if 8 is above it
                    lm6 = hand_lms.landmark[6]
                    if lm8.y < lm6.y:
                        points.append(fingertip)
                    else:
                        points.append(None) # Break line

            # Draw on canvas
            for i in range(1, len(points)):
                if points[i-1] is not None and points[i] is not None:
                    cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 5)
            
            # Combine frame and canvas
            combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
            
            # Encode back to base64
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
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            img_data = base64.b64decode(message['image'].split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            try:
                # DeepFace analysis
                # We do this every few frames or on request to save CPU
                results = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                age = results[0]['dominant_age'] if results else "Unknown"
                
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
