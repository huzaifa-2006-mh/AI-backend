import os
import asyncio
import gc
import json
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime

# Environment settings for memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model placeholders (LAZY LOADING)
hands_model = None
deepface_ready = False

@app.get("/api")
async def root():
    return {"status": "Online", "mode": "Instant-Start Ready"}

@app.get("/api/logs")
async def get_logs():
    # Simple mock logs for stability if DB is slow
    return [{"id": 1, "timestamp": datetime.utcnow().isoformat(), "feature_type": "system", "result_value": "Engine Ready"}]

@app.websocket("/api/ws/air-writing")
async def air_writing_websocket(websocket: WebSocket):
    global hands_model
    await websocket.accept()
    
    # Load Mediapipe only when needed
    if hands_model is None:
        import cv2
        import numpy as np
        import mediapipe as mp
        hands_model = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    else:
        import cv2
        import numpy as np

    canvas = None
    points = []
    
    try:
        while True:
            data = await websocket.receive_text()
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

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_model.process(rgb_frame)
            
            tip = None
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    lm8 = hand_lms.landmark[8]
                    lm6 = hand_lms.landmark[6]
                    cx, cy = int(lm8.x * w), int(lm8.y * h)
                    tip = (cx, cy)
                    if lm8.y < lm6.y:
                        points.append(tip)
                    else:
                        points.append(None)

            for i in range(1, len(points)):
                if points[i-1] and points[i]:
                    cv2.line(canvas, points[i-1], points[i], (0, 255, 255), 7)
            
            combined = cv2.addWeighted(frame, 0.7, canvas, 0.3, 0)
            _, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
            encoded = base64.b64encode(buffer).decode('utf-8')
            
            await websocket.send_text(json.dumps({
                "image": f"data:image/jpeg;base64,{encoded}",
                "fingertip": tip
            }))
            
            del frame, rgb_frame, combined
            gc.collect()
                
    except:
        pass

@app.websocket("/api/ws/age-detection")
async def age_detection_websocket(websocket: WebSocket):
    await websocket.accept()
    
    # Imports inside to prevent startup crash
    import cv2
    import numpy as np
    from deepface import DeepFace
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            img_bytes = base64.b64decode(msg['image'].split(',')[1])
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            try:
                # Process age
                results = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
                if results:
                    age = results[0]['dominant_age']
                    await websocket.send_text(json.dumps({
                        "age": age,
                        "confidence": 0.9,
                        "new_log": False
                    }))
            except:
                pass
            
            del frame
            gc.collect()
    except:
        pass

# Static serving
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
    # Use port 5000 as required by Replit
    uvicorn.run(app, host="0.0.0.0", port=5000)



