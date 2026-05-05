import os
import sys

print("Testing imports...")

try:
    print("Importing FastAPI...")
    from fastapi import FastAPI
    print("FastAPI imported.")
except Exception as e:
    print(f"FastAPI failed: {e}")

try:
    print("Importing uvicorn...")
    import uvicorn
    print("uvicorn imported.")
except Exception as e:
    print(f"uvicorn failed: {e}")

try:
    print("Importing cv2...")
    import cv2
    print("cv2 imported.")
except Exception as e:
    print(f"cv2 failed: {e}")

try:
    print("Importing mediapipe...")
    import mediapipe as mp
    print("mediapipe imported.")
except Exception as e:
    print(f"mediapipe failed: {e}")

try:
    print("Importing numpy...")
    import numpy as np
    print("numpy imported.")
except Exception as e:
    print(f"numpy failed: {e}")

try:
    print("Importing DeepFace (this might take a while or crash)...")
    from deepface import DeepFace
    print("DeepFace imported.")
except Exception as e:
    print(f"DeepFace failed: {e}")

print("All imports tested.")
