import os
import gc
from pathlib import Path
import cv2
import numpy as np
from mtcnn import MTCNN


VIDEO_PATH = "/dataset/FaceForensics"
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")

CONFIDENCE_THRESHOLD = 0.9
SKIP_FRAMES = 9                    # 0 = processa todos; N = pula N
MAX_FRAMES_PER_VIDEO = 30          # Limite de frames com rosto
MAX_READ_LIMIT = 1000              # Limite de leitura total de frames
MAX_VIDEOS = None
MIN_FACE_SIZE = 80
JPG_QUALITY = 95

SIGLIP2_INPUT_SIZE = 224   
SIGLIP2_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
SIGLIP2_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

detector = None

def get_detector():
    global detector
    if detector is None:
        detector = MTCNN()
    return detector

def normalize_siglip2(face_rgb: np.ndarray, target_size=SIGLIP2_INPUT_SIZE) -> np.ndarray:
    resized = cv2.resize(face_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    normalized = (resized.astype(np.float32) / 255.0 - SIGLIP2_MEAN) / SIGLIP2_STD
    return normalized

def save_face(face_rgb: np.ndarray, filepath_base: Path):
    norm_face = normalize_siglip2(face_rgb)
    np.save(f"{filepath_base}_siglip2.npy", norm_face)
    
    bgr_face = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{filepath_base}.jpg", bgr_face, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])

def detect_faces(frame: np.ndarray, confidence=CONFIDENCE_THRESHOLD) -> list:
    mtcnn_detector = get_detector()
    results = mtcnn_detector.detect_faces(frame)
    faces = []
    
    for r in results:
        if r.get("confidence", 0) < confidence:
            continue
        
        x, y, w, h = r["box"]
        x, y = max(0, x), max(0, y)
        
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue
        
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue
        
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        faces.append({
            "bbox": [x, y, x+w, y+h],
            "confidence": float(r["confidence"]),
            "face_rgb": face_rgb
        })
    return faces

def process_video(video_path: str, output_dir: str, max_frames=MAX_FRAMES_PER_VIDEO,
                  skip_frames=SKIP_FRAMES, max_read_limit=MAX_READ_LIMIT) -> int:

    video_path = Path(video_path)
    frames_dir = Path(output_dir) / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERRO] Falha ao abrir o vídeo: {video_path}")
        return 0

    frame_idx, processed_idx, saved_count = 0, 0, 0

    try:
        while True:
            if max_frames is not None and saved_count >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            if max_read_limit is not None and processed_idx >= max_read_limit:
                break

            faces = detect_faces(frame)
            if faces:
                for i, face in enumerate(faces):
                    filepath_base = frames_dir / f"frame_{frame_idx+1:06d}_face_{i+1:03d}"
                    save_face(face["face_rgb"], filepath_base)
                saved_count += 1

            frame_idx += 1
            processed_idx += 1

    finally:
        cap.release()
        gc.collect()

    return saved_count
