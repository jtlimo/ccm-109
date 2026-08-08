import gc
from pathlib import Path
import cv2
import numpy as np
from mtcnn import MTCNN
import config

detector = None

cfg = config.get_config()


def get_detector():
    global detector
    if detector is None:
        detector = MTCNN()
    return detector

def normalize_siglip2(face_rgb: np.ndarray) -> np.ndarray:
    resized = cv2.resize(face_rgb, (cfg['img_size'], cfg['img_size']), interpolation=cv2.INTER_CUBIC)
    normalized = (resized.astype(np.float32) / 255.0 - cfg['siglip_mean']) / cfg['siglip_std']
    return normalized

def save_face(face_rgb: np.ndarray, filepath_base: Path):
    norm_face = normalize_siglip2(face_rgb)
    np.save(f"{filepath_base}_siglip2.npy", norm_face)
    
    bgr_face = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{filepath_base}.jpg", bgr_face, [cv2.IMWRITE_JPEG_QUALITY, cfg['jpg_quality']])

def detect_faces(frame: np.ndarray, confidence=cfg["confidence"]) -> list:
    mtcnn_detector = get_detector()
    results = mtcnn_detector.detect_faces(frame)
    faces = []
    
    for r in results:
        if r.get("confidence", 0) < confidence:
            continue
        
        x, y, w, h = r["box"]
        x, y = max(0, x), max(0, y)
        
        if w < cfg["min_face_size"] or h < cfg["min_face_size"]:
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

def process_video(video_path: str, output_dir: str) -> int:
    max_frames = cfg["max_frames"]
    skip_frames = cfg["skip_frames"]
    max_read_limit = cfg["max_read_limit"]

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

def preprocess_video(video_path, output_dir=None):
    output_dir = output_dir or cfg["pred_output_dir"]
    video_path = Path(video_path)
    video_name = video_path.stem
    
    n_faces = process_video(video_path, output_dir)
    
    if n_faces == 0:
       print(f"[INFO] Nenhuma face detectada em {video_path.name}")
       return {"frame_paths": []}

    video_dir = Path(output_dir) / video_name
    frame_paths = sorted(video_dir.glob("*.jpg"))


    if frame_paths is None or len(frame_paths) == 0:
         return {"frame_paths": []}

    return {
        "frame_paths": [str(p) for p in frame_paths],
    }
