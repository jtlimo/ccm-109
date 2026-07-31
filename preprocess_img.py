import cv2
import numpy as np
from mtcnn import MTCNN
import numpy as np
import tensorflow as tf
import gc
from pathlib import Path

VIDEO_PATH = "/dataset/FaceForensics"
VIDEO_EXTENSIONS = ("*.mp4", "*.avi", "*.mov", "*.mkv")
CONFIDENCE_THRESHOLD = 0.9
ALIGN = True
SKIP_FRAMES = 9                    # 0 = todos; N = pula N (processa 1 a cada N+1)
MAX_FRAMES_PER_VIDEO = 30          # None = todos os frames válidos
MAX_READ_LIMIT = 1000              # None = sem limite de leitura
MAX_VIDEOS = None                  # None = todos os vídeos
MIN_FACE_SIZE = 80                 # mínimo de pixels (largura/altura)
JPG_QUALITY = 95

SIGLIP2_INPUT_SIZE = 224   
SIGLIP2_MEAN = np.array([0.5, 0.5, 0.5])
SIGLIP2_STD = np.array([0.5, 0.5, 0.5])

detector = MTCNN()


def preprocess_siglip2(face_rgb, target_size=SIGLIP2_INPUT_SIZE):
    resized = cv2.resize(face_rgb, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return (resized.astype(np.float32) / 255.0 - SIGLIP2_MEAN) / SIGLIP2_STD

def save_face(face_rgb, filepath_base):
    np.save(str(filepath_base) + "_siglip2.npy", preprocess_siglip2(face_rgb))
    cv2.imwrite(str(filepath_base) + ".jpg", cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])



def detect_faces(frame, confidence=CONFIDENCE_THRESHOLD):
    results = detector.detect_faces(frame)
    
    faces = []
    for r in results:
        conf = r.get("confidence", 0)
        if conf < confidence:
            continue
        
        x, y, w, h = r["box"]
        if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
            continue
        
        face_crop = frame[y:y+h, x:x+w]
        if face_crop.size == 0:
            continue
        
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        faces.append({
            "bbox": [x, y, x+w, y+h],
            "confidence": float(conf),
            "face_rgb": face_rgb
        })
    
    return faces


def process_video(video_path, output_dir, max_frames=MAX_FRAMES_PER_VIDEO,
                  skip_frames=SKIP_FRAMES, max_read_limit=MAX_READ_LIMIT):
    
    video_path = Path(video_path)
    frames_dir = Path(output_dir) / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERRO] Não abriu: {video_path}")
        return 0

    frame_idx = 0
    processed_idx = 0
    saved_count = 0

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
            if not faces:
                frame_idx += 1
                processed_idx += 1
                continue

            for i, face in enumerate(faces):
                filepath_base = frames_dir / f"frame_{frame_idx+1:06d}_face_{i+1:03d}"
                save_face(face["face_rgb"], filepath_base)

            saved_count += 1
            frame_idx += 1
            processed_idx += 1

            if frame_idx % 100 == 0:
                tf.keras.backend.clear_session()
                gc.collect()

    finally:
        cap.release()
        gc.collect()

    print(f"[OK] {video_path.name}: {saved_count} faces salvas em {frames_dir}")
    return saved_count
