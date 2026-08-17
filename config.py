import ipywidgets as widgets
from IPython.display import display
import glob
import os
import numpy as np

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

dataset_widget = widgets.Dropdown(
    options=['Celeb-DF','Custom', 'FF++'],
    value='FF++',
    description='Dataset:',
    style={'description_width': '100px'},
    layout=widgets.Layout(width='400px')
)

threshold_widget = widgets.FloatSlider(
    value=0.5,
    min=0.1,
    max=0.9,
    step=0.05,
    description='Threshold:',
    style={'description_width': '100px'},
    layout=widgets.Layout(width='400px')
)

batch_size_widget = widgets.IntSlider(
    value=32,
    min=8,
    max=128,
    step=8,
    description='Batch size:',
    style={'description_width': '100px'},
    layout=widgets.Layout(width='400px')
)

aggregation_widget = widgets.Dropdown(
    options=['median', 'mean', 'max', 'trimmed_mean'],
    value='median',
    description='Agregação:',
    style={'description_width': '100px'},
    layout=widgets.Layout(width='400px')
)

trim_percent_widget = widgets.IntSlider(
    value=10,
    min=0,
    max=50,
    step=5,
    description='Trim %:',
    style={'description_width': '100px'},
    layout=widgets.Layout(width='400px')
)

def choice_config():
    print("=" * 60)
    print("CONFIGURAÇÕES")
    print("=" * 60)
    display(
    dataset_widget, 
    threshold_widget, 
    batch_size_widget,
    aggregation_widget,
    trim_percent_widget
)

def get_config():    
    cfg = {
        "dataset": dataset_widget.value,
        "threshold": threshold_widget.value,
        "batch_size": batch_size_widget.value,
        "aggregation": aggregation_widget.value,
        "trim_percent": trim_percent_widget.value,
        
        "img_size": SIGLIP2_INPUT_SIZE,  
        "confidence": CONFIDENCE_THRESHOLD,
        "min_face_size": MIN_FACE_SIZE,
        "skip_frames": SKIP_FRAMES,
        "max_frames": MAX_FRAMES_PER_VIDEO,
        "max_read_limit": MAX_READ_LIMIT,
        "jpg_quality": JPG_QUALITY,
        "video_extensions": VIDEO_EXTENSIONS,
        "siglip_mean": SIGLIP2_MEAN,
        "siglip_std": SIGLIP2_STD
    }
    
    DATASET_PATHS = {
        'Celeb-DF': '/dataset/Celeb-DF',
        'Custom': '/dataset/Custom',
        'FF++': '/dataset/FF',
    }
    
    base_path = DATASET_PATHS.get(cfg['dataset'], '')
    cfg["dataset_path"] = base_path
    cfg["pred_output_dir"] = os.path.join(base_path, "predictions")
    cfg["frames_output_dir"] = os.path.join(base_path, "frames")
    cfg["fake_dir"] = os.path.join(base_path, "fake")
    cfg["real_dir"] = os.path.join(base_path, "real")
    
    return cfg
