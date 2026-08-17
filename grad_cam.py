from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import preprocess as pi


def denormalize_siglip2(img) -> np.ndarray:
    img_arr = np.array(img, dtype=np.float32)
    

    if img_arr.min() < 0:
        img_arr = (img_arr * 0.5) + 0.5

    elif img_arr.max() > 1.0:
        img_arr = img_arr / 255.0

    img_arr = np.clip(img_arr, 0.0, 1.0)
    return (img_arr * 255).astype(np.uint8)


def preprocess_frame_for_cam(frame, img_size=224) -> np.ndarray:
    if isinstance(frame, (str, Path)):
        img = cv2.imread(str(frame))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(frame)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if img.dtype != np.uint8 and img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    normalized = pi.normalize_siglip2(img)
    if normalized.ndim == 3:
        normalized = np.expand_dims(normalized, axis=0)
    return normalized


def make_gradcampp_vit_heatmap(detector, image, pred_index=0):
    tensor_img = preprocess_frame_for_cam(image)         
    backbone = detector.backbone
    classifier = detector.classifier

    grad_input = keras.layers.Input(shape=(224, 224, 3), name="cam_input")

    x = grad_input
    tokens_3d = None

    for layer in backbone.vision_encoder.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        x = layer(x)
        if (
            hasattr(x, "shape")
            and len(x.shape) == 3
            and x.shape[1] is not None
            and x.shape[1] > 1
        ):
            tokens_3d = x

    cls_token = x

    for layer in classifier.layers:
        if isinstance(layer, keras.layers.InputLayer):
            continue
        cls_token = layer(cls_token)

    if tokens_3d is None:
        raise ValueError(
            "Saída 3D de patches não encontrada no encoder visual."
        )

    grad_model = Model(inputs=grad_input, outputs=[tokens_3d, cls_token])
    grad_model.trainable = False

    with tf.GradientTape() as tape:
        tokens, preds = grad_model(tensor_img, training=False)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, tokens)

    conv_output = tokens[0]
    grads_val = grads[0]

    first_derivative = grads_val
    second_derivative = grads_val * grads_val
    third_derivative = second_derivative * grads_val

    alpha_num = second_derivative
    alpha_denom = (
        2.0 * second_derivative + third_derivative * conv_output + 1e-8
    )
    alphas = alpha_num / tf.where(alpha_denom != 0, alpha_denom, 1e-8)

    weights = tf.reduce_sum(
        tf.maximum(first_derivative, 0.0) * alphas, axis=0
    )
    patch_embeddings = conv_output[1:]

    cam = tf.reduce_sum(weights * patch_embeddings, axis=-1)
    cam = tf.maximum(cam, 0)
    cam = cam / (tf.reduce_max(cam) + 1e-8)

    grid_size = int(np.sqrt(cam.shape[0]))
    cam_2d = tf.reshape(
        cam[: grid_size * grid_size], (grid_size, grid_size)
    )

    heatmap = tf.image.resize(
        cam_2d[None, :, :, None], [224, 224], method="bicubic"
    )[0, :, :, 0].numpy()
    heatmap = cv2.GaussianBlur(heatmap, (11, 11), 0)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)

    pred_prob = float(preds[0][pred_index])

    return heatmap, pred_prob


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha=0.40
) -> np.ndarray:
    rgb_img = denormalize_siglip2(image)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(rgb_img, 1 - alpha, heatmap_color, alpha, 0)


def explain_video_frame(detector, frame, is_temporal=False, frame_detector=None):
    model_for_cam = frame_detector if (is_temporal and frame_detector is not None) else detector

    if model_for_cam is None:
        raise ValueError("Para explicações em modelos temporais, 'frame_detector' deve ser fornecido.")
    
    heatmap, prob = make_gradcampp_vit_heatmap(model_for_cam, frame)
    return {
        "heatmap": heatmap,
        "pred_label": "fake" if prob > 0.5 else "real",
        "pred_prob": prob,
    }


def plot_gradcam_explanations(
    original_frames, explanations, video_label, figsize=(20, 7)
):
    n = len(explanations)
    if n == 0:
        print("[AVISO] Nenhuma explicação para plotar.")
        return None

    fig, axes = plt.subplots(2, n, figsize=figsize)
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (frame, expl) in enumerate(zip(original_frames, explanations)):
        rgb_disp = denormalize_siglip2(frame)

        frame_idx = expl.get("frame_idx", i)
        prob_fake = expl.get("frame_prob", expl.get("pred_prob", 0.0))

        axes[0, i].imshow(rgb_disp)
        axes[0, i].axis("off")
        axes[0, i].set_title(
            f"Frame #{frame_idx}\nProb FAKE: {prob_fake * 100:.1f}%",
            fontsize=10,
            fontweight="bold",
            color="#1e293b"
        )

        overlay = overlay_heatmap(rgb_disp, expl["heatmap"], alpha=0.40)
        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")

        is_fake = prob_fake > 0.5
        title_color = "#dc2626" if is_fake else "#16a34a"
        lbl_str = "FAKE" if is_fake else "REAL"

        axes[1, i].set_title(
            f"Grad-CAM++ [{lbl_str}]\nScore: {prob_fake:.3f}",
            fontsize=10,
            fontweight="bold",
            color=title_color,
        )

    label_color = "#dc2626" if video_label.lower() == "fake" else "#16a34a"
    plt.suptitle(
        f"Explicabilidade Grad-CAM++ | Vídeo: {video_label.upper()}\n"
        f"(Regiões Quentes = Focos de Atenção da Rede)",
        fontsize=13,
        fontweight="bold",
        color=label_color,
        y=1.03,
    )
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    
    plt.show()
    return fig
