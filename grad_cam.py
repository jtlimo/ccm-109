from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
import preprocess as pi


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
    calibrated_heatmap = heatmap * pred_prob

    return calibrated_heatmap, pred_prob


def overlay_heatmap(
    image: np.ndarray, heatmap: np.ndarray, alpha=0.45
) -> np.ndarray:
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    return cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)


def explain_video_frame(detector, frame):
    heatmap, prob = make_gradcampp_vit_heatmap(detector, frame)
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
        img_disp = np.array(frame)
        if img_disp.max() <= 1.0:
            img_disp = (img_disp * 255).astype(np.uint8)

        axes[0, i].imshow(img_disp)
        axes[0, i].axis("off")
        axes[0, i].set_title(
            f"Frame {expl['frame_idx']}\nProb FAKE: {expl['frame_prob']*100:.1f}%",
            fontsize=11,
            fontweight="bold",
        )

        overlay = overlay_heatmap(img_disp, expl["heatmap"], alpha=0.45)
        axes[1, i].imshow(overlay)
        axes[1, i].axis("off")

        is_fake = expl["frame_prob"] > 0.5
        title_color = "#dc2626" if is_fake else "#16a34a"
        lbl_str = "FAKE" if is_fake else "REAL"

        axes[1, i].set_title(
            f"Grad-CAM++ [{lbl_str}]\nAtivação: {expl['frame_prob']:.3f}",
            fontsize=11,
            fontweight="bold",
            color=title_color,
        )

    plt.suptitle(
        f"Explicabilidade Grad-CAM++ | Vídeo: {video_label.upper()}\n"
        f"(Azul/Ciano = Região Neutra/Real | Amarelo/Vermelho = Artefato de Manipulação/Fake)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.show()
    return fig
