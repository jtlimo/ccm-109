import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import matplotlib.cm as cm

print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")


def make_gradcampp_vit_heatmap(model, image, target_layer_name=None, pred_index=None):    
    if target_layer_name is None:      
        for layer in reversed(model.layers):
            output_shape = layer.output_shape
            if output_shape is not None and len(output_shape) == 3 and output_shape[1] is not None:
                target_layer_name = layer.name
                print(f"[Auto-detect] Camada alvo: {target_layer_name} | Shape: {output_shape}")
                break
        
        if target_layer_name is None:
            candidates = [(l.name, l.output_shape) for l in model.layers 
                          if l.output_shape is not None and len(l.output_shape) == 3]
            raise ValueError(f"Não encontrou camada 3D automática. Candidatas: {candidates}")
    
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[
            model.get_layer(target_layer_name).output,
            model.output
        ]
    )
    
    with tf.GradientTape() as tape:
        transformer_output, preds = grad_model(image, training=False)
        
    
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        
        class_channel = preds[:, pred_index]
    
    grads = tape.gradient(class_channel, transformer_output)
    
    if grads is None:
        raise RuntimeError("Gradiente é None. Verifique se a camada alvo está conectada ao output.")
    
        
    
    conv_output = transformer_output[0]  
    grads_val = grads[0]                 
    
    first_derivative = grads_val
    second_derivative = grads_val * grads_val
    third_derivative = second_derivative * grads_val
    
    alpha_num = second_derivative
    alpha_denom = 2.0 * second_derivative + third_derivative * conv_output + 1e-8
    
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom) * 1e-8)
    
    alphas = alpha_num / alpha_denom
    
    weights = tf.reduce_sum(
        tf.maximum(first_derivative, 0.0) * alphas,
        axis=0 
    )
    
    patch_embeddings = conv_output[1:] 
    
    cam = tf.reduce_sum(weights * patch_embeddings, axis=-1)
    cam = tf.maximum(cam, 0)
    
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    

    n_patches = cam.shape[0]
    grid_size = int(np.sqrt(n_patches))
    
    if grid_size * grid_size != n_patches:
        print(f"[AVISO] n_patches={n_patches} não é quadrado perfeito. Tentando grid {grid_size}x{grid_size}")
        cam = cam[:grid_size * grid_size]
    
    cam_2d = tf.reshape(cam, (grid_size, grid_size))
    
    img_size = image.shape[1]
    heatmap = tf.image.resize(
        cam_2d[None, :, :, None],
        [img_size, img_size],
        method="bilinear"
    )[0, :, :, 0]
    
    heatmap = heatmap.numpy()
    
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    
    pred_prob = float(preds[0][pred_index])
    pred_index = int(pred_index)
    
    return heatmap, pred_index, pred_prob

def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    
    return overlayed


def plot_gradcampp_vit(image, heatmap, pred_label, pred_prob, figsize=(14, 4)):
    SIGLIP_MEAN = np.array([0.5, 0.5, 0.5])
    SIGLIP_STD = np.array([0.5, 0.5, 0.5])
    
    img_vis = image.copy()
    if img_vis.max() <= 1.5:
        img_vis = (img_vis * SIGLIP_STD + SIGLIP_MEAN)
    img_vis = np.clip(img_vis, 0, 1)
    
    overlayed = overlay_heatmap(img_vis, heatmap, alpha=0.5)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(img_vis)
    axes[0].set_title("Imagem Original")
    axes[0].axis("off")
    
    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM++ Heatmap")
    axes[1].axis("off")
    
    axes[2].imshow(overlayed)
    axes[2].set_title(f"Overlay | {pred_label.upper()} ({pred_prob:.3f})")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

def explain_video_frame(model, frame_tensor, target_layer_name=None, cfg=None):
    if frame_tensor.ndim == 3:
        frame_tensor = np.expand_dims(frame_tensor, axis=0)
    
    heatmap, pred_idx, prob = make_gradcampp_vit_heatmap(
        model, frame_tensor, target_layer_name, pred_index=None
    )
    
    label = "fake" if pred_idx == 1 else "real"
    
    return {
        "heatmap": heatmap,
        "pred_label": label,
        "pred_prob": prob,
        "pred_index": pred_idx
    }


def explain_video_frames(model, frames_tensor, frame_indices=None, target_layer_name=None):
    if frame_indices is None:
        frame_indices = range(len(frames_tensor))
    
    explanations = []
    for idx in frame_indices:
        frame = frames_tensor[idx]
        exp = explain_video_frame(model, frame, target_layer_name)
        exp["frame_idx"] = idx
        explanations.append(exp)
    
    return explanations


def plot_video_explanations(original_frames, explanations, max_frames=8, figsize=(16, 8)):
    n_show = min(len(explanations), max_frames)
    cols = 4
    rows = (n_show + cols - 1) // cols
    
    fig, axes = plt.subplots(rows * 2, cols, figsize=figsize)
    if rows * 2 == 2:
        axes = axes.reshape(2, -1)
    
    for i, exp in enumerate(explanations[:n_show]):
        col = i % cols
        row = i // cols
        
        frame = original_frames[exp["frame_idx"]]
        heatmap = exp["heatmap"]
        label = exp["pred_label"]
        prob = exp["pred_prob"]
        
        axes[row * 2, col].imshow(frame)
        axes[row * 2, col].set_title(f"Frame {exp['frame_idx']} | {label.upper()}")
        axes[row * 2, col].axis("off")
        
        overlayed = overlay_heatmap(frame, heatmap, alpha=0.5)
        axes[row * 2 + 1, col].imshow(overlayed)
        axes[row * 2 + 1, col].set_title(f"Grad-CAM++ ({prob:.3f})")
        axes[row * 2 + 1, col].axis("off")
    
    for i in range(n_show, rows * cols):
        col = i % cols
        row = i // cols
        if row * 2 < axes.shape[0]:
            axes[row * 2, col].axis("off")
            axes[row * 2 + 1, col].axis("off")
    
    plt.tight_layout()
    plt.show()


def inspect_model_layers(model):
    print(f"{'Layer #':<8} {'Name':<35} {'Output Shape':<30} {'Type'}")
    print("=" * 90)
    
    for i, layer in enumerate(model.layers):
        shape = str(layer.output_shape)
        cls = layer.__class__.__name__
        print(f"{i:<8} {layer.name:<35} {shape:<30} {cls}")
    
    print("\\n[CAMADAS 3D — candidatas para Grad-CAM++]")
    for i, layer in enumerate(model.layers):
        shape = layer.output_shape
        if shape is not None and len(shape) == 3:
            print(f"  {layer.name}: {shape}")


def suggest_target_layer(model):
    candidates = []
    for layer in reversed(model.layers):
        shape = layer.output_shape
        if shape is not None and len(shape) == 3:
            candidates.append((layer.name, shape, layer.__class__.__name__))
    
    print("Candidatas (da mais profunda para a mais rasa):")
    for name, shape, cls in candidates[:5]:
        print(f"  • {name} | {shape} | {cls}")
    
    if candidates:
        print(f"\\n[SUGESTÃO] Use: target_layer_name='{candidates[0][0]}'")
    else:
        print("[ERRO] Nenhuma camada 3D encontrada. O modelo pode não ser um ViT.")
