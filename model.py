import tensorflow as tf
from tensorflow import keras

def build_classifier(feature_dim: int = 768, l2_reg: float = 1e-4) -> keras.Model:
    inputs = keras.layers.Input(shape=(feature_dim,), name="features_input")
    
    x = keras.layers.Dense(
        256, activation='relu', 
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(inputs)
    x = keras.layers.Dropout(0.5)(x)
    
    x = keras.layers.Dense(
        128, activation='relu', 
        kernel_regularizer=keras.regularizers.l2(l2_reg)
    )(x)
    x = keras.layers.Dropout(0.4)(x)
    
    outputs = keras.layers.Dense(1, activation='sigmoid', name="sigmoid_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="deepfake_classifier")
    return model

def build_temporal_classifier(sequence_length: int = 32, feature_dim: int = 768, dropout_rate: float = 0.3) -> keras.Model:
    inputs = keras.layers.Input(shape=(sequence_length, feature_dim), name="sequence_input")
    
    x = keras.layers.Bidirectional(
        keras.layers.GRU(64, return_sequences=False), 
        name="bigru_layer"
    )(inputs)
    x = keras.layers.Dropout(dropout_rate)(x)
    
    outputs = keras.layers.Dense(1, activation='sigmoid', name="sigmoid_output")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="temporal_deepfake_classifier")
    return model

class DeepfakeDetector:
    def __init__(self, backbone_path: str, classifier_path: str, img_size: int = 224):
        self.img_size = img_size
        self.backbone = self._load_and_freeze(backbone_path, "Backbone")
        self.classifier = self._load_and_freeze(classifier_path, "Classifier")
        self.model = self._build_e2e_model()
        self._verify_frozen()

    def _load_and_freeze(self, path: str, name: str) -> keras.Model:
        print(f"[INFO] Carregando {name} de: {path}")
        model = keras.models.load_model(path)
        model.trainable = False
        return model

    def _build_e2e_model(self) -> keras.Model:
        image_input = keras.layers.Input(
            shape=(self.img_size, self.img_size, 3), 
            name="images"
        )
        
        features = self.backbone.get_vision_embeddings(image_input)
        
        x = features
        for layer in self.classifier.layers:
            if isinstance(layer, keras.layers.InputLayer):
                continue
            x = layer(x)

        e2e_model = keras.Model(inputs=image_input, outputs=x, name="end_to_end_detector")
        e2e_model.trainable = False
        return e2e_model

    def _verify_frozen(self):
        trainable_params = sum(
            keras.backend.count_params(w) for w in self.model.trainable_weights
        )
        assert trainable_params == 0, f"Erro: O modelo possui {trainable_params} parâmetros treináveis!"
        print(f"[OK] Modelo End-to-End congelado ({self.model.count_params():,} parâmetros).")
