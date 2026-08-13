# CCM-109

Modelo de detecção de deepfake usando visão computacional.

## Descrição

Projeto acadêmico (UFABC – CCM-109) para classificação de imagens como reais ou falsas (deepfakes). Utiliza extração de features com modelos pré-treinados de visão e classificadores supervisionados.

## Estrutura

```
ccm-109/
├── train.ipynb              # Treinamento do modelo
├── prediction.ipynb         # Inferência e predição
├── face_extraction.ipynb    # Extração de faces do dataset
├── face_extraction_torch    # Extração de faces do dataset utilizando pytorch no macOs
├── grad_cam.py              # Visualização de atenção (Grad-CAM)
├── preprocess.py            # Pré-processamento de imagens
├── preprocess_macos.py      # Pré-processamento de imagens no MacOs Mini M1
├── regenerate_kernel.py     # Registro do kernel Jupyter com CUDA
├── requirements.txt         # Dependências Python
├── devbox.json              # Configuração do ambiente de desenvolvimento
└── .devbox/                 # Ambiente Devbox (gerado automaticamente)
```

## Ambiente

O projeto usa [Devbox](https://www.jetify.com/devbox) para gerenciar o ambiente com Python 3.12 e CUDA.

```bash
devbox shell
```

O hook inicial ativa o venv e registra o kernel Jupyter com suporte a CUDA.

## Dependências

```bash
pip install -r requirements.txt
```

Principais: Jupyter, TensorFlow/Keras, Keras Hub (SigLIP2), NumPy, Matplotlib.

## Datasets

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [CelebDF](https://github.com/yuezunli/celeb-deepfakeforensics)
- Dataset customizado (próprio)

## Uso

Execute os notebooks na ordem:

1. `face_extraction.ipynb` — extrai faces dos vídeos
2. `train.ipynb` — treina o modelo
3. `prediction.ipynb` — avalia e prediz

Use `grad_cam.py` para visualizar as regiões de atenção do modelo.

Caso esteja utilizando MacOs a extração de faces deve ser executada através do notebook `face_extraction_torch.ipynb` por conta de limitações técnicas do tensorflow/keras em combinação com o processador da Apple M1.

## Licença

GPL-3.0
