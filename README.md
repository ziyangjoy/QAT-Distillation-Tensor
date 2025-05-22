# Quantization-Aware and Tensor-Compressed Training of Transformers

This repository contains the implementation of the methods described in the paper:

> **Quantization-Aware and Tensor-Compressed Training of Transformers for Natural Language Understanding**  
> Zi Yang, Samridhi Choudhary, Siegfried Kunzmann, Zheng Zhang  
> [arXiv:2306.01076](https://www.isca-archive.org/interspeech_2023/yang23s_interspeech.pdf)

## Overview

Transformer models deliver outstanding performance on many NLU tasks, but their size makes them hard to deploy on edge devices. This work proposes:

- **Tensor Compression**: Reducing model size using low-rank tensor formats (e.g., TT, TTM).
- **Quantization-Aware Training**: Learnable scale-aware quantization during training.
- **Layer-by-Layer Distillation**: Task-specific knowledge distillation from a fine-tuned teacher model to a compact tensorized student model.

This leads to up to **88× compression, 23x FLOPs reduction** with minimal loss.

## Features

- Low-rank tensor compression (TT, TTM) for embedding and linear layers.
- Learnable quantization during training for low-precision weights/activations.
- Optional knowledge distillation to boost performance of compressed models.
- Supports BERT models on GLUE/ATIS datasets.

## Installation

```bash
git clone https://github.com/ziyangjoy/QAT-Distillation-Tensor.git
pip install -r requirements.txt
```

## Usage
```bash
sh run_distill.sh
```

## Main Results
| Model              | Precision | Size (MB)     | FLOPs (G)     | MNLI | QNLI | SST-2 | MRPC |
|-------------------|-----------|---------------|---------------|------|------|-------|------|
| BERT-base [2]     | FP₃₂      | 423 (1×)      | 20.3 (1×)     | 83.4 | 91.2 | 92.8  | 87.7 |
| DistilBERT [14]   | FP₃₂      | 254 (1.7×)    | 10.1 (2×)     | 82.2 | 89.2 | 91.3  | 87.5 |
| BinaryBERT [19]   | INT₁      | 16.5 (26×)    | 3.1 (7×)      | 84.2 | 91.5 | 92.6  | 85.5 |
| LadaBERT-4 [22]   | FP₃₂      | 42 (10×)      | —             | 75.8 | 75.1 | 84.0  | —    |
| **Rank 50**       |           |               |               |      |      |       |      |
|                   | FP₃₂      | 99 (4×)       | 3.8 (5×)      | 82.1 | 89.1 | 90.0  | 86.5 |
|                   | INT₈      | 24.3 (17×)    | 3.8 (5×)      | 80.7 | 88.1 | 89.6  | 85.8 |
|                   | INT₄      | 12.1 (35×)    | 1.9 (11×)     | 79.7 | 87.9 | 89.2  | 85.5 |
| **Rank 30**       |           |               |               |      |      |       |      |
|                   | FP₃₂      | 39 (11×)      | 1.8 (11×)     | 80.1 | 88.1 | 89.3  | 85.1 |
|                   | INT₈      | 9.5 (45×)     | 1.8 (11×)     | 78.3 | 87.2 | 89.2  | 85.0 |
|                   | INT₄      | 4.8 (88×)     | 0.9 (23×)     | 77.4 | 86.9 | 88.3  | 84.8 |

## Citation
```
@inproceedings{yang2023quantization,
  title={Quantization-aware and Tensor-compressed Training of Transformers for Natural Language Understanding},
  author={Yang, Zi and Choudhary, Samridhi and Kunzmann, Siegfried and Zhang, Zheng},
  booktitle={INTERSPEECH},
  year={2023}
}
```