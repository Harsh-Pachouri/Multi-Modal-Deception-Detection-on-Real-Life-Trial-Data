# Multimodal Deception Detection on Real‑Life Trial Data

## Overview
This repository contains the complete **multimodal deception‑detection pipeline** developed for our comparative study on *Real‑Life Trial* courtroom videos. Four progressively refined models are provided together with training scripts, pretrained checkpoints, and the accompanying paper. The final hybrid model (Model 3) fuses **BERT‑based text embeddings, acoustic features, and facial cues** and attains **94.51 % accuracy (AUC 0.94)** under strict subject‑level LOSO cross‑validation.citeturn0file3  

---

## Quick Start
```bash
# 1. Clone the repo
$ git clone https://github.com/<your‑org>/multimodal‑deception.git
$ cd multimodal‑deception

# 2. Create and activate a virtual environment (optional but recommended)
$ python -m venv venv
$ source venv/bin/activate      # Windows: .\venv\Scripts\activate

# 3. Install dependencies
$ pip install -r requirements.txt

# 4. Download and  the Real‑Life Trial dataset
$ mkdir -p data/
$ unzip /path/to/Real-life_Deception_Detection_2016.zip -d data/

# 5. Train / evaluate (example: final hybrid model)
$ python scripts/train.py --model 3 --data_dir data/Real-life_Deception_Detection_2016 \
                          --checkpoint_dir checkpoints/model3
```

> **Note**  All models support the same CLI; use `--model {0|1|2|3}` to pick an architecture.

---

## Dataset
| Name | Source | Modality Coverage | License |
|------|--------|-------------------|---------|
| **Real‑Life Trial (2016)** | M. Umut Şen *et al.* (IEEE T‑AC) | Text ✧ Audio ✧ Video | Academic / research only |

The dataset is **not redistributed** here for legal reasons. Please obtain it from the original authors or your institutional repository and place it under `data/`.citeturn0file3  
source : https://lit.eecs.umich.edu/downloads.html#Real-life%20Deception

---

## Models & Results
| ID | Architecture | Accuracy | AUC |
|----|--------------|----------|-----|
| 1 | Baseline + BERT | 69.14 % | 0.76 |
| 2 | HSTA Deep Network | 62.71 % | 0.67 |
| 3 | **Complex Hybrid (Final)** | **94.51 %** | **0.94** |

*Metrics are averaged over three random seeds under Leave‑One‑Subject‑Out (LOSO) cross‑validation.*citeturn0file3  

---

## Repository Layout
```
├── models/
│   ├── model1/                             # Baseline + BERT
│   ├── model2/                             # HSTA deep network
│   └── model3/                             # Complex hybrid (final)
└── README.md                               # <‑‑ you are here
```

---

## Dependencies
The project is tested on **Python ≥ 3.9**. Key packages:
- `torch` (≥ 2.2)
- `transformers` (≥ 4.38)
- `scikit‑learn`
- `torchvision`
- `opencv‑python`
- `face_recognition`
- `librosa`, `moviepy`

Install them in one go:
```bash
pip install -r requirements.txt
```

---


All experiments will print overall LOSO metrics comparable to the paper.citeturn0file0turn0file1turn0file2  

---

---

## License
This work is released under the **MIT License**. See `LICENSE` for full text.

---

## Acknowledgements
- U. M. Sen, V. Perez-Rosas, B. Yanikoglu, M. Abouelenien, M. Burzo and R. Mihalcea, "Multimodal Deception Detection using Real-Life Trial Data," in IEEE Transactions on Affective Computing, vol. 13, no. 1, pp. 306-319, 2022. [PDF]
- The open‑source libraries that made the project possible (PyTorch, Transformers, scikit‑learn, etc.).

---

Happy researching — pull requests are welcome!

