# nikita-llm

Custom LLM training project scaffolded with [`@theanikrtgiri/create-llm`].

This repo contains a **fully working pipeline** for training a small GPT-like model on local machine (macOS, CPU). The goal of the project is to learn how to **create, train and serve** your own LLM, and then wrap it into an AI agent (for example, a “data scientist agent”).

---

## 1. Project goals

- ✅ scaffold LLM project locally (no cloud required)
- ✅ train tokenizer on **own** data
- ✅ prepare dataset and run training loop
- ✅ save checkpoints and best model
- ✅ run text generation / chat on the trained model
- 🛠 next: wrap model into API (FastAPI) and build agent on top

---

## 2. Tech stack

- **Python** 3.12
- **PyTorch** 2.x
- **Transformers**
- **Gradio** (for chat UI)
- **create-llm** CLI (project bootstrap)
- OS: **macOS / Apple Silicon (M2)**

---

## 3. Project structure

```text
.
├── data/             # raw and processed data
├── tokenizer/        # tokenizer training script + tokenizer.json
├── training/         # main training loop, callbacks, dashboard
├── evaluation/       # generation and evaluation scripts
├── models/           # model architectures (nano, tiny, small, base)
├── checkpoints/      # saved models (ignored in git)
├── logs/             # training logs (ignored in git)
├── llm.config.js     # main config (model + training)
└── README.md
```



## 4. How to run

1. Clone / open project
```
git clone https://github.com/NikitaMarshchonok/nikita-llm.git
cd nikita-llm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Add data
  ```
mkdir -p data/raw
curl https://www.gutenberg.org/files/100/100-0.txt > data/raw/shakespeare.txt
```  


3. Train tokenizer
```

```
