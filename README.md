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
python tokenizer/train.py --data data/raw/
```


4. Prepare dataset
```
python data/prepare.py
```

5. Train model
```
python training/train.py
```

6. Generate text from checkpoint
```
python evaluation/generate.py \
  --checkpoint checkpoints/checkpoint-best.pt \
  --prompt "Hello, I am Nikita and I am building my own LLM"
```


## 5. Current training run (local, CPU)

device: cpu

steps: ≈ 2000 (config was 10000, but early checkpoint is enough)

time: ~1 hour on MacBook Pro (M2Pro)

best checkpoint: checkpoints/checkpoint-best.pt

validation loss: ≈ 0.08

perplexity: ≈ 1.09

trainer warning: “Perplexity < 1.1 → overfitting”



## 6. Why overfitting happened

small dataset

small model that learns fast

long training (2000+ steps)

same domain text

How to fix:

add more data (data/raw/)

increase dropout in llm.config.js from 0.2 → 0.3

reduce training.max_steps (e.g. 3000 instead of 10000)



## 7. Next steps (agent plan)

✅ train and get checkpoint

🟡 add simple API (FastAPI) around evaluation/generate.py

🟡 add “tools”: load CSV, describe, train simple model

🟡 build loop: LLM decides → tool runs → LLM объясняет

🟡 UI (Gradio / React) → “Nikita DS Agent”

So the final goal: “my own small LLM + agent that helps with data-analysis tasks”
