# WhatsApp AI Clone (Gemma 3 1B QLoRA Fine-tuning)

This project allows you to fine-tune a Large Language Model (Gemma 3 1B Instruct) on your personal WhatsApp chat data. By leveraging QLoRA, the entire training process can fit within an 8GB VRAM GPU (like the RTX 4060). 

After training, you can chat with the AI in your terminal or via a Streamlit web interface, and it will respond in your conversational style!

---

## 🚀 Quick Start Guide

### 1. Installation

1. Make sure you have Python 3.10+ installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Prepare Your Data

1. Export your WhatsApp chat (without media) as a `.txt` file.
2. Place the exported `.txt` files inside the `data/` folder.
3. Open `config.py` and ensure `TARGET_PERSONA` is set to the exact name you use in your WhatsApp chats (e.g., `"Rudra"`).
4. Run the preprocessing script:
   ```bash
   python preprocess.py
   ```
   *This will parse the chat format, clean system messages, and create `train.jsonl` and `val.jsonl` in the `data/` directory.*

### 3. Fine-Tune the Model

Start the training process:
```bash
python train.py
```
**Hardware Considerations:**
- QLoRA 4-bit quantization reduces memory usage drastically.
- Gradient Accumulation allows us to simulate larger batch sizes without running Out-Of-Memory (OOM).
- Expected VRAM usage: ~5.5GB - 7GB.
- Expected Time: Depends heavily on your dataset size, but typically 30 mins to a few hours on an RTX 4060.

### 4. Chat with your AI!

**Option A: Terminal Interface**
```bash
python inference.py
```

**Option B: Streamlit Web UI**
```bash
streamlit run app.py
```

---

## 🧠 Concepts Explained

### Tokenization
Before text is fed to a language model, it must be converted into numbers. A tokenizer breaks words into smaller pieces ("tokens") and assigns an ID to each. For Gemma, it also handles formatting the conversation structure using a Chat Template (`<start_of_turn>`, `<end_of_turn>`).

### Fine-Tuning
Fine-tuning is taking a pre-trained general knowledge model (like Gemma 3) and training it slightly more on a very specific dataset (your WhatsApp chats). This shifts the model's behavior to sound like you.

### LoRA (Low-Rank Adaptation)
Training all 1 Billion parameters of Gemma is impossible on an 8GB GPU. LoRA freezes the original model weights and injects tiny, trainable "adapters" into the model's attention layers. Instead of training 1B parameters, you only train a few million.

### QLoRA
QLoRA takes LoRA a step further by **Quantizing** the base model down to 4-bit precision (instead of 16-bit or 32-bit). This drastically cuts down the VRAM required to load the model, making it possible to train on consumer GPUs like the RTX 4060.

---

## 📉 Handling Small Datasets

If you have less than 1,000 conversational pairs in your WhatsApp history, the model is at risk of **overfitting** (memorizing exact phrases instead of learning your style). 

**Mitigations implemented in this project:**
1. **System Prompt**: A strong system prompt in `config.py` acts as an anchor. It tells the model to "be you" so it relies on its base knowledge of conversations while sprinkling in your specific style.
2. **Small Batch Sizes**: `PER_DEVICE_TRAIN_BATCH_SIZE=1` with Gradient Accumulation.
3. **Weight Decay**: Set in `config.py` to prevent weights from growing too large.

**Further Data Augmentation Tips:**
- You can manually duplicate some pairs and paraphrase the 'User' side slightly to create more diverse prompts for the same response.
- Combine multiple chat exports into the `data/` folder before running `preprocess.py`.

---

## 🎤 Sample Interview Questions & Answers

**Q: Why use QLoRA instead of standard full fine-tuning?**
A: Full fine-tuning requires keeping optimizer states for all parameters, which requires roughly 4-6x the memory of the model size. For a 1B parameter model, that easily exceeds an 8GB VRAM GPU. QLoRA loads the base model in 4-bit precision and only trains a tiny LoRA adapter, fitting comfortably in 8GB.

**Q: How do you handle continuous messages from the same sender in WhatsApp?**
A: In `preprocess.py`, consecutive messages from the same person are grouped together separated by newlines before being split into Instruction-Response pairs. This preserves context.

**Q: What if the model hallucinates facts not in the chat?**
A: We are doing Instruction Tuning primarily for *style transfer*, not factual retrieval. The model relies on its pre-trained knowledge base to answer novel questions but uses the finetuned weights to format the response in the target persona's style.

**Q: How does gradient accumulation help on your RTX 4060?**
A: Because memory is limited, we can only fit a batch size of 1. Gradient accumulation allows us to process 8 separate batches of 1, summing up the gradients, and then taking a single optimization step. This effectively simulates a batch size of 8, smoothing out the learning curve without crashing the GPU.

---

## 📊 Actual Training Results (RTX 4060 8GB)

This model was successfully trained on **389 WhatsApp conversation pairs** extracted from 5 chat exports. Training completed in **~9 minutes 21 seconds** on an RTX 4060 8GB GPU.

### Training Metrics Summary

| Epoch | Train Loss | Token Accuracy | Eval Loss |
|-------|-----------|---------------|-----------|
| 0.23  | 6.26      | 33.4%         | —         |
| 0.46  | 3.56      | 58.3%         | —         |
| 0.91  | 2.70      | 63.7%         | —         |
| 1.14  | 2.58      | 64.7%         | 2.37      |
| 1.59  | 2.55      | 63.7%         | —         |
| 2.27  | 2.28      | 66.8%         | 2.30      |
| 2.96  | **1.98**  | **71.6%**     | —         |
| **Final** | **2.81 avg** | **68.5%** | **2.30** |

### What Each Metric Means

- **Train Loss** — How wrong the model's predictions were on training data. Dropped from **6.26 → 1.98** over 3 epochs. Lower is better.
- **Token Accuracy** — How often the model predicted the correct next token. Improved from **33% → 71.6%**.
- **Eval Loss** — Loss on unseen validation data. Went from **2.37 → 2.30** (steadily decreasing = **no overfitting**).
- **Entropy** — How confident/decisive the model's predictions are. Dropped from **3.04 → 1.85**, meaning the model became significantly more decisive and learned the conversational style.
- **Total Training Time**: 561 seconds (~9.4 minutes)
- **Peak VRAM Usage**: ~7.5GB out of 8GB

### Key Hyperparameters Used

```python
MODEL_NAME = "google/gemma-3-1b-it"
LORA_R = 16
LORA_ALPHA = 32
MAX_SEQ_LENGTH = 256
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8   # effective batch size = 8
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3
OPTIMIZER = "paged_adamw_32bit"
BF16 = True  # bfloat16 mixed precision (RTX 4060 native)
```

### Output Files

The trained LoRA adapter is saved to:
```
outputs/gemma-whatsapp-lora/
├── adapter_config.json
├── adapter_model.safetensors
└── tokenizer files...
```

To use it, run `python inference.py` (terminal) or `streamlit run app.py` (web UI).
