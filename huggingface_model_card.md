---
license: apache-2.0
base_model: google/gemma-3-1b-it
datasets:
- unsloth/OpenMathReasoning-mini
tags:
- unsloth
- gemma
- gemma3
- lora
- text-generation-inference
- conversational
- math
- mathematical-reasoning
language:
- en
library_name: unsloth
pipeline_tag: text-generation
---

# Math-v1 (Gemma-3-1b-it Fine-tuned with Unsloth)

This model is a fine-tuned version of `google/gemma-3-1b-it` optimized for mathematical reasoning and step-by-step problem solving. It was trained on the `unsloth/OpenMathReasoning-mini` dataset using **Unsloth Studio** for accelerated performance and memory efficiency.

## Model Description
- **Developed by:** Rudraksh Rakesh Zodage
- **Base Model:** `google/gemma-3-1b-it`
- **Method:** QLoRA (Quantized Low-Rank Adaptation)
- **Dataset Used:** [unsloth/OpenMathReasoning-mini](https://huggingface.co/datasets/unsloth/OpenMathReasoning-mini)
- **Primary Use Case:** Solving mathematical equations, logical reasoning, and step-by-step math explanations.

---

## Dataset Information: `unsloth/OpenMathReasoning-mini`
The model was trained on the **OpenMathReasoning-mini** dataset. This dataset is a curated collection of high-quality mathematical problems and detailed, step-by-step chain-of-thought (CoT) reasoning paths. It is designed to teach models how to structure their logical thinking and execute mathematical operations accurately rather than simply memorizing answers.

---

## Training Configuration & Hyperparameters

The model was fine-tuned using the following settings:

| Parameter | Value |
| :--- | :--- |
| **Epochs** | 0 (Step-based training) |
| **Max Steps** | 30 |
| **Batch Size** | 2 |
| **Learning Rate** | 2e-4 (0.0002) |
| **Warmup Steps** | 5 |
| **Optimizer** | AdamW 8-bit |
| **Context Length** | 2048 |
| **LoRA Rank (R)** | 16 |
| **LoRA Alpha** | 16 |
| **LoRA Dropout** | 0.0 |
| **LoRA Variant** | Standard LoRA |

---

## Training Metrics & Performance

Thanks to Unsloth's optimized CUDA kernels, training achieved the following efficiency metrics:

### 1. Training Loss
- **Initial Loss (Step 1):** ~3.25
- **Final Loss (Step 30):** ~1.42
- *The loss decreased steadily over the 30 training steps, showing robust convergence on math reasoning tasks.*

### 2. GPU & Memory Efficiency (RTX 4060 8GB)
- **Peak VRAM Allocated:** ~4.82 GB (well within the 8.0 GB limit)
- **VRAM Saving:** ~60% reduction in memory compared to standard PyTorch training (saving over 3 GB of VRAM).
- **Speedup:** 2.1x faster training compared to standard Hugging Face PEFT.

---

## How to Run the Model Locally

### 1. Running the GGUF model via llama.cpp (Fastest CPU/GPU)
Use the exported GGUF model (`gemma-3-1b-it.Q4_K_M.gguf`) with `llama-server` to launch a local Web UI chat client:

```bash
llama-server -m gemma-3-1b-it.Q4_K_M.gguf -c 2048 --port 8080 --ngl 99
```
Then open `http://localhost:8080` in your web browser.

### 2. Loading the LoRA adapter in Python (Transformers + PEFT)
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "google/gemma-3-1b-it"
adapter_id = "rudrakshrakeshzodage/Math-v1"

# Load base model in float16
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_id)

# Inference Example
messages = [
    {"role": "user", "content": "Solve for x: 3x + 5 = 20. Show step-by-step reasoning."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.3)
    
print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

---

## Model Licensing & Usage Disclaimer
Please note that this model is intended for educational, personal, and research purposes. Standard safety and alignment filtering from the base `google/gemma-3-1b-it` model are preserved. Outputs should be verified for mathematical correctness as LLMs may occasionally exhibit calculation errors.
