import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Create necessary directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Target persona configuration (Assume "Rudra" is the assistant we want to mimic)
TARGET_PERSONA = "Rudra"

# Preprocessing Config
RAW_DATA_PATH = DATA_DIR  # Will scan all .txt files here
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.jsonl")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val.jsonl")
VAL_SPLIT_RATIO = 0.1  # 10% for validation
MAX_SEQ_LENGTH = 256   # Reduced from 512 to prevent OOM on 8GB VRAM during evaluation

# Model Config
MODEL_NAME = "google/gemma-3-1b-it"  # Hugging Face model hub name
FINAL_ADAPTER_DIR = os.path.join(OUTPUTS_DIR, "gemma-whatsapp-lora")

# LoRA & QLoRA Config
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Gemma uses different projection layers, these are standard for llama/gemma architectures
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training Hyperparameters (Optimized for RTX 4060 8GB VRAM)
PER_DEVICE_TRAIN_BATCH_SIZE = 1  # Very small batch size to fit in 8GB
PER_DEVICE_EVAL_BATCH_SIZE = 1   # Keep eval batch size at 1 to avoid OOM
GRADIENT_ACCUMULATION_STEPS = 8  # Increased from 4 to compensate for smaller batch
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 3             # Adjust based on dataset size (if dataset is very small, might need 5-10)
WARMUP_RATIO = 0.03
LOGGING_STEPS = 10
SAVE_STEPS = 50
EVAL_STEPS = 50
WEIGHT_DECAY = 0.001
MAX_GRAD_NORM = 0.3
OPTIMIZER = "paged_adamw_32bit"
LR_SCHEDULER_TYPE = "cosine"
BF16 = True   # RTX 4060 supports bfloat16. BF16 is more numerically stable and does NOT need a gradient scaler.
FP16 = False  # Do NOT use FP16 together with BF16 — that causes a gradient scaler crash.

# System Prompt for the dataset
# A good system prompt helps the model understand its role, especially crucial for small datasets.
SYSTEM_PROMPT = f"You are {TARGET_PERSONA}. Respond to the messages in your typical conversational WhatsApp style, keeping it casual, short, and natural."
