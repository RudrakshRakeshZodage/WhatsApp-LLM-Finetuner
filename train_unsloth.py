import os
# Disable Rust hf_transfer which deadlocks/stalls on Windows, and reuse local HF cache instead
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import config

def format_chat_template(example, tokenizer):
    """
    Formats the 'messages' list in the dataset using the model's chat template.
    """
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

def main():
    print("Loading configuration...")
    
    # 1. Load Dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": config.TRAIN_DATA_PATH,
        "validation": config.VAL_DATA_PATH
    })
    
    # 2. Load Model and Tokenizer using Unsloth
    print(f"Loading model and tokenizer with Unsloth: {config.MODEL_NAME}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = config.MODEL_NAME,
        max_seq_length = config.MAX_SEQ_LENGTH,
        load_in_4bit = True,
        dtype = None, # Auto-detect float16/bfloat16
        device_map = "cuda:0"
    )
    
    # 3. Setup LoRA Adapters using Unsloth's highly optimized Peft loader
    print("Setting up Unsloth optimized LoRA configuration...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = config.LORA_R,
        target_modules = config.TARGET_MODULES,
        lora_alpha = config.LORA_ALPHA,
        lora_dropout = config.LORA_DROPOUT,
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Fast, memory-saving gradient checkpointing
        random_state = 3407,
    )
    
    # 4. Format Dataset
    print("Formatting dataset with chat template...")
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        desc="Applying chat template"
    )
    
    # 5. Training Arguments (trl SFTConfig)
    print("Configuring Training Arguments...")
    training_args = SFTConfig(
        output_dir=config.MODELS_DIR,
        per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=config.LEARNING_RATE,
        logging_steps=config.LOGGING_STEPS,
        max_length=config.MAX_SEQ_LENGTH,
        num_train_epochs=config.NUM_TRAIN_EPOCHS,
        save_steps=config.SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        optim="adamw_8bit", # Use 8-bit Adam for speed and memory savings
        fp16=config.FP16,
        bf16=config.BF16,
        max_grad_norm=config.MAX_GRAD_NORM,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        weight_decay=config.WEIGHT_DECAY,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,
        dataset_text_field="text",
        report_to="none",
    )
    
    # 6. Trainer
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        args=training_args,
    )
    
    # 7. Train
    print("Starting Training with Unsloth (this will be 2-3x faster)...")
    trainer.train()
    
    # 8. Save Model
    print(f"Saving final adapter to {config.FINAL_ADAPTER_DIR}...")
    model.save_pretrained(config.FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(config.FINAL_ADAPTER_DIR)
    print("Training complete and adapter saved successfully!")

if __name__ == "__main__":
    main()
