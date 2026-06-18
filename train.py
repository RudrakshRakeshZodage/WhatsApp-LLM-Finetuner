import os
import torch

# Set this BEFORE importing anything else - reduces VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import config

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def format_chat_template(example, tokenizer):
    """
    Formats the 'messages' list in the dataset using the model's chat template.
    """
    # apply_chat_template returns a string if tokenize=False
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

def main():
    print("Loading configuration...")
    
    # 1. Load Dataset
    print("Loading dataset...")
    # Dataset structure should be 'messages' list of dicts.
    dataset = load_dataset("json", data_files={
        "train": config.TRAIN_DATA_PATH,
        "validation": config.VAL_DATA_PATH
    })
    
    # 2. Setup Quantization (BitsAndBytes 4-bit)
    # Optimized for RTX 4060 8GB
    print("Setting up QLoRA 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if config.BF16 else torch.float16,
    )

    # 3. Load Tokenizer
    print(f"Loading Tokenizer {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    
    # Pad token setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix for FP16 training with padding

    # 4. Format Dataset
    print("Formatting dataset with chat template...")
    # Map the formatting function to convert 'messages' into a single 'text' string per sample
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        desc="Applying chat template"
    )

    # 5. Load Model
    print(f"Loading Model {config.MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Disable caching for training (saves memory)
    model.config.use_cache = False
    
    # Prepare model for k-bit training - enables gradient checkpointing to save activation memory
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 6. Setup LoRA
    print("Setting up LoRA configuration...")
    peft_config = LoraConfig(
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        r=config.LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.TARGET_MODULES,
    )
    

    # 7. Training Arguments
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
        optim=config.OPTIMIZER,
        fp16=config.FP16,
        bf16=config.BF16,
        max_grad_norm=config.MAX_GRAD_NORM,
        warmup_ratio=config.WARMUP_RATIO,
        lr_scheduler_type=config.LR_SCHEDULER_TYPE,
        weight_decay=config.WEIGHT_DECAY,
        per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,  # Critical: keep at 1 to avoid OOM
        gradient_checkpointing=True,   # Saves activation memory at cost of ~20% speed
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        report_to="none", # Disables wandb/tensorboard logging for simplicity
    )

    # 8. Trainer
    print("Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    # 9. Train
    print("Starting Training...")
    # Log memory stats
    print(f"Memory allocated before training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    trainer.train()

    print("Training Complete!")
    print(f"Memory allocated after training: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    # 10. Save Model
    print(f"Saving final adapter to {config.FINAL_ADAPTER_DIR}...")
    trainer.model.save_pretrained(config.FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(config.FINAL_ADAPTER_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
