import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import config

# Memory optimization config not needed for float16 inference

def clean_response(text):
    """
    Post-process the model's response to strip any hallucinated WhatsApp
    timestamp patterns that leaked from training data.
    e.g. '//, PM - Rudra:' or '5/20/26, 5:31 PM - Name:'
    """
    # Remove WhatsApp timestamp lines: 'M/D/YY, h:mm PM - Name: ...'
    text = re.sub(r"\d{0,2}/?/?\s*,?\s*:?\s*[⸱\u202f]*\s*(AM|PM)\s*-\s*\S+.*", "", text)
    # Remove lines that are just punctuation or slashes
    lines = [l for l in text.splitlines() if l.strip() not in ["", "//", "//,", ",", ":", "/"]]
    return "\n".join(lines).strip()

def main():
    # --- GPU Check ---
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[OK] GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
    else:
        print("[WARNING] No GPU detected! Running on CPU - this will be VERY slow.")
        print("   Make sure you installed the CUDA version of PyTorch.")

    print(f"[1/5] Loading Base Model ({config.MODEL_NAME}) in float16...")
    print("      (No 4-bit needed for inference — 1B model fits in 2GB VRAM)")
    
    import time
    t0 = time.time()

    # For INFERENCE: load in float16 directly — fast and no quantization overhead.
    # 4-bit is only needed for TRAINING to save memory.
    # Gemma 1B in float16 = ~2GB VRAM, easily fits in 8GB.
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda:0",       # Explicitly use the RTX 4060
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"[2/5] Base model loaded in {time.time()-t0:.1f}s! "
          f"VRAM used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # Load Tokenizer
    print("[3/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[3/5] Tokenizer loaded!")
        
    print(f"[4/5] Loading LoRA Adapter from {config.FINAL_ADAPTER_DIR}...")
    try:
        model = PeftModel.from_pretrained(base_model, config.FINAL_ADAPTER_DIR)
        print("[4/5] Adapter loaded successfully!")
    except Exception as e:
        print(f"[4/5] Could not load adapter: {e}")
        print("Falling back to base model for inference.")
        model = base_model

    print("[5/5] Ready!")


    print("\n" + "="*50)
    print("Welcome to the WhatsApp AI Chatbot!")
    print(f"You are chatting with an AI trained to mimic {config.TARGET_PERSONA}.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("="*50 + "\n")

    # Conversation history
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT}
    ]

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
                
            # Append user message to history
            messages.append({"role": "user", "content": user_input})
            
            # Format using chat template
            prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize and send to GPU
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            # Generate response - keep short like real WhatsApp messages
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,      # WhatsApp messages are SHORT
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.4, # Strongly penalize repetition
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            raw_response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            
            # Clean hallucinated WhatsApp timestamp patterns from the response
            response = clean_response(raw_response)
            
            if not response:
                response = "Ha"  # Fallback to a natural short reply
            
            print(f"{config.TARGET_PERSONA} (AI): {response}")
            
            # Append AI response to history to maintain context
            messages.append({"role": "assistant", "content": response.strip()})
            
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
