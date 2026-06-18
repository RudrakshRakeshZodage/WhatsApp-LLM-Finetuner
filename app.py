import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import config

# Streamlit page config
st.set_page_config(
    page_title=f"Chat with {config.TARGET_PERSONA}",
    page_icon="💬",
    layout="centered"
)

# Load model and tokenizer (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model_and_tokenizer():
    st.info(f"Loading Base Model ({config.MODEL_NAME}) and Adapter in float16...")
    
    # Load base model in float16 directly - fits in ~2GB VRAM and loads in seconds
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    try:
        model = PeftModel.from_pretrained(base_model, config.FINAL_ADAPTER_DIR)
    except Exception as e:
        st.warning(f"Could not load LoRA adapter (Did you run train.py?). Using base model instead.")
        model = base_model
        
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Title and Description
st.title(f"WhatsApp AI: {config.TARGET_PERSONA}")
st.write(f"This AI has been fine-tuned using QLoRA to mimic {config.TARGET_PERSONA}'s WhatsApp conversational style.")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT}
    ]

# Display chat messages from history (excluding system prompt)
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Type your message here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Format for model
    formatted_prompt = tokenizer.apply_chat_template(
        st.session_state.messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with st.spinner(f"{config.TARGET_PERSONA} is typing..."):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.strip())
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.strip()})
