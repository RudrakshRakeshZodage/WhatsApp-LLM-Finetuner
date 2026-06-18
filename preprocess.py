import os
import re
import json
import random
import glob
from config import RAW_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, VAL_SPLIT_RATIO, TARGET_PERSONA, SYSTEM_PROMPT

def clean_message(text):
    """
    Removes URLs, numbers, and leaked WhatsApp timestamp patterns from messages.
    """
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Remove WhatsApp timestamp continuation lines like '5/20/26, 7:43 PM - Rudra: ...'
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[\u202f]?(?:AM|PM)\s*-\s*.*", "", text)
    # Remove standalone numbers
    text = re.sub(r"\b\d+\b", "", text)
    return text.strip()

def parse_whatsapp_chat(filepath):
    """
    Parses a WhatsApp chat export file.
    Format usually: 'M/D/YY, h:mm AM/PM - SENDER: Message'
    Returns a list of dictionaries with 'sender' and 'message'.
    """
    parsed_data = []
    
    # Regex to match the standard WhatsApp date/time and sender prefix
    # Example: '5/20/26, 5:31 PM - HEMANG: Waiting for this message'
    # \u202f is the narrow no-break space sometimes present before AM/PM
    pattern = re.compile(r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s+(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm|\u202f[aApP][mM]))\s+-\s+(.*?):\s+(.*)$")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        current_sender = None
        current_message = []
        
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Skip system messages like "Messages and calls are end-to-end encrypted..."
            if "Messages and calls are end-to-end encrypted" in line:
                continue
                
            match = pattern.match(line)
            if match:
                # We found a new message line
                date, time, sender, message = match.groups()
                
                # Filter out system or omitted media messages
                if message == "<Media omitted>" or "This message was deleted" in message:
                    continue
                    
                message = clean_message(message)
                if not message:
                    continue
                
                # Save previous message block if exists
                if current_sender is not None and current_message:
                    parsed_data.append({
                        "sender": current_sender,
                        "message": "\n".join(current_message)
                    })
                
                current_sender = sender
                current_message = [message]
            else:
                # This line doesn't start with the date prefix, so it's a continuation of the previous message
                if current_sender is not None:
                    cleaned_line = clean_message(line)
                    if cleaned_line:
                        current_message.append(cleaned_line)
                    
        # Append the very last message
        if current_sender is not None and current_message:
            parsed_data.append({
                "sender": current_sender,
                "message": "\n".join(current_message)
            })
            
    return parsed_data

def group_consecutive_messages(parsed_data):
    """
    Groups consecutive messages from the same sender into a single block.
    """
    if not parsed_data:
        return []
        
    grouped_data = []
    current_sender = parsed_data[0]["sender"]
    current_message = [parsed_data[0]["message"]]
    
    for item in parsed_data[1:]:
        if item["sender"] == current_sender:
            current_message.append(item["message"])
        else:
            grouped_data.append({
                "sender": current_sender,
                "message": "\n".join(current_message)
            })
            current_sender = item["sender"]
            current_message = [item["message"]]
            
    # Append the last block
    if current_message:
        grouped_data.append({
            "sender": current_sender,
            "message": "\n".join(current_message)
        })
        
    return grouped_data

def extract_conversation_pairs(grouped_data, target_persona):
    """
    Extracts User -> Target Persona conversational pairs.
    Returns a list of (instruction, response) tuples.
    """
    pairs = []
    
    for i in range(len(grouped_data) - 1):
        msg1 = grouped_data[i]
        msg2 = grouped_data[i+1]
        
        # If the first message is NOT the target persona, and the second message IS the target persona
        if msg1["sender"] != target_persona and msg2["sender"] == target_persona:
            instruction = msg1["message"]
            response = msg2["message"]
            pairs.append((instruction, response))
            
    return pairs

def create_jsonl_dataset(pairs, output_path):
    """
    Saves the pairs into a JSONL format suitable for Hugging Face datasets.
    We use the ChatML / Gemma prompt format essentially by structuring it with roles.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for instruction, response in pairs:
            # We create a conversation structure
            # For Gemma 3, typical chat template is applied by the tokenizer later.
            # We will save it as a 'messages' array.
            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": instruction},
                    {"role": "assistant", "content": response}
                ]
            }
            f.write(json.dumps(record) + "\n")

def main():
    print("Starting Data Preprocessing...")
    
    all_pairs = []
    
    # Find all .txt files in the data directory
    txt_files = glob.glob(os.path.join(RAW_DATA_PATH, "*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {RAW_DATA_PATH}. Please add your exported WhatsApp chats.")
        return
        
    for file_path in txt_files:
        print(f"Processing {os.path.basename(file_path)}...")
        
        # 1. Parse raw lines
        parsed_data = parse_whatsapp_chat(file_path)
        
        # 2. Group consecutive messages
        grouped_data = group_consecutive_messages(parsed_data)
        
        # 3. Extract pairs where the Target Persona is responding
        pairs = extract_conversation_pairs(grouped_data, TARGET_PERSONA)
        
        all_pairs.extend(pairs)
        
    print(f"Total conversation pairs extracted: {len(all_pairs)}")
    
    if len(all_pairs) < 1000:
        print("---")
        print("NOTE: You have a small dataset (< 1000 pairs).")
        print("Small datasets are prone to overfitting.")
        print("To prevent overfitting, we have set a strong SYSTEM_PROMPT.")
        print("Data augmentation suggestions: Paraphrase user messages, manually add variations of greetings, or use a larger learning rate and fewer epochs.")
        print("---")
        
    # Shuffle pairs for random train/val split
    random.shuffle(all_pairs)
    
    split_idx = int(len(all_pairs) * (1 - VAL_SPLIT_RATIO))
    train_pairs = all_pairs[:split_idx]
    val_pairs = all_pairs[split_idx:]
    
    print(f"Train split: {len(train_pairs)} pairs")
    print(f"Validation split: {len(val_pairs)} pairs")
    
    # Create JSONL files
    create_jsonl_dataset(train_pairs, TRAIN_DATA_PATH)
    create_jsonl_dataset(val_pairs, VAL_DATA_PATH)
    
    print(f"Saved dataset to {TRAIN_DATA_PATH} and {VAL_DATA_PATH}")

if __name__ == "__main__":
    main()
