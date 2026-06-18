import os
import re
import json
import random
import glob
from config import RAW_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH, VAL_SPLIT_RATIO, TARGET_PERSONA, SYSTEM_PROMPT

def clean_message(text):
    """
    Removes URLs, emails, phone numbers, numbers, emojis/icons, 
    and leaked WhatsApp timestamp/sender metadata.
    """
    if not text:
        return ""
        
    # Replace unicode spaces and markers
    text = text.replace('\u200e', '').replace('\u200f', '').replace('\u202f', ' ')
    
    cleaned_lines = []
    for line in text.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
            
        # Skip system lines (including group icon updates, etc.)
        if any(x in line_strip.lower() for x in [
            "messages and calls are end-to-end encrypted",
            "you deleted this message",
            "this message was deleted",
            "pinned a message",
            "joined using this group",
            "created group",
            "changed the subject",
            "changed this group's icon",
            "changed group info",
            "disappearing messages",
            "group's icon",
            "changed the group icon",
            "icon",
            "<media omitted>"
        ]):
            continue
            
        # Remove Email Addresses
        line_strip = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", line_strip)
        
        # Remove URLs/Links (even without http/https or www, e.g. domain.com/path)
        line_strip = re.sub(r"\b(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+(?:\/\S*)?\b", "", line_strip)
        
        # Remove phone numbers (e.g. +91 98765 43210, 98765-43210, 9876543210)
        line_strip = re.sub(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b", "", line_strip)
        
        # Remove stand-alone numbers of any length
        line_strip = re.sub(r"\b\d+\b", "", line_strip)
        
        # Remove edited message marker
        line_strip = re.sub(r"(?i)<\s*this message was edited\s*>", "", line_strip)
        
        # Remove leaked timestamp patterns and remnants:
        line_strip = re.sub(r"(?i)//,\s*:?\s*(?:AM|PM)?\s*-\s*[^:]+:\s*", "", line_strip)
        line_strip = re.sub(r"(?i)//,\s*:?\s*(?:AM|PM)?\s*-\s*.*", "", line_strip)
        line_strip = re.sub(r"(?i)(?://,\s*)?\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s*-\s*[^:]+:\s*", "", line_strip)
        line_strip = re.sub(r"(?i)(?://,\s*)?\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:AM|PM)?\s*-\s*.*", "", line_strip)
        
        # Remove Emojis and Unicode Icons/Symbols
        # Matches emoji ranges, miscellaneous symbols, dingbats, emoticons, etc.
        line_strip = re.compile(
            '['
            '\U0001f300-\U0001f5ff'  # Symbols & Pictographs
            '\U0001f600-\U0001f64f'  # Emoticons
            '\U0001f680-\U0001f6ff'  # Transport & Map Symbols
            '\U0001f900-\U0001f9ff'  # Supplemental Symbols
            '\U0001fa70-\U0001faff'  # Symbols Extended
            '\u2600-\u26ff'          # Misc Symbols
            '\u2700-\u27bf'          # Dingbats
            '\u2000-\u3000'          # Special spaces and punctuation range
            ']+', 
            re.UNICODE
        ).sub('', line_strip)
        
        # Remove stand-alone punctuation fragments that often leak
        if line_strip.strip() in ["//", "//,", ",", ":", "/", "\\", ".", "?", "!"]:
            continue
            
        final_line = line_strip.strip()
        if final_line:
            cleaned_lines.append(final_line)
            
    return "\n".join(cleaned_lines).strip()

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
    Filters out non-conversational style elements (e.g. long copy-pastes, emails, phone numbers, code).
    """
    pairs = []
    
    # Regex to detect emails and phone numbers
    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
    
    for i in range(len(grouped_data) - 1):
        msg1 = grouped_data[i]
        msg2 = grouped_data[i+1]
        
        # If msg1 is NOT target persona, and msg2 IS target persona (response)
        if msg1["sender"] != target_persona and msg2["sender"] == target_persona:
            instruction = msg1["message"].strip()
            response = msg2["message"].strip()
            
            # --- Style-based Filters ---
            
            # 1. Discard if response is empty or just punctuation
            if not response or response in [".", ",", "?", "!", "-", "_", "...", ".."]:
                continue
                
            # 2. Keep responses short and punchy (WhatsApp style). Discard long paragraphs (> 200 chars).
            # This filters out non-conversational copy-pastes or long technical explanations.
            if len(response) > 200:
                continue
                
            # 3. Discard long instructions too (> 300 chars) to maintain clean short context
            if len(instruction) > 300:
                continue
                
            # 4. Filter out personal identifiers (emails/phone numbers) to prevent memorization
            if email_pattern.search(response) or email_pattern.search(instruction):
                continue
                
            # Simple check to filter out long number sequences (like phone numbers/account IDs)
            digits_in_resp = sum(c.isdigit() for c in response)
            digits_in_inst = sum(c.isdigit() for c in instruction)
            if digits_in_resp >= 10 or digits_in_inst >= 10:
                continue
            
            # 5. Filter out code blocks or markdown tables
            if "```" in response or "```" in instruction or "|" in response:
                continue
                
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
