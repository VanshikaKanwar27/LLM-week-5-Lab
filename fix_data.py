import os
import json

def antigravity_fix():
    data_dir = 'data'
    files = ['user_subset.json', 'item_subset.json', 'review_subset.json', 'test_review_subset.json']
    
    print("🚀 Antigravity Agent: Commencing data alignment...")

    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"⚠️ Skipping {file_name}: File not found in /data folder.")
            continue

        try:
            # Step 1: Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Step 2: Check if it's already a list (proper JSON)
            first_char = lines[0].strip()[0] if lines else ""
            if first_char == '[':
                print(f"✅ {file_name} is already in correct JSON format.")
                continue

            # Step 3: Convert JSONL to a proper JSON List
            data_list = [json.loads(line.strip()) for line in lines if line.strip()]
            
            # Step 4: Write it back out as a single JSON Array
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_list, f, indent=4)
            
            print(f"✨ Successfully converted {file_name} to JSON Array.")

        except Exception as e:
            print(f"❌ Error fixing {file_name}: {e}")

    print("🏁 All data is now gravity-compliant and ready for CrewAI.")

if __name__ == "__main__":
    antigravity_fix()