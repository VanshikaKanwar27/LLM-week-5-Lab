import json
import os

files_to_fix = [
    r"c:\Users\vansh\Downloads\item_subset.json",
    r"c:\Users\vansh\OneDrive\Desktop\New folder\AgentReview\review_subset.json",
    r"c:\Users\vansh\OneDrive\Desktop\New folder\AgentReview\test_review_subset.json",
    r"c:\Users\vansh\OneDrive\Desktop\New folder\AgentReview\user_subset.json",
    r"c:\Users\vansh\OneDrive\Desktop\New folder\AgentReview\data\item_subset.json",
    r"c:\Users\vansh\OneDrive\Desktop\New folder\AgentReview\data\review_subset.json",
    r"c:\Users\vansh\OneDrive\Desktop\New folder\AgentReview\data\user_subset.json"
]

for file_path in files_to_fix:
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    if not lines:
        continue
        
    s = lines[0].strip()
    if s.startswith("[") and lines[-1].strip().endswith("]"):
        continue # Already formatted
        
    print(f"Fixing {file_path}")
    objects = []
    for line in lines:
        if line.strip():
            try:
                objects.append(json.loads(line))
            except:
                pass
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(objects, f, indent=2)

print("Done fixing json files.")
