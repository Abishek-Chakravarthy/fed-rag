import os
import glob
import json
import csv

base_dir = "/Users/abishekchakravarthy/FInal_year_project/fed-rag/zz_coderuns/das_fedrag/exp_05_results"
log_file = os.path.join(base_dir, "das_fedrag_exp_05_log.md")

with open(log_file, "a") as f:
    f.write("\n\n---\n\n## 6. Detailed Round-by-Round Metrics (from CSVs)\n\n")
    
    csv_files = glob.glob(os.path.join(base_dir, "*/*.csv"))
    csv_files.sort()
    
    for csv_path in csv_files:
        exp_name = os.path.basename(os.path.dirname(csv_path))
        f.write(f"### {exp_name} - Round Metrics\n\n")
        
        with open(csv_path, "r") as cf:
            reader = csv.reader(cf)
            rows = list(reader)
            if not rows:
                continue
                
            header = rows[0]
            f.write("| " + " | ".join(header) + " |\n")
            f.write("|" + "|".join(["---" for _ in header]) + "|\n")
            
            for row in rows[1:]:
                f.write("| " + " | ".join(row) + " |\n")
        f.write("\n\n")
        
    f.write("\n\n---\n\n## 7. Acceptance Reports\n\n")
    
    acc_files = glob.glob(os.path.join(base_dir, "*/acceptance_*.json"))
    acc_files.sort()
    
    for acc_path in acc_files:
        exp_name = os.path.basename(os.path.dirname(acc_path))
        f.write(f"### {exp_name} - Acceptance Report\n\n")
        with open(acc_path, "r") as acc_f:
            data = json.load(acc_f)
            f.write("```json\n")
            f.write(json.dumps(data, indent=2))
            f.write("\n```\n\n")

print("Done appending to markdown.")
