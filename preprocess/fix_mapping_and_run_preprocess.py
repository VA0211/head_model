import os
import subprocess

# --- CONFIGURATION ---
# NOTE: Use forward slashes '/' which work on both Python and Bash to avoid confusion.
project_root = "D:/VA/coding/project/bioinformatics"
preprocess_dir = f"{project_root}/code/head_model/preprocess"
mapping_file_path = f"{preprocess_dir}/mapping_file.txt"

# Arguments for the bash script
tar_file = f"{project_root}/data/GSE211692_RAW.tar"
output_dir = f"{project_root}/code/head_model/data_clean_all"  # New clean folder
temp_dir = f"{project_root}/code/head_model/data_clean_tmp"

# --- STEP 1: FORCE CLEAN THE MAPPING FILE ---
print(f"1. Reading {mapping_file_path} in binary mode...")

with open(mapping_file_path, 'rb') as f:
    content = f.read()

# Count how many bad characters exist
bad_chars = content.count(b'\r')
print(f"   Found {bad_chars} Windows carriage return characters.")

if bad_chars > 0:
    print("   Stripping characters...")
    # Replace \r\n (Windows) and \r (Mac) with \n (Linux)
    new_content = content.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
    
    with open(mapping_file_path, 'wb') as f:
        f.write(new_content)
    print("   [SUCCESS] File cleaned and saved.")
else:
    print("   [OK] File was already clean.")

# --- STEP 2: RUN THE PREPROCESSING IMMEDIATELY ---
print("-" * 50)
print("2. Launching run_preprocess.sh now...")
print("-" * 50)

# Use the /d/ style path for Bash arguments to ensure compatibility
bash_mapping_path = "/d/VA/coding/project/bioinformatics/code/head_model/preprocess/mapping_file.txt"
bash_tar_path = "/d/VA/coding/project/bioinformatics/data/GSE211692_RAW.tar"
bash_out_dir = "/d/VA/coding/project/bioinformatics/code/head_model/data_clean_all"
bash_temp_dir = "/d/VA/coding/project/bioinformatics/code/head_model/data_clean_tmp"

cmd = [
    "bash", "run_preprocess.sh",
    "-1",                # n_features
    bash_tar_path,      # Input tar
    bash_out_dir,       # Output dir
    bash_temp_dir,      # Working dir
    bash_mapping_path   # Mapping file
]

# Run the command
result = subprocess.run(cmd, cwd=preprocess_dir)

if result.returncode == 0:
    print("\n[SUCCESS] Preprocessing complete!")
    print(f"Results should be in: {output_dir}")
else:
    print("\n[FAIL] The bash script encountered an error.")