# 1. Pick one random file from your downloaded tar
tar tf /d/VA/coding/project/bioinformatics/data/GSE211692_RAW.tar | head -n 1 > test_file.txt
raw_name=$(cat test_file.txt)
echo "---------------------------------------------------"
echo "1. Raw file in tar:      $raw_name"

# 2. Simulate the script's renaming logic
# Remove .gz
no_gz=${raw_name%.gz}
# Cut prefix (Simulate the 'cut -d' command)
clean_name=$(echo "$no_gz" | cut -d '_' -f 2-)
echo "2. Script converts to:   $clean_name"

# 3. Check for hidden Windows characters in mapping file
echo "3. Checking for Windows line endings (\r)..."
CR_COUNT=$(grep -U $'\015' /d/VA/coding/project/bioinformatics/code/head_model/preprocess/mapping_file.txt | wc -l)
if [ "$CR_COUNT" -gt 0 ]; then
    echo "   [FAIL] FOUND $CR_COUNT LINES WITH WINDOWS RETURNS (\r)!"
    echo "   This is likely the cause of the error."
else
    echo "   [OK] Line endings look correct (Unix style)."
fi

# 4. Check if the file exists in mapping file
echo "4. Searching for match in mapping file..."
# We use grep to see if the clean name exists in the text file
if grep -q "$clean_name" /d/VA/coding/project/bioinformatics/code/head_model/preprocess/mapping_file.txt; then
    echo "   [SUCCESS] Found '$clean_name' in mapping file!"
else
    echo "   [FAIL] Could not find '$clean_name' in mapping_file.txt"
    echo "   First 3 lines of mapping file look like:"
    head -n 3 /d/VA/coding/project/bioinformatics/code/head_model/preprocess/mapping_file.txt
fi
echo "---------------------------------------------------"