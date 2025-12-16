import json

english_file = 'NLLB.en-mos.en'
moore_file   = 'NLLB.en-mos.mos'
output_file  = 'nllb_en_mos.json'
json_data = []

print("File read")

try:
    # Open both files simultaneously and read them line-by-line
    with open(english_file, 'r', encoding='utf-8') as f_en, \
         open(moore_file, 'r', encoding='utf-8') as f_mos:
        
        # 'zip' pairs the lines together automatically
        for i, (en_line, mos_line) in enumerate(zip(f_en, f_mos)):
            
            # Create a dictionary for this pair
            entry = {
                "id": i,  # Optional: adds a number to each pair
                "en": en_line.strip(),
                "mos": mos_line.strip()
            }
            json_data.append(entry)

    # 3. SAVE TO JSON
    print(f"Combined {len(json_data)} pairs. Saving to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # ensure_ascii=False ensures Mooré characters (like 'ɛ' or 'ɩ') save correctly
        json.dump(json_data, f_out, ensure_ascii=False, indent=2)

    print("Success")

except FileNotFoundError:
    print("Error: Could not find the text files")
except Exception as e:
    print(f"An error occurred: {e}")