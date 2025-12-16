import json
import random

# CONFIGURATION
INPUT_FILE = 'nllb_en_mos.json'
TEST_SIZE = 200

def create_split():
    print(f"Loading {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    # Shuffle to ensure random distribution
    random.seed(42)
    random.shuffle(data)

    # Split into Test and Corpus (Train)
    test_data = data[:TEST_SIZE]
    corpus_data = data[TEST_SIZE:]

    # Assign difficulty based on Mossi sentence length
    test_data.sort(key=lambda x: len(x['mos']))
    third = len(test_data) // 3
    
    for i, item in enumerate(test_data):
        if i < third:
            item['source'] = 'easy'
        elif i < third * 2:
            item['source'] = 'medium'
        else:
            item['source'] = 'hard'

    # Save Files
    print(f"Saving 'mossi_corpus.json' ({len(corpus_data)} items)...")
    with open('mossi_corpus.json', 'w', encoding='utf-8') as f:
        json.dump(corpus_data, f, ensure_ascii=False, indent=2)

    print(f"Saving 'mossi_test.json' ({len(test_data)} items)...")
    with open('mossi_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("Success")

if __name__ == "__main__":
    create_split()