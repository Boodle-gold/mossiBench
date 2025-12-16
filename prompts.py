from corpus import lang2tokenizer

model_to_chat_template = {
    'qwen': "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
}

def get_word_explanation_prompt(text, src_lang, dictionary):
    """
    Generates vocabulary hints using the dictionary.
    Only works if the dictionary matches the source language (e.g., Mossi -> English).
    """
    # If no dictionary or the source language isn't Mossi (since your dict is Mos->En), skip hints
    if dictionary is None or src_lang != 'mos':
        return ""

    tokenizer = lang2tokenizer.get(src_lang, None)
    if not tokenizer: return ""

    tokens = tokenizer.tokenize(text, remove_punc=True)
    prompt = "## Vocabulary Hints:\n"
    found = False

    for word in tokens:
        # 1. Exact Match
        exact = dictionary.get_meanings_by_exact_match(word, max_num_meanings=1)
        if exact:
            # Clean up citation text if present in the definition
            defn = exact[0].split('[cite')[0].strip()
            prompt += f"- '{word}' means: {defn}\n"
            found = True
            continue # Skip fuzzy if exact match is found

        # 2. Fuzzy Match (Optional fallback)
        fuzzy = dictionary.get_meanings_by_fuzzy_match(word, top_k=1)
        if fuzzy:
            match_word = fuzzy[0]['word']
            defn = fuzzy[0]['meanings'][0].split('[cite')[0].strip()
            prompt += f"- '{word}' (similar to '{match_word}') means: {defn}\n"
            found = True

    return (prompt + "\n") if found else ""


def construct_prompt_mos2en(src_sent, dictionary, parallel_corpus, args):
    # 1. Retrieve similar sentences from the corpus
    retrieved = []
    if args.num_parallel_sent > 0:
        retrieved = parallel_corpus.search_by_bm25(src_sent, query_lang='mos', top_k=args.num_parallel_sent)

    prompt = ""
    
    # 2. Add Few-Shot Context (Examples)
    if retrieved:
        prompt += "Translate the following Mossi sentences into English.\n\n"
        for item in retrieved:
            pair = item['pair']
            prompt += f"Mossi: {pair['mos']}\n"
            prompt += f"English: {pair['en']}\n\n"

    # 3. Add the Target Sentence to translate
    prompt += f"Mossi: {src_sent}\n"
    
    # 4. Add Dictionary Hints (if any found)
    prompt += get_word_explanation_prompt(src_sent, 'mos', dictionary)
    
    prompt += "English:"
    
    return prompt

def construct_prompt_en2mos(src_sent, dictionary, parallel_corpus, args):
    # 1. Retrieve similar sentences
    retrieved = []
    if args.num_parallel_sent > 0:
        retrieved = parallel_corpus.search_by_bm25(src_sent, query_lang='en', top_k=args.num_parallel_sent)

    prompt = ""
    
    # 2. Add Few-Shot Context
    if retrieved:
        prompt += "Translate the following English sentences into Mossi.\n\n"
        for item in retrieved:
            pair = item['pair']
            prompt += f"English: {pair['en']}\n"
            prompt += f"Mossi: {pair['mos']}\n\n"

    # 3. Add Target
    prompt += f"English: {src_sent}\n"
    prompt += "Mossi:"
    return prompt