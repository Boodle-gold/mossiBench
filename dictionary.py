import json
from rapidfuzz import process, fuzz

class WordDictionary():
    def __init__(self, src_lang, tgt_lang, dict_path):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.dict_path = dict_path
        self.load_dict()
    
    def load_dict(self):
        self.word_dict = {}
        print(f"Loading dictionary from {self.dict_path}...")
        
        try:
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Your structure: [{"word": "...", "definition": "..."}]
            for item in data:
                headword = item.get('word', '').strip()
                # definition can be a string or list, normalize to list
                definition = item.get('definition', '')
                if isinstance(definition, str):
                    definition = [definition]
                
                if headword:
                    self.word_dict[headword] = definition
                    
            self.choices = list(self.word_dict.keys())
            print(f"Dictionary loaded with {len(self.choices)} entries.")
            
        except Exception as e:
            print(f"Error loading dictionary: {e}")
            self.word_dict = {}
            self.choices = []

    def get_meanings_by_exact_match(self, word, max_num_meanings=None):
        if word in self.word_dict:
            meanings = self.word_dict[word]
            if max_num_meanings:
                return meanings[:max_num_meanings]
            return meanings
        return None
        
    def get_meanings_by_fuzzy_match(self, word, top_k=1, max_num_meanings_per_word=1):
        # Only run fuzzy match if we have choices and word is reasonably long
        if not self.choices or len(word) < 2:
            return []

        output = []
        # rapidfuzz returns (match, score, index)
        results = process.extract(word, self.choices, scorer=fuzz.WRatio, limit=top_k)
        
        for match in results:
            match_word = match[0]
            score = match[1]
            
            if score > 85: # High threshold to prevent bad noise
                meanings = self.word_dict[match_word]
                if max_num_meanings_per_word:
                    meanings = meanings[:max_num_meanings_per_word]
                    
                output.append({
                    "word": match_word,
                    "meanings": meanings,
                    "score": score
                })
        return output