import json
import os
import pickle
from rank_bm25 import BM25Okapi
from tokenizer import *

lang2tokenizer = {
    'en': EngTokenizer(),
    'eng': EngTokenizer(),
    'mos': MosTokenizer(),
}

class ParallelCorpus():
    def __init__(self, src_lang, tgt_lang, corpus_path, construct_bm25=True):
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.corpus_path = corpus_path
        self.load_corpus()
        if construct_bm25:
            self.construct_bm25()
            # Reversed index is optional unless you need specific word search

    def __len__(self):
        return len(self.corpus) 

    def __getitem__(self, idx):
        return self.corpus[idx]
    
    def load_corpus(self):
        print(f"Loading corpus from {self.corpus_path}...")
        self.corpus = []
        try:
            with open(self.corpus_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Normalize data to list of dicts with known keys
            for item in data:
                # Support your structure keys: 'mos', 'en'
                self.corpus.append({
                    self.src_lang: item.get(self.src_lang, ""),
                    self.tgt_lang: item.get(self.tgt_lang, ""),
                    'source': item.get('source', 'n/a')
                })
        except Exception as e:
            print(f"Error loading corpus: {e}")
            raise

    def construct_bm25(self):
        # 1. Define Cache Filename
        cache_path = self.corpus_path + ".bm25.pkl"
        
        # 2. Try Loading Cache
        if os.path.exists(cache_path):
            print(f"Loading cached BM25 index from {cache_path}...")
            try:
                with open(cache_path, 'rb') as f:
                    self.bm25 = pickle.load(f)
                return  # Exit function early if successful
            except Exception as e:
                print(f"Cache load failed ({e}). Rebuilding index...")

        # 3. Build Index (If no cache or load failed)
        self.bm25 = {}
        for lang in [self.src_lang, self.tgt_lang]:
            tokenizer = lang2tokenizer.get(lang, Tokenizer())
            print(f"Building BM25 index for {lang}...")
            
            # Use cut_for_search only if Chinese
            if lang == 'zh':
                tokenized_corpus = [tokenizer.tokenize(doc[lang], remove_punc=True, cut_for_search=True) for doc in self.corpus]
            else:
                tokenized_corpus = [tokenizer.tokenize(doc[lang], remove_punc=True) for doc in self.corpus]
                
            self.bm25[lang] = BM25Okapi(tokenized_corpus)

        # 4. Save Cache
        print(f"Saving BM25 index to {cache_path}...")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.bm25, f)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")
    
    def search_by_bm25(self, text, query_lang='src', top_k=5):
        target_lang = self.src_lang if query_lang == 'src' else self.tgt_lang
        tokenizer = lang2tokenizer.get(target_lang, Tokenizer())

        if target_lang == 'zh':
            query = tokenizer.tokenize(text, remove_punc=True, cut_for_search=True)
        else:
            query = tokenizer.tokenize(text, remove_punc=True)

        doc_scores = self.bm25[target_lang].get_scores(query)
        top_k_idx = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i])[-top_k:]
        
        return [{"pair": self.corpus[i], "score": doc_scores[i]} for i in top_k_idx]