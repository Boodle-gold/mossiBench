import re
import jieba

class Tokenizer():
    def __init__(self):
        pass

    def tokenize(self, text, remove_punc=False):
        text = text.lower()    
        if remove_punc:
            # Common punctuation removal
            for punc in "，。、；！？「」『』【】（）《》“”…,.;?!":
                text = text.replace(punc, " ")
            text = re.sub(r'\d+', '', text)
        else:
            # Add spaces around punctuation
            for punc in ",.;?!":
                text = text.replace(punc, " " + punc + " ")
        
        # Split by whitespace
        tokenized_text = text.split(" ")
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text

class EngTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

class MosTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    # Mossi specific characters (like ã, õ, ɛ, ɩ) are handled fine by python strings
    pass

class ZhTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def tokenize(self, text, remove_punc=False, do_cut_all=False, cut_for_search=False):
        text = text.lower()
        if remove_punc:
            for punc in "，。、；！？「」『』【】（）《》“”…":
                text = text.replace(punc, "")
            text = re.sub(r'\d+', '', text)
        
        if cut_for_search:
            tokenized_text = jieba.lcut_for_search(text)
        else:
            tokenized_text = jieba.lcut(text, cut_all=do_cut_all)
        tokenized_text = [word.strip() for word in tokenized_text if word.strip() != ""]
        return tokenized_text