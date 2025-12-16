import json
import argparse
import numpy as np
from sacrebleu.metrics import BLEU, CHRF

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--leveled', action='store_true')
    args = parser.parse_args()

    # Metrics
    bleu = BLEU(lowercase=True)
    chrf = CHRF(word_order=2)

    data = []
    with open(args.output_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    refs = [[d['gold'] for d in data]]
    preds = [d['pred'] for d in data]

    print("Overall:")
    print("BLEU:", bleu.corpus_score(preds, refs).score)
    print("CHRF:", chrf.corpus_score(preds, refs).score)

    if args.leveled:
        for level in ['easy', 'medium', 'hard']:
            subset = [d for d in data if d['source'] == level]
            if not subset: continue
            
            sub_refs = [[d['gold'] for d in subset]]
            sub_preds = [d['pred'] for d in subset]
            
            print(f"\nLevel: {level} ({len(subset)} items)")
            print("BLEU:", bleu.corpus_score(sub_preds, sub_refs).score)
            print("CHRF:", chrf.corpus_score(sub_preds, sub_refs).score)