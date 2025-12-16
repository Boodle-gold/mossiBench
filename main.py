import argparse
import os
import json
import random
import numpy as np
from tqdm import tqdm

from dictionary import WordDictionary
from corpus import ParallelCorpus
# We don't import load_model if we are in API mode, to save RAM
from model import get_pred_api
from prompts import construct_prompt_mos2en, construct_prompt_en2mos

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Resources
    parser.add_argument('--src_lang', type=str, default='mos')
    parser.add_argument('--tgt_lang', type=str, default='en')
    parser.add_argument('--dict_path', type=str, default='dictionary.json')
    parser.add_argument('--corpus_path', type=str, default='mossi_corpus.json')
    parser.add_argument('--test_data_path', type=str, default='mossi_test.json')
    
    # API CONFIGURATION (New)
    parser.add_argument('--use_api', action='store_true', help="Use an API instead of local model")
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--base_url', type=str, default="https://api.openai.com/v1", help="Change this for other providers")
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini') 

    # Local Model Config (Ignored if --use_api is set)
    parser.add_argument('--model_path', type=str, default=None) 
    parser.add_argument('--no_vllm', action='store_true')
    parser.add_argument('--n_gpu', type=int, default=1)

    # Generation Config
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--top_p', type=float, default=0.9)

    # Prompt Config
    parser.add_argument('--prompt_type', type=str, default='mos2en', choices=['mos2en', 'en2mos'])
    parser.add_argument('--num_parallel_sent', type=int, default=3)
    
    # Output
    parser.add_argument('--output_path', type=str, default=None)

    args = parser.parse_args()

    # 1. Load Resources (Always Local)
    dictionary = WordDictionary(args.src_lang, args.tgt_lang, args.dict_path)
    parallel_corpus = ParallelCorpus(args.src_lang, args.tgt_lang, args.corpus_path)
    test_data = json.load(open(args.test_data_path, 'r'))

    # 2. Setup Model (API vs Local)
    llm = None
    tokenizer = None
    client = None

    if args.use_api:
        from openai import OpenAI
        print(f"Connecting to API: {args.base_url}")
        # If api_key is not passed, look for env var
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please provide --api_key or set OPENAI_API_KEY environment variable.")
            
        client = OpenAI(api_key=api_key, base_url=args.base_url)
    else:
        # Local Loading Logic
        if not args.model_path:
             raise ValueError("You must provide --model_path for local execution")
             
        if args.no_vllm:
            llm, tokenizer = load_model(args.model_name, args.model_path, args.n_gpu, use_vllm=False)
        else:
            # Import only if needed
            from vllm import SamplingParams
            llm = load_model(args.model_name, args.model_path, args.n_gpu, use_vllm=True)

    # 3. Setup Prompt Function
    prompt_funcs = {
        'mos2en': construct_prompt_mos2en,
        'en2mos': construct_prompt_en2mos
    }
    prompt_func = prompt_funcs[args.prompt_type]

    # 4. Output Config
    if not args.output_path:
        mode = "api" if args.use_api else "local"
        args.output_path = f"output_{args.src_lang}2{args.tgt_lang}_{mode}.jsonl"
    
    fout = open(args.output_path, 'w', encoding='utf-8')
    print(f"Writing results to {args.output_path}...")

    # 5. Inference Loop
    for item in tqdm(test_data):
        src_sent = item[args.src_lang]
        
        # A. Construct Prompt (Happens Locally)
        prompt = prompt_func(src_sent, dictionary, parallel_corpus, args)
        
        # B. Generate (Happens via API or Local)
        pred = ""
        if args.use_api:
            pred = get_pred_api(client, args.model_name, prompt, args)
        elif args.no_vllm:
            pred = get_pred_no_vllm(llm, tokenizer, prompt, args)
        else:
            # vLLM logic
            sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_new_tokens)
            pred = get_pred(llm, sampling_params, prompt)
            
        # C. Save
        output_obj = {
            "query": src_sent,
            "gold": item[args.tgt_lang],
            "pred": pred,
            "prompt": prompt,
            "source": item.get('source', 'n/a')
        }
        fout.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
        fout.flush()

    fout.close()
    print("Done. You can now run eval.py on the output file.")