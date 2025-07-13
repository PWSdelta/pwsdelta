

import argparse
import time
import sys
import re
import json
import csv
from hf_models import load_qwen3_generation, load_qwen3_reranker, load_qwen3_embeddings

import requests


import os




# Only use qwen2.5:7b-instruct for all tasks
EMBED_MODEL = "qwen2.5:7b-instruct"
OLLAMA_MODELS = {
    "qwen2.5:7b-instruct": {
        "gen": load_qwen3_generation(model_name="qwen2.5:7b-instruct")
    }
}
EMBED_PIPE = load_qwen3_embeddings(model_name=EMBED_MODEL)
RERANKER_PIPE = load_qwen3_reranker(model_name=EMBED_MODEL)

def get_available_models():
    return list(OLLAMA_MODELS.keys())

def call_ollama_generation(prompt, model_name, max_new_tokens=128):
    gen_pipe = OLLAMA_MODELS[model_name]["gen"]
    out = gen_pipe(prompt, max_new_tokens=max_new_tokens)
    raw = out[0]['generated_text'] if isinstance(out, list) else out['generated_text']
    return raw

def call_ollama_embedding(sentences, model_name=None):
    return EMBED_PIPE.encode(sentences)

def call_ollama_reranker(query, docs, model_name=None):
    import numpy as np
    if not docs:
        return []
    texts = [query] + docs
    embs = RERANKER_PIPE(texts)
    # Only keep docs with valid embeddings
    valid_indices = [i for i, e in enumerate(embs[1:]) if e and len(e) == len(embs[0])]
    if not embs or len(embs[0]) == 0:
        print(f"[ERROR] Query embedding is empty for embedding model {EMBED_MODEL}. Skipping reranking.")
        return []
    if not valid_indices:
        print(f"[ERROR] All doc embeddings are empty or mismatched for embedding model {EMBED_MODEL}. Skipping reranking.")
        return []
    q_emb = np.array(embs[0])
    sims = []
    for i, e in enumerate(embs[1:]):
        if e and len(e) == len(q_emb):
            d = np.array(e)
            sim = float(np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d) + 1e-8))
            sims.append((i, sim))
        else:
            print(f"[WARN] Skipping doc {i} due to empty or mismatched embedding.")
    return sims

def parse_translation_output(output):
    # Extract only Japanese sentences, remove commentary and non-Japanese explanations
    lines = output.splitlines()
    jp_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Remove lines that look like commentary, explanations, or start with '以下', 'こちら', etc.
        if re.match(r'^(以下|こちら|注|和訳|翻訳|Here|Note|In English|\d+\.|[A-Za-z])', l):
            continue
        # Heuristic: Japanese sentences usually contain Hiragana/Katakana/Kanji
        if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', l):
            jp_lines.append(l)
    return {'japanese': ' '.join(jp_lines).strip()}

def parse_backtranslation_output(output):
    # Remove commentary, explanations, and only keep main English sentences
    lines = output.splitlines()
    en_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # Remove lines that look like commentary, explanations, or start with 'Here', 'Note', 'In English', etc.
        if re.match(r'^(Here|Note|In English|\d+\.|[\u3040-\u30ff\u4e00-\u9fff])', l):
            continue
        # Heuristic: English sentences contain mostly ASCII and punctuation
        if re.match(r'^[A-Za-z0-9 ,\.!?\'\"\-]+$', l):
            # Remove leading/trailing quotes
            l = l.strip('"\'')
            en_lines.append(l)
    return {'english': ' '.join(en_lines).strip()}


def strip_note(text):
    if not text:
        return text
    # Remove 'Note:' and everything after, case-insensitive
    return re.split(r'(?i)\bnote:', text)[0].strip()

def update_translation_histogram(all_results):
    histogram = {
        "japanese": {},
        "jp_romaji": {},
        "back_english": {}
    }

    for result in all_results:
        japanese = result.get("japanese", "")
        jp_romaji = result.get("jp_romaji", "")
        back_english = result.get("back_english", "")

        if japanese:
            histogram["japanese"][japanese] = histogram["japanese"].get(japanese, 0) + 1
        if jp_romaji:
            histogram["jp_romaji"][jp_romaji] = histogram["jp_romaji"].get(jp_romaji, 0) + 1
        if back_english:
            histogram["back_english"][back_english] = histogram["back_english"].get(back_english, 0) + 1

    return histogram




def run_translation(model_name, text, runs=14, delay=0):
    all_results = []
    prime_translation = {
        'input_text': text,
        'model': model_name,
        'japanese': "",
        'back_english': ""
    }

    # 1. Generate translations using Ollama generation (31 runs)
    for i in range(31):
        print(f"[INFO] Run {i+1} for model {model_name}...")
        jp_raw = call_ollama_generation(f"Translate all of the following English sentences to Japanese, preserving each sentence, as if you were speaking in a generally polite, but not overly formal, manner:\n\n{text}", model_name)
        parsed_jp = parse_translation_output(jp_raw)
        back_en_raw = call_ollama_generation(f"Translate this to English:\n\n{parsed_jp['japanese']}", model_name)
        parsed_en = parse_backtranslation_output(back_en_raw)

        result = {
            'run': i+1,
            'input_text': text,
            'model': model_name,
            'japanese': parsed_jp['japanese'],
            'back_english': parsed_en['english']
        }
        all_results.append(result)

    # 2. Compute semantic similarity using Ollama reranker or embeddings
    back_english_list = [r['back_english'] for r in all_results]
    sims = call_ollama_reranker(text, back_english_list, model_name)
    # sims is a list of (idx, sim) for valid runs only
    if not sims:
        print(f"[WARN] No valid embeddings for reranking. Skipping reranking and fusion for this run.")
        return {
            'prime_translation': prime_translation,
            'all_runs': all_results,
            'translation_histogram': {},
            'translation_occurrences': {}
        }
    # Only keep valid runs
    scored = [(sim, all_results[idx]) for idx, sim in sims]
    scored.sort(reverse=True, key=lambda x: x[0])

    # Top 3 and 4th-14th for fusion
    top3 = [r['japanese'] for _, r in scored[:3]]
    fourth_to_14th = [r['japanese'] for _, r in scored[3:14]]

    # LLM Fusion of top 3
    fuse_prompt_top3 = """Fuse these Japanese sentences into one natural, fluent Japanese translation that preserves all the original meaning, is not overly formal, and is suitable for a general audience. Only output the Japanese translation, no commentary.\n\n"""
    for idx, jp in enumerate(top3):
        fuse_prompt_top3 += f"{idx+1}. {jp}\n"
    fused_top3 = call_ollama_generation(fuse_prompt_top3, model_name)
    fused_top3 = parse_translation_output(fused_top3)['japanese']

    # LLM Fusion of 4th-14th
    fuse_prompt_4_14 = """Fuse these Japanese sentences into one natural, fluent Japanese translation that preserves all the original meaning, is not overly formal, and is suitable for a general audience. Only output the Japanese translation, no commentary.\n\n"""
    for idx, jp in enumerate(fourth_to_14th):
        fuse_prompt_4_14 += f"{idx+4}. {jp}\n"
    fused_4_14 = call_ollama_generation(fuse_prompt_4_14, model_name)
    fused_4_14 = parse_translation_output(fused_4_14)['japanese']

    # Final LLM Fusion of the two fusions
    final_fuse_prompt = """Fuse these two Japanese translations into one final, natural, fluent Japanese translation that preserves all the original meaning, is not overly formal, and is suitable for a general audience. Only output the Japanese translation, no commentary.\n\n1. """ + fused_top3 + "\n2. " + fused_4_14 + "\n"
    final_fused_japanese = call_ollama_generation(final_fuse_prompt, model_name)
    final_fused_japanese = parse_translation_output(final_fused_japanese)['japanese']

    # Backtranslate the final fused Japanese to English using the LLM
    final_fused_back_en = call_ollama_generation(f"Translate this to English. Only output the English translation, no commentary or explanation:\n\n{final_fused_japanese}", model_name)
    final_fused_back_en = parse_backtranslation_output(final_fused_back_en)['english']

    prime_translation['japanese'] = final_fused_japanese
    prime_translation['back_english'] = final_fused_back_en
    prime_translation['top3_fused'] = fused_top3
    prime_translation['4_14_fused'] = fused_4_14
    prime_translation['top_japanese'] = top3
    prime_translation['top_back_english'] = [r['back_english'] for _, r in scored[:3]]

    # Update histogram after all runs
    translation_histogram = {
        "japanese": {},
        "back_english": {}
    }
    for result in all_results:
        japanese = result.get("japanese", "")
        back_english = result.get("back_english", "")
        if japanese:
            translation_histogram["japanese"][japanese] = translation_histogram["japanese"].get(japanese, 0) + 1
        if back_english:
            translation_histogram["back_english"][back_english] = translation_histogram["back_english"].get(back_english, 0) + 1

    # Build histogram of unique English-to-Japanese and Japanese-to-English translations
    en_to_jp_hist = {}
    jp_to_en_hist = {}
    for result in all_results:
        en = result.get("input_text", "")
        jp = result.get("japanese", "")
        back_en = result.get("back_english", "")
        if en and jp:
            if en not in en_to_jp_hist:
                en_to_jp_hist[en] = {}
            en_to_jp_hist[en][jp] = en_to_jp_hist[en].get(jp, 0) + 1
        if jp and back_en:
            if jp not in jp_to_en_hist:
                jp_to_en_hist[jp] = {}
            jp_to_en_hist[jp][back_en] = jp_to_en_hist[jp].get(back_en, 0) + 1

    # Return all results for this model
    return {
        'prime_translation': prime_translation,
        'all_runs': all_results,
        'translation_histogram': translation_histogram,
        'translation_occurrences': {
            'english_to_japanese': en_to_jp_hist,
            'japanese_to_english': jp_to_en_hist
        }
    }

# CLI entry point

# No main() CLI for Hugging Face models; only Ollama pipeline is used




# No health/model check needed for Hugging Face

def menu():
    print("\nCLI Menu:")
    print("1. Enter text")
    print("2. Use default text (Programming and Food)")
    print("3. Read from a .txt file")
    print("4. Read from a .csv file")
    print("5. Read from a .json file")
    print("0. Exit")

    choice = input("Enter your choice: ")
    return choice

def process_txt_file(file_path, model_name, runs):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            text = line.strip()
            if text:
                print(f"Processing: {text}")
                run_translation(model_name, text, runs)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
    except Exception as e:
        print(f"[ERROR] An error occurred while processing the file: {e}")


if __name__ == "__main__":
    # Multi-model support: Qwen2.5:7b-instruct and Qwen3:8b-instruct
    available_models = get_available_models()
    print(f"Available models: {', '.join(available_models)}")

    while True:
        choice = menu()
        if choice == "1":
            text = input("Enter text to translate: ")
            print(f"\n=== Running for models: {', '.join(available_models)} ===")
            results = []
            for model in available_models:
                print(f"[INFO] Running {model}...")
                res = run_translation(model, text, runs=14)
                results.append({'model': model, 'result': res})
            with open('latest_translation.json', 'w', encoding='utf-8') as jf:
                json.dump({'models': results}, jf, ensure_ascii=False, indent=2)
            print("[INFO] Saved all model results to latest_translation.json")
        elif choice == "2":
            default_text = "I love programming. I am learning about AI and Python now. I enjoy cooking and exploring new cuisines."
            print(f"Using default text: {default_text}")
            print(f"\n=== Running for models: {', '.join(available_models)} ===")
            results = []
            for model in available_models:
                print(f"[INFO] Running {model}...")
                res = run_translation(model, default_text, runs=14)
                results.append({'model': model, 'result': res})
            with open('latest_translation.json', 'w', encoding='utf-8') as jf:
                json.dump({'models': results}, jf, ensure_ascii=False, indent=2)
            print("[INFO] Saved all model results to latest_translation.json")
        elif choice == "3":
            file_path = input("Enter the path to the .txt file: ")
            print("[INFO] Multi-model batch for .txt not implemented yet.")
            # You could implement similar logic for batch files if needed
        elif choice == "4":
            print("[INFO] CSV processing is not implemented yet.")
        elif choice == "5":
            print("[INFO] JSON processing is not implemented yet.")
        elif choice == "0":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")
