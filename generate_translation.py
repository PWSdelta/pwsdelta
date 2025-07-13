import argparse
import requests
import time
import sys
import re
import json
import csv

OLLAMA_URL = "http://localhost:11434/api/generate"

def get_available_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m['name'] for m in data.get('models', [])]
    except Exception as e:
        print(f"[ERROR] Could not fetch available models: {e}")
        return []

def call_ollama_model(model_name, prompt):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # Prefer 'completion', but fallback to 'response' (for instruct/chat models)
        result = data.get("completion")
        if result is None:
            result = data.get("response")
        if not result:
            print("[DEBUG] Ollama API response:", data)
            return "[No completion]"
        return result
    except Exception as e:
        print(f"[DEBUG] Ollama API exception: {e}")
        return f"Error: {e}"

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

    # 1. Generate 14 translations as before
    for i in range(runs):
        jp = call_ollama_model(model_name, f"Translate all of the following English sentences to Japanese, preserving each sentence, as if you were speaking in a generally polite, but not overly formal, manner:\n\n{text}")
        parsed_jp = parse_translation_output(jp)
        back_en = call_ollama_model(model_name, f"Translate this to English:\n\n{parsed_jp['japanese']}")
        parsed_en = parse_backtranslation_output(back_en)

        print(f"RUN {i+1}:")
        print(f"  {parsed_jp['japanese']}")
        print(f"  {parsed_en['english'] if parsed_en['english'] else '[No English back-translation]'}\n")

        result = {
            'run': i+1,
            'input_text': text,
            'model': model_name,
            'japanese': parsed_jp['japanese'],
            'back_english': parsed_en['english']
        }
        all_results.append(result)

    # 2. Compute semantic similarity between each back-translation and the input text
    def semantic_similarity(a, b):
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0
        return len(set_a & set_b) / len(set_a | set_b)

    scored = []
    for r in all_results:
        sim = semantic_similarity(text, r['back_english'])
        scored.append((sim, r))
    scored.sort(reverse=True, key=lambda x: x[0])
    top_n = 3
    top_japanese = []
    top_back_english = []
    for _, r in scored[:top_n]:
        top_japanese.append(r['japanese'])
        top_back_english.append(r['back_english'])

    # 3. Fuse the top 3 Japanese translations using the LLM
    fuse_prompt = """Fuse these Japanese sentences into one natural, fluent Japanese translation that preserves all the original meaning, is not overly formal, and is suitable for a general audience. Only output the Japanese translation, no commentary.\n\n"""
    for idx, jp in enumerate(top_japanese):
        fuse_prompt += f"{idx+1}. {jp}\n"
    fused_japanese = call_ollama_model(model_name, fuse_prompt)
    fused_japanese = parse_translation_output(fused_japanese)['japanese']

    # Backtranslate the fused Japanese to English using the LLM
    fused_back_en = call_ollama_model(model_name, f"Translate this to English. Only output the English translation, no commentary or explanation:\n\n{fused_japanese}")
    fused_back_en = parse_backtranslation_output(fused_back_en)['english']

    prime_translation['japanese'] = fused_japanese
    prime_translation['back_english'] = fused_back_en
    prime_translation['top_japanese'] = top_japanese
    prime_translation['top_back_english'] = top_back_english

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
def main():
    parser = argparse.ArgumentParser(description="Qwen3 Translation CLI")
    parser.add_argument("model", choices=MODELS, help="Qwen3 model to use")
    parser.add_argument("text", help="English text to translate (wrap in quotes)")
    parser.add_argument("--runs", type=int, default=5, help="Number of translation runs (default: 14)")
    args = parser.parse_args()

    print(f"Using model: {args.model}")
    print(f"Translating: {args.text}")
    run_translation(args.model, args.text, runs=args.runs)



def ollama_health_check():
    try:
        resp = requests.get("http://localhost:11434/", timeout=31)
        if resp.status_code == 200:
            print("Ollama server is reachable.")
            return True
        else:
            print(f"Ollama server responded with status: {resp.status_code}")
            return False
    except Exception as e:
        print(f"Could not reach Ollama server: {e}")
        print("Please start Ollama with: ollama serve")
        return False

def ollama_model_check(model_name):
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        available = [m['name'] for m in data.get('models', [])]
        if model_name in available:
            return True
        else:
            print(f"Model '{model_name}' is not available in your local Ollama.")
            print(f"Available models: {available}")
            print(f"To pull the model, run: ollama pull {model_name}")
            return False
    except Exception as e:
        print(f"Could not check models: {e}")
        return False

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
    if not ollama_health_check():
        sys.exit(1)

    available_models = get_available_models()
    if not available_models:
        print("No models available in Ollama. Please pull a model with 'ollama pull <modelname>' and try again.")
        sys.exit(1)

    default_model = "qwen3:1.8b"
    model = default_model if default_model in available_models else available_models[0]
    print(f"Using model: {model}")

    def select_models_menu():
        print("\nAvailable models:")
        for idx, m in enumerate(available_models):
            print(f"{idx+1}. {m}")
        print("a. All models above")
        print("0. Cancel")
        sel = input("Select model(s) to test (comma-separated numbers, or 'a' for all): ")
        if sel.strip().lower() == 'a':
            return available_models
        if sel.strip() == '0':
            return []
        try:
            idxs = [int(x)-1 for x in sel.split(',') if x.strip().isdigit()]
            return [available_models[i] for i in idxs if 0 <= i < len(available_models)]
        except Exception:
            print("Invalid selection.")
            return []

    while True:
        choice = menu()
        if choice == "1":
            text = input("Enter text to translate: ")
            models_to_run = select_models_menu()
            if not models_to_run:
                continue
            results = []
            for m in models_to_run:
                print(f"\n=== Running for model: {m} ===")
                res = run_translation(m, text, runs=14)
                results.append({'model': m, 'result': res})
            with open('latest_translation.json', 'w', encoding='utf-8') as jf:
                json.dump({'models': results}, jf, ensure_ascii=False, indent=2)
            print("[INFO] Saved all model results to latest_translation.json")
        elif choice == "2":
            default_text = "I love programming. I am learning about AI and Python now. I enjoy cooking and exploring new cuisines."
            print(f"Using default text: {default_text}")
            models_to_run = select_models_menu()
            if not models_to_run:
                continue
            results = []
            for m in models_to_run:
                print(f"\n=== Running for model: {m} ===")
                res = run_translation(m, default_text, runs=14)
                results.append({'model': m, 'result': res})
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
