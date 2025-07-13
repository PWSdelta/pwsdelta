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
    # Extracts Japanese translation from the model output
    japanese = []
    lines = output.splitlines()
    in_japanese = False

    for line in lines:
        l = line.strip()
        if not l:
            continue
        if re.search(r'(Japanese|日本語|translation into Japanese)', l, re.I):
            in_japanese = True
            continue
        if in_japanese:
            # Collect only the actual translation
            japanese.append(l)

    return {
        'japanese': '\n'.join(japanese).strip()
    }

def parse_backtranslation_output(output):
    # Extracts English translation from the model output
    english = []
    lines = output.splitlines()
    in_english = False

    for line in lines:
        l = line.strip()
        if not l:
            continue
        if re.search(r'(English|translation into English)', l, re.I):
            in_english = True
            continue
        if in_english:
            # Collect only the actual translation
            english.append(l)

    return {
        'english': '\n'.join(english).strip()
    }


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

    for i in range(runs):
        print(f"\nRun {i+1}:")
        prompt = f"Translate this to Japanese:\n\n{text}"
        print(f"[PROMPT to Ollama]: {prompt}")
        jp = call_ollama_model(model_name, prompt)
        parsed_jp = parse_translation_output(jp)

        # Update prime translation fields
        prime_translation['japanese'] = parsed_jp['japanese']

        back_prompt = f"Translate this to English:\n\n{parsed_jp['japanese']}"
        print(f"[PROMPT to Ollama]: {back_prompt}")
        back_en = call_ollama_model(model_name, back_prompt)
        parsed_en = parse_backtranslation_output(back_en)

        prime_translation['back_english'] = parsed_en['english']

        result = {
            'run': i+1,
            'input_text': text,
            'model': model_name,
            'japanese': parsed_jp['japanese'],
            'back_english': parsed_en['english']
        }
        all_results.append(result)

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

    # Save prime translation, all results, and histograms to JSON (no CSV)
    if all_results:
        with open('latest_translation.json', 'w', encoding='utf-8') as jf:
            json.dump({
                'prime_translation': prime_translation,
                'all_runs': all_results,
                'translation_histogram': translation_histogram,
                'translation_occurrences': {
                    'english_to_japanese': en_to_jp_hist,
                    'japanese_to_english': jp_to_en_hist
                }
            }, jf, ensure_ascii=False, indent=2)
        print("[INFO] Saved prime translation, all results, and histograms to latest_translation.json")

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

    while True:
        choice = menu()
        if choice == "1":
            text = input("Enter text to translate: ")
            run_translation(model, text, runs=14)
        elif choice == "2":
            default_text = "I love programming. I am learning about AI and Python now. I enjoy cooking and exploring new cuisines."
            print(f"Using default text: {default_text}")
            run_translation(model, default_text, runs=14)
        elif choice == "3":
            file_path = input("Enter the path to the .txt file: ")
            process_txt_file(file_path, model, runs=14)
        elif choice == "4":
            print("[INFO] CSV processing is not implemented yet.")
        elif choice == "5":
            print("[INFO] JSON processing is not implemented yet.")
        elif choice == "0":
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")
