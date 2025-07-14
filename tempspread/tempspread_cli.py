# Standalone CLI for 31-run translation at multiple temperatures (no external project imports)
import argparse
import json
import os
import re
import requests

def call_ollama_generation(prompt, model_name, max_new_tokens=128, temperature=0.7):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": {"num_predict": max_new_tokens, "temperature": temperature}
    }
    try:
        resp = requests.post(url, json=payload, timeout=120, stream=True)
        output = ""
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = line.decode("utf-8")
                if data.startswith('{'):
                    chunk = json.loads(data)
                    if 'response' in chunk:
                        output += chunk['response']
            except Exception as e:
                print(f"[DEBUG] JSON parse error: {e} for data: {data}")
                continue
        return output.strip()
    except Exception as e:
        print(f"[ERROR] Ollama generation failed: {e}")
        return "[Ollama error: no output]"

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
        if re.match(r'^[A-Za-z0-9 ,\.\!\?\'\"\-]+$', l):
            # Remove leading/trailing quotes
            l = l.strip('"\'')
            en_lines.append(l)
    return {'english': ' '.join(en_lines).strip()}

def run_31_translations(text, model, temperature):
    global all_results, output_file, text_arg, model_arg
    for i in range(31):
        prompt = f"Translate all of the following English sentences to Japanese, preserving each sentence, as if you were speaking in a generally polite, but not overly formal, manner:\n\n{text}"
        jp_raw = call_ollama_generation(prompt, model, max_new_tokens=128, temperature=temperature)
        parsed_jp = parse_translation_output(jp_raw)
        run_num = i+1
        pad = ' ' if run_num < 10 else ''
        print(f"[TEMP {temperature}] Run {run_num}/31...{pad} {parsed_jp['japanese']}")
        # Backtranslate to English
        if parsed_jp['japanese']:
            back_en_raw = call_ollama_generation(f"Translate this to English. Only output the English translation, no commentary or explanation:\n\n{parsed_jp['japanese']}", model, max_new_tokens=128, temperature=temperature)
            parsed_en = parse_backtranslation_output(back_en_raw)
            backtranslation = parsed_en['english']
        else:
            backtranslation = ''
        result = {
            'japanese': parsed_jp['japanese'],
            'backtranslation': backtranslation,
            'input_text': text,
            'temperature': temperature
        }
        all_results.append(result)
        # Write after every run
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({'text': text_arg, 'model': model_arg, 'results': all_results}, f, ensure_ascii=False, indent=2)
    return None

def main():
    parser = argparse.ArgumentParser(description="Run 31 translations at different temperatures and compare results.")
    parser.add_argument('--text', type=str, required=True, help='Input English text to translate')
    parser.add_argument('--model', type=str, default='qwen2.5:7b-instruct', help='Model name')
    parser.add_argument('--output', type=str, default='tempspread_results.json', help='Output JSON file')
    args = parser.parse_args()

    temps = [0.1, 0.3, 0.5, 0.7, 0.9]
    global all_results, temp, output_file, text_arg, model_arg
    all_results = []
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Write initial empty structure
    output_file = args.output
    text_arg = args.text
    model_arg = args.model
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({'text': text_arg, 'model': model_arg, 'results': []}, f, ensure_ascii=False, indent=2)
    all_results = []
    try:
        for temp in temps:
            run_31_translations(args.text, args.model, temp)
    finally:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({'text': text_arg, 'model': model_arg, 'results': all_results}, f, ensure_ascii=False, indent=2)
    if not all_results:
        print(f"[ERROR] No results were written to {args.output}")
    else:
        print(f"[INFO] Saved all results to {args.output}")

if __name__ == '__main__':
    main()
