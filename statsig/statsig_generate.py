import requests
import re
import json
import argparse
import os

def should_stop_early(num_runs, min_runs=10, window=5):
    # No early stopping for 31/14/3 workflow
    return False

def call_ollama_generation(prompt, model_name="qwen2.5:7b-instruct", max_new_tokens=128, temperature=0.5):
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
            except Exception:
                continue
        return output.strip()
    except Exception as e:
        print(f"[ERROR] Ollama generation failed: {e}")
        return "[Ollama error: no output]"

def parse_translation_output(output):
    lines = output.splitlines()
    jp_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if re.match(r'^(以下|こちら|注|和訳|翻訳|Here|Note|In English|\d+\.|[A-Za-z])', l):
            continue
        if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', l):
            jp_lines.append(l)
    return ' '.join(jp_lines).strip()

def parse_backtranslation_output(output):
    lines = output.splitlines()
    en_lines = []
    for line in lines:
        l = line.strip()
        if not l:
            continue
        if re.match(r'^(Here|Note|In English|\d+\.|[\u3040-\u30ff\u4e00-\u9fff])', l):
            continue
        if re.match(r'^[A-Za-z0-9 ,\.!?\'\"\-]+$', l):
            l = l.strip('"\'')
            en_lines.append(l)
    return ' '.join(en_lines).strip()

def count_japanese_chars(text):
    hiragana = sum(1 for c in text if '\u3040' <= c <= '\u309f')
    katakana = sum(1 for c in text if '\u30a0' <= c <= '\u30ff')
    kanji = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return {'hiragana': hiragana, 'katakana': katakana, 'kanji': kanji}

def run_stat_sig_batch(input_text, temperature=0.5, min_runs=10, max_runs=31):
    results = []
    for i in range(31):
        prompt = f"Translate all of the following English sentences to Japanese, preserving each sentence, as if you were speaking in a generally polite, but not overly formal, manner:\n\n{input_text}"
        jp_raw = call_ollama_generation(prompt, max_new_tokens=128, temperature=temperature)
        parsed_jp = parse_translation_output(jp_raw)
        char_counts = count_japanese_chars(parsed_jp)
        if parsed_jp:
            back_en_raw = call_ollama_generation(f"Translate this to English. Only output the English translation, no commentary or explanation:\n\n{parsed_jp}", max_new_tokens=128, temperature=temperature)
            parsed_en = parse_backtranslation_output(back_en_raw)
            backtranslation = parsed_en
        else:
            backtranslation = ''
        result = {
            'run': i+1,
            'input_text': input_text,
            'japanese': parsed_jp,
            'backtranslation': backtranslation,
            'temperature': temperature,
            'hiragana': char_counts['hiragana'],
            'katakana': char_counts['katakana'],
            'kanji': char_counts['kanji']
        }
        results.append(result)
        print(f"Run {i+1}: {parsed_jp} [ひ:{char_counts['hiragana']} カ:{char_counts['katakana']} 漢:{char_counts['kanji']}]" )

    # Remove lowest 17 by length (shortest Japanese outputs)
    sorted_results = sorted(results, key=lambda r: len(r['japanese']), reverse=True)
    top_14 = sorted_results[:14]
    top_3 = sorted_results[:3]

    # For top_14 and top_3, include both Japanese and backtranslation (always from raw runs)
    def find_backtranslation(jp):
        for r in results:
            if r['japanese'] == jp:
                return r['backtranslation']
        return ''

    top_14_out = [{
        'japanese': r['japanese'],
        'backtranslation': find_backtranslation(r['japanese'])
    } for r in top_14]
    top_3_out = [{
        'japanese': r['japanese'],
        'backtranslation': find_backtranslation(r['japanese'])
    } for r in top_3]

    def llm_merge(jp_list, label):
        merge_prompt = f"You are an expert Japanese translator. Merge the following {len(jp_list)} Japanese translations into a single, best, natural, and accurate Japanese translation. Only output the merged Japanese translation.\n\n" + "\n\n".join(jp_list)
        merged = call_ollama_generation(merge_prompt, max_new_tokens=256, temperature=temperature)
        print(f"[LLM Merge: {label}]\n{merged}\n")
        return merged.strip()

    merged_14 = llm_merge([r['japanese'] for r in top_14], "top_14")
    merged_3 = llm_merge([r['japanese'] for r in top_3], "top_3")
    final_merged = llm_merge([merged_14, merged_3], "final_merge")

    # Backtranslations for merged outputs
    def get_backtranslation(jp):
        back_en_raw = call_ollama_generation(
            f"Translate this to English. Only output the English translation, no commentary or explanation:\n\n{jp}",
            max_new_tokens=256, temperature=temperature)
        return parse_backtranslation_output(back_en_raw)

    merged_14_backtranslation = get_backtranslation(merged_14)
    merged_3_backtranslation = get_backtranslation(merged_3)
    final_merged_backtranslation = get_backtranslation(final_merged)

    return {
        'all_results': results,
        'top_14': top_14_out,
        'top_3': top_3_out,
        'merged_14': merged_14,
        'merged_14_backtranslation': merged_14_backtranslation,
        'merged_3': merged_3,
        'merged_3_backtranslation': merged_3_backtranslation,
        'final_merged': final_merged,
        'final_merged_backtranslation': final_merged_backtranslation
    }

def main():
    parser = argparse.ArgumentParser(description="Run stat-sig batch translation and save results as JSON (no embeddings).")
    parser.add_argument('--temperatures', type=float, nargs='*', default=[0.5], help='Sampling temperatures (space separated, e.g. 0.1 0.5 0.9)')
    parser.add_argument('--min_runs', type=int, default=10, help='Minimum runs before checking for significance')
    parser.add_argument('--max_runs', type=int, default=31, help='Maximum number of runs')
    parser.add_argument('input_text', nargs='?', default='', help='Input English text to translate (last argument, optional)')
    args = parser.parse_args()

    all_temp_results = []
    for temp in args.temperatures:
        print(f"\n=== Running batch for temperature {temp} ===")
        results = run_stat_sig_batch(
            args.input_text,
            temperature=temp
        )
        out_data = {
            'input_text': args.input_text,
            'temperature': temp,
            'all_results': results['all_results'],
            'top_14': results['top_14'],
            'top_3': results['top_3'],
            'merged_14': results['merged_14'],
            'merged_14_backtranslation': results['merged_14_backtranslation'],
            'merged_3': results['merged_3'],
            'merged_3_backtranslation': results['merged_3_backtranslation'],
            'final_merged': results['final_merged'],
            'final_merged_backtranslation': results['final_merged_backtranslation']
        }
        all_temp_results.append(out_data)

    # Save all results in a single JSON file (list of dicts, one per temperature)
    with open('latest_translation.json', 'w', encoding='utf-8') as f:
        json.dump(all_temp_results, f, ensure_ascii=False, indent=2)
    print(f"Wrote results for {len(all_temp_results)} temperature(s) to latest_translation.json")

    # Write CSV for all 31 runs for each temperature
    import csv
    with open('latest_translation.csv', 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['temperature', 'run', 'japanese', 'backtranslation', 'hiragana', 'katakana', 'kanji', 'length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for temp_result in all_temp_results:
            for r in temp_result['all_results']:
                writer.writerow({
                    'temperature': temp_result['temperature'],
                    'run': r['run'],
                    'japanese': r['japanese'],
                    'backtranslation': r['backtranslation'],
                    'hiragana': r['hiragana'],
                    'katakana': r['katakana'],
                    'kanji': r['kanji'],
                    'length': len(r['japanese'])
                })
    print("Wrote all runs for all temperatures to latest_translation.csv")

if __name__ == '__main__':
    main()
