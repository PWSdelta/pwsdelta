import json
import re
import numpy as np
import argparse

# --- Parsing functions (from cli_gen_prime.py) ---
def parse_translation_output(output):
    lines = output.splitlines() if isinstance(output, str) else []
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
    lines = output.splitlines() if isinstance(output, str) else []
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

# --- Main aggregation logic ---
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def dummy_embed(text):
    # Dummy embedding: hash-based vector for demonstration (replace with real model for production)
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(384)

def aggregate_prime_translation(results, input_text, model_name):
    # 1. Collect all backtranslations
    all_jp = [r['japanese'] for r in results if r.get('japanese')]
    all_back_en = [r['backtranslation'] for r in results if r.get('backtranslation')]
    # 2. Compute similarities (dummy embedding)
    sims = []
    input_emb = dummy_embed(input_text)
    for idx, back_en in enumerate(all_back_en):
        if not back_en:
            continue
        emb = dummy_embed(back_en)
        sim = cosine_similarity(input_emb, emb)
        sims.append((idx, sim))
    if not sims:
        return {}
    # 3. Sort by similarity
    sims.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in sims]
    # 4. Top 3 and 4th-14th
    top3 = [all_jp[i] for i in top_indices[:3]]
    fourth_to_14th = [all_jp[i] for i in top_indices[3:14]]
    # 5. LLM fusion (here, just join for demo)
    fused_top3 = ' '.join(top3)
    fused_4_14 = ' '.join(fourth_to_14th)
    final_fused_japanese = fused_top3 + ' ' + fused_4_14
    # 6. Backtranslate (pick most similar for demo)
    final_fused_back_en = all_back_en[top_indices[0]] if top_indices else ''
    # 7. Build block
    return {
        'input_text': input_text,
        'model': model_name,
        'japanese': final_fused_japanese,
        'back_english': final_fused_back_en,
        'top3_fused': fused_top3,
        '4_14_fused': fused_4_14,
        'top_japanese': top3,
        'top_back_english': [all_back_en[i] for i in top_indices[:3]]
    }

def main():
    parser = argparse.ArgumentParser(description='Aggregate tempspread_results.json to produce prime_translation block.')
    parser.add_argument('--input', type=str, default='tempspread_results.json', help='Input JSON file')
    parser.add_argument('--output', type=str, default='tempspread_prime_translation.json', help='Output JSON file')
    args = parser.parse_args()
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = data['results']
    input_text = data.get('text', '')
    model_name = data.get('model', 'qwen2.5:7b-instruct')
    # Group results by temperature
    temp_groups = {}
    for r in results:
        temp = r.get('temperature')
        if temp is None:
            continue
        temp_groups.setdefault(str(temp), []).append(r)
    prime_translations = {}
    for temp, group in temp_groups.items():
        prime_translations[temp] = aggregate_prime_translation(group, input_text, model_name)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'prime_translations': prime_translations}, f, ensure_ascii=False, indent=2)
    print(f"Wrote prime_translations for all temperatures to {args.output}")

if __name__ == '__main__':
    main()
