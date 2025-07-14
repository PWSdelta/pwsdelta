import json
import sys
from hf_models import load_qwen3_reranker

# Usage: python aggregate_distributed_results.py batch_translations.json
if len(sys.argv) < 2:
    print("Usage: python aggregate_distributed_results.py batch_translations.json")
    sys.exit(1)

input_file = sys.argv[1]
with open(input_file, 'r', encoding='utf-8') as f:
    results = json.load(f)

# Only keep results with a non-empty 'japanese' field
valid_results = [r for r in results if r.get('japanese')]
if not valid_results:
    print("No valid results with 'japanese' field found.")
    sys.exit(1)

# Rerank using Ollama reranker (cosine similarity)

# Prepare reranker input as a single list: [query] + candidates
text = valid_results[0].get('input_text', '')
back_english_list = [r.get('backtranslation', '') or r.get('back_english', '') for r in valid_results]
reranker = load_qwen3_reranker()
embs = reranker([text] + back_english_list)
import numpy as np
if not embs or len(embs[0]) == 0:
    print("[ERROR] Query embedding is empty for reranking. Skipping reranking and fusion.")
    sys.exit(1)
q_emb = np.array(embs[0])
sims = []
for i, e in enumerate(embs[1:]):
    if e and len(e) == len(q_emb):
        d = np.array(e)
        sim = float(np.dot(q_emb, d) / (np.linalg.norm(q_emb) * np.linalg.norm(d) + 1e-8))
        sims.append((i, sim))
    else:
        print(f"[WARN] Skipping doc {i} due to empty or mismatched embedding.")
scored = [(sim, valid_results[idx]) for idx, sim in sims]
scored.sort(reverse=True, key=lambda x: x[0])

# Top 3 and 4th-14th for fusion
fuse_top3 = [r['japanese'] for _, r in scored[:3]]
fuse_4_14 = [r['japanese'] for _, r in scored[1:14]]

# LLM Fusion of top 3
from cli_gen_prime import call_ollama_generation, parse_translation_output, parse_backtranslation_output
model_name = valid_results[0].get('model', 'qwen2.5:7b-instruct')
fuse_prompt_top3 = """Fuse these Japanese sentences into one natural, fluent Japanese translation that preserves all the original meaning, is not overly formal, and is suitable for a general audience. Only output the Japanese translation, no commentary.\n\n"""
for idx, jp in enumerate(fuse_top3):
    fuse_prompt_top3 += f"{idx+1}. {jp}\n"
fused_top3 = call_ollama_generation(fuse_prompt_top3, model_name)
fused_top3 = parse_translation_output(fused_top3)['japanese']

# LLM Fusion of 4th-14th
fuse_prompt_4_14 = """Fuse these Japanese sentences into one natural, fluent Japanese translation that preserves all the original meaning, is not overly formal, and is suitable for a general audience. Only output the Japanese translation, no commentary.\n\n"""
for idx, jp in enumerate(fuse_4_14):
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

# Build histogram
from collections import Counter
t_hist = Counter([r['japanese'] for r in valid_results])
b_hist = Counter([r.get('backtranslation', '') or r.get('back_english', '') for r in valid_results])

summary = {
    'prime_translation': {
        'input_text': text,
        'model': model_name,
        'japanese': final_fused_japanese,
        'back_english': final_fused_back_en,
        'top3_fused': fused_top3,
        '4_14_fused': fused_4_14,
        'top_japanese': fuse_top3,
        'top_back_english': [r.get('backtranslation', '') or r.get('back_english', '') for _, r in scored[:3]]
    },
    'all_runs': valid_results,
    'translation_histogram': {
        'japanese': dict(t_hist),
        'back_english': dict(b_hist)
    }
}

with open('distributed_aggregate.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print('[INFO] Aggregation complete. Results saved to distributed_aggregate.json')
