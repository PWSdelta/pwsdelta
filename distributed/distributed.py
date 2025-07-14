import requests
import time
import sys
import json

# Usage: python distributed_batch_translate.py --model qwen2.5:7b-instruct --runs 31 --server http://localhost:5000 --timeout 600 --text "Your sentence here."
import argparse
parser = argparse.ArgumentParser(description='Distributed batch translation (enqueue and wait for results)')
parser.add_argument('--model', default='qwen2.5:7b-instruct')
parser.add_argument('--runs', type=int, default=31)
parser.add_argument('--server', default='http://localhost:5000')
parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds to wait for all results')
parser.add_argument('--text', required=True, help='English text to translate (2 sentences recommended)')
args = parser.parse_args()

# Prepend starter instruction
starter = "Translate this English text into Japanese in a polite, friendly manner, but not over the top.\n\n"
full_text = starter + args.text.strip()

# 1. Enqueue jobs
payload = {'text': full_text, 'model': args.model, 'runs': args.runs}
resp = requests.post(f'{args.server}/start-distributed', json=payload)
if resp.status_code != 200:
    print('[ERROR] Failed to enqueue jobs:', resp.text)
    sys.exit(1)
print(f'[INFO] Enqueued {args.runs} jobs for distributed translation.')


# 2. Poll for results (ensure unique work_ids)
start = time.time()
while True:
    r = requests.get(f'{args.server}/distributed-results')
    if r.status_code != 200:
        print('[ERROR] Failed to get results:', r.text)
        sys.exit(1)
    data = r.json()
    results = data.get('results', [])
    # Only count unique work_ids
    unique_results = {}
    for res in results:
        work_id = None
        # Try to get work_id from metadata or top-level
        if isinstance(res, dict):
            work_id = res.get('work_id')
            if not work_id and 'metadata' in res:
                work_id = res['metadata'].get('work_id')
            if not work_id and 'metadata' in res and 'run' in res['metadata']:
                # fallback: use run as pseudo-id if work_id missing
                work_id = res['metadata']['run']
        if work_id is not None:
            unique_results[work_id] = res
    if len(unique_results) >= args.runs:
        print(f'[INFO] All {args.runs} unique results collected.')
        results = list(unique_results.values())
        break
    print(f'[INFO] {len(unique_results)}/{args.runs} unique results ready. Waiting...')
    if time.time() - start > args.timeout:
        print('[ERROR] Timeout waiting for results.')
        sys.exit(1)
    time.sleep(1)


# 3. Save results to distributed_aggregate.json (for direct aggregation)
with open('distributed_aggregate.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('[INFO] Saved all results to distributed_aggregate.json')


# 4. Directly aggregate results in Python (incorporated logic from aggregate_distributed_results.py)
from hf_models import load_qwen3_reranker
import numpy as np
from cli_gen_prime import call_ollama_generation, parse_translation_output, parse_backtranslation_output
from collections import Counter

# Only keep results with a non-empty 'japanese' field
valid_results = [r for r in results if r.get('japanese')]
if not valid_results:
    print("No valid results with 'japanese' field found.")
    sys.exit(1)

# Rerank using Ollama reranker (cosine similarity)
text = valid_results[0].get('input_text', '')
back_english_list = [r.get('backtranslation', '') or r.get('back_english', '') for r in valid_results]
reranker = load_qwen3_reranker()
embs = reranker([text] + back_english_list)
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
fuse_4_14 = [r['japanese'] for _, r in scored[3:14]]

# LLM Fusion of top 3
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
