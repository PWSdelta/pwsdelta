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

# 2. Poll for results
start = time.time()
while True:
    r = requests.get(f'{args.server}/distributed-results')
    if r.status_code != 200:
        print('[ERROR] Failed to get results:', r.text)
        sys.exit(1)
    data = r.json()
    results = data.get('results', [])
    if len(results) >= args.runs:
        print(f'[INFO] All {args.runs} results collected.')
        break
    print(f'[INFO] {len(results)}/{args.runs} results ready. Waiting...')
    if time.time() - start > args.timeout:
        print('[ERROR] Timeout waiting for results.')
        sys.exit(1)
    time.sleep(3)

# 3. Save results
with open('batch_translations.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print('[INFO] Saved all results to batch_translations.json')
