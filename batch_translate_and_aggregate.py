import subprocess
import json
from collections import Counter, defaultdict

# Settings
text = "Hello, I appreciate you joining and letting this translate for us."
model = "qwen2.5:7b-instruct"
runs = 31

results = []

# Run translate_one.py 31 times
for i in range(1, runs+1):
    proc = subprocess.run([
        'python3', 'translate_one.py', '--text', text, '--model', model, '--run', str(i)
    ], capture_output=True, text=True)
    try:
        result = json.loads(proc.stdout)
        results.append(result)
    except Exception as e:
        print(f"Error parsing run {i}: {e}\nOutput: {proc.stdout}")

# Build histogram of translations
translations = [r['translation'] for r in results]
hist = Counter(translations)
top_14 = hist.most_common(14)

# Find the 3rd best (by frequency)
third_best = hist.most_common(3)[-1] if len(hist) >= 3 else None

# Expanded occurrences: translation -> list of run numbers
occurrences = defaultdict(list)
for r in results:
    occurrences[r['translation']].append(r['metadata']['run'])

# Output summary
print("Top 14 translations:")
for t, count in top_14:
    print(f"{count}x: {t}")

if third_best:
    print("\n3rd best translation:")
    print(f"{third_best[1]}x: {third_best[0]}")

print("\nHistogram:", dict(hist))
print("\nExpanded occurrences (translation -> runs):")
for t, runs in occurrences.items():
    print(f"{t}: {runs}")

# Save all results to a file for further analysis
with open('batch_translations.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
