import sys
import argparse
from hf_models import load_qwen3_generation
import json

def translate_one(text, model_name, run):
    prompt = f"Translate all of the following English sentences to Japanese, preserving each sentence, as if you were speaking in a generally polite, but not overly formal, manner:\n\n{text}"
    gen = load_qwen3_generation(model_name=model_name)
    jp = gen(prompt)[0]['generated_text']
    back_prompt = f"Translate this to English:\n\n{jp}"
    en = gen(back_prompt)[0]['generated_text']
    result = {
        'prompt': prompt,
        'english_native': text,
        'translation': jp.strip(),
        'backtranslation': en.strip(),
        'metadata': {
            'model': model_name,
            'run': run,
            'back_prompt': back_prompt
        }
    }
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate one run (for distributed worker)')
    parser.add_argument('--text', required=True)
    parser.add_argument('--model', default='qwen2.5:7b-instruct')
    parser.add_argument('--run', type=int, default=1)
    args = parser.parse_args()
    result = translate_one(args.text, args.model, args.run)
    print(json.dumps(result, ensure_ascii=False))
