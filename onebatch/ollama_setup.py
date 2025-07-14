# --- OLLAMA GENERATION (Qwen2.7:7b-Instruct) ---

import subprocess
import requests
import json

# Use Ollama for all tasks (generation, embedding, reranking)
OLLAMA_MODEL = "qwen2.5:7b-instruct"

def _ollama_generate(prompt, model=OLLAMA_MODEL, max_new_tokens=128, temperature=0.7):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
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
        return [{"generated_text": output.strip()}]
    except Exception as e:
        print(f"[ERROR] Ollama generation failed: {e}")
        return [{"generated_text": "[Ollama error: no output]"}]

def load_qwen3_generation(model_name=None, force_cpu=False):
    model = model_name if model_name is not None else OLLAMA_MODEL
    print(f"[INFO] Using Ollama for generation: {model}. Make sure Ollama is running and model is pulled.")
    def generate(prompt, max_new_tokens=128, temperature=0.7, **kwargs):
        return _ollama_generate(prompt, model=model, max_new_tokens=max_new_tokens, temperature=temperature)
    return generate

def load_qwen3_reranker(model_name=None, force_cpu=False):
    # Use Ollama for reranking: returns embeddings, use cosine similarity
    model = model_name if model_name is not None else OLLAMA_MODEL
    print(f"[INFO] Using Ollama for reranking/embeddings: {model}. Cosine similarity will be used.")
    def ollama_embed(texts):
        # Accepts a list of texts, returns list of embeddings
        url = "http://localhost:11434/api/embeddings"
        results = []
        for t in texts:
            payload = {"model": model, "prompt": t}
            try:
                resp = requests.post(url, json=payload, timeout=60)
                data = resp.json()
                if "embedding" in data:
                    results.append(data["embedding"])
                else:
                    print(f"[ERROR] Ollama embedding response missing 'embedding' key. Full response: {data}")
                    results.append([0.0]*1024)
            except Exception as e:
                print(f"[ERROR] Ollama embedding failed: {e}")
                results.append([0.0]*1024)
        return results
    return ollama_embed

def load_qwen3_embeddings(model_name=None, force_cpu=False):
    # Use Ollama for embeddings
    model = model_name if model_name is not None else OLLAMA_MODEL
    print(f"[INFO] Using Ollama for embeddings: {model}.")
    def ollama_embed(texts):
        url = "http://localhost:11434/api/embeddings"
        results = []
        for t in texts:
            payload = {"model": model, "prompt": t}
            try:
                resp = requests.post(url, json=payload, timeout=60)
                data = resp.json()
                if "embedding" in data:
                    results.append(data["embedding"])
                else:
                    print(f"[ERROR] Ollama embedding response missing 'embedding' key. Full response: {data}")
                    results.append([0.0]*1024)
            except Exception as e:
                print(f"[ERROR] Ollama embedding failed: {e}")
                results.append([0.0]*1024)
        return results
    return ollama_embed

if __name__ == "__main__":
    # Minimal embedding test for debugging
    print("[TEST] Running minimal embedding test for reranker and embeddings...")
    reranker = load_qwen3_reranker("qwen2.5:7b-instruct")
    embedder = load_qwen3_embeddings("qwen2.5:7b-instruct")
    test_texts = [
        "This is a test sentence.",
        "Another test sentence for embedding.",
        "日本語のテスト文です。"
    ]
    print("[TEST] Reranker embeddings:")
    print(reranker(test_texts))
    print("[TEST] Embedding function:")
    print(embedder(test_texts))
    print("[TEST] If you see nonzero vectors above, embedding works. If all zeros, Ollama embedding is broken or model does not support embeddings.")


