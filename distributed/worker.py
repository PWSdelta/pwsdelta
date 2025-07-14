import requests
import time
import sys
import json
import subprocess

def get_work(server_url):
    try:
        resp = requests.get(f'{server_url}/get-work')
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception as e:
        print(f"[WORKER] Error getting work: {e}")
        return None

def submit_translation(server_url, work_id, translation_result):
    try:
        # Ensure work_id is included in the result dict for uniqueness
        if isinstance(translation_result, dict):
            translation_result['work_id'] = work_id
        resp = requests.post(f'{server_url}/submit-translation', json={
            'work_id': work_id,
            'result': translation_result
        })
        return resp.status_code == 200
    except Exception as e:
        print(f"[WORKER] Error submitting translation: {e}")
        return False

def do_translation(task):
    text = task['text']
    run = task['run']
    model = task['model']
    # Call translate_one.py as a subprocess
    try:
        proc = subprocess.run([
            sys.executable, 'translate_one.py',
            '--text', text,
            '--model', model,
            '--run', str(run)
        ], capture_output=True, text=True, timeout=600)
        # The script prints a dict, so eval is safe here (or use json if you change print to json)
        out = proc.stdout.strip()
        if out.startswith('{') and out.endswith('}'):  # crude check
            result = eval(out)
        else:
            result = {'japanese': '', 'run': run, 'model': model, 'input_text': text, 'error': out}
        return result
    except Exception as e:
        return {'japanese': '', 'run': run, 'model': model, 'input_text': text, 'error': str(e)}

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python worker.py <server_url>")
        sys.exit(1)
    server_url = sys.argv[1]
    print(f"[WORKER] Starting worker for server: {server_url}")
    while True:
        task = get_work(server_url)
        if not task or not task.get('work_id'):
            print("[WORKER] No work available. Sleeping...")
            time.sleep(5)
            continue
        print(f"[WORKER] Got work: {task}")
        result = do_translation(task)
        success = submit_translation(server_url, task['work_id'], result)
        if success:
            print(f"[WORKER] Submitted result for work_id {task['work_id']}")
        else:
            print(f"[WORKER] Failed to submit result for work_id {task['work_id']}")
        time.sleep(1)
