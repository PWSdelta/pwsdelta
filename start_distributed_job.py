import sys
import requests
import json

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Start distributed translation job')
    parser.add_argument('--text', required=True, help='Text to translate')
    parser.add_argument('--model', default='qwen2.5:7b-instruct', help='Model name')
    parser.add_argument('--runs', type=int, default=14, help='Number of runs')
    parser.add_argument('--server', default='http://localhost:5000', help='Flask server URL')
    args = parser.parse_args()

    payload = {
        'text': args.text,
        'model': args.model,
        'runs': args.runs
    }
    resp = requests.post(f'{args.server}/start-distributed', json=payload)
    if resp.status_code == 200:
        print('[INFO] Distributed job started:', resp.json())
    else:
        print('[ERROR] Failed to start distributed job:', resp.text)
        sys.exit(1)

if __name__ == '__main__':
    main()
