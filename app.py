from flask import Flask, request, jsonify, render_template_string
import subprocess
import json
import os
import threading

app = Flask(__name__)

# --- Distributed Work Queue (in-memory for demo) ---
work_queue = []
work_results = {}
work_id_counter = [1]
work_lock = threading.Lock()

def add_translation_jobs(text, model, runs=14):
    with work_lock:
        for run in range(1, runs+1):
            work_id = work_id_counter[0]
            work_id_counter[0] += 1
            work_queue.append({'work_id': work_id, 'text': text, 'model': model, 'run': run})

@app.route('/get-work', methods=['GET'])
def get_work():
    with work_lock:
        if work_queue:
            job = work_queue.pop(0)
            return jsonify(job)
        else:
            return jsonify({'work_id': None})

@app.route('/submit-translation', methods=['POST'])
def submit_translation():
    data = request.get_json(force=True)
    work_id = data.get('work_id')
    result = data.get('result')
    if not work_id or not result:
        return jsonify({'error': 'Missing work_id or result'}), 400
    with work_lock:
        work_results[work_id] = result
    # Optionally, trigger more work or aggregation here
    return jsonify({'status': 'ok'})

@app.route('/start-distributed', methods=['POST'])
def start_distributed():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    model = data.get('model', 'qwen2.5:7b-instruct')
    runs = int(data.get('runs', 14))
    if not text:
        return jsonify({'error': 'No text provided.'}), 400
    add_translation_jobs(text, model, runs)
    return jsonify({'status': 'jobs added', 'jobs': runs})

@app.route('/distributed-results', methods=['GET'])
def distributed_results():
    # Return all collected results so far
    with work_lock:
        results = list(work_results.values())
    return jsonify({'results': results, 'count': len(results)})

@app.route('/generate', methods=['POST'])
def generate_translation():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided.'}), 400
    try:
        result = subprocess.run(
            ['python', 'cli_gen_prime.py'],
            input=f'1\n{text}\n0\n',
            capture_output=True, text=True, timeout=600
        )
        json_path = os.path.join(os.path.dirname(__file__), 'latest_translation.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            output = json.load(f)
        return jsonify({'cli_stdout': result.stdout, 'cli_stderr': result.stderr, 'result': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def view_results():
    json_path = os.path.join(os.path.dirname(__file__), 'latest_translation.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    model_results = data.get('models')
    if not model_results:
        model_results = [{'model': data.get('prime_translation', {}).get('model', 'unknown'), 'result': data}]
    app.jinja_env.filters['zip'] = zip

    # UI for style/tone selection and comparison
    styles = [
        'basic', 'polite', 'casual', 'formal', 'business', 'youthful', 'elderly', 'feminine', 'masculine', 'kansai', 'samurai', 'anime', 'robotic', 'poetic', 'concise', 'verbose'
    ]
    selected_style = request.form.get('style', 'basic')
    selected_run = int(request.form.get('run', '1'))
    # Find the first model's all_runs for demo
    all_runs = model_results[0]['result'].get('all_runs', [])
    base_jp = all_runs[selected_run-1]['japanese'] if all_runs and 1 <= selected_run <= len(all_runs) else ''
    # Generate specialized translations for the selected style
    specialized_jp = ''
    diff = ''
    if request.method == 'POST' and base_jp:
        # Use the LLM to generate a specialized translation (simulate)
        # In real use, call the backend or Ollama for this
        specialized_jp = f"[{selected_style.capitalize()} style] {base_jp}"
        # Compute a simple diff (for demo, just show both)
        diff = f"<b>Base:</b> {base_jp}<br><b>{selected_style.capitalize()}:</b> {specialized_jp}"

    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Translation Results</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
        <style>
            body { background: #f8f9fa; }
            .japanese { font-family: 'Noto Sans JP', sans-serif; font-size: 1.2em; }
            .back-english { color: #555; }
            .prime { background: #e3fcec; border-left: 5px solid #28a745; }
            .table-fullwidth { width: 100% !important; table-layout: auto; }
            .card-body { overflow-x: auto; }
            .centered-table { margin-left: auto; margin-right: auto; }
        </style>
    </head>
    <body>
    <div class="container my-4">
        <h1 class="mb-4 text-center">Translation Results (Multi-Model)</h1>
        <form method="post" class="mb-4">
            <div class="row g-2 align-items-end">
                <div class="col-auto">
                    <label for="run" class="form-label">Select Run:</label>
                    <select name="run" id="run" class="form-select">
                        {% for run in range(1, all_runs|length+1) %}
                        <option value="{{run}}" {% if run == selected_run %}selected{% endif %}>{{run}}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="col-auto">
                    <label for="style" class="form-label">Select Style/Tone:</label>
                    <select name="style" id="style" class="form-select">
                        {% for s in styles %}
                        <option value="{{s}}" {% if s == selected_style %}selected{% endif %}>{{s.capitalize()}}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="col-auto">
                    <button type="submit" class="btn btn-success">Generate Specialized Translation</button>
                </div>
            </div>
        </form>
        {% if diff %}
        <div class="alert alert-info">{{diff|safe}}</div>
        {% endif %}
        <div class="row justify-content-center">
        {% for model_result in model_results %}
            {% set model = model_result.model %}
            {% set res = model_result.result %}
            {% set prime = res.prime_translation %}
            {% set top3 = prime.top_japanese|zip(prime.top_back_english) %}
            <div class="col-12 col-lg-10 col-xl-8 mb-4">
                <div class="card h-100">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">Model: {{model}}</h4>
                    </div>
                    <div class="card-body">
                        <div class="prime mb-3 p-2">
                            <h5>Prime Translation</h5>
                            <table class="table table-bordered table-sm mb-0">
                                <tbody>
                                    <tr>
                                        <th scope="row">Input</th>
                                        <td>{{prime.input_text}}</td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Model</th>
                                        <td>{{prime.model}}</td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Prime Japanese</th>
                                        <td class="japanese">{{prime.japanese}}</td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Prime Back-translation</th>
                                        <td class="back-english">{{prime.back_english}}</td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Top 3 Fused</th>
                                        <td class="japanese">{{prime.top3_fused}}</td>
                                    </tr>
                                    <tr>
                                        <th scope="row">4th-14th Fused</th>
                                        <td class="japanese">{{prime['4_14_fused']}}</td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Top 3 Japanese</th>
                                        <td>
                                            <ol class="mb-0">
                                            {% for jp in prime.top_japanese %}
                                                <li class="japanese">{{jp}}</li>
                                            {% endfor %}
                                            </ol>
                                        </td>
                                    </tr>
                                    <tr>
                                        <th scope="row">Top 3 Back-translation</th>
                                        <td>
                                            <ol class="mb-0">
                                            {% for en in prime.top_back_english %}
                                                <li class="back-english">{{en}}</li>
                                            {% endfor %}
                                            </ol>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                        <h6>All Runs</h6>
                        <div class="table-responsive">
                        <table class="table table-sm table-bordered table-fullwidth centered-table">
                            <thead class="table-light">
                                <tr>
                                    <th>Run</th>
                                    <th>Japanese</th>
                                    <th>Back-translation</th>
                                </tr>
                            </thead>
                            <tbody>
                            {% for run in res.all_runs %}
                                <tr>
                                    <td>{{run.run}}</td>
                                    <td class="japanese">{{run.japanese}}</td>
                                    <td class="back-english">{{run.back_english}}</td>
                                </tr>
                            {% endfor %}
                            </tbody>
                        </table>
                        </div>
                        <h6>Translation Histogram</h6>
                        <div class="row">
                            <div class="col-6">
                                <h6>Japanese</h6>
                                <ul>
                                {% for jp, count in res.translation_histogram.japanese.items() %}
                                    <li><span class="japanese">{{jp}}</span> <span class="badge bg-primary">{{count}}</span></li>
                                {% endfor %}
                                </ul>
                            </div>
                            <div class="col-6">
                                <h6>Back-translation</h6>
                                <ul>
                                {% for en, count in res.translation_histogram.back_english.items() %}
                                    <li><span class="back-english">{{en}}</span> <span class="badge bg-secondary">{{count}}</span></li>
                                {% endfor %}
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        {% endfor %}
        </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    </body>
    </html>
    '''
    return render_template_string(html, model_results=model_results, styles=styles, selected_style=selected_style, selected_run=selected_run, all_runs=all_runs, diff=diff)

if __name__ == '__main__':
    app.run(debug=True)
