from flask import Flask, render_template_string, send_from_directory
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Load the latest_translation.json file
    json_path = os.path.join(os.path.dirname(__file__), 'latest_translation.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # If multi-model format, use data['models'], else fallback to old format
    model_results = data.get('models')
    if not model_results:
        # fallback: wrap old format for backward compatibility
        model_results = [{'model': data.get('prime_translation', {}).get('model', 'unknown'), 'result': data}]

    # Bootstrap HTML template
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
        </style>
    </head>
    <body>
    <div class="container my-4">
        <h1 class="mb-4">Translation Results (Multi-Model)</h1>
        <div class="row">
        {% for model_result in model_results %}
            {% set model = model_result.model %}
            {% set res = model_result.result %}
            {% set prime = res.prime_translation %}
            {% set top3 = prime.top_japanese|zip(prime.top_back_english) %}
            <div class="col-lg-6 col-xl-4 mb-4">
                <div class="card h-100">
                    <div class="card-header bg-primary text-white">
                        <h4 class="mb-0">Model: {{model}}</h4>
                    </div>
                    <div class="card-body">
                        <div class="prime mb-3 p-2">
                            <h5>Prime Translation</h5>
                            <div><b>Input:</b> {{prime.input_text}}</div>
                            <div class="japanese"><b>Japanese:</b> {{prime.japanese}}</div>
                            <div class="back-english"><b>Back-translation:</b> {{prime.back_english}}</div>
                        </div>
                        <div class="mb-3">
                            <h6>Top 3 Japanese & Back-Translations</h6>
                            <ol>
                            {% for jp, en in top3 %}
                                <li>
                                    <div class="japanese"><b>Japanese:</b> {{jp}}</div>
                                    <div class="back-english"><b>Back-translation:</b> {{en}}</div>
                                </li>
                            {% endfor %}
                            </ol>
                        </div>
                        <h6>All Runs</h6>
                        <table class="table table-sm table-bordered">
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
    return render_template_string(html, model_results=model_results)

if __name__ == '__main__':
    app.run(debug=True)
