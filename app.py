
from flask import Flask, render_template_string, abort
import json
import os

app = Flask(__name__)

TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Batch Japanese Translations</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 2rem; }
        .table-responsive { margin-bottom: 2rem; }
        pre { white-space: pre-wrap; word-break: break-word; }
    </style>
</head>
<body>
<div class="container">
    <h1 class="mb-4">Batch Japanese Translations</h1>
    <h5>Input Text</h5>
    <div class="alert alert-secondary">{{ data['input_text'] }}</div>
    <h5>Temperature: <span class="badge bg-info">{{ data['temperature'] }}</span></h5>
    <div class="table-responsive">
        <table class="table table-striped table-bordered table-hover align-middle">
            <thead class="table-dark">
                <tr>
                    <th>Run</th>
                    <th>Japanese</th>
                    <th>Backtranslation</th>
                    <th>ひらがな</th>
                    <th>カタカナ</th>
                    <th>漢字</th>
                    <th>Length</th>
                </tr>
            </thead>
            <tbody>
            {% for r in data['all_results'] %}
                <tr>
                    <td>{{ r['run'] }}</td>
                    <td><pre>{{ r['japanese'] }}</pre></td>
                    <td><pre>{{ r['backtranslation'] }}</pre></td>
                    <td>{{ r['hiragana'] }}</td>
                    <td>{{ r['katakana'] }}</td>
                    <td>{{ r['kanji'] }}</td>
                    <td>{{ r['japanese']|length }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
    <h4 class="mt-4">Merged Results</h4>
    <div class="mb-3">
        <strong>Top 14 Merge:</strong>
        <div class="alert alert-success"><pre>{{ data['merged_14'] }}</pre></div>
    </div>
    <div class="mb-3">
        <strong>Top 3 Merge:</strong>
        <div class="alert alert-primary"><pre>{{ data['merged_3'] }}</pre></div>
    </div>
    <div class="mb-3">
        <strong>Final Merge:</strong>
        <div class="alert alert-warning"><pre>{{ data['final_merged'] }}</pre></div>
    </div>
</div>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
'''

@app.route("/")
def index():
    json_path = os.path.join(os.path.dirname(__file__), "latest_results.json")
    if not os.path.exists(json_path):
        abort(404, description="latest_results.json not found.")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return render_template_string(TEMPLATE, data=data)

if __name__ == "__main__":
    app.run(debug=True)
