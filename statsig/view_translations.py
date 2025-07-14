from flask import Flask, render_template_string, send_from_directory
import json
import os

app = Flask(__name__)

@app.route('/')
def index():
    # Load CSV for all raw runs
    import csv
    csv_path = os.path.join(os.path.dirname(__file__), 'latest_translation.csv')
    csv_rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['temperature'] = float(row['temperature'])
            row['run'] = int(row['run'])
            row['hiragana'] = int(row['hiragana'])
            row['katakana'] = int(row['katakana'])
            row['kanji'] = int(row['kanji'])
            row['length'] = int(row['length'])
            csv_rows.append(row)

    # Load JSON for merged results only
    json_path = os.path.join(os.path.dirname(__file__), 'latest_translation.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Group CSV rows by temperature
    from collections import defaultdict
    temp_to_runs = defaultdict(list)
    for row in csv_rows:
        temp_to_runs[row['temperature']].append(row)

    # Build model_results for each temperature batch
    model_results = []
    for batch in json_data:
        temp = batch['temperature']
        runs = temp_to_runs[temp]
        # Sort by Japanese length (desc), best first
        runs_sorted = sorted(runs, key=lambda r: r['length'], reverse=True)
        winner = runs_sorted[0]
        top_3 = runs_sorted[:3]
        top_14 = runs_sorted[:14]
        # Use merged results from JSON
        merged = {
            'merged_14': batch.get('merged_14'),
            'merged_14_backtranslation': batch.get('merged_14_backtranslation'),
            'merged_3': batch.get('merged_3'),
            'merged_3_backtranslation': batch.get('merged_3_backtranslation'),
            'final_merged': batch.get('final_merged'),
            'final_merged_backtranslation': batch.get('final_merged_backtranslation')
        }
        model_results.append({
            'model': f"T={temp}",
            'input_text': batch.get('input_text', ''),
            'temperature': temp,
            'all_results': runs,
            'winner': winner,
            'top_3': top_3,
            'top_14': top_14,
            **merged
        })
    # Bootstrap HTML template with Bootstrap nav-pills for temperature selection
    app.jinja_env.filters['zip'] = zip
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
        <h1 class="mb-4 text-center">Translation Results</h1>
        <ul class="nav nav-pills mb-3 justify-content-center" id="temp-pills" role="tablist">
        {% for result in model_results %}
            <li class="nav-item" role="presentation">
                <button class="nav-link {% if loop.first %}active{% endif %}" id="pill-{{loop.index0}}" data-bs-toggle="pill" data-bs-target="#tab-{{loop.index0}}" type="button" role="tab" aria-controls="tab-{{loop.index0}}" aria-selected="{{ 'true' if loop.first else 'false' }}">
                    T={{result.temperature}}
                </button>
            </li>
        {% endfor %}
        </ul>
        <div class="tab-content" id="temp-pills-content">
        {% for result in model_results %}
        <div class="tab-pane fade {% if loop.first %}show active{% endif %}" id="tab-{{loop.index0}}" role="tabpanel" aria-labelledby="pill-{{loop.index0}}">
            <div class="row justify-content-center">
                <div class="col-12 col-lg-10 col-xl-8 mb-4">
                    <div class="card h-100">
                        <div class="card-header bg-primary text-white">
                            <h4 class="mb-0">T={{result.temperature}}</h4>
                        </div>
                        <div class="card-body">
                            <div class="prime mb-3 p-2">
                                <h5>Merged Results</h5>
                                <table class="table table-bordered table-sm mb-0">
                                    <tbody>
                                        <tr><th scope="row">Input</th><td>{{result.input_text}}</td></tr>
                                        <tr><th scope="row">Temperature</th><td>{{result.temperature}}</td></tr>
                                        <tr><th scope="row">Merged 14</th><td class="japanese">{{result.merged_14}}</td></tr>
                                        <tr><th scope="row">Merged 14 Backtranslation</th><td class="back-english">{{result.merged_14_backtranslation}}</td></tr>
                                        <tr><th scope="row">Merged 3</th><td class="japanese">{{result.merged_3}}</td></tr>
                                        <tr><th scope="row">Merged 3 Backtranslation</th><td class="back-english">{{result.merged_3_backtranslation}}</td></tr>
                                        <tr><th scope="row">Final Merged</th><td class="japanese">{{result.final_merged}}</td></tr>
                                        <tr><th scope="row">Final Merged Backtranslation</th><td class="back-english">{{result.final_merged_backtranslation}}</td></tr>
                                    </tbody>
                                </table>
                            </div>
                            <h6>Winner</h6>
                            <div class="table-responsive mb-3">
                            <table class="table table-sm table-bordered table-fullwidth centered-table">
                                <thead class="table-light">
                                    <tr>
                                        <th>Japanese</th>
                                        <th>Hiragana</th>
                                        <th>Katakana</th>
                                        <th>Kanji</th>
                                        <th>Length</th>
                                        <th>Backtranslation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td class="japanese">{{result.winner.japanese}}</td>
                                        <td>{{result.winner.hiragana}}</td>
                                        <td>{{result.winner.katakana}}</td>
                                        <td>{{result.winner.kanji}}</td>
                                        <td>{{result.winner.length}}</td>
                                        <td class="back-english">{{result.winner.backtranslation}}</td>
                                    </tr>
                                </tbody>
                            </table>
                            </div>
                            <h6>Top 3 Results</h6>
                            <div class="table-responsive mb-3">
                            <table class="table table-sm table-bordered table-fullwidth centered-table">
                                <thead class="table-light">
                                    <tr>
                                        <th>#</th>
                                        <th>Japanese</th>
                                        <th>Hiragana</th>
                                        <th>Katakana</th>
                                        <th>Kanji</th>
                                        <th>Length</th>
                                        <th>Backtranslation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                {% for r in result.top_3 %}
                                    <tr>
                                        <td>{{loop.index}}</td>
                                        <td class="japanese">{{r.japanese}}</td>
                                        <td>{{r.hiragana}}</td>
                                        <td>{{r.katakana}}</td>
                                        <td>{{r.kanji}}</td>
                                        <td>{{r.length}}</td>
                                        <td class="back-english">{{r.backtranslation}}</td>
                                    </tr>
                                {% endfor %}
                                </tbody>
                            </table>
                            </div>
                            <h6>Top 14 Results</h6>
                            <div class="table-responsive mb-3">
                            <table class="table table-sm table-bordered table-fullwidth centered-table">
                                <thead class="table-light">
                                    <tr>
                                        <th>#</th>
                                        <th>Japanese</th>
                                        <th>Hiragana</th>
                                        <th>Katakana</th>
                                        <th>Kanji</th>
                                        <th>Length</th>
                                        <th>Backtranslation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                {% for r in result.top_14 %}
                                    <tr>
                                        <td>{{loop.index}}</td>
                                        <td class="japanese">{{r.japanese}}</td>
                                        <td>{{r.hiragana}}</td>
                                        <td>{{r.katakana}}</td>
                                        <td>{{r.kanji}}</td>
                                        <td>{{r.length}}</td>
                                        <td class="back-english">{{r.backtranslation}}</td>
                                    </tr>
                                {% endfor %}
                                </tbody>
                            </table>
                            </div>
                            <h6>All 31 Runs</h6>
                            <div class="table-responsive mb-3">
                            <table class="table table-sm table-bordered table-hover table-fullwidth centered-table runs-table">
                                <thead class="table-light">
                                    <tr>
                                        <th>Run</th>
                                        <th>Japanese</th>
                                        <th>Hiragana</th>
                                        <th>Katakana</th>
                                        <th>Kanji</th>
                                        <th>Length</th>
                                        <th>Backtranslation</th>
                                    </tr>
                                </thead>
                                <tbody>
                                {% for run in result.all_results %}
                                    <tr>
                                        <td>{{run.run}}</td>
                                        <td class="japanese">{{run.japanese}}</td>
                                        <td>{{run.hiragana}}</td>
                                        <td>{{run.katakana}}</td>
                                        <td>{{run.kanji}}</td>
                                        <td>{{run.length}}</td>
                                        <td class="back-english">{{run.backtranslation}}</td>
                                    </tr>
                                {% endfor %}
                                </tbody>
                            </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        {% endfor %}
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.6/js/dataTables.bootstrap5.min.js"></script>
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/dataTables.bootstrap5.min.css"/>
    <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
    <script>
    $(document).ready(function() {
        $('.runs-table').DataTable({
            pageLength: 10,
            lengthMenu: [5, 10, 15, 31],
            order: [[ 0, 'asc' ]],
            columns: [
                { type: 'num' }, // Run
                { type: 'string' }, // Japanese
                { type: 'string' }, // Backtranslation
                { type: 'num' }, // Hiragana
                { type: 'num' }, // Katakana
                { type: 'num' }, // Kanji
                { type: 'num' }  // Length
            ]
        });
    });
    </script>
    </body>
    </html>
    '''
    return render_template_string(html, model_results=model_results)
    return render_template_string(html, model_results=model_results)

if __name__ == '__main__':
    app.run(debug=True)
