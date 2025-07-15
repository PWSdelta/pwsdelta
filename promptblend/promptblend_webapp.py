from flask import Flask, render_template_string, request
from promptblend_generate import run_stat_sig_batch

app = Flask(__name__)

DEFAULT_PROMPT = "Translate all of the following English sentences to Japanese, preserving each sentence, as if you were speaking in a generally polite, but not overly formal, manner:"
DEFAULT_TEMP = 0.5

@app.route('/', methods=['GET', 'POST'])
def index():
    user_prompt = DEFAULT_PROMPT
    input_text = ''
    results = None
    if request.method == 'POST':
        user_prompt = request.form.get('user_prompt', DEFAULT_PROMPT)
        input_text = request.form.get('input_text', '')
        # Use the user prompt as the base for the batch
        # For now, just run the batch with the user prompt and input text
        # You can expand to allow more prompt variations if desired
        results = run_stat_sig_batch(input_text, temperature=DEFAULT_TEMP)
    return render_template_string('''
    <h2>PromptBlend 31/14/3 Experiment</h2>
    <form method="post">
        <label>Base Prompt:</label><br>
        <input type="text" name="user_prompt" value="{{ user_prompt }}" style="width:80%"><br><br>
        <label>Text to Translate:</label><br>
        <textarea name="input_text" rows="4" cols="80">{{ input_text }}</textarea><br><br>
        <input type="submit" value="Run 31/14/3 Batch">
    </form>
    {% if results %}
    <h3>Final Merged Translation</h3>
    <div style="border:1px solid #ccc;padding:10px;margin-bottom:10px;">{{ results['final_merged'] }}</div>
    <h4>Backtranslation</h4>
    <div style="border:1px solid #eee;padding:10px;margin-bottom:10px;">{{ results['final_merged_backtranslation'] }}</div>
    <h4>Top 3 Translations</h4>
    <table border="1" cellpadding="4">
        <tr><th>#</th><th>Japanese</th><th>Backtranslation</th></tr>
        {% for row in results['top_3'] %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ row['japanese'] }}</td>
            <td>{{ row['backtranslation'] }}</td>
        </tr>
        {% endfor %}
    </table>
    <h4>Top 14 Translations</h4>
    <table border="1" cellpadding="4">
        <tr><th>#</th><th>Japanese</th><th>Backtranslation</th></tr>
        {% for row in results['top_14'] %}
        <tr>
            <td>{{ loop.index }}</td>
            <td>{{ row['japanese'] }}</td>
            <td>{{ row['backtranslation'] }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
    ''', user_prompt=user_prompt, input_text=input_text, results=results)

if __name__ == '__main__':
    app.run(debug=True)
