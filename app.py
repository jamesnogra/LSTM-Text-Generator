from flask import Flask, request, Response, render_template
from flask.json import jsonify
from flask_cors import CORS #pip install -U flask-cors
from serve import get_model_api

app = Flask(__name__)
CORS(app) # needed for cross-domain requests, allow everything by default
model_api = get_model_api()

# form page
@app.route('/', methods=['GET'])
def main():
	return render_template('index.html')

# API route
@app.route('/api', methods=['POST'])
def api():
	seed_text = request.form.get('seed_text')
	result = model_api(seed_text)
	return result

if __name__ == '__main__':
	app.run(debug=True, port=8080, host='0.0.0.0')