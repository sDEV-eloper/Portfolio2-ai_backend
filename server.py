from flask import Flask, request, jsonify
from flask_cors import CORS
from main import modify_input

app = Flask(__name__)
CORS(app)

@app.route('/api/home', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
        input_text = data.get('inputData') # Use 'inputData' as the key
        trained_text = modify_input(input_text)
        return jsonify({'inputData': trained_text})  # Use 'inputData' as the key

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8080)
