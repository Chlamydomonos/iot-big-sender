from flask import Flask, request, send_file
from encode import encode, save_wave
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    text = data['text']
    wave = encode(text)
    save_wave(wave, 'output.wav')
    return {'success': True }


@app.route('/waves/<name>', methods=['GET'])
def get_wave(name):
    return send_file('output.wav')
