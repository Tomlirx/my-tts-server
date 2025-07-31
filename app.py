import os
from flask import Flask, request, send_file, jsonify
from piper.voice import PiperVoice
from pathlib import Path

app = Flask(__name__)

# --- Model Loading ---
model_path = Path(__file__).parent / 'models' / 'en_US-ljspeech-medium.onnx'
voice = None
try:
    voice = PiperVoice.load(str(model_path))
    print("Voice model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load voice model. Error: {e}")

# --- API Endpoint ---
@app.route('/generate-speech', methods=['POST'])
def generate_speech():
    if not voice:
        return jsonify({"error": "Server error: Voice model not loaded"}), 500

    data = request.get_json()
    text_to_speak = data.get('text', '')
    if not text_to_speak:
        return jsonify({"error": "No text provided"}), 400

    output_path = Path('output.wav')
    try:
        with output_path.open("wb") as wav_file:
            voice.synthesize(text_to_speak, wav_file)
        return send_file(output_path, mimetype='audio/wav')
    except Exception as e:
        print(f"Error during audio synthesis: {e}")
        return jsonify({"error": "Failed to generate audio"}), 500
    finally:
        if output_path.exists():
            os.remove(output_path)

# --- Health Check for Render ---
@app.route('/')
def health_check():
    return "TTS Server is running.", 200

# --- Server Start ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)