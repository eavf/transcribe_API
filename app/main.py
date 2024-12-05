import os
from flask import Flask, request, jsonify
import speech_recognition as sr
from moviepy.editor import AudioFileClip

app = Flask(__name__)

# Endpoint to transcribe audio file
@app.route('/transcribe/audio', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(file)
        with audio_file as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return jsonify({'transcription': text})
        except sr.UnknownValueError:
            return jsonify({'error': 'Google Speech Recognition could not understand the audio'}), 400
        except sr.RequestError as e:
            return jsonify({'error': f'Could not request results from Google Speech Recognition service; {e}'}), 500

# Endpoint to transcribe audio from video file
@app.route('/transcribe/video', methods=['POST'])
def transcribe_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        video_path = os.path.join('/tmp', file.filename)
        file.save(video_path)
        audio_path = os.path.join('/tmp', 'audio.wav')
        clip = AudioFileClip(video_path)
        clip.audio.write_audiofile(audio_path)
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(audio_path)
        with audio_file as source:
            audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return jsonify({'transcription': text})
        except sr.UnknownValueError:
            return jsonify({'error': 'Google Speech Recognition could not understand the audio'}), 400
        except sr.RequestError as e:
            return jsonify({'error': f'Could not request results from Google Speech Recognition service; {e}'}), 500
        finally:
            os.remove(video_path)
            os.remove(audio_path)

# Endpoint to transcribe audio with quality parameter
@app.route('/transcribe/quality', methods=['POST'])
def transcribe_quality():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    quality = request.form.get('quality', 'standard')
    if file:
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(file)
        with audio_file as source:
            audio_data = recognizer.record(source)
        try:
            text = transcribe_with_quality(audio_data, quality)
            return jsonify({'transcription': text})
        except sr.UnknownValueError:
            return jsonify({'error': 'Google Speech Recognition could not understand the audio'}), 400
        except sr.RequestError as e:
            return jsonify({'error': f'Could not request results from Google Speech Recognition service; {e}'}), 500

# Function to handle different models based on quality
def transcribe_with_quality(audio_data, quality):
    recognizer = sr.Recognizer()
    if quality == 'high':
        # Use a high-quality model (e.g., Google Web Speech API)
        return recognizer.recognize_google(audio_data)
    elif quality == 'medium':
        # Use a medium-quality model (e.g., Google Web Speech API with less confidence)
        return recognizer.recognize_google(audio_data, show_all=True)['alternative'][0]['transcript']
    else:
        # Use a standard model (e.g., Google Web Speech API with default settings)
        return recognizer.recognize_google(audio_data)

# Endpoint to transcribe audio from microphone
@app.route('/transcribe/microphone', methods=['POST'])
def transcribe_microphone():
    quality = request.form.get('quality', 'standard')
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio_data = recognizer.listen(source)
    try:
        text = transcribe_with_quality(audio_data, quality)
        return jsonify({'transcription': text})
    except sr.UnknownValueError:
        return jsonify({'error': 'Google Speech Recognition could not understand the audio'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Could not request results from Google Speech Recognition service; {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
