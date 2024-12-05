import os
from flask import Flask, request, jsonify
import speech_recognition as sr
from moviepy.editor import AudioFileClip
import numpy as np

import wave
import deepspeech
import azure.cognitiveservices.speech as speechsdk


app = Flask(__name__)

# Load DeepSpeech model
deepspeech_model_path = 'models/deepspeech-0.9.3-models.pbmm'
deepspeech_scorer_path = 'models/deepspeech-0.9.3-models.scorer'
deepspeech_model = deepspeech.Model(deepspeech_model_path)
deepspeech_model.enableExternalScorer(deepspeech_scorer_path)

# Azure Speech Service configuration
azure_speech_key = 'YOUR_AZURE_SPEECH_KEY'
azure_service_region = 'YOUR_AZURE_SERVICE_REGION'


@app.route('/transcribe/model', methods=['POST'])

def transcribe_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    model = request.form.get('model', 'google')
    quality = request.form.get('quality', 'standard')

    if file:
        # Save the uploaded file temporarily
        audio_file_path = f'temp_{file.filename}'
        file.save(audio_file_path)

        try:
            if model == 'deepspeech':
                # Use DeepSpeech for transcription
                transcription = transcribe_with_deepspeech(audio_file_path)
                return jsonify({'transcription': transcription})

            elif model == 'azure':
                # Use Microsoft Azure Speech Service for transcription
                transcription = transcribe_with_azure(audio_file_path)
                return jsonify({'transcription': transcription})

            else:
                return jsonify({'error': 'Invalid model selected'}), 400

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up the temporary file
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)


def transcribe_with_deepspeech(audio_file):
    # Load the audio file
    with wave.open(audio_file, 'rb') as wf:
        # Ensure the audio file is in the correct format (16-bit PCM, mono, 16kHz)
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio file must be mono PCM WAV with 16kHz sample rate.")

        # Read the audio data
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)

    # Perform inference using DeepSpeech
    text = deepspeech_model.stt(audio_data)
    return text


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


def transcribe_with_deepspeech(audio_file):
    # Load the audio file
    with wave.open(audio_file, 'rb') as wf:
        # Ensure the audio file is in the correct format (16-bit PCM, mono, 16kHz)
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
            raise ValueError("Audio file must be mono PCM WAV with 16kHz sample rate.")

        # Read the audio data
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)

    # Perform inference using DeepSpeech
    text = deepspeech_model.stt(audio_data)
    return text


def transcribe_with_azure(audio_file):
    # Set up Azure Speech Service
    speech_config = speechsdk.SpeechConfig(subscription=azure_speech_key, region=azure_service_region)
    audio_input = speechsdk.AudioConfig(filename=audio_file)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)

    # Start recognition
    result = speech_recognizer.recognize_once()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    else:
        return "Could not recognize speech"
    

if __name__ == '__main__':
    app.run(debug=True)
