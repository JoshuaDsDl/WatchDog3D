#!/usr/bin/env python

import flask
from flask_compress import Compress
from flask import abort, make_response, request, jsonify, send_from_directory
from os import path, environ
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import cv2
import numpy as np
import requests
import time
import base64

from lib.detection_model import load_net, detect

THRESH = 0.02  # The threshold for a box to be considered a positive detection
SESSION_TTL_SECONDS = 60*2
llm_rate_limit_seconds = 30
auto_analysis_enabled = True
auto_analysis_interval_minutes = 3
llm_model = 'openai/gpt-5-nano'
llm_prompt = "Est-ce que tu identifies un problème en particulier avec cette pièce imprimée en 3D?\nNe parle pas de l'imprimante en elle même et ne soit pas trop strict et ne voit pas des problèmes là où il n'y en a pas. Si tu réponds Oui, ça aura des conséquences il faut que tu sois certain. Si tu ne vois même pas l'objet réponds qu'il n'y a pas de problème.\n\nRéponds de cette manière SEULEMENT :\n\nhasproblem: \"OUI/NON\"\n\nSi \"OUI\", alors on ajoute :\nname: \"Nom du problème\"\ncause: \"Cause potentielle du problème\"\nsolution: \"Solution au problème\""
last_auto_analysis_time = 0

# Sentry
if environ.get('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=environ.get('SENTRY_DSN'),
        integrations=[FlaskIntegration(), ],
    )

app = flask.Flask(__name__)
Compress(app)

status = dict()
detection_times = []
llm_call_times = []

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = environ.get('DEBUG') == 'True'

model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main = load_net(path.join(model_dir, 'model.cfg'), path.join(model_dir, 'model.meta'))


# Store for pending analysis
pending_analysis = {'active': False, 'image': None, 'is_auto': False}

@app.route('/predict', methods=['POST'])
def predict():
    global detection_times, llm_call_times, THRESH, llm_rate_limit_seconds, pending_analysis, auto_analysis_enabled, auto_analysis_interval_minutes, last_auto_analysis_time
    if 'image' not in request.files:
        return jsonify([]), 400
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    detections = detect(net_main, img, thresh=THRESH)
    print(f"Detections: {detections}")
    scale_x = 640 / width
    scale_y = 480 / height
    result = []
    for label, conf, (xc, yc, w, h) in detections:
        if conf > THRESH:
            x1 = (xc - w / 2) * scale_x
            y1 = (yc - h / 2) * scale_y
            x2 = (xc + w / 2) * scale_x
            y2 = (yc + h / 2) * scale_y
            result.append({'box': [x1, y1, x2, y2], 'label': label, 'confidence': conf})
    
    current_time = time.time()
    
    # Check for defect-based analysis
    if len(result) > 0:
        detection_times.append(current_time)
        detection_times = [t for t in detection_times if current_time - t < 30]
        if len(detection_times) >= 3:
            # Check LLM rate limiting
            llm_call_times = [t for t in llm_call_times if current_time - t < llm_rate_limit_seconds]

            if len(llm_call_times) == 0 and not pending_analysis['active']:
                print("Triggering defect-based deep analysis")
                # Store image for analysis
                _, buffer = cv2.imencode('.jpg', img)
                pending_analysis = {
                    'active': True,
                    'image': base64.b64encode(buffer).decode('utf-8'),
                    'is_auto': False
                }
                # Return signal to start analysis overlay
                return jsonify({'deep_analysis_starting': True, 'is_auto': False})
    
    # Check for automatic periodic analysis
    if auto_analysis_enabled and not pending_analysis['active']:
        time_since_last_auto = current_time - last_auto_analysis_time
        if time_since_last_auto >= (auto_analysis_interval_minutes * 60):
            print(f"Triggering automatic preventive analysis (last: {time_since_last_auto:.0f}s ago)")
            # Store image for analysis
            _, buffer = cv2.imencode('.jpg', img)
            pending_analysis = {
                'active': True,
                'image': base64.b64encode(buffer).decode('utf-8'),
                'is_auto': True
            }
            # Return signal to start analysis overlay
            return jsonify({'deep_analysis_starting': True, 'is_auto': True})

    return jsonify(result)

@app.route('/analyze', methods=['POST'])
def analyze():
    global pending_analysis, detection_times, llm_call_times, last_auto_analysis_time, llm_model, llm_prompt
    
    if not pending_analysis['active']:
        return jsonify({'error': 'No pending analysis'}), 400
    
    img_base64 = pending_analysis['image']
    is_auto = pending_analysis['is_auto']
    pending_analysis = {'active': False, 'image': None, 'is_auto': False}
    
    # Update last auto analysis time if this was an auto analysis
    if is_auto:
        last_auto_analysis_time = time.time()
    
    print("Calling LLM for 3D print analysis")
    try:
        llm_response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers={
            'Authorization': f'Bearer {environ.get("OPENROUTER_API_KEY")}',
            'Content-Type': 'application/json'
        }, json={
            'model': llm_model,
            'messages': [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': llm_prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_base64}'}}
                ]
            }]
        })
        print(f"LLM response status: {llm_response.status_code}")
        if llm_response.status_code == 200:
            content = llm_response.json()['choices'][0]['message']['content']
            print(f"LLM content: {content}")
            lines = content.split('\n')
            hasproblem = None
            name = None
            cause = None
            solution = None
            for line in lines:
                if line.startswith('hasproblem:'):
                    hasproblem = line.split(':', 1)[1].strip().strip('"')
                elif line.startswith('name:'):
                    name = line.split(':', 1)[1].strip().strip('"')
                elif line.startswith('cause:'):
                    cause = line.split(':', 1)[1].strip().strip('"')
                elif line.startswith('solution:'):
                    solution = line.split(':', 1)[1].strip().strip('"')
            print(f"Parsed: hasproblem={hasproblem}, name={name}, cause={cause}, solution={solution}")
            if hasproblem == 'OUI':
                print("Returning problems")
                detection_times.clear()
                llm_call_times.append(time.time())
                return jsonify({'problems': [{'name': name, 'cause': cause, 'solution': solution}], 'deep_analysis': True})
            else:
                print("No problem detected by LLM")
                detection_times.clear()
                llm_call_times.append(time.time())
                return jsonify({'deep_analysis': True})
        else:
            print(f"LLM failed: {llm_response.text}")
            detection_times.clear()
            return jsonify({'deep_analysis': True})
    except Exception as e:
        print(f"LLM error: {e}")
        detection_times.clear()
        return jsonify({'deep_analysis': True, 'error': str(e)})

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>', methods=['GET'])
def serve_static(filename):
    return send_from_directory('.', filename)

@app.route('/hc/', methods=['GET'])
def health_check():
    return 'ok' if net_main is not None else 'error'

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global THRESH, llm_rate_limit_seconds, auto_analysis_enabled, auto_analysis_interval_minutes, llm_model, llm_prompt
    if request.method == 'POST':
        data = request.get_json()
        THRESH = data.get('thresh', THRESH)
        llm_rate_limit_seconds = data.get('llm_rate_limit', llm_rate_limit_seconds)
        auto_analysis_enabled = data.get('auto_analysis_enabled', auto_analysis_enabled)
        auto_analysis_interval_minutes = data.get('auto_analysis_interval', auto_analysis_interval_minutes)
        llm_model = data.get('llm_model', llm_model)
        llm_prompt = data.get('llm_prompt', llm_prompt)
        return jsonify({'status': 'ok'})
    else:
        return jsonify({
            'thresh': THRESH,
            'llm_rate_limit': llm_rate_limit_seconds,
            'auto_analysis_enabled': auto_analysis_enabled,
            'auto_analysis_interval': auto_analysis_interval_minutes,
            'llm_model': llm_model,
            'llm_prompt': llm_prompt
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, threaded=False)
