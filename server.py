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

THRESH = 0.1  # The threshold for a box to be considered a positive detection
SESSION_TTL_SECONDS = 60*2

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


@app.route('/predict', methods=['POST'])
def predict():
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

    global detection_times, llm_call_times
    if len(result) > 0:
        detection_times.append(time.time())
        detection_times = [t for t in detection_times if time.time() - t < 30]
        if len(detection_times) >= 3:
            # Check LLM rate limiting (max once per minute)
            current_time = time.time()
            llm_call_times = [t for t in llm_call_times if current_time - t < 60]  # Keep only calls within last minute

            if len(llm_call_times) == 0:
                print("Calling LLM for 3D print analysis")
                llm_call_times.append(current_time)
                # Call LLM
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                llm_response = requests.post('https://openrouter.ai/api/v1/chat/completions', headers={
                    'Authorization': f'Bearer {environ.get("OPENROUTER_API_KEY")}',
                    'Content-Type': 'application/json'
                }, json={
                    'model': 'x-ai/grok-4-fast:free',
                    'messages': [{
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': "Est-ce que tu identifies un problème en particulier avec cette pièce imprimée en 3D?\nNe soit pas trop strict et ne voit pas des problèmes là où il n'y en a pas. Si tu ne vois même pas l'objet réponds qu'il n'y a pas de problème.\n\nRéponds de cette manière SEULEMENT :\n\nhasproblem: \"OUI/NON\"\n\nSi \"OUI\", alors on ajoute :\nname: \"Nom du problème\"\ncause: \"Cause potentielle du problème\"\nsolution: \"Solution au problème\""},
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
                        return jsonify({'problems': [{'name': name, 'cause': cause, 'solution': solution}], 'stop': True})
                    else:
                        print("No problem detected by LLM")
                        detection_times.clear()
                else:
                    print(f"LLM failed: {llm_response.text}")
                    detection_times.clear()

    return jsonify(result)

@app.route('/', methods=['GET'])
def index():
    return send_from_directory('.', 'index.html')

@app.route('/hc/', methods=['GET'])
def health_check():
    return 'ok' if net_main is not None else 'error'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, threaded=False)
