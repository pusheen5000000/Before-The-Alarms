import json, urllib.request, uuid, os, numpy as np

def post_wav(path, url='http://127.0.0.1:5000/predict'):
    boundary = uuid.uuid4().hex
    with open(path, 'rb') as fh:
        payload = fh.read()
    body = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(path)}"\r\n'
        f'Content-Type: audio/wav\r\n\r\n'
    ).encode() + payload + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(url, data=body, method='POST')
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req, timeout=120) as res:
        return res.status, json.loads(res.read())

tests = [
    (r'1\fold1\102305-6-0-0.wav', 'gun_shot'),
    (r'1\fold10\100648-1-0-0.wav', 'air_conditioner'),
    (r'1\fold10\100648-1-1-0.wav', 'air_conditioner #2'),
    (r'1\fold1\103074-7-0-0.wav', 'jackhammer'),
    (r'1\fold1\103074-7-1-0.wav', 'jackhammer #2'),
    (r'1\fold2\104817-4-0-0.wav', 'drilling'),
    (r'1\fold1\101415-3-0-2.wav', 'dog_bark'),
    (r'1\fold1\101415-3-0-3.wav', 'dog_bark #2'),
    (r'1\fold1\102842-3-0-1.wav', 'dog_bark #3'),
    (r'1\fold10\100795-3-0-0.wav', 'dog_bark fold10'),
    (r'1\fold10\101382-2-0-10.wav', 'children_playing'),
    (r'1\fold10\102857-5-0-0.wav', 'engine_idling'),
    (r'1\fold10\102857-5-0-1.wav', 'engine_idling #2'),
]

results = {"background": 0, "gunshot": 0, "chainsaw": 0, "firework": 0, "vehicle": 0}
for path, note in tests:
    status, data = post_wav(path)
    cls = data['class']
    conf = data['confidence']
    results[cls] = results.get(cls, 0) + 1
    veh = data['scores'].get('vehicle', 0)
    gun = data['scores'].get('gunshot', 0)
    top3 = [(d['label'], round(d['score'], 3)) for d in data['top_labels'][:3]]
    print(f'{note:22s} -> {cls:12s} conf={conf:.3f}  veh={veh:.3f} gun={gun:.3f}  yamnet={top3}')

print(f'\nSummary: {results}')
