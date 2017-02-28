import subprocess
import urllib
import json
from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def healthcheck():
    return 'OK'

@app.route('/segment', methods=['POST'])
def segment():
    url = request.get_json(force=True)['url']
    urllib.urlretrieve(url, './model/image.png')

    code = subprocess.call([
        'python',
        'TensorBox/evaluate.py',
        '--weights', './model/save.ckpt-5000',
        '--test_boxes', './model/test_boxes.json',
        '--logdir', './model'
    ], stderr=subprocess.STDOUT)

    if code != 0:
        raise Exception('Failed to evaluate the model', code)

    with open('./model/save.ckpt-5000.test_boxes.json') as output:
        data = json.load(output)

    return jsonify(data[0]['rects'])

if __name__ == '__main__':
    app.run(port=80, host='0.0.0.0')
