import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc

import os, sys
tensor_box_path = os.path.abspath(os.path.join('TensorBox'))
sys.path.append(tensor_box_path)

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes

import cv2
import argparse

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1])
    expname = '_' + args.expname if args.expname else ''
    image_dir = '%s/images_%s_%d%s' % (os.path.dirname(args.weights), os.path.basename(args.test_boxes)[:-5], weights_iteration, expname)
    return image_dir

def get_results(args, H):
    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, args.weights)

        pred_annolist = al.AnnoList()

        true_annolist = al.parse(args.test_boxes)
        data_dir = os.path.dirname(args.test_boxes)
        image_dir = get_image_dir(args)
        subprocess.call('mkdir -p %s' % image_dir, shell=True)
        for i in range(len(true_annolist)):
            true_anno = true_annolist[i]
            orig_img = imread('%s/%s' % (data_dir, true_anno.imageName))[:,:,:3]
            img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
            feed = {x_in: img}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            pred_anno = al.Annotation()
            pred_anno.imageName = true_anno.imageName
            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)

            pred_anno.rects = rects
            pred_anno.imagePath = os.path.abspath(data_dir)
            pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1])
            pred_annolist.append(pred_anno)

            imname = '%s/%s' % (image_dir, os.path.basename(true_anno.imageName))
            misc.imsave(imname, new_img)
            if i % 25 == 0:
                print(i)
    return pred_annolist, true_annolist

import urllib
import json
import codecs
import tempfile
import shutil
from dotmap import DotMap
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def healthcheck():
    return 'OK'

@app.route('/segment', methods=['POST'])
def segment():
    temp_dir_path = tempfile.mkdtemp()
    try:
        url = request.get_json(force=True)['url']
        img_path = temp_dir_path + '/' + 'image.png'
        urllib.urlretrieve(url, img_path)

        test_boxes_path = temp_dir_path + '/' + 'test_boxes.json'
        test_boxes = [{
            'image_path': 'image.png',
            'rects': []
        }]

        with open(test_boxes_path, 'wb') as f:
            json.dump(test_boxes, codecs.getwriter('utf-8')(f), ensure_ascii=False)

        args = DotMap()
        args.weights = os.getcwd() + '/model/save.ckpt-10000'
        args.expname = ''
        args.test_boxes = test_boxes_path
        args.gpu = 0
        args.logdir = 'output'
        args.iou_threshold = 0.5
        args.tau = 0.25
        args.min_conf = 0.2
        args.show_suppressed = True
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

        hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
        with open(hypes_file, 'r') as f:
            H = json.load(f)
        pred_annolist, true_annolist = get_results(args, H)

        return jsonify(pred_annolist[0].writeJSON()['rects'])
    finally:
        shutil.rmtree(temp_dir_path)

if __name__ == '__main__':
    app.run(port=80, processes=4, host='0.0.0.0')
