import sys, skvideo.io, json, base64
import os.path
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import argparse
import sys
import time
import helper
import scipy.misc
import timeit
import sklearn
import sklearn.metrics
import cv2
#import sklearn.metrics.f1_score

from scipy.misc import imread, imresize
from glob import glob

file = sys.argv[-1]
FLAGS = None
image_shape = (160, 576)

def load_graph(graph_file):
    config = tf.ConfigProto()
    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess, ops

def benchmark(sess, video, gt_images_name, binary=True):
    answer_key = {}
    g = sess.graph
    x = g.get_tensor_by_name('image_input:0')
    keep_prob = g.get_tensor_by_name('keep_prob:0')
    out = g.get_tensor_by_name('prediction_softmax:0')
    output_compare = []
    times = []
    num_classes = 2
    streets_im = []
    frame = 0
    output_dir = os.path.join('runs', str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    for rgb_frame in video:
        rgb_frame_h = scipy.misc.imresize(rgb_frame, image_shape)
        rgb_frame_m = rgb_frame_h.reshape(1, image_shape[0], image_shape[1], 3)
        im_softmax = sess.run(out, {x: rgb_frame_m, keep_prob: 1.0})
        road_pix = im_softmax[:, 1].reshape(image_shape[0], image_shape[1])

        road_seg = (road_pix > 0.5).reshape(image_shape[0], image_shape[1], 1)

        mask = np.dot(road_seg, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(rgb_frame_h)
        street_im.paste(mask, box=None, mask=mask)
        streets_im.append(np.asarray(street_im))
        scipy.misc.imsave(os.path.join(output_dir, gt_images_name[frame]), street_im)
        frame += 1
    skvideo.io.vwrite('road_semantic_seg_video.mp4', streets_im)
    return output_compare

def main(_):
    data_dir = './data'
    helper.maybe_download_pretrained_vgg(data_dir)
    gt_images = []
    gt_images_name = []
    image_paths = glob(os.path.join(data_dir, 'data_road/testing/image_2', '*.png'))
    for image_file in image_paths:
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        print(image.shape)
        gt_images.append(image)
        gt_images_name.append(image_file.split('/')[-1])
    tf.reset_default_graph()
    graph_def = tf.GraphDef()
    tf.import_graph_def(graph_def, name='')
    sess, ops = load_graph('./frozen_model/graph.pb')
    output_predicted = benchmark(sess, gt_images, gt_images_name)

if __name__ == '__main__':
    tf.app.run(main=main)
