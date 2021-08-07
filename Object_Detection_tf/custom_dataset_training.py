# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 12:31:33 2021

@author: jiashangliu
"""

# -- Import libraries
import numpy as np
import pandas as pd
import cv2
import os
import io
import uuid
import time
import object_detection
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import namedtuple
import argparse
# -----------------------

CUSTOM_MODEL_NAME = 'centernet_hg104_cus'
PRETRAINED_MODEL_NAME = 'centernet_hg104_512x512_coco17_tpu-8_voc'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz'

LABEL_MAP_NAME = 'label_map.pbtxt'

root_dir = "Project"
work_dir = "data_models"
tf_m = 'pre_trained_models'
my_tf_m = 'my_trained_models'

paths = {
    'WORKSPACE_PATH': os.path.join(root_dir, work_dir),
    'APIMODEL_PATH': os.path.join(root_dir,'tf_model_garden'),
    'ANNOTATION_PATH': os.path.join(root_dir, work_dir,'annot'),
    'IMAGE_PATH': os.path.join(root_dir, work_dir,'images'),
    'MODEL_PATH': os.path.join(root_dir, work_dir,my_tf_m),
    'PRETRAINED_MODEL_PATH': os.path.join(root_dir, work_dir,tf_m),
    'CHECKPOINT_PATH': os.path.join(root_dir, work_dir,my_tf_m,CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join(root_dir, work_dir,my_tf_m,CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join(root_dir, work_dir,my_tf_m,CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join(root_dir, work_dir,my_tf_m,CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join(root_dir,'protoc')
 }


files = {
    'PIPELINE_CONFIG':os.path.join(root_dir, work_dir,my_tf_m, CUSTOM_MODEL_NAME, 'pipeline.config'),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


for path in paths.values():
    if not os.path.exists(path):

        if os.name == 'posix': # for mac and linux
            #!mkdir -p {paths}
            os.makedirs(path)

        if os.name == 'nt': # for windows
            !mkdir {path}
            
            
            
# -- Train the model ---------------------------------------------------------------------------------------------------
TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

command = "python {} --model_dir={} --pipeline_config_path={} --num_train_steps=6000".format(TRAINING_SCRIPT,\
                                                        paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'])

print(command)
!{command}


# -- Evaluate the model ------------------------------------------------------------------------------------------------
command = "python {} --model_dir={} --pipeline_config_path={} --checkpoint_dir={}".format(TRAINING_SCRIPT,\
                            paths['CHECKPOINT_PATH'],files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'])

print(command)
!{command}


# -- Make prediction, detect objects------------------------------------------------------------------------------------

# -- First load the saved model from checkpoints
import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()



# -- Prediction, object detection, images
import cv2
import numpy as np
from matplotlib import pyplot as plt
#matplotlib.use('module://backend_interagg')

# --  define the function
#@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'extra_tests', '3.jpg')
%matplotlib inline
IMAGE_PATH
img = cv2.imread(IMAGE_PATH)
plt.imshow(img)
plt.show()


image_np = np.array(img)

input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes']+label_id_offset,
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            min_score_thresh=.8,
            agnostic_mode=False)

plt.subplots()
plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
plt.show()


# -- Live Prediction  ------------------------------------------------


cap = cv2.VideoCapture(0) # try other channels if "0" did not worked, line "1" or "2" !!!!!!!!!!!!!!!
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 25, (600, 1200))

while cap.isOpened():
    ret, frame = cap.read()
    image_np = np.array(frame)
    # out.write(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + label_id_offset,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.8,
        agnostic_mode=False)

    cv2.imshow('object detection', cv2.resize(image_np_with_detections, (800, 600)))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
    
# -- Freeze our model  ------------------------------------------------
FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')
command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={}\
 --output_directory={}".format(FREEZE_SCRIPT, files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['OUTPUT_PATH'])

print(command)
!{command}