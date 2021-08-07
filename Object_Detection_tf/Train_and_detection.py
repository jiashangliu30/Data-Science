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
# ----------------------------------------------------------------------------------------------------------------------

# -- Check Versions
# print(pd.__version__)
# print(np.__version__)
# print(tf.__version__)

# my versisons
# 1.1.3
# 1.19.2
# 2.5.0

# ----------------------------------------------------------------------------------------------------------------------

#scraping
# -- Image Collecting Tool
python image_scraping.py --search "--insert keyword--" --num_images 50 --directory "C:\Users\jiash\Desktop\images"


# creating tf_record for the pascal_VOC dataset
# I used the script that came with the object detection API "create_pascal_tf_record.py"
# configure your own path to continue

# Train tfrecord
!python C:\Users\jiash\anaconda3\Lib\site-packages\object_detection\dataset_tools\create_pascal_tf_record.py \
    --label_map_path=C:\Users\jiash\Downloads\Project\data_models\annot/label_map.pbtxt \
    --data_dir=C:\Users\jiash\Desktop\VOCtrainval_06-Nov-2007\VOCdevkit --year=VOC2007 --set=train \
    --output_path=C:\Users\jiash\Desktop\VOCdevkitpascal_train.record    
# Test tfrecord
!python C:\Users\jiash\anaconda3\Lib\site-packages\object_detection\dataset_tools\create_pascal_tf_record.py \
    --label_map_path=C:\Users\jiash\Downloads\Project\data_models\annot/label_map.pbtxt \
    --data_dir=C:\Users\jiash\Desktop\VOCtrainval_06-Nov-2007\VOCdevkit --year=VOC2007 --set=val \
    --output_path=C:\Users\jiash\Desktop\VOCdevkitpascal_val.record   
    
    
# -- Name and url of the model downloaded form TF2 Model zoo:
# -- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
# -- Get the link by copying the link for the model that you want to train from the model zoo
# -- Note that if you changed the model you HAVE TO CHANGE BOTH OF THE FOLLOWINGS


# CUSTOM_MODEL_NAME = 'resnet50'
# PRETRAINED_MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640_coco17_tpu-8'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz'

CUSTOM_MODEL_NAME = 'centernet_hg104'
PRETRAINED_MODEL_NAME = 'centernet_hg104_512x512_coco17_tpu-8'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200713/centernet_hg104_512x512_coco17_tpu-8.tar.gz'

# CUSTOM_MODEL_NAME = 'efficientdet_d0'
# PRETRAINED_MODEL_NAME = 'efficientdet_d0_coco17_tpu-32'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'


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


# -- Update the config file --------------------------------------------------------------------------------------------

# -- Step 1
# -- copy the model config from the pre_trained model to my_training_model folder !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# -- Update Config For Transfer Learning
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

config

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)


# pipeline_config.model.ssd.num_classes = 20
# pipeline_config.train_config.batch_size = 4
# pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'checkpoint', 'ckpt-0')
# pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
# pipeline_config.train_input_reader.label_map_path= files['LABELMAP']
# pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
# pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
# pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(paths['ANNOTATION_PATH'], 'test.record')]


# config_text = text_format.MessageToString(pipeline_config)
# with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
#     f.write(config_text)




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

! tensorboard --logdir=C:\Users\jiash\Desktop\Project\data_models\my_trained_models\resnet50\evaluation

# Then open the url address in a browser, that will open tensorboard for you

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
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-7')).expect_partial()



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

IMAGE_PATH = os.path.join(paths['IMAGE_PATH'], 'extra_tests', 'people.jpg')
%matplotlib inline
IMAGE_PATH
img = cv2.imread(IMAGE_PATH)
plt.imshow(img)
plt.show()

"""
# -- where working with grayscale images
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplots()
plt.imshow(img_gray, cmap='gray')
plt.show()

# -- Convert to RGB with 3 channels
img_rgb = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2RGB)
plt.subplots()
plt.imshow(img_rgb, cmap='gray')
plt.show()
"""

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

#plt.isinteractive()
#plt.interactive(False)
#plt.show(block=True)


# -- Real time detection -----------------------------------------------------------------------------------------------
# -- this use your video camera
#!pip uninstall opencv-python-headless -y

cap = cv2.VideoCapture(0) # try other channels if "0" did not worked, line "1" or "2" !!!!!!!!!!!!!!!
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    image_np = np.array(frame)

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

# -- Freeze our model (same format as the tensorflow pretrained models) ------------------------------------------------
FREEZE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'exporter_main_v2.py ')
command = "python {} --input_type=image_tensor --pipeline_config_path={} --trained_checkpoint_dir={}\
 --output_directory={}".format(FREEZE_SCRIPT, files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['OUTPUT_PATH'])

print(command)
!{command}

# -- Save our model in TF Jason format ---------------------------------------------------------------------------------
# !pip install tensorflowjs
command = "tensorflowjs_converter --input_format=tf_saved_model --output_node_names='detection_boxes,detection_classes,\
detection_features,detection_multiclass_scores,detection_scores,num_detections,raw_detection_boxes,raw_detection_scores'\
--output_format=tfjs_graph_model --signature_name=serving_default {} {}".format(os.path.join(paths['OUTPUT_PATH'],\
                                                                                    'saved_model'), paths['TFJS_PATH'])

print(command)
!{command}

# -- Save our model in TFLite format -----------------------------------------------------------------------------------
TFLITE_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'export_tflite_graph_tf2.py ')

command = "python {} --pipeline_config_path={} --trained_checkpoint_dir={} --output_directory={}".format(TFLITE_SCRIPT,\
                                            files['PIPELINE_CONFIG'], paths['CHECKPOINT_PATH'], paths['TFLITE_PATH'])

print(command)
!{command}


FROZEN_TFLITE_PATH = os.path.join(paths['TFLITE_PATH'], 'saved_model')
TFLITE_MODEL = os.path.join(paths['TFLITE_PATH'], 'saved_model', 'detect.tflite')

command = "tflite_convert \
--saved_model_dir={} \
--output_file={} \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1',\
'TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=FLOAT \
--allow_custom_ops".format(FROZEN_TFLITE_PATH, TFLITE_MODEL, )

print(command)
!{command}

# -- Zip and export model
!tar -czf models.tar.gz {paths['CHECKPOINT_PATH']}









