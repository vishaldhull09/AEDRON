# flask_objectdetection_app

Dependencies--

1) Flask
 
 --apt-get update
 
 --pip install Flask==0.12.2 WTForms==2.1 Flask_WTF==0.14.2 Werkzeug==0.12.2

2) tensorflow

3) OpenCV


4) Installtion Process
-- apt-get update

-- apt-get install -y protobuf-compiler python-pil python-lxml python-pip python-dev git

--git clone https://github.com/tensorflow/models

-- cd models-master/research

-- protoc object_detection/protos/*.proto --python_out=.

5) Set MODEL_BASE in detector_app.py to the path of the object detection api ../models-master/research


## Download the pretrained model binaries

Models available at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

There are five pretrained models that can be used by the application.
 They have diffrent characteristics in terms of accuracy and speed.
 You can change the model used by the application by setting
 the PATH_TO_CKPT to point the frozen weights of the required model.

You specify one of the following models.

- ssd_mobilenet_v1_coco_11_06_2017
- ssd_inception_v2_coco_11_06_2017
- rfcn_resnet101_coco_11_06_2017
- faster_rcnn_resnet101_coco_11_06_2017
- faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017




## After setting the paths run this on terminal

-- cd ~/realtimeCV/obj_detect_multi

-- export FLASK_APP=detector.app

-- flask run


