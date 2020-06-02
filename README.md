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
-- cd models-master/research
-- protoc object_detection/protos/*.proto --python_out=.

5) Set MODEL_BASE in detector_app.py to the path of the object detection api ../models-master/research
6)
-- cd ~/realtimeCV/obj_detect_multi
-- export FLASK_APP=detector.app
-- flask run


