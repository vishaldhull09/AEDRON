import io
import base64
import sys
import tempfile
import cv2
import time
import argparse
import datetime
import numpy as np
from queue import Queue
from threading import Thread

MODEL_BASE = r'/home/vishal/Desktop/flask_app/realtimeCV/models-master/research'
sys.path.append(MODEL_BASE)
sys.path.append(MODEL_BASE + '/object_detection')
sys.path.append(MODEL_BASE + '/slim')

from flask import Flask,jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import Response
from flask import url_for
from flask import session
from flask_wtf.file import FileField
import numpy as np
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from utils import label_map_util
from utils import visualization_utils as vis_util
from werkzeug.datastructures import CombinedMultiDict
from wtforms import Form
from wtforms import ValidationError
from cv2 import imencode
from app_utils import draw_boxes_and_labels
app = Flask(__name__)



BASE_CKPT = '/home/vishal/Desktop/flask_app/realtimeCV/' 
PATH_TO_CKPT = BASE_CKPT +  'ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
PATH_TO_LABELS = MODEL_BASE + '/object_detection/data/mscoco_label_map.pbtxt'
MODELS = {
    'Mobilenet': 'ssd_mobilenet_v1_coco_2018_01_28',
    'Inception': 'ssd_inception_v2_coco_2018_01_28',
    'Resnet101': 'rfcn_resnet101_coco_2018_01_28',
    'FasterRCNN_ResNet101': 'faster_rcnn_resnet101_coco_2018_01_28'
}

content_types = {'jpg': 'image/jpeg',
                 'jpeg': 'image/jpeg',
                 'png': 'image/png'}
extensions = sorted(content_types.keys())

# Helper Functions
class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src, width, height):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.src = src
        self.width = width
        self.height = height

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def init(self):
        self.stream = cv2.VideoCapture(self.src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        (self.grabbed, self.frame) = self.stream.read()

    def start(self):
        # start the thread to read frames from the video stream
        self.camthread = Thread(target=self.update, args=())
        self.camthread.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def is_image():
  def _is_image(form, field):
    if not field.data:
      raise ValidationError()
    elif field.data.filename.split('.')[-1].lower() not in extensions:
      raise ValidationError()

  return _is_image


def draw_bounding_box_on_image(image, box, color='red', thickness=4):
  draw = ImageDraw.Draw(image)
  im_width, im_height = image.size
  ymin, xmin, ymax, xmax = box
  (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
  print(left)
  draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)


def encode_image(image):
  image_buffer = io.BytesIO()
  image.save(image_buffer, format='PNG')
  mime_str = 'data:image/png;base64,'
  imgstr = '{0!s}'.format(base64.b64encode(image_buffer.getvalue()))
  quote_index = imgstr.find("b'")
  end_quote_index = imgstr.find("'", quote_index+2)
  imgstr = imgstr[quote_index+2:end_quote_index]
  imgstr = mime_str + imgstr
  #imgstr = 'data:image/png;base64,{0!s}'.format(
      #base64.b64encode(image_buffer.getvalue()))
  return imgstr

# Webcam feed Helper
def worker(input_q, output_q):
    detection_graph = model.detection_graph
    sess = model.sess
    fps = FPS().start()
    while True:
        fps.update()
        frame = input_q.get()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_q.put(detect_objects_webcam(frame_rgb, sess, detection_graph))

    fps.stop()
    sess.close()

# detector for web camera
def detect_objects_webcam(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    rect_points, class_names, class_colors = draw_boxes_and_labels(
        boxes=np.squeeze(boxes),
        classes=np.squeeze(classes).astype(np.int32),
        scores=np.squeeze(scores),
        category_index=model.category_index,
        min_score_thresh=.5
    )
    return dict(rect_points=rect_points, class_names=class_names, class_colors=class_colors)

# Image class
class PhotoForm(Form):
  input_photo = FileField(validators=[is_image()])


class VideoForm(Form):
    input_video = FileField()

# Obect Dection Class
class ObjectDetector(object):

  def __init__(self, PATH_TO_CKPT):
    self.PATH_TO_CKPT = PATH_TO_CKPT
    self.detection_graph = self._build_graph()
    self.sess = tf.Session(graph=self.detection_graph)

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    

  def _build_graph(self):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph

  def _load_image_into_numpy_array(self, image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

  def detect(self, image):
    image_np = self._load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    graph = self.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    (boxes, scores, classes, num_detections) = self.sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    boxes, scores, classes, num_detections = map(
        np.squeeze, [boxes, scores, classes, num_detections])

    return boxes, scores, classes.astype(int), num_detections.astype(int)

# Detection function
def detect_objects(model, image_path):
  image = Image.open(image_path).convert('RGB')
  boxes, scores, classes, num_detections = model.detect(image)
  image.thumbnail((480, 480), Image.ANTIALIAS)

  new_images = {}
  for i in range(num_detections):
    if scores[i] < 0.7: continue
    cls = classes[i]
    if cls not in new_images.keys():
      new_images[cls] = image.copy()
    print("drawing boxes")  
    draw_bounding_box_on_image(new_images[cls], boxes[i],
                               thickness=int(scores[i]*10)-4)

  result = {}
  #result['original'] = encode_image(image.copy())
  result['detections'] = {}
  result['detections']['original'] = encode_image(image.copy())

  for cls, new_image in new_images.items():
    category = model.category_index[cls]['name']
    #result[category] = encode_image(new_image)
    result['detections'][category] = encode_image(new_image)

  return result


@app.route('/')
def main_display():
    photo_form = PhotoForm(request.form)
    video_form = VideoForm(request.form)
    return render_template('main.html', photo_form=photo_form, video_form=video_form, models=list(MODELS.keys()), result={})

@app.route('/imgproc', methods=['GET', 'POST'])
def imgproc():
    video_form = VideoForm(request.form)
    form = PhotoForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST' and form.validate():
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            form.input_photo.data.save(temp)
            temp.flush()
            results = {}
            print('detecting')
            start_time = time.time();
            for model_name, model_path in MODELS.items():
                model_start_time = time.time()
                PATH_TO_CKPT = BASE_CKPT + model_path + '/frozen_inference_graph.pb'
                model = ObjectDetector(PATH_TO_CKPT)
                results[model_name] = detect_objects(model, temp.name)
                results[model_name]['time'] = round(time.time() - model_start_time, 2)
            total_time = round(time.time() - start_time, 2)
            print('detected')
        photo_form = PhotoForm(request.form)
        return render_template('main.html', photo_form=photo_form, video_form=video_form, models=list(MODELS.keys()), results=results, total_time=total_time)    
    else:
        return redirect(url_for('main_display'))


@app.route('/vidproc', methods=['GET', 'POST'])
def vidproc():
    global PATH_TO_CKPT
    print('vidproc')
   
    form = VideoForm(CombinedMultiDict((request.files, request.form)))
    if request.method == 'POST':
        PATH_TO_CKPT = BASE_CKPT +  MODELS[request.form['model']] + '/frozen_inference_graph.pb'
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            form.input_video.data.save(temp)
            temp.flush()
            session['vid'] = temp.name
        video_form = VideoForm(request.form)
        photo_form = PhotoForm(request.form)

        return render_template('video.html', photo_form=photo_form, video_form=video_form, models=list(MODELS.keys()), video=True)


foundClasses = []

@app.route('/temp')
def temp():
    return jsonify(foundClasses)

@app.route('/vidpros')
def vidpros():
    model = ObjectDetector(PATH_TO_CKPT)
    global foundClasses
    foundClasses = []
    graph = model.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')
    
    vid_source = cv2.VideoCapture(session['vid'])
    
    def generate(image_tensor, boxes, scores, classes, num_detections):
        ret, frame = vid_source.read()
        while ret:
            image_np_expanded = np.expand_dims(frame, axis=0)

            (boxes_t, scores_t, classes_t, num_detections_t) = model.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            boxes_t, scores_t, classes_t, num_detections_t = map(np.squeeze, [boxes_t, scores_t, classes_t, num_detections_t])

            for i in range(num_detections_t.astype(int)):
                if scores_t[i] < 0.5: continue
                foundClass = model.category_index[classes_t[i]]['name']
                if foundClass not in foundClasses:
                    foundClasses.append(foundClass)


            vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            boxes_t,
            classes_t.astype(np.int32),
            scores_t,
            model.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
            
            payload = cv2.imencode('.jpg', frame)[1].tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')

            ret, frame = vid_source.read()
        print(foundClasses)
    
    return Response(generate(image_tensor, boxes, scores, classes, num_detections), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/json',methods=['GET'])
def json():
    graph = model.detection_graph
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')
    


    vid_source = cv2.VideoCapture(session['vid'])
    print("vid src")
    
    def generate(image_tensor, boxes, scores, classes, num_detections):
        ret, frame = vid_source.read()
        #label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
	    #print(label_map)
	    #categories = label_map_util.convert_label_map_to_categories(
	     #   label_map, max_num_classes=90, use_display_name=True) 
	    #category_index = label_map_util.create_category_index(categories)
        # tensor code
        result2={}
        index=0
        while ret:
            #image_np = model._load_image_into_numpy_array(frame)
            #result2=[]
            image_np_expanded = np.expand_dims(frame, axis=0)

            (boxes_t, scores_t, classes_t, num_detections_t) = model.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            im,clas=vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes_t),
            np.squeeze(classes_t).astype(np.int32),
            np.squeeze(scores_t),
            model.category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

            #print(display_str)
            #image_pil = Image.fromarray(np.uint8(frame)).convert('RGB')
            #result2.append(set(classes_t[np.where(classes_t>1)]))
            if len(clas)>0 and (clas not in result2.values()): 
                 result2[index]= list(str(clas))

            #if(max(classes_t).any()>1):
            #	result2.append(int(max(classes_t)))
            #	print(result2)
            	#print(result2)

            payload = cv2.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
            ret, frame = vid_source.read()
            index+=1
        #print(result2)   
        return result2
            

    print("Before return")
    #result3=[]
    result3=generate(image_tensor, boxes, scores, classes, num_detections)
    print(result3)
    #print(list(result3))
    return jsonify(result3)


@app.route('/realproc', methods=['GET', 'POST'])
def realproc():
    return render_template('realtime.html')


@app.route('/realstop', methods=['GET', 'POST'])
def realstop():
    photo_form = PhotoForm(request.form)
    video_form = VideoForm(request.form)
    if request.method == 'POST':
        print("In - Stop - POST")
        if request.form['realstop'] == 'Stop Web Cam':
            print(request.form['realstop'])
            fps_init.stop()
            video_init.stop()
            video_init.update()
            print("Stopped")
    return render_template('main.html', photo_form=photo_form, video_form=video_form)


@app.route('/realpros')
def realpros():
    print("in real pros")
    input_q = Queue(5)
    output_q = Queue()
    for i in range(1):
        t = Thread(target=worker, args=(input_q, output_q))
        t.daemon = True
        t.start()

    video_init.init()
    video_capture = video_init.start()
    fps = fps_init.start()
    def generate():
        print("in gen real pros")
        frame = video_capture.read()
        while video_capture.grabbed:
            print("in while gen real pros")
            input_q.put(frame)
            t = time.time()

            if output_q.empty():
                pass
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                data = output_q.get()
                rec_points = data['rect_points']
                class_names = data['class_names']
                class_colors = data['class_colors']
                for point, name, color in zip(rec_points, class_names, class_colors):
                    cv2.rectangle(frame, (int(point['xmin'] * 480), int(point['ymin'] * 360)),
                                  (int(point['xmax'] * 480), int(point['ymax'] * 360)), color, 3)
                    cv2.rectangle(frame, (int(point['xmin'] * 480), int(point['ymin'] * 360)),
                                  (int(point['xmin'] * 480) + len(name[0]) * 6,
                                   int(point['ymin'] * 360) - 10), color, -1, cv2.LINE_AA)
                    cv2.putText(frame, name[0], (int(point['xmin'] * 480), int(point['ymin'] * 360)), font,
                                0.3, (0, 0, 0), 1)

                payload = cv2.imencode('.jpg', frame)[1].tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + payload + b'\r\n')
                frame = video_capture.read()
                #video_capture.update()
            print("out of while")
            fps.update()


    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

#import time
#start_time = time.time()
#model = ObjectDetector(PATH_TO_CKPT)
#print('model took', time.time() - start_time)
video_init = WebcamVideoStream(src=0, width=480, height=360)
fps_init = FPS()

app.secret_key = 'super secret key'

if __name__ == '__main__':

    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.run(debug=True)
