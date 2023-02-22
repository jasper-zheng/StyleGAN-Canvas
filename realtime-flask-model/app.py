import os
from sys import stdout
import logging

from flask import Flask, render_template
from flask_socketio import SocketIO, emit


from processor import Processor
from model import Pipeline

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

async_mode = None
app = Flask(__name__)
app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = os.urandom(24).hex()
app.config['DEBUG'] = True
socketio = SocketIO(app)

# quality from server to client (0 - 1, default 0.75)
quality = 0.5

processor = Processor(Pipeline(), quality = quality)
layer_names = processor.model_backend.get_layer_names()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


@socketio.on('input_frame', namespace='/demo')
def process_frame(input):
    input = input.split(",")[1]
    processor.enqueue_input(input)
    image_data = processor.get_frame()
    # out_data = "data:image/jpeg;base64," + str(image_data[0], "utf-8")
    # out_data_x = "data:image/jpeg;base64," + str(image_data[1], "utf-8")
    # emit('processed_frame', {'image_data': out_data, 'image_data_x': out_data_x}, namespace='/demo')
    out_data = "data:image/jpeg;base64," + str(image_data, "utf-8")
    emit('processed_frame', {'image_data': out_data}, namespace='/demo')

@socketio.on('config_update', namespace='/demo')
def update_configs(name, input):
    # print(input)
    processor.model_backend.update_configs(name, input)

@socketio.on('config_clear', namespace='/demo')
def clear_configs(name):
    # print(input)
    processor.model_backend.update_configs(name, None)

@socketio.on('change_cluster_demo', namespace='/demo')
def change_cluster_demo(layer_name, cluster_numbers, img):
    img = img.split(",")[1]
    # print(type(cluster_idx))
    img = processor.model_backend.get_cluster_demo(layer_name, cluster_numbers, img)
    image_datas = {}
    for idx, this_img in enumerate(img):
        image_datas[f'c{idx}'] = "data:image/jpeg;base64," + str(this_img, "utf-8")

    emit('return_cluster_demo', image_datas, namespace='/demo')

    # image_data = "data:image/jpeg;base64," + str(img, "utf-8")
    # emit('return_cluster_demo', {'image_data': image_data}, namespace='/demo')

@socketio.on('regenerate_cluster', namespace='/demo')
def regenerate_cluster(layer_name, img, num_of_clusters, cur_cluster_selection):
    img = img.split(",")[1]
    print(f'regenerating cluster for {layer_name} with {int(num_of_clusters)} cluster')
    img = processor.model_backend.regenerate_cluster(layer_name, img, int(num_of_clusters), cur_cluster_selection = int(cur_cluster_selection))
    # image_data = "data:image/jpeg;base64," + str(img, "utf-8")
    # emit('return_cluster_demo', {'image_data': image_data}, namespace='/demo')
    image_datas = {}
    print(f'return {len(img)}')
    for idx, this_img in enumerate(img):
        image_datas[f'c{idx}'] = "data:image/jpeg;base64," + str(this_img, "utf-8")

    emit('return_cluster_demo', image_datas, namespace='/demo')

@socketio.on('connect', namespace='/demo')
def test_connection():
    emit('set_layer_names', {'names': layer_names}, namespace='/demo')
    print("client connected")
    app.logger.info("client connected")

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    exit(0)

if __name__ == '__main__':

    socketio.run(app,port=5000)
