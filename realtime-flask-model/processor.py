
from PIL import Image
import threading
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64

class Processor(object):
    def __init__(self, model_backend, quality = 0.75):
        self.quality = int(quality*100)
        self.to_process = []
        self.to_output = []
        self.model_backend = model_backend

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return
        # input is an ascii string.
        input_str = self.to_process.pop(0)
        # convert it to a pil image
        input_img = base64_to_pil_image(input_str)

        ######## Calling the backend model ##########
        if input_img is not None:
            output_img = self.model_backend(input_img)

        ######## Calling the backend model ##########

        # output_str is a base64 string in ascii
            output_str = pil_image_to_base64(output_img, quality = self.quality)
        # convert eh base64 string in ascii to base64 string in _bytes_
            self.to_output.append(output_str)

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)

class MyModel(object):
    def __init__(self):
        pass

    def generate(self, img):
        return img.transpose(Image.FLIP_LEFT_RIGHT)
