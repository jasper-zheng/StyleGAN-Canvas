# Realtime Flask Model (Webcam)  

![coverimg](/docs/realtime-flask.gif)   

This repository demonstrates **deploying a real-time image-to-image translation model** via [Flask](https://flask.palletsprojects.com/en/2.2.x/) and [SocketIO](https://socket.io/docs/v4/).

## How it Works  

We're using [Flask-SocketIO](https://flask-socketio.readthedocs.io/en/latest/) since it provides bi-directional communications between the web client and the model. We take inputs from the webcam via web clients, and send them to the model deployed on the flask server. The model processes the frames and returns them to the web client.  

## Requirements

We provided a Gaussian blur model for basic pipeline demonstration, and a pix2stylegan3 de-blurring model, which is based on StyleGAN3 and requires more computing power.

### GaussianBlur model only:  

```python
bidict==0.22.0  
click==8.1.3
Flask==2.2.0
Flask-SocketIO==5.2.0
h11==0.13.0
importlib-metadata==4.12.0
itsdangerous==2.1.2
Jinja2==3.1.2
MarkupSafe==2.1.1
python-engineio==4.3.3
python-socketio==5.7.1
typing_extensions==4.3.0
Werkzeug==2.2.1
wsproto==1.1.0
zipp==3.8.1
```

### Additional requirements for the pix2stylegan3
The same as [StyleGAN3](https://github.com/NVlabs/stylegan3#requirements)  

## Limitations  
It doesn't handle multiple clients.

## Quickstart  

Clone the repository by running:
```python
git clone https://github.com/jasper-zheng/realtime-flask-model.git  
cd realtime-flask-model
```

If using the pix2StyleGAN3 model, please download the [trained model](https://drive.google.com/file/d/1_j_zeBwnBkZgit2ozZ8_bR59z2jN9JUv/view?usp=sharing) and place it in `saved_models` folder.  
`gdown` command: `gdown 1_j_zeBwnBkZgit2ozZ8_bR59z2jN9JUv -O ./saved_models/demo_model.pkl`   


### Define the processing pipeline  

1. Specify the processing pipeline in `Class Pipeline` in `model.py`. The `forward() ` method takes a PIL format frame and returns the processed frame in the same format.

> The current code is for the pix2StyleGAN3 model. For the basic gaussian blur model, please remove the current Pipeline class and uncomment the Pipeline class below it.

2. The size of the input frame, frame rate, and quality is defined in `static/js/main.js`, currently set to 256, 100ms and 0.75  

3. The size of the return frame depends on your processing pipeline in `model.py`.  

4. The model is added to the server in `app.py` (usually don't need to change this one).  

### Activate the server   

Port number is defined in `app.py`, currently set to 5000. Then run:   

```python   
cd realtime-flask-model
python app.py
```

The application is now running on `http://127.0.0.1:5000/`   


## Deploying on Cloud GPUs [Optional]  
To deploy the model on cloud GPUs (e.g. Colab or Paperspace's Gradient Notebook), [LocalXpose](https://localxpose.io/docs/) or [ngrok](https://ngrok.com/docs) is recommended.
