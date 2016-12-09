# Deep Learning: Behavioral Cloning

In this project, I host all software I've written by hand to train a deep learning neural network to drive a car around a test track AND a never-before-seen track. The latter track demonstrates how well my network modeling generalizes to unseen roadways and conditions. This is going to be epic. Stay tuned!

#### Validating Your Network

You can validate your model by launching the simulator and entering autonomous mode.

The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

##### Install Python Dependencies with Anaconda
```
$ conda install numpy
$ conda install -c conda-forge flask-socketio
$ conda install -c conda-forge eventlet
$ conda install pillow
$ conda install h5py
```
Install Python Dependencies with pip
```
$ pip install keras
```

Download [drive.py](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5821235d_drive/drive.py)

```
$ wget https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5821235d_drive/drive.py
```

Run Server

```
$ python drive.py model.json
```

If you're using Docker for this project: docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starer-kit python drive.py model.json or docker run -it --rm -p 4567:4567 -v ${pwd}:/src udacity/carnd-term1-starer-kit python drive.py model.json. Port 4567 is used by the simulator to communicate.
Once the model is up and running in drive.py, you should see the car move around (and hopefully not off) the track!