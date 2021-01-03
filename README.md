# DeepFake Detection

In this repository, we train a deep learning model on fake and real videos. Create ```conda``` environment using  ```environment.yml```

## Training

We deploy [Xception Net](https://arxiv.org/abs/1610.02357) since it is proven to work well for lot of DeepFake tasks. Since we have a relatively small dataset to train, we make use of pre-trained model extracted from [this link](http://kaldir.vc.in.tum.de/FaceForensics/models/faceforensics++_models.zip). We also make use of some of the functions and network definition from [faceforensics](https://github.com/agpeshal/FaceForensics/tree/master/classification).

This is to capture faces from video frames and use those images to learn. The prediction is based on the average prediction on the faces of the video frames

First to extract images from videos run the following command. Ensure that fake videos are in ```vidoes/fake``` and real under ```videos/real``` with ```.mp4``` file format.

```bash
python video_to_image.py --video_dir videos --image_dir images --interval 20
```

```--interval``` is used to set the frequency of the frames from which you capture the faces.

Extract the locations of images for each video in a text file to support data loading

```bash
python data_dir.py
```

To start the training and validation process, run

```bash
python train.py
mlflow ui
```

You can monitor training with *mlflow* at http://localhost:5000



# Inference

To infer on a new video, one must download the new video in ```videos/infer``` and the name must be in {number}.mp4 format. Then run

```bash
python infer.py
```

This will load the pretrained model obtained after training and store the prediction in ```results.txt```



# REST API and Docker

### Rest API

- Use Flask micro-service to start a server that listens to HTTP post requests for accepting the client video data (payload=video data)
- Once the request is received, save the videos in the appropriate folder and start inference



## Docker File

- Use python3.8 docker image as the base image
- Install the required modules
- Clone the git repo in the docker image
- Start the flask service with an open port