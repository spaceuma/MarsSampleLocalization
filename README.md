## Hardware-accelerated Mars Sample Localization via deep transfer learning from photorealistic simulations
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6542933.svg)](https://doi.org/10.5281/zenodo.6542933)

Code associated to the article: **"Hardware-accelerated Mars Sample Localization via deep transfer learning from photorealistic simulations".**


## Docker configuration 

An Ubuntu host system is needed to run the files located at the repo, as we use a Nvidia GPU to train the network. First of all, we must install the docker core:

```bash
$ sudo apt-get update

$ sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common
    
$ curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

$ sudo apt-key fingerprint 0EBFCD88

$ sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
   
$ sudo apt-get update
$ sudo apt-get install docker-ce docker-ce-cli containerd.io

```
We install our NVIDIA card's drivers and the modules that let us use them in our docker environment:

```bash
$ sudo ubuntu-drivers autoinstall            ## Auto-installs Nvidia drivers
$ sudo apt-get install -y nvidia-docker2 nvidia-container-runtime
```

After that, we must build the configured docker container for this project:

```bash
$ docker build . -f ./dockerfile/yolo_compiler.Dockerfile -t MarsSampleLocalization 
```

We run the docker image:

```bash
$ xhost +local:docker  ## To let docker use the screen

$ docker run -e DISPLAY=$DISPLAY -v /your/cloned/repo/location:/opt \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --rm --gpus all \
  --privileged -v /dev/bus/usb:/dev/bus/usb \
  -t MarsSampleLocalization \
  bin/bash 
```

## Lab tests videos

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/8_ymP6bg6-c/0.jpg)](https://www.youtube.com/watch?v=8_ymP6bg6-c)


## Citation

If this work was helpful for your research, please consider citing the following BibTeX entry:
```BibTeX
Fill
```

## License

This repository is released under the MIT open source license as found in the [LICENSE](LICENSE) file.


## General File Tree

```
.
├── 3rdparty_coral
│   ├── coral_inference
│   ├── darknet_to_keras
│   └── keras_to_tflite
|
├── 3rdparty_darknet
|
├── datasets
│   ├── field_tests_images
│   ├── sample_orientation_images
│   └── train_images
|
├── dockerfile
|
├── output_files
│   ├── compiled_coral
│   ├── coral_detection
│   ├── darknet_detections
│   ├── darknet_weights
│   ├── fieldtest
│   ├── keras_conversion
│   ├── log
│   ├── orientation_images
│   └── quantized_tflite
├── src
│   ├── example_files
|   |
|   ├── coralmodule.py
│   ├── cameradepth.py
|   ├── main_coraldetect.py
│   ├── main_examples.py
│   ├── main_fieldtest.py
│   ├── main_orientation.py
│   └── sample.py
|
├── training_files
|    ├── cfg
|    ├── darknet_data
|    └── generated_weights
| 
├── darknet_to_coral.sh
├── detect_coral.sh
├── detect_datasets.sh
├── generate_data.sh
├── test_coral.sh
├── test_datasets.sh
├── test_fieldtest.sh
├── test_orientation.sh
└── train_datasets.sh
```
