# unstop

A simple python script for [neural style transfer](https://en.wikipedia.org/wiki/Neural_Style_Transfer), using the pre-trained [fast arbitrary image style transfer](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2) model.

## Set up

1. Download or clone the code.
2. Inside of the unstop directory create a virtual environment, activate the
environment, and install the modules.
```
# Create a virtual environment:
python3 -m venv unstop

# Activate the environment:
source unstop/bin/activate

# Install python modules:
pip3 install -r requirements
```
## Running

1. After set up, you only have to activate the environment and run the script.
```
source unstop/bin/activate
python3 unst.py
```
2. When finished, deactivate the environment.
```
deactivate
```
## Notes

There are example images in the /data directory. The script applied
the style from "style.jpeg" to "content.jpeg" and generates "stylized.jpeg".
Unfortunately tensorflow has a lot of dependencies, so it is highly
recommended to use a virtual environment. Unstop may not work with future
versions of the required modules. It works the newest versions, but has been
confirmed to work with at least:
```
numpy==1.22.1
tensorflow==2.7.0
tensorflow-hub==0.12.0.
```
