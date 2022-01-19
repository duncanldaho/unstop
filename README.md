# unstop
A simple python script for [Neural Style Transfer](https://en.wikipedia.org/wiki/Neural_Style_Transfer).<br>

## Instructions:
1. Download or clone the code.
2. Inside of the main directory create a python virtual environment, activate the virtual environment, and install the modules.
```
python3 -m venv venv_name
source venv_name/bin/activate
pip3 install -r requirements.txt
```
3. Run the python script.
```
python3 unst.py
```
The script uses the images in the data directory. It applies the style from "style.jpeg" to "content.jpeg" and generates "stylized.jpeg".
