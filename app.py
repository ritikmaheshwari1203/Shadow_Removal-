# coding=utf8
#Access-Control-Allow-Origin: http://127.0.0.1:5500/

import os
import torch
import argparse
from torch.backends import cudnn
from models.DSANet import build_net
# from train import _train
# from eval import _eval
# import numpy as np
# import random
from torchvision import transforms
from PIL import Image
import torch.nn.functional as f

from torchvision.transforms import functional as F
from flask import Flask, render_template, redirect, url_for, request,jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
# import os
import warnings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

app = Flask(__name__)
CORS(app)


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = build_net()
model = model.cuda()
state_dict = torch.load("pretrained\model.pkl",map_location="cpu")
model.load_state_dict(state_dict['model'])

@app.route("/")
def index():
    return render_template("index.html")

# @app.route("/")
# def index():
#     return render_template("index.html")



# @app.route('/upload', methods=['POST'])
@app.post("/upload")
def main():

    if 'image' not in request.files:
        return {'No image part in the request':400}

    image = request.files['image']

    if image.filename == '':
        return {'No selected image':400}

    if image:
        filename = secure_filename(image.filename)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        cudnn.benchmark = True
        factor=4
        preprocess = transforms.Compose([
        transforms.Resize((450,450)),
        transforms.ToTensor()  # Convert the image to a tensor
        
  
            ])

        image_path = "uploads/"+filename
        input_img = Image.open(image_path)
        width, height = input_img.size


        print("image reading")
        input_tensor = preprocess(input_img)
        input_tensor = input_tensor.unsqueeze(0)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_tensor = f.pad(input_tensor, (0, padw, 0, padh), 'reflect')


        input_tensor = input_tensor.cuda()

        pred = model(input_tensor)[2]
        pred_clip = torch.clamp(pred, 0, 1)
        pred_clip += 0.5 / 255

        pred_clip = f.interpolate(pred_clip,(height,width),mode="bilinear")

        pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
        filename = "output_"+filename
        pred.save("static/"+filename)
        torch.cuda.empty_cache()

        # return {"image":filename}
        return jsonify({"image_url": url_for('static', filename=filename)})





if __name__ == '__main__':

    app.run(host="0.0.0.0",debug=True, port=5002)
    
