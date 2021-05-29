import os
from inference import execute_vr
from flask import Flask, request, abort, jsonify, send_file
import torch
from lib import nets

UPLOAD_DIRECTORY = "./uploads"
DOWNLOAD_DIRECTORY = "./downloads"
MODEL_DIRECTORY = "./models"

api = Flask(__name__)

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

if not os.path.exists(DOWNLOAD_DIRECTORY):
    os.makedirs(DOWNLOAD_DIRECTORY)

print('loading model...', end=' ')
arg_pretrained_model = 'v4_sr44100_hl512_nf2048.pth'
arg_gpu = 0
arg_n_fft = 2048
device = torch.device('cpu')
model = nets.CascadedASPPNet(arg_n_fft)
model_path = os.path.join(MODEL_DIRECTORY, arg_pretrained_model)
model.load_state_dict(torch.load(model_path, map_location=device))
if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(arg_gpu))
    model.to(device)
print('done')


# Main API
@api.route("/api/vrv/<uid>", methods=["POST"])
def process_vrv(uid):
    if "/" in uid:
        # Return 400 BAD REQUEST
        abort(400, "No subdirectories allowed")

    referer = request.headers.get("Referer")
    print('referer: ', referer)
    if "songmaker" not in referer.lower():
        print('Illegal header')
        abort(400, "No subdirectories allowed")

    input_path = os.path.join(UPLOAD_DIRECTORY, '{}.mp3'.format(uid))
    uploaded_file = request.files['file']
    uploaded_file.save(input_path)

    output_path = execute_vr(model, device, input_path, DOWNLOAD_DIRECTORY, uid)
    if os.path.isfile(output_path):
        os.remove(input_path)
        return send_file(output_path)
    else:
        abort(400, "Empty output file")

@api.route("/api/vrv_done/<uid>")
def cleanup_vrv(uid):
    if "/" in uid:
        # Return 400 BAD REQUEST
        abort(400, "No subdirectories allowed")
    referer = request.headers.get("Referer")
    print('referer: ', referer)

    # uploaded file
    path = os.path.join(UPLOAD_DIRECTORY, '{}.mp3'.format(uid))
    if os.path.exists(path):
        os.remove(path)
    # downloaded file
    path = os.path.join(DOWNLOAD_DIRECTORY, '{}.mp3'.format(uid))
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(DOWNLOAD_DIRECTORY, '{}.wav'.format(uid))
    if os.path.exists(path):
        os.remove(path)

    return "", 200

@api.route("/api/files")
def list_files():
    referer = request.headers.get("Referer")
    print('referer: ', referer)
    if "songmaker" not in referer.lower():
        print('Illegal header')
        abort(400, "No subdirectories allowed")

    files = []
    for filename in os.listdir(DOWNLOAD_DIRECTORY):
        path = os.path.join(DOWNLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)

@api.route("/ping")
def ping():
    return "pong", 200


if __name__ == "__main__":
    api.run(debug=True, port=5000)
