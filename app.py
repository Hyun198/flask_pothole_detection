from pyexpat import model
from flask import Flask, render_template, request, session , url_for
import torch
from werkzeug.utils import secure_filename
import os
from GPX_read import *
from map import *
from frame import *
from model import *
from make_vid import *
#from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
app.secret_key = 'You will never guess'


#model = torch.hub.load('.', 'custom', path='bestofbest.pt', source='local')


VIDEO_UPLOAD_FOLDER = os.path.join('static','video')
GPX_UPLOAD_FOLDER = os.path.join('static','gpx')

app.config['VIDEO_UPLOAD_FOLDER'] = VIDEO_UPLOAD_FOLDER
app.config['GPX_UPLOAD_FOLDER'] = GPX_UPLOAD_FOLDER


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["GET","POST"])
def upload():
    return render_template("public/upload.html")


@app.route("/upload2", methods=["GET","POST"])
def upload2():
    if request.method == 'POST':
        uploaded_vid = request.files['uploaded-file']
        vid_filename = secure_filename(uploaded_vid.filename)
        uploaded_vid.save(os.path.join(app.config['VIDEO_UPLOAD_FOLDER'],vid_filename))
        session['uploaded_vid_file_path'] = os.path.join(app.config['VIDEO_UPLOAD_FOLDER'],vid_filename)

    return render_template('public/upload2.html')    


@app.route('/upload_gpx',methods=["GET","POST"])
def upload_gpx():
    if request.method == 'POST':
        uploaded_gpx = request.files['uploaded-gpxfile']
        gpx_filename = secure_filename(uploaded_gpx.filename)
        uploaded_gpx.save(os.path.join(app.config['GPX_UPLOAD_FOLDER'],gpx_filename))

    vid_file_path = session.get('uploaded_vid_file_path',None)

    GPX_path = "C:/Users/user-pc/Desktop/visual/project/static/gpx/"
    new_path = GPX_path + gpx_filename
    #read gpx file
    read_gpx(new_path)

    #read csv coordinates
    csv_path = 'C:/Users/user-pc/Desktop/visual/project/static/csv/route_df.csv'
    read_csv(csv_path)

    #frame divide
    frame_path = 'C:/Users/user-pc/Desktop/visual/project/static/video'
    frame(frame_path)

    #frame detect/run model
    run_model()
    make_vid()

    return render_template('public/display_vid.html', upload_vid=vid_file_path, path = frame_path)


@app.route('/map')
def map():
    return render_template('map.html')

@app.route('/gallery')
def gallery():

    hists = os.listdir('C:/Users/user-pc/Desktop/visual/project/static/result_img/')
    hists = [''+ file for file in hists]

    return render_template('gallery.html',hists=hists)


if __name__ == "__main__":
    app.run(debug=True)