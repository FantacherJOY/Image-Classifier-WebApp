from flask import render_template,Flask, request, url_for
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
model = load_model('facefeatures_new_model.h5')
# from forms import image_file
import os
# from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config["IMAGE_UPLOADS"] = "static/profile_pics"
# img_dr='static/profile_pics'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
# db = SQLAlchemy(app)

# class User(db.Model):
#     image_file = db.Column(db.String(20), nullable=False, default='default.jpg')
#     def __repr__(self):
#         return f"User('{self.image_file}')"

# def save_picture(form_picture):
#     # random_hex = secrets.token_hex(8)
#     # _, f_ext = os.path.splitext(form_picture.filename)
#     # picture_fn = random_hex + f_ext
#     form_picture.save(os.path.join(app.root_path, 'static/profile_pics', form_picture))

#     return


def imagePred():
    pred_set = ImageDataGenerator(rescale = 1./255)
    pred_set = pred_set.flow_from_directory('static',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
    a=model.predict(pred_set)
    b=a.argmax(axis=-1)
    for i in range(len(b)):
        if b[i]==1:
            b='dog'
        else:
            b='cat'
    return b
def rmb():
    path = 'static/profile_pics'
    # filelist = glob.glob(os.path.join(path, "*.bak"))
    # for f in filelist:
    #     os.remove(f)
        
    # filelist = [ f for f in os.listdir(path) if f.endswith(".jpg",".png") ]
    # for f in filelist:
    #     os.remove(os.path.join(path, f))
    for root, dirs, files in os.walk(path):
        for file in files:
            os.remove(os.path.join(root, file))
    return  
# print(imagePred())
# img_cat=imagePred()
@app.route("/")
@app.route("/home", methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        rmb()
        if request.files:
            image = request.files["image"]
            img=os.path.join(app.config["IMAGE_UPLOADS"], image.filename)
            image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            img_cat=imagePred()
            # os.remove(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
            return render_template('result.html', img_cat=img_cat, img=img)
        # os.remove(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
    return render_template('home.html', title="IMG App")



# @app.route("/upload-image", methods=["GET", "POST"])
# def upload_image():

#     if request.method == "POST":

#         if request.files:

#             image = request.files["image"]

#             image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))

#             print("Image saved")

#             return redirect(request.url)

#     return render_template("public/upload_image.html")
        
if __name__ == '__main__':
    app.run(debug=True)
