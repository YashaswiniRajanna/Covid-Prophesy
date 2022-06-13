import numpy as np
from flask import Flask, render_template, request,jsonify
import pickle


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    if(prediction==0.0):
        output="Does NOT have COVID"
    else:
        output="HAS COVID"
    return render_template("index.html", prediction_text = "The patient {}".format(output))

    # if(prediction[0]==0.):
    #     output='DOESNT HAVE covid'
    # else:
    #     output=' HAS covid'
    # return render_template("index.html", prediction_text="THE PATIENT{} with the probability {}" .prediction.predformat(output))

    # if request.method=="POST":
    #     name=request.form["username"]
    # # data1=request.form['bp']
    # # data2=request.form['fev']
    # # data3=request.form['cou']
    # # data4=request.form['st']
    # # data5=request.form['ht']
    # # data6=request.form['at']
    # # data7=request.form['con']
    # # data8=request.form['attend']
    # # data9=request.form['visit']
    # # data10=request.form['fam']
    # # arr=np.array([[data1,data2,data3,data4,data5,data6,data7,data8,data9,data10]])
    # # pred=model.predict(arr)
    # return render_template("after.html",n=name)




if __name__ == "__main__":
    app.run(debug=True)
