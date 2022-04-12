import numpy as np
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
import pickle
#from ckd_prediction_ import sc

app = Flask(__name__)# app creation
model = pickle.load(open('lgmodel.pkl', 'rb'))# lloading pkl
scalr = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():                                #home page 
    return render_template('index.html')
 

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [[float(x) for x in request.form.values()]]
    final_fea=scalr.transform(int_features)
   
   
    #features=sc.transform(int_features)
    prediction = model.predict(final_fea)
    pred2 = model.predict_proba(final_fea)
    output2 = '{0:.{1}f}'.format(pred2[0][1],2)

    if(prediction[0]==0.):
        output='DOESNT HAVE CKD'
    else:
        output=' HAS CKD'
    return render_template('index.html', prediction_text='THE PATIENT  {} with the probability {}'.format(output, output2))


if __name__ == "__main__":
    app.run(debug=True)