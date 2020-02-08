# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 03:26:10 2020

@author: User
"""

from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
@app.route('/')
def hello_world():
    return render_template("loan_prediction.html")
@app.route("/check_status", methods=["POST","GET"])
def check_status():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proba(final)
    output="{0:.{1}f}".format(prediction[0][1], 2)
    if output>str(0.5):
        return render_template("loan_prediction.html",prediction_text="This Client is going to get House insurance, get the money set.\n the estimated probability is {}".format(output))
    else:
        return render_template("loan_prediction.html", prediction_text= "This Client won't be here for his house insurance\n the estimated probability is {}".format(output))
if __name__ =="__main__":
    app.run()