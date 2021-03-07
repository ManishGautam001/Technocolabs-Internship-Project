from flask import Flask, request, render_template
import pickle
import pandas as pd
import math
import numpy as np
app = Flask(__name__)

model = pickle.load(open("mlp.pkl","rb"))




@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')



@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        myDict = request.form
        Recency = int(myDict['Recency'])
        Frequency = int(myDict['Frequency'])
        Monetary = float(math.log10(int(myDict['Monetary'])))
        Time = int(myDict['Time'])
        prediction=model.predict([[Recency,Frequency,Monetary,Time]])
        output = prediction[0]
        return render_template('show.html',prediction_text=output)


if __name__ == "__main__":
    app.run(debug=True)