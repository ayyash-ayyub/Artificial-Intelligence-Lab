from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

model_file = open('model_emas.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')


@app.route('/')
def index():
    return render_template('index.html', hasil=0)

@app.route('/predict', methods=['POST'])
def predict():
    
    tahun=float(request.form['tahun'])
    
    x=np.array([[tahun]])
    
    prediction=model.predict(x)
    output = round(prediction[0],2)

    return render_template('index.html', hasil=output)

if __name__ == '__main__':
    app.run(debug=True)
