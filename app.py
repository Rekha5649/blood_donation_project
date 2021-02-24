import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.bin','rb'))

@app.route('/')
def home():
    return render_template('project.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.arrary(int_features)]
    prediction = model.predict(final_features[1:])
    if prediction == 0:
        return render_template('project.html', prediction_text = f'{final_features[0]}, you are not able to donate the blood')

    else:
        return render_template('project.html', prediction_text = f'{final_features[0]}, you are allright and able to donate the blood')

if __name__ == '__main__':
    app.run(debug=True)