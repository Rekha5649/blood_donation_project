import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
model_normalizer = pickle.load(open('model_normalizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('project.html')

@app.route('/predict', methods = ['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    int_features = [np.array(int_features)]
    int_features = model_normalizer.transform(int_features)
    print('-----------------------------------------------',int_features,'--------------------------------------------------------------')
    final_features = [np.array(int_features)]
    print('-----------------------------------------------',len(final_features),'-----------------------------------------------------')
    prediction = model.predict(final_features)
   
    if prediction == 0:
        return render_template('project.html', prediction_text = f'Result: You are not able to donate the blood.')

    else:
        return render_template('project.html', prediction_text = f'Result: You are able to donate the blood.')

if __name__ == '__main__':
    app.run(debug=True)