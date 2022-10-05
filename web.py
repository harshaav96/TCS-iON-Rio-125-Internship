import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    features = [int(x) for x in request.form.values()]
    print(features)
    final_feat = [np.array(features)]
    print(final_feat)
    prediction = model.predict(final_feat)
    print("prediction:",prediction)
    if prediction == 0:
        prediction='Low cost'
    elif  prediction == 1:
        prediction='Medium cost'
    elif prediction == 2:
        prediction='High cost'
    else:
        prediction='Very high cost'   
    return render_template('result.html', prediction_text = 'Price Score is {}'.format(prediction))

if __name__ == "__main__":
    app.run(port=8000)
