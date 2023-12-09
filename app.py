

import numpy as np
from flask import Flask, request, render_template ,session, url_for,abort,redirect
import pickle

app=Flask(__name__)
model1=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Newpage')
def new_page():
    return render_template('Newpage.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features=[float(x) for x in request.form.values()]
    print(int_features)
    final_features=[np.array(int_features)]
  
    output=model1.predict(final_features)
    if output == 0:
        outp="negative"
    else:
        outp="positive"
    if outp=="positive":
        #return render_template('index.html',prediction_text='Your result is {}'.format(outp))
        return render_template('result.html',prediction_text='DEMENTED')
    else:
        return render_template('result.html',prediction_text='NON-DEMENTED')


if __name__=="__main__":
    app.run(debug=True)
    
    
    # {{url_for('static', filename='css/style.css')}}">