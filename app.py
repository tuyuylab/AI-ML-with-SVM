from flask import Flask, render_template, request
import pickle
import numpy as np
import gzip


with gzip.open('model.pkl', 'rb') as f:
    model = pickle.load(f)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('crop4.html')

@app.route('/predict', methods = ['POST'])
def recomend_crop():
    age = float(request.form.get('age'))
    anaemia = float(request.form.get('anaemia'))
    creatinine_phosphokinase = float(request.form.get('creatinine_phosphokinase'))
    diabetes = float(request.form.get('diabetes'))
    ejection_fraction = float(request.form.get('ejection_fraction'))
    high_blood_pressure = float(request.form.get('high_blood_pressure'))
    platelets = float(request.form.get('platelets'))
    serum_creatinine = float(request.form.get('serum_creatinine'))
    serum_sodium = float(request.form.get('serum_sodium'))
    sex = float(request.form.get('sex'))
    smoking = float(request.form.get('smoking'))
    time = float(request.form.get('time'))


    # prediction
    crops = ['Hearat Fail','Heart Healthy']
    result = model.predict(np.array([age	,anaemia,	creatinine_phosphokinase,	diabetes,	ejection_fraction,	high_blood_pressure,	platelets,	serum_creatinine	,serum_sodium,	sex,	smoking,	time
]).reshape(1,12))
    index = int(result[0])
    # image = Image.open(f"crops/{crops[index]}.jpg")
    # image.show()
    result = crops[index]
    return render_template('crop4.html', result = result)


if __name__ == '__main__':
    app.run(debug = True,port=5001)
