from flask import Flask, render_template, url_for, request
from flask_material import Material
import numpy as np
from sklearn.externals.joblib import load as ld

app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze',methods=['POST'])
def analyze():
    if request.method=='POST':
        sepal_length = request.form['sepal_length']
        sepal_width = request.form['sepal_width']
        petal_length = request.form['petal_length']
        petal_width = request.form['petal_width']

        dt = [sepal_length,sepal_width,petal_length,petal_width]
        X = np.array([float(i) for i in dt]).reshape(1,-1)

        mymodel = ld("data/iris_test.sav")
        result = mymodel.predict(X)
        result = result[0]

    return render_template('index.html', sepal_length=sepal_length,
                           sepal_width=sepal_width,
                           petal_length=petal_length,
                           petal_width=petal_width,
                           result=result)

if __name__ == '__main__':
    app.run()
