import pandas as pd
import pickle
from flask import Flask, render_template, request
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))
df.head()

# Create a Flask app
app = Flask(__name__)

@app.route('/')
def index():

    locations=sorted(df['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    
        # Get the input data from the form
       # input_data = request.form

        # Convert the input data to a Pandas DataFrame
       # input_data = pd.DataFrame(input_data, index=[0])

        # Run the model and get the prediction result
       # result = predict(input_data)
      location=request.form.get('location')
      bhk=request.form.get('bhk')
      bath =request.form.get('bath')
      sqft=request.form.get('total_sqft')
      
      print(location,bhk,bath,sqft)
      input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','BHK'])
      prediction=pipe.predict(input)[0]
     # input=pd.DataFrame([['1st Phase JP Nagar',np.log(1875),3,3]],columns=['location','total_sqft','bath','BHK'])
     # prediction=pipe.predict(input)[0]
      prediction=np.exp(prediction)
      prediction=np.round(prediction,2)
    
      return render_template('result.html', result=prediction)
    

if __name__=="__main__":
    app.run(debug=True)
     