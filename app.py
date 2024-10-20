import pickle
from flask import Flask, request, app, render_template, jsonify
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))
inverse = pickle.load(open('inverse.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_data = [
            int(data['Type']),  # Ensure 'Type' is consistent with your training data (object type might need to be handled)
            float(data['Air temperature [K]']),
            float(data['Process temperature [K]']),
            int(data['Rotational speed [rpm]']),
            float(data['Torque [Nm]']),
            int(data['Tool wear [min]']),
        ]
        # print("Input Data:\n", input_data)
        
          # Column names to match the training data
        column_names = ['Type', 'Air temperature [K]', 'Process temperature [K]', 
                        'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        input_df = pd.DataFrame([input_data], columns=column_names)
        # print("Input DataFrame:\n", input_df)

        
        prediction = model.predict(input_df)
        print("Prediction:", prediction)

        # Map the prediction if available in the inverse dictionary
        mapped_prediction = pd.Series(prediction).map(inverse).iloc[0]
        # print("Mapped Prediction:", mapped_prediction)
        return jsonify(prediction_text=str(mapped_prediction))
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify(prediction_text= "An error occurred during prediction")

if __name__ == "__main__":
    app.run(debug=True)

