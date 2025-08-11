from flask import Flask, request, render_template
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from the pickle file
with open('heart_disease_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')  # Render the homepage with the form

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = [
            float(request.form['age']),
            float(request.form['sex']),
            float(request.form['cp']),
            float(request.form['trestbps']),
            float(request.form['chol']),
            float(request.form['fbs']),
            float(request.form['restecg']),
            float(request.form['thalach']),
            float(request.form['exang']),
            float(request.form['oldpeak']),
            float(request.form['slope']),
            float(request.form['ca']),
            float(request.form['thal'])
        ]
        
        # Convert to numpy array and reshape for prediction
        input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

        # Make prediction
        prediction = model.predict(input_data_as_numpy_array)

        # Map prediction to result
        result = "The person has heart disease" if prediction[0] == 1 else "The person does not have heart disease"
        
        # Return result as plain text to be displayed on the page
        return result

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
