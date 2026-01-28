from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
model = joblib.load('model/house_price_model.pkl')
scaler = joblib.load('model/scaler.pkl')
label_encoder = joblib.load('model/label_encoder.pkl')

# Get the list of neighborhoods from the label encoder
neighborhoods = list(label_encoder.classes_)

@app.route('/')
def home():
    return render_template('index.html', neighborhoods=neighborhoods)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        overall_qual = int(request.form['overall_qual'])
        gr_liv_area = float(request.form['gr_liv_area'])
        total_bsmt_sf = float(request.form['total_bsmt_sf'])
        garage_cars = int(request.form['garage_cars'])
        year_built = int(request.form['year_built'])
        neighborhood = request.form['neighborhood']
        
        # Encode the neighborhood
        neighborhood_encoded = label_encoder.transform([neighborhood])[0]
        
        # Create feature array
        features = np.array([[overall_qual, gr_liv_area, total_bsmt_sf, 
                             garage_cars, year_built, neighborhood_encoded]])
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Return the prediction
        return render_template('index.html', 
                             neighborhoods=neighborhoods,
                             prediction_text=f'Predicted House Price: ${prediction:,.2f}',
                             overall_qual=overall_qual,
                             gr_liv_area=gr_liv_area,
                             total_bsmt_sf=total_bsmt_sf,
                             garage_cars=garage_cars,
                             year_built=year_built,
                             selected_neighborhood=neighborhood)
    
    except Exception as e:
        return render_template('index.html', 
                             neighborhoods=neighborhoods,
                             prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)