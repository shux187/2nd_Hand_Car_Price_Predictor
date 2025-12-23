import pickle
import pandas as pd 
import numpy as np 
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# --- 1. DATA LOADING ---
# Load your dataset to know which models belong to which brands
DATA_PATH = r"C:\Users\Administrator\Desktop\ML-Projects\2nd_Hand_Car_Price_Predictor\data\raw_data.xlsx"
try:
    df = pd.read_excel(DATA_PATH)
    # Ensure columns are stripped of whitespace
    df.columns = df.columns.str.strip()
except Exception as e:
    print(f"Error loading Excel file: {e}")
    df = pd.DataFrame()

# --- 2. MODEL LOADING ---
# Use relative paths or full paths as needed
model = pickle.load(open('./models/Model.pkl', 'rb'))
Brand_Encoder = pickle.load(open('./models/Brand_Encoder.pkl', 'rb'))
Model_Encoder = pickle.load(open('./models/Model_Encoder.pkl', 'rb'))
OneHot_Encoder = pickle.load(open('./models/OneHot_Encoder.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

# --- 3. THE MISSING ROUTE (Fixes the 404) ---
@app.route('/get_models/<brand>')
def get_models(brand):
    if df.empty:
        return jsonify({"models": [], "error": "Dataset not loaded"}), 500
    
    # Filter models based on the selected brand
    filtered_models = df[df['Brand'].str.lower() == brand.lower()]['Model'].unique().tolist()
    filtered_models.sort() # Sort alphabetically for better UX
    
    return jsonify({"models": filtered_models})

@app.route('/test-models')
def test_models():
    """Debug route to check available brands and models"""
    if not df.empty:
        result = "<h1>Available Brands and Models</h1>"
        for brand in sorted(df['Brand'].unique()):
            models = df[df['Brand'] == brand]['Model'].unique().tolist()
            result += f"<h3>{brand} ({len(models)} models)</h3>"
            result += f"<p>{', '.join(models[:10])}{'...' if len(models) > 10 else ''}</p>"
        return result
    else:
        return "<h1>Dataset not loaded</h1>"

@app.route("/predict", methods=['POST'])
@app.route("/predict", methods=['POST'])
def predict():
    try:
        # 1. Get the data regardless of format (JSON or Form)
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # 2. Extract values safely from the 'data' variable
        Brand = data.get('Brand')
        Model = data.get('Model')
        Fuel = data.get('Fuel')
        Transmission = data.get('Transmission')
        Year = int(data.get('Year', 2022))
        EngineSize = float(data.get('EngineSize', 2.0))
        Mileage = int(data.get('Mileage', 50000))
        Doors = int(data.get('Doors', 4))
        OwnerCount = int(data.get('OwnerCount', 1))

        # 3. Create a DataFrame for the input data
        input_df = pd.DataFrame({
            'Brand': [Brand],
            'Model': [Model],
            'Fuel': [Fuel],
            'Transmission': [Transmission],
            'Year': [Year],
            'EngineSize': [EngineSize],
            'Mileage': [Mileage],
            'Doors': [Doors],
            'OwnerCount': [OwnerCount]
        })

        # 4. Encoding Logic
        input_df['Encoded_Brand'] = input_df['Brand'].map(Brand_Encoder).fillna(0)
        input_df['Encoded_Model'] = input_df['Model'].map(Model_Encoder).fillna(0)
        input_df.drop(['Brand', 'Model'], axis=1, inplace=True)

        # One-hot encode Fuel and Transmission
        categorical_cols = ['Fuel', 'Transmission']
        encoded_array = OneHot_Encoder.transform(input_df[categorical_cols])
        encoded_df = pd.DataFrame(encoded_array, columns=OneHot_Encoder.get_feature_names_out(categorical_cols))

        # Merge encoded columns
        input_df_encoded = input_df.drop(columns=categorical_cols).reset_index(drop=True)
        input_data = pd.concat([input_df_encoded, encoded_df], axis=1)

        # 5. Make prediction
        prediction = model.predict(input_data)
        # Convert to standard float because numpy types aren't JSON-friendly
        output = round(float(prediction[0]), 2)

        # 6. Send response back based on how it was requested
        result_text = f"Car is worth at: $ {output}" if output > 0 else "Sorry, you cannot sell this car."
        
        if request.is_json:
            return jsonify({
                "prediction": output,
                "prediction_text": result_text
            })
        else:
            return render_template('index.html', prediction_text=result_text)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)