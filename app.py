import pandas as pd
import requests
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load and prepare the dataset
data = pd.read_csv('cardio_train.csv', delimiter=';')
X = data.drop(columns=['id', 'cardio'])
y = data['cardio']
X = pd.get_dummies(X, columns=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# Save the model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Reload model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to fetch ThinkSpeak data
def fetch_thinkspeak_data():
    CHANNEL_ID = '2720455'
    READ_API_KEY = '56QHX9QZLRIWROGK'
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1'

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        latest_entry = data['feeds'][0]
        return {
            'cholesterol': int(latest_entry['field1']),
            'glucose': int(latest_entry['field2']),
            'ap_hi': int(latest_entry['field3']),
            'ap_lo': int(latest_entry['field4'])
        }
    except Exception as err:
        print("Error fetching ThinkSpeak data:", err)
        return None

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    user_data = request.get_json()
    user_data['age'] = user_data['age'] * 365
    user_data['height'] = user_data['height'] * 2.54

    # Fetch ThinkSpeak data
    sensor_data = fetch_thinkspeak_data()
    if not sensor_data:
        return jsonify({"error": "Failed to retrieve sensor data"}), 500

    data_input = pd.DataFrame({
        'age': [user_data['age']],
        'gender': [user_data['gender']],
        'height': [user_data['height']],
        'weight': [user_data['weight']],
        'ap_hi': [sensor_data['ap_hi']],
        'ap_lo': [sensor_data['ap_lo']],
        'cholesterol': [sensor_data['cholesterol']],
        'gluc': [sensor_data['glucose']],
        'smoke': [user_data['smoke']],
        'alco': [user_data['alco']],
        'active': [user_data['active']]
    })

    data_input = pd.get_dummies(data_input, columns=['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'])
    data_input = data_input.reindex(columns=X.columns, fill_value=0)
    data_scaled = scaler.transform(data_input)

    prediction = model.predict(data_scaled)[0]
    result = {"prediction": int(prediction)}

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
