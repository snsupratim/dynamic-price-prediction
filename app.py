from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Update your model path and load the model
ride_cost_model_path = 'dynamic-pricing.pkl'  # Replace with your actual model path
with open(ride_cost_model_path, 'rb') as file:
    ride_cost_model = pickle.load(file)

app = Flask(__name__)

# Define the route for prediction
@app.route('/ride_cost_predict', methods=['POST'])
def ride_cost_predict():
    # Extract data from form for ride cost prediction
    number_of_riders = request.form.get('Number_of_Riders', type=float)
    number_of_drivers = request.form.get('Number_of_Drivers', type=float)
    vehicle_type = request.form.get('Vehicle_Type')  # Vehicle type as string
    expected_ride_duration = request.form.get('Expected_Ride_Duration', type=float)

    # Convert vehicle type to numerical (premium = 1, economy = 0)
    if vehicle_type.lower() == 'premium':
        vehicle_type = 1
    elif vehicle_type.lower() == 'economy':
        vehicle_type = 0
    else:
        return render_template('ride_cost.html', prediction_text='Invalid Vehicle Type. Please enter "premium" or "economy".')

    # Validate inputs
    if (number_of_riders is None or number_of_drivers is None or
        vehicle_type is None or expected_ride_duration is None):
        return render_template('ride_cost.html', prediction_text='Invalid input. Please provide all fields.')

    # Create numpy array for prediction
    final_features = np.array([[number_of_riders, number_of_drivers,
                                vehicle_type, expected_ride_duration]])

    # Make ride cost prediction
    predicted_cost = ride_cost_model.predict(final_features)[0]

    return render_template('ride_cost.html', prediction_text='Predicted Adjusted Ride Cost: ${:.2f}'.format(predicted_cost))

# Define another route for the main page
@app.route('/')
def main_page():
    return render_template('ride_cost.html')

if __name__ == "__main__":
    app.run(debug=True)
