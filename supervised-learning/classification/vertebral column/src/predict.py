import numpy as np

def predict(trained_model, scaler, input_data, model_key='xgb_model'):
    input_data = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    # Extracting Individual Models From Dictionary
    model = trained_model[model_key]
    prediction = model.predict(input_scaled)

    return prediction[0]
