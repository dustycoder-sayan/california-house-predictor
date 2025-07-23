from flask import Flask, request, jsonify
from flask_cors import cross_origin
from numpy import array
from joblib import load

app = Flask(__name__)

HOUSE_PRICE_PREDICTOR_FILE = "model/house_pricer.joblib"
HOUSE_PRICE_SCALER_FILE = "model/house_pricing_scaler.joblib"

house_price_scaler = load(HOUSE_PRICE_SCALER_FILE)
house_price_predictor = load(HOUSE_PRICE_PREDICTOR_FILE)

@app.route("/", methods=["POST"])
def predict_house_price():
    request_data = request.json
    if request_data:
        try:
            MedInc = float(request_data.get("MedInc"))
            HouseAge = float(request_data.get("HouseAge"))
            AveRooms = float(request_data.get("AveRooms"))
            AveBedrms = float(request_data.get("AveBedrms"))
            Population = float(request_data.get("Population"))
            AveOccup = float(request_data.get("AveOccup"))
            Latitude = float(request_data.get("Latitude"))
            Longitude = float(request_data.get("Longitude"))
        except:
            raise ValueError("House Data not propagated as expected")
        
        price = array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
        price = house_price_scaler.transform(price)

        target_name = house_price_predictor.predict(price).tolist()[0]
        return jsonify({"house_price": target_name})
    else:
        raise ValueError("House Data not sent with Request")

if __name__ == "__main__":
    app.run(debug=True)