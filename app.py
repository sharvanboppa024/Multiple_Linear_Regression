
from flask import Flask, render_template, request
import joblib
import numpy as np
import traceback

app = Flask(__name__)

# Load your existing pickle model
model = joblib.load("MLR_MODEL.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    values = {
        "rnd": request.form.get("rnd", "").strip(),
        "admin": request.form.get("admin", "").strip(),
        "marketing": request.form.get("marketing", "").strip(),
        "state": request.form.get("state", "").strip()
    }

    try:
        # Convert integers and decimals to float
        rnd = float(values["rnd"])
        admin = float(values["admin"])
        marketing = float(values["marketing"])
        state = float(values["state"])

        # Validate numeric inputs
        inputs = [rnd, admin, marketing, state]

        if not all(np.isfinite(inputs)):
            raise ValueError("All inputs must be finite numbers.")

        if state not in [0, 1, 2]:
            raise ValueError("Select a valid state.")

        # Four numeric features in the model's expected order
        input_data = np.array(
            [[rnd, admin, marketing, state]],
            dtype=float
        )

        print("Input sent to model:", input_data)
        print("Model expects features:", model.n_features_in_)

        # Make prediction
        result = model.predict(input_data)[0]

        print("Predicted profit:", result)

        return render_template(
            "index.html",
            prediction_text=f"{result:,.2f}",
            form_values=values
        )

    except Exception as e:
        traceback.print_exc()

        return render_template(
            "index.html",
            error_text=f"{type(e).__name__}: {e}",
            form_values=values
        )


if __name__ == "__main__":
    app.run(debug=True)
