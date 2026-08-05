"""Simple HTTP server for TEC surrogate model predictions."""
import json
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler

import joblib
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(PROJECT / "data" / "models" / "comsol_latent_surrogate.joblib")


def predict(inputs: list[float]) -> dict:
    """Run PCA + SVR prediction pipeline."""
    X = np.array([inputs])  # (1, 10)
    Xs = MODEL["input_scaler"].transform(X)

    # Predict PCA scores
    latent = MODEL["regressor"].predict(Xs)  # (1, k)

    # Reconstruct full output
    Ys = MODEL["pca"].inverse_transform(latent)  # (1, D)
    Y = MODEL["output_scaler"].inverse_transform(Ys)[0]

    # Map back to named outputs
    result = {}
    for i, name in enumerate(MODEL["output_names"]):
        result[name] = float(Y[i])

    # 6 scalars
    for name in MODEL["scalar_names"]:
        result[name] = result.get(name, None)

    # COP surface as 3x8 grid
    cop = np.array([result.get(f"COP_dT{dT}K_I0_{cur}", np.nan)
                     for dT in MODEL["delta_t_values"]
                     for cur in MODEL["currents"]]).reshape(
        len(MODEL["delta_t_values"]), len(MODEL["currents"]))

    result["cop_surface"] = cop.tolist()
    result["currents"] = MODEL["currents"]
    result["delta_t_values"] = MODEL["delta_t_values"]
    result["latent"] = latent[0].tolist()

    return result


class APIHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/predict":
            self.send_error(405, "Use POST")
        elif self.path == "/" or self.path == "/index.html":
            self.path = "/legacy/web/predict_ui.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/predict":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            data = json.loads(body)

            try:
                result = predict(data["inputs"])
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode())
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main():
    port = 9876
    server = HTTPServer(("localhost", port), APIHandler)
    print(f"TEC Surrogate UI: http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
