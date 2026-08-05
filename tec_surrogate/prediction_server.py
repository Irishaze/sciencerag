"""Local HTTP server for the TEC latent-surrogate prediction UI."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import joblib
import numpy as np

from physics_foundation import MultiPairFieldService


PROJECT = Path(__file__).parent
WEB_ROOT = PROJECT / "web"
MODEL_PATH = PROJECT / "data" / "models" / "comsol_latent_surrogate.joblib"
DATASET_PATH = PROJECT / "data" / "processed" / "comsol_report_dataset.npz"
SUMMARY_PATH = PROJECT / "outputs" / "comsol_latent_training.json"
FIELD_CHECKPOINT_PATH = PROJECT / "outputs" / "component_graph_seponet_20pairs.pt"
FIELD_SUMMARY_PATH = PROJECT / "outputs" / "component_graph_seponet_20pairs.json"
FIELD_CASE_PATH = PROJECT / "data" / "component_cases" / "tec_1pair_dset3.npz"

INPUT_LABELS = {
    "length_mm": ("总长", "mm", 0.01),
    "width_mm": ("总宽", "mm", 0.01),
    "height_mm": ("总高", "mm", 0.01),
    "d_conductor_um": ("导体厚度", "um", 0.1),
    "d_ceramics_mm": ("陶瓷厚度", "mm", 0.001),
    "leg_length_mm": ("臂截面长", "mm", 0.001),
    "leg_width_mm": ("臂截面宽", "mm", 0.001),
    "pitch_mm": ("间距", "mm", 0.001),
    "n_pairs": ("PN 对数", "对", 1),
    "Tref_K": ("热侧温度", "K", 0.1),
}

SCALAR_LABELS = {
    "delta_T_max_K": ("最大温差", "K", 2),
    "optimal_current_A": ("最优电流", "A", 3),
    "optimal_voltage_V": ("最优电压", "V", 4),
    "total_resistance_ohm": ("总电阻", "ohm", 5),
    "max_heat_dissipation_W": ("最大热耗散", "W", 4),
    "figure_of_merit_1_per_K": ("品质因子", "1/K", 6),
}


class PredictionService:
    def __init__(self) -> None:
        if not MODEL_PATH.exists() or not DATASET_PATH.exists() or not SUMMARY_PATH.exists():
            raise FileNotFoundError("Train the report model before starting the prediction UI")
        self.model = joblib.load(MODEL_PATH)
        self.dataset = np.load(DATASET_PATH)
        self.summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.X = np.asarray(self.dataset["X"], dtype=float)
        self.Y = np.asarray(self.dataset["Y"], dtype=float)
        self.input_names = list(self.model["input_names"])
        self.scalar_names = list(self.model["scalar_names"])

        training_scaled = self.model["output_scaler"].transform(self.Y)
        training_latent = self.model["pca"].transform(training_scaled)
        self.training_latent_2d = np.column_stack(
            [
                training_latent[:, 0],
                training_latent[:, 1] if training_latent.shape[1] > 1 else np.zeros(len(training_latent)),
            ]
        )

    def metadata(self) -> dict[str, Any]:
        inputs = []
        defaults = np.median(self.X, axis=0)
        for index, name in enumerate(self.input_names):
            label, unit, step = INPUT_LABELS[name]
            minimum = float(self.X[:, index].min())
            maximum = float(self.X[:, index].max())
            default = float(defaults[index])
            if name == "n_pairs":
                default = float(round(default))
            inputs.append(
                {
                    "name": name,
                    "label": label,
                    "unit": unit,
                    "step": step,
                    "min": minimum,
                    "max": maximum,
                    "default": default,
                }
            )

        scalar_outputs = []
        for name in self.scalar_names:
            label, unit, precision = SCALAR_LABELS[name]
            scalar_outputs.append(
                {"name": name, "label": label, "unit": unit, "precision": precision}
            )

        return {
            "model": {
                "type": self.model["model_type"],
                "latent_dim": int(self.model["latent_dim"]),
                "sample_count": len(self.X),
                "retained_variance": self.summary["retained_variance"],
                "cop_cv_mae": self.summary["cross_validation"]["cop"]["mae"],
                "cop_cv_r2": self.summary["cross_validation"]["cop"]["r2_variance_weighted"],
            },
            "inputs": inputs,
            "scalar_outputs": scalar_outputs,
            "currents": [float(value) for value in self.model["currents"]],
            "delta_t_values": [float(value) for value in self.model["delta_t_values"]],
            "training_latent": self.training_latent_2d.tolist(),
            "training_files": list(self.model["training_filenames"]),
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        supplied = payload.get("inputs")
        if not isinstance(supplied, dict):
            raise ValueError("inputs must be an object")

        values = []
        warnings = []
        for index, name in enumerate(self.input_names):
            if name not in supplied:
                raise ValueError(f"Missing input: {name}")
            try:
                value = float(supplied[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid numeric input: {name}") from exc
            if not np.isfinite(value):
                raise ValueError(f"Input must be finite: {name}")
            if name == "n_pairs" and not np.isclose(value, round(value)):
                raise ValueError("PN pair count must be an integer")
            minimum = float(self.X[:, index].min())
            maximum = float(self.X[:, index].max())
            if value < minimum or value > maximum:
                warnings.append(
                    {
                        "name": name,
                        "label": INPUT_LABELS[name][0],
                        "value": value,
                        "min": minimum,
                        "max": maximum,
                    }
                )
            values.append(value)

        X = np.asarray(values, dtype=float)[None, :]
        input_scaled = self.model["input_scaler"].transform(X)
        latent = self.model["regressor"].predict(input_scaled)
        if latent.ndim == 1:
            latent = latent[:, None]
        output_scaled = self.model["pca"].inverse_transform(latent)
        output = self.model["output_scaler"].inverse_transform(output_scaled)[0]
        n_scalars = len(self.scalar_names)
        cop = output[n_scalars:].reshape(
            len(self.model["delta_t_values"]), len(self.model["currents"])
        )

        return {
            "scalars": {
                name: float(output[index]) for index, name in enumerate(self.scalar_names)
            },
            "cop_surface": cop.tolist(),
            "latent": latent[0].tolist(),
            "latent_2d": [float(latent[0, 0]), float(latent[0, 1] if latent.shape[1] > 1 else 0.0)],
            "outside_training_range": warnings,
        }


class PredictionHandler(BaseHTTPRequestHandler):
    service: PredictionService
    field_service: MultiPairFieldService | None = None

    @classmethod
    def get_field_service(cls) -> MultiPairFieldService:
        if cls.field_service is None:
            cls.field_service = MultiPairFieldService(
                FIELD_CHECKPOINT_PATH,
                FIELD_CASE_PATH,
                FIELD_SUMMARY_PATH,
            )
        return cls.field_service

    def log_message(self, format_string: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format_string % args}")

    def _send_json(self, body: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/meta":
            self._send_json(self.service.metadata())
            return
        if path == "/api/field-meta":
            self._send_json(self.get_field_service().metadata())
            return
        if path == "/health":
            self._send_json({"status": "ok"})
            return

        relative = "index.html" if path == "/" else unquote(path.lstrip("/"))
        candidate = (WEB_ROOT / relative).resolve()
        try:
            candidate.relative_to(WEB_ROOT.resolve())
        except ValueError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in {"/api/predict", "/api/field-predict"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("Invalid request size")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if path == "/api/field-predict":
                self._send_json(
                    self.get_field_service().predict(
                        n_pairs=payload.get("n_pairs"),
                        current_A=float(payload.get("current_A")),
                        hot_temperature_K=float(payload.get("hot_temperature_K")),
                    )
                )
            else:
                self._send_json(self.service.predict(payload))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            print(f"Prediction failed: {exc}")
            self._send_json({"error": "Prediction failed"}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    PredictionHandler.service = PredictionService()
    PredictionHandler.field_service = None
    server = ThreadingHTTPServer((args.host, args.port), PredictionHandler)
    print(f"TEC prediction UI: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
