"""Bundle the legacy TEC latent-space predictor and trained model into one HTML file."""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HTML = ROOT / "legacy" / "web" / "predict_ui.html"
MODEL_PATH = ROOT / "data" / "models" / "comsol_latent_surrogate.joblib"
OUTPUT_HTML = ROOT / "legacy" / "web" / "tec_latent_predictor_standalone.html"


def array(value):
    return value.tolist()


def export_model(model: dict) -> dict:
    regressors = []
    for estimator in model["regressor"].estimators_:
        regressors.append(
            {
                "support_vectors": array(estimator.support_vectors_),
                "dual_coef": array(estimator.dual_coef_[0]),
                "intercept": float(estimator.intercept_[0]),
                "gamma": float(estimator._gamma),
            }
        )

    return {
        "sample_count": len(model["training_filenames"]),
        "input_mean": array(model["input_scaler"].mean_),
        "input_scale": array(model["input_scaler"].scale_),
        "output_mean": array(model["output_scaler"].mean_),
        "output_scale": array(model["output_scaler"].scale_),
        "pca_mean": array(model["pca"].mean_),
        "pca_components": array(model["pca"].components_),
        "output_names": model["output_names"],
        "regressors": regressors,
    }


def main() -> None:
    model = joblib.load(MODEL_PATH)
    model_json = json.dumps(export_model(model), ensure_ascii=False, separators=(",", ":"))
    html = SOURCE_HTML.read_text(encoding="utf-8")
    html = html.replace(
        "PCA+SVR | 29 样本 | 潜维度 5 (99.5% 方差) | 实时 API 预测",
        f"PCA+SVR | {len(model['training_filenames'])} 样本 | 潜维度 5 (99.5% 方差) | 离线单文件预测",
    )
    html = html.replace(
        "const INPUT_NAMES =",
        f"const MODEL = {model_json};\n\nconst INPUT_NAMES =",
        1,
    )

    offline_predictor = r'''async function fetchPredict() {
  document.getElementById('error-msg').textContent = '';
  document.getElementById('status').textContent = '计算中...';
  try {
    const started = performance.now();
    const scaled = inputs.map((value, i) => (value - MODEL.input_mean[i]) / MODEL.input_scale[i]);
    const latent = MODEL.regressors.map(regressor => {
      let prediction = regressor.intercept;
      for (let row = 0; row < regressor.support_vectors.length; row++) {
        const support = regressor.support_vectors[row];
        let squaredDistance = 0;
        for (let col = 0; col < scaled.length; col++) {
          const difference = scaled[col] - support[col];
          squaredDistance += difference * difference;
        }
        prediction += regressor.dual_coef[row] * Math.exp(-regressor.gamma * squaredDistance);
      }
      return prediction;
    });

    const outputScaled = MODEL.pca_mean.map((mean, outputIndex) => {
      let value = mean;
      for (let component = 0; component < latent.length; component++) {
        value += latent[component] * MODEL.pca_components[component][outputIndex];
      }
      return value;
    });
    const output = outputScaled.map((value, i) => value * MODEL.output_scale[i] + MODEL.output_mean[i]);
    const data = {latent, inference_ms: performance.now() - started};
    MODEL.output_names.forEach((name, i) => { data[name] = output[i]; });
    updateDisplay(data);
    document.getElementById('status').textContent = `就绪 · 本地推理 ${data.inference_ms.toFixed(2)} ms`;
  } catch(e) {
    document.getElementById('error-msg').textContent = '预测失败: ' + e.message;
    document.getElementById('status').textContent = '错误';
  }
}

const predict = debounce'''
    pattern = re.compile(
        r"async function fetchPredict\(\) \{.*?\n\}\n\nconst predict = debounce",
        re.DOTALL,
    )
    html, count = pattern.subn(offline_predictor, html, count=1)
    if count != 1:
        raise RuntimeError("Could not locate the API prediction function in the legacy HTML")

    OUTPUT_HTML.write_text(html, encoding="utf-8", newline="\n")
    print(OUTPUT_HTML)


if __name__ == "__main__":
    main()
