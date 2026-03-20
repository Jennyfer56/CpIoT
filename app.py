"""
API REST para predição de diagnósticos ortopédicos da coluna vertebral.
Modelo: Random Forest (acurácia ~82.8%)
"""

import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Carregar modelo e label encoder na inicialização
with open("modelo.pkl", "rb") as f:
    modelo = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "servico": "API de Diagnóstico Ortopédico - Coluna Vertebral",
        "versao": "1.0",
        "modelo": "Random Forest",
        "acuracia": "82.80%",
        "endpoints": {
            "GET  /": "Informações da API",
            "GET  /health": "Status da API",
            "POST /predict": "Realiza predição com os 6 biomarcadores (V1-V6)",
            "POST /predict/batch": "Predição em lote (múltiplos pacientes)"
        },
        "classes": list(le.classes_)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modelo_carregado": True}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Recebe JSON com os 6 biomarcadores e retorna o diagnóstico.

    Exemplo de body:
    {
        "V1": 63.03,
        "V2": 22.55,
        "V3": 39.61,
        "V4": 40.47,
        "V5": 98.67,
        "V6": -0.25
    }
    """
    data = request.get_json(force=True)

    if not data:
        return jsonify({"erro": "Body JSON não encontrado."}), 400

    campos = ["V1", "V2", "V3", "V4", "V5", "V6"]
    faltando = [c for c in campos if c not in data]
    if faltando:
        return jsonify({
            "erro": f"Campos obrigatórios ausentes: {faltando}",
            "campos_necessarios": campos
        }), 400

    try:
        entrada = np.array([[float(data[c]) for c in campos]])
        pred_num = modelo.predict(entrada)[0]
        pred_proba = modelo.predict_proba(entrada)[0]
        diagnostico = le.inverse_transform([pred_num])[0]

        probabilidades = {
            le.classes_[i]: round(float(p), 4)
            for i, p in enumerate(pred_proba)
        }

        return jsonify({
            "diagnostico": diagnostico,
            "codigo": int(pred_num),
            "probabilidades": probabilidades,
            "entrada": {c: data[c] for c in campos}
        })

    except (ValueError, TypeError) as e:
        return jsonify({"erro": f"Valor inválido: {str(e)}"}), 400


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    Recebe uma lista de pacientes e retorna diagnósticos para todos.

    Exemplo de body:
    {
        "pacientes": [
            {"V1": 63.03, "V2": 22.55, "V3": 39.61, "V4": 40.47, "V5": 98.67, "V6": -0.25},
            {"V1": 39.06, "V2": 10.06, "V3": 25.02, "V4": 29.00, "V5": 114.41, "V6": 4.56}
        ]
    }
    """
    data = request.get_json(force=True)

    if not data or "pacientes" not in data:
        return jsonify({"erro": "Campo 'pacientes' (lista) é obrigatório."}), 400

    pacientes = data["pacientes"]
    if not isinstance(pacientes, list) or len(pacientes) == 0:
        return jsonify({"erro": "'pacientes' deve ser uma lista não vazia."}), 400

    campos = ["V1", "V2", "V3", "V4", "V5", "V6"]
    resultados = []

    for i, p in enumerate(pacientes):
        faltando = [c for c in campos if c not in p]
        if faltando:
            resultados.append({
                "paciente": i + 1,
                "erro": f"Campos ausentes: {faltando}"
            })
            continue
        try:
            entrada = np.array([[float(p[c]) for c in campos]])
            pred_num = modelo.predict(entrada)[0]
            diagnostico = le.inverse_transform([pred_num])[0]
            resultados.append({
                "paciente": i + 1,
                "diagnostico": diagnostico,
                "codigo": int(pred_num)
            })
        except (ValueError, TypeError) as e:
            resultados.append({"paciente": i + 1, "erro": str(e)})

    return jsonify({"total": len(pacientes), "resultados": resultados})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
