# 🦴 API de Diagnóstico Ortopédico — Coluna Vertebral

> **Checkpoint 01 — Machine Learning**  
> Disciplina: Disruptive Architectures: IOT, IA & Generative AI  
> Turma: 2TDS — 1º semestre de 2026  
> Professor: André Tritiack

---

## 📋 Descrição

API REST que recebe dados biomecânicos de um paciente e retorna o diagnóstico ortopédico da coluna vertebral, utilizando um modelo de **Random Forest** treinado com 310 registros anonimizados.

**Classes possíveis:**
| Código | Diagnóstico |
|--------|-------------|
| 0 | Disk Hernia (Hérnia de Disco) |
| 1 | Normal |
| 2 | Spondylolisthesis (Espondilolistese) |

**Acurácia do modelo:** `82.80%`

---

## 📁 Estrutura do Repositório

```
ml-coluna-vertebral/
├── questao_01.csv                              # Dataset original
├── train_model.py                              # Script de treinamento
├── app.py                                      # API Flask
├── modelo.pkl                                  # Modelo treinado (Random Forest)
├── label_encoder.pkl                           # Encoder de classes
├── requirements.txt                            # Dependências Python
├── Dockerfile                                  # Container para deploy
├── Procfile                                    # Para deploy no Railway/Render
├── Exercicio_Machine_Learning_2TDS_...ipynb    # Notebook completo
└── README.md
```

---

## ⚙️ Execução Local

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU_USUARIO/ml-coluna-vertebral.git
cd ml-coluna-vertebral
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. (Opcional) Retreinar o modelo

```bash
python train_model.py
```

### 4. Iniciar a API

```bash
python app.py
```

A API estará disponível em: `http://localhost:5000`

---

## 🔌 Endpoints

### `GET /`
Retorna informações sobre a API.

### `GET /health`
Verifica se a API está em funcionamento.

```bash
curl http://localhost:5000/health
```

### `POST /predict`
Realiza a predição para um único paciente.

**Body (JSON):**
```json
{
  "V1": 63.03,
  "V2": 22.55,
  "V3": 39.61,
  "V4": 40.47,
  "V5": 98.67,
  "V6": -0.25
}
```

**Exemplo com cURL:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": 63.03, "V2": 22.55, "V3": 39.61, "V4": 40.47, "V5": 98.67, "V6": -0.25}'
```

**Resposta:**
```json
{
  "diagnostico": "Disk Hernia",
  "codigo": 0,
  "probabilidades": {
    "Disk Hernia": 0.72,
    "Normal": 0.18,
    "Spondylolisthesis": 0.10
  },
  "entrada": {
    "V1": 63.03, "V2": 22.55, "V3": 39.61,
    "V4": 40.47, "V5": 98.67, "V6": -0.25
  }
}
```

### `POST /predict/batch`
Realiza predição para múltiplos pacientes.

```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "pacientes": [
      {"V1": 63.03, "V2": 22.55, "V3": 39.61, "V4": 40.47, "V5": 98.67, "V6": -0.25},
      {"V1": 39.06, "V2": 10.06, "V3": 25.02, "V4": 29.00, "V5": 114.41, "V6": 4.56}
    ]
  }'
```

---

## ☁️ Deploy na Nuvem

### Opção A — Railway (mais simples, gratuito)

1. Crie conta em [railway.app](https://railway.app)
2. Clique em **New Project → Deploy from GitHub Repo**
3. Selecione este repositório
4. Railway detecta o `Procfile` automaticamente
5. Em **Variables**, adicione: `PORT=5000`
6. Sua URL pública será gerada automaticamente

### Opção B — AWS App Runner

```bash
# 1. Build e push da imagem Docker para ECR
aws ecr create-repository --repository-name ml-coluna-vertebral
docker build -t ml-coluna-vertebral .
docker tag ml-coluna-vertebral:latest <AWS_ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com/ml-coluna-vertebral:latest
aws ecr get-login-password | docker login --username AWS --password-stdin <ECR_URI>
docker push <ECR_URI>:latest

# 2. No console AWS, crie um App Runner apontando para a imagem ECR
# Porta: 5000
```

### Opção C — Azure Container Apps

```bash
# 1. Build e push para Azure Container Registry
az group create --name rg-ml-coluna --location brazilsouth
az acr create --resource-group rg-ml-coluna --name mlcolunaacr --sku Basic
az acr login --name mlcolunaacr
docker build -t mlcolunaacr.azurecr.io/ml-coluna-vertebral:latest .
docker push mlcolunaacr.azurecr.io/ml-coluna-vertebral:latest

# 2. Criar Container App
az containerapp create \
  --name ml-coluna-vertebral \
  --resource-group rg-ml-coluna \
  --image mlcolunaacr.azurecr.io/ml-coluna-vertebral:latest \
  --target-port 5000 \
  --ingress external
```

### Opção D — OCI Container Instances

```bash
# 1. Criar namespace e repositório no OCIR
docker tag ml-coluna-vertebral <REGION>.ocir.io/<NAMESPACE>/ml-coluna-vertebral:latest
docker push <REGION>.ocir.io/<NAMESPACE>/ml-coluna-vertebral:latest

# 2. No console OCI: Developer Services → Containers → Container Instances
# Configurar porta 5000, e o container usará a imagem acima
```

---

## 🧪 Testando a API publicada

Substitua `<URL_DA_API>` pela URL gerada no deploy:

```bash
# Health check
curl https://<URL_DA_API>/health

# Predição
curl -X POST https://<URL_DA_API>/predict \
  -H "Content-Type: application/json" \
  -d '{"V1": 50.0, "V2": 15.0, "V3": 35.0, "V4": 35.0, "V5": 120.0, "V6": 5.0}'
```

---

## 📊 Resultados dos Modelos

| Modelo | Acurácia |
|--------|----------|
| Random Forest (100 árvores) | **82.80%** |
| KNN (k=5) | 78.49% |

O modelo Random Forest foi selecionado por apresentar maior acurácia e foi salvo como `modelo.pkl`.

---

## 📦 Dependências

```
flask==3.1.0
scikit-learn==1.6.1
numpy==2.2.3
pandas==2.2.3
gunicorn==23.0.0
```
"# CpIoT" 
