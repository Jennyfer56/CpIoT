"""
Treinamento de Modelos de Machine Learning
Dataset: Diagnóstico de Problemas Ortopédicos na Coluna Vertebral
Disciplina: DISRUPTIVE ARCHITECTURES: IOT, IA & GENERATIVE AI
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# 1. Criação do dataframe
# ─────────────────────────────────────────────
df = pd.read_csv("questao_01.csv", index_col=0)

# Visualizar o tamanho do dataframe
print("=== Tamanho do dataframe ===")
print(df.shape)  # (310, 7)

# Visualizar os primeiros registros
print("\n=== Primeiros registros ===")
print(df.head())

# Informações dos atributos
print("\n=== Informações dos atributos ===")
print(df.info())

# Estatísticas descritivas
print("\n=== Estatísticas descritivas ===")
print(df.describe())

# ─────────────────────────────────────────────
# 2. Conversão de dados categóricos em numéricos
# ─────────────────────────────────────────────
le = LabelEncoder()
df["diagnostic"] = le.fit_transform(df["diagnostic"])
# Disk Hernia=0, Normal=1, Spondylolisthesis=2

print("\n=== Coluna 'diagnostic' após transformação ===")
print(df["diagnostic"].value_counts())
print("Classes:", list(le.classes_))

# Salvar o LabelEncoder para usar na API
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# ─────────────────────────────────────────────
# 3. Variáveis X e y
# ─────────────────────────────────────────────
X = df[["V1", "V2", "V3", "V4", "V5", "V6"]]
y = df["diagnostic"].values

print("\n=== Primeiros registros de X ===")
print(X.head())

print("\n=== X como array Numpy ===")
print(X.values[:5])

print("\n=== y como array Numpy ===")
print(y[:10])

# ─────────────────────────────────────────────
# 4. Separação treino / teste  (70% / 30%)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

print(f"\n=== Separação treino/teste ===")
print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# ─────────────────────────────────────────────
# 5. Modelo 1 — Random Forest
# ─────────────────────────────────────────────
modelo1 = RandomForestClassifier(n_estimators=100, random_state=42)
modelo1.fit(X_train, y_train)

y_predict_1 = modelo1.predict(X_test)

print("\n=== y_predict_1 (Random Forest) ===")
print(y_predict_1)

# ─────────────────────────────────────────────
# 6. Modelo 2 — KNN
# ─────────────────────────────────────────────
modelo2 = KNeighborsClassifier(n_neighbors=5)
modelo2.fit(X_train, y_train)

y_predict_2 = modelo2.predict(X_test)

print("\n=== y_predict_2 (KNN) ===")
print(y_predict_2)

# ─────────────────────────────────────────────
# 7. Acurácia dos modelos
# ─────────────────────────────────────────────
acc1 = accuracy_score(y_test, y_predict_1)
acc2 = accuracy_score(y_test, y_predict_2)

print(f"\n=== Acurácia Modelo 1 (Random Forest): {acc1:.4f} ({acc1*100:.2f}%) ===")
print(f"=== Acurácia Modelo 2 (KNN):           {acc2:.4f} ({acc2*100:.2f}%) ===")

print("\n=== Relatório detalhado — Random Forest ===")
print(classification_report(y_test, y_predict_1, target_names=le.classes_))

print("=== Relatório detalhado — KNN ===")
print(classification_report(y_test, y_predict_2, target_names=le.classes_))

# ─────────────────────────────────────────────
# 8. Salvar o melhor modelo como modelo.pkl
# ─────────────────────────────────────────────
modelo = modelo1 if acc1 >= acc2 else modelo2
modelo_nome = "Random Forest" if acc1 >= acc2 else "KNN"
print(f"\nModelo salvo: {modelo_nome} (acurácia: {max(acc1, acc2)*100:.2f}%)")

with open("modelo.pkl", "wb") as f:
    pickle.dump(modelo, f)

print("Arquivos salvos: modelo.pkl | label_encoder.pkl")
