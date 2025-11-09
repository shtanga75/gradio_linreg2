"""
Gradio-приложение для обученной модели линейной регрессии.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import pandas as pd

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


MODEL_PATH = os.path.join("models", "model.joblib")

X_all, y_all, coef_true = make_regression(
    n_samples=500,
    n_features=4,
    n_informative=2,
    noise=10.0,
    coef=True,
    random_state=212862
)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=212862)
'''idx = np.arange(X_all.shape[0])
train_idx, test_idx = train_test_split(idx, shuffle=True, random_state=0, test_size=0.2)
X_train = X_all[train_idx]
y_train = y_all[train_idx]
X_test = X_all[test_idx]
y_test = y_all[test_idx]'''

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. Положите model.joblib в папку models/")

model = joblib.load(MODEL_PATH)

#R^2
y_test_pred = model.predict(X_test)
r2 = r2_score(y_test, y_test_pred)


try:
    coefs = np.array(model.coef_)
except Exception:
    coefs = np.ravel(np.array(getattr(model, "coef_", np.zeros(X_all.shape[1]))))

most_important_idx = int(np.argmax(np.abs(coefs)))  # индекс 0-based

#LaTeX
coef_rounded = [round(float(c), 2) for c in coefs]
intercept = round(float(getattr(model, "intercept_", 0.0)), 2)
terms = [f"{coef_rounded[i]} x_{i+1}" for i in range(len(coef_rounded))]
equation_latex = r"$y = " + " + ".join(terms) + f" + {intercept}$"

#функция предсказания для Gradio
def predict_and_plot(x1, x2, x3, x4):
    x_in = np.array([x1, x2, x3, x4]).reshape(1, -1)
    #предсказание
    y_pred = float(model.predict(x_in)[0])

    #scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_train[:, most_important_idx], y_train, alpha=0.6, label="Обучающие точки")
    #пользовательская точка
    ax.scatter(x_in[0, most_important_idx], y_pred, color="red", s=80, label="Ваш ввод (предсказание)")
    ax.set_xlabel(f"Признак {most_important_idx + 1}")
    ax.set_ylabel("Целевая переменная y")
    ax.set_title("Scatter: наиболее важный признак vs y (обучение)")

    mean_other = X_train.mean(axis=0)
    xs = np.linspace(X_train[:, most_important_idx].min(), X_train[:, most_important_idx].max(), 100)
    #Формируем матрицу для предсказаний
    X_line = np.tile(mean_other, (len(xs), 1))
    X_line[:, most_important_idx] = xs
    ys_line = model.predict(X_line)
    ax.plot(xs, ys_line, linestyle="--", linewidth=2, label="Линия регрессии (по одному признаку)")

    ax.legend()
    fig.tight_layout()

    #Возвращаем предсказание
    return round(y_pred, 3), fig

#Gradio
description_md = (
    f"**R² на тестовой выборке:** {r2:.3f}  \n\n"
    f"**Уравнение регрессии:**  \n\n"
    f"{equation_latex}"
)

inputs = [
    gr.Number(label="Признак 1 (x1)"),
    gr.Number(label="Признак 2 (x2)"),
    gr.Number(label="Признак 3 (x3)"),
    gr.Number(label="Признак 4 (x4)"),
]

outputs = [
    gr.Number(label="Предсказанное y"),
    gr.Plot(label="Scatter: признак vs y"),
]

title = "Gradio: Линейная регрессия — предсказание и визуализация"

iface = gr.Interface(
    fn=predict_and_plot,
    inputs=inputs,
    outputs=outputs,
    description=description_md,
    title=title,
    flagging_mode="never"
)

if __name__ == "__main__":
    iface.launch(share=False)
