"""
Gradio-приложение для обученной модели линейной регрессии.
"""

import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import pandas as pd

from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


MODEL_PATH = os.path.join("models", "model.joblib")
TRAIN_DATA_PATH = os.path.join("data", "train.csv")
TEST_DATA_PATH = os.path.join("data", "test.csv")

#проверка
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}. Запустите lin_reg.py для создания модели.")

if not os.path.exists(TRAIN_DATA_PATH):
    raise FileNotFoundError(f"Обучающие данные не найдены: {TRAIN_DATA_PATH}. Запустите lin_reg.py для создания данных.")

if not os.path.exists(TEST_DATA_PATH):
    raise FileNotFoundError(f"Тестовые данные не найдены: {TEST_DATA_PATH}. Запустите lin_reg.py для создания данных.")

#Загрузка модели
model = joblib.load(MODEL_PATH)

#Загрузка данных
train_data = pd.read_csv(TRAIN_DATA_PATH)
test_data = pd.read_csv(TEST_DATA_PATH)

X_train = train_data[['x1', 'x2', 'x3', 'x4']].values
y_train = train_data['y'].values
X_test = test_data[['x1', 'x2', 'x3', 'x4']].values
y_test = test_data['y'].values

#R^2
y_test_pred = model.predict(X_test)
r2 = r2_score(y_test, y_test_pred)

try:
    coefs = np.array(model.coef_)
except Exception:
    coefs = np.ravel(np.array(getattr(model, "coef_", np.zeros(X_train.shape[1]))))

most_important_idx = int(np.argmax(np.abs(coefs)))

# LaTeX
coef_rounded = [round(float(c), 2) for c in coefs]
intercept = round(float(getattr(model, "intercept_", 0.0)), 2)
terms = [f"{coef_rounded[i]} \\cdot x_{i+1}" for i in range(len(coef_rounded))]
equation_latex = r"$$y = " + " + ".join(terms) + f" + {intercept}$$"

#функция предсказания для Gradio
def predict_and_plot(x1, x2, x3, x4):
    x_in = np.array([x1, x2, x3, x4]).reshape(1, -1)
    # предсказание
    y_pred = float(model.predict(x_in)[0])

    # scatter plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X_train[:, most_important_idx], y_train, alpha=0.6, label="Обучающие точки")
    # пользовательская точка
    ax.scatter(x_in[0, most_important_idx], y_pred, color="red", s=80, label="Ваш ввод (предсказание)")
    ax.set_xlabel(f"Признак {most_important_idx + 1}")
    ax.set_ylabel("Целевая переменная y")
    ax.set_title("Scatter: наиболее важный признак vs y (обучение)")

    mean_other = X_train.mean(axis=0)
    xs = np.linspace(X_train[:, most_important_idx].min(), X_train[:, most_important_idx].max(), 100)
    # Формируем матрицу для предсказаний
    X_line = np.tile(mean_other, (len(xs), 1))
    X_line[:, most_important_idx] = xs
    ys_line = model.predict(X_line)
    ax.plot(xs, ys_line, linestyle="--", linewidth=2, label="Линия регрессии (по одному признаку)")

    ax.legend()
    fig.tight_layout()

    #предсказание
    return round(y_pred, 3), fig

#Gradio
description_md = (
    f"**R² на тестовой выборке:** {r2:.3f}\n\n"
    f"**Уравнение регрессии:**\n\n"
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
