
import os
import joblib
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

def main():
    
    model = joblib.load("models/model.joblib")

    X_all, y_all, coef_true = make_regression(
        n_samples=500,
        n_features=4,
        n_informative=2,
        noise=10.0,
        coef=True,
        random_state=212862
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=212862
    )

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    x_example = np.array([[1, 1, 1, 1]])
    y_example_pred = model.predict(x_example)[0]

    #результаты
    print("=== Проверка модели ===")
    print(f"R² на тестовой выборке: {r2:.3f}")
    print(f"Пример ввода: {x_example}")
    print(f"Предсказанное значение: {y_example_pred:.3f}")
    print(f"Коэффициенты модели: {model.coef_}")
    print(f"Свободный член (intercept): {model.intercept_:.3f}")

if __name__ == "__main__":
    main()
