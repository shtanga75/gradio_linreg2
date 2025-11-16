import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def test_model():
    """
    Тест модели: загружаем модель и данные из файлов,
    проверяем, что модель работает корректно на тестовой выборке.
    """
    
    # Загрузка модели
    model = joblib.load("../models/model.joblib")
    
    # Загрузка тестовых данных из файла
    test_data = pd.read_csv("../data/test.csv")
    X_test = test_data[['x1', 'x2', 'x3', 'x4']].values
    y_test = test_data['y'].values
    
    #предсказание на тестовой выборке
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    #проверка, что R² > 0
    assert r2 > 0, f"R² на тестовой выборке должен быть > 0, получено: {r2:.3f}"
    assert r2 > 0.5, f"R² слишком низкий, получено: {r2:.3f}"
    
    # Проверка на нескольких конкретных примерах из тестовой выборки
    #берём первые 5 примеро
    for i in range(min(5, len(X_test))):
        x_example = X_test[i:i+1]
        y_true = y_test[i]
        y_pred_example = model.predict(x_example)[0]
        
        # Проверяем, что предсказание близко к истинному значению
        # (допускаем погрешность до 20% от диапазона y)
        y_range = y_test.max() - y_test.min()
        tolerance = 0.2 * y_range
        error = abs(y_pred_example - y_true)
        
        assert error < tolerance, f"Пример {i}: ошибка предсказания слишком велика. y_true={y_true:.3f}, y_pred={y_pred_example:.3f}, error={error:.3f}"
    
    #проверка коэффициентов модели
    assert hasattr(model, 'coef_'), "Модель должна иметь коэффициенты (coef_)"
    assert len(model.coef_) == 4, f"Модель должна иметь 4 коэффициента, получено: {len(model.coef_)}"
    
    #проверка, что коэффициенты не нулевые
    assert not np.allclose(model.coef_, 0), "Коэффициенты модели не должны быть все нулевыми"
    
    print("=== Все тесты пройдены ===")
    print(f"R² на тестовой выборке: {r2:.3f}")
    print(f"Коэффициенты модели: {model.coef_}")
    print(f"Свободный член (intercept): {model.intercept_:.3f}")
    print("\nПримеры предсказаний:")
    for i in range(min(3, len(X_test))):
        x_example = X_test[i:i+1]
        y_true = y_test[i]
        y_pred_example = model.predict(x_example)[0]
        print(f"  Пример {i+1}: x={x_example[0]}, y_true={y_true:.3f}, y_pred={y_pred_example:.3f}")

if __name__ == "__main__":
    test_model()
