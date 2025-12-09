"""
Тесты для модуля вычисления суммы ряда.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import task_package.zad as sc


def test_get_control_value() -> None:
    """Тест вычисления контрольного значения."""
    x = math.pi
    expected = -math.log(2)
    result = sc.get_control_value(x)
    assert math.isclose(
        result, expected, rel_tol=1e-10
    ), f"Ожидалось {expected}, получено {result}"


def test_calculate_analytical_sum_for_pi() -> None:
    """Тест аналитического решения для x=π."""
    x = math.pi
    expected = -math.log(2)
    result = sc.calculate_analytical_sum(x)
    assert math.isclose(
        result, expected, rel_tol=1e-10
    ), f"Ожидалось {expected}, получено {result}"


def test_calculate_partial_sum() -> None:
    """Тест вычисления частичной суммы."""
    x = math.pi
    epsilon = 1e-7
    results_list = [0.0, 0.0]

    # Вычисляем сумму для первых 10 членов
    result = sc.calculate_partial_sum(x, 1, 10, epsilon, results_list, 0)

    # Проверяем, что результат корректно записан в список
    assert math.isclose(
        results_list[0], result, rel_tol=1e-10
    ), "Результат не сохранен в список"

    # Проверяем, что результат конечный
    assert not math.isnan(result), "Результат не должен быть NaN"
    assert not math.isinf(result), "Результат не должен быть бесконечностью"


def test_single_threaded_calculation() -> None:
    """Тест однопоточного вычисления."""
    x = math.pi
    epsilon = 1e-7
    analytical_sum = -math.log(2)

    result, terms_counted = sc.calculate_series_sum_single_threaded(x, epsilon)

    # Проверяем корректность типа и значения
    assert isinstance(result, float), "Результат должен быть float"
    assert isinstance(terms_counted, int), "Количество членов должно быть int"
    assert terms_counted > 0, "Должно быть просуммировано хотя бы 1 член"

    # Проверяем точность
    error = abs(result - analytical_sum)
    assert error < 1e-5, f"Слишком большая погрешность: {error}"


def test_multi_threaded_calculation() -> None:
    """Тест многопоточного вычисления."""
    x = math.pi
    epsilon = 1e-7
    analytical_sum = -math.log(2)

    # Тестируем с разным количеством потоков
    for num_threads in [1, 2, 4]:
        result = sc.calculate_series_sum_multi_threaded(x, epsilon, num_threads)

        assert isinstance(
            result, float
        ), f"Результат должен быть float для {num_threads} потоков"
        assert not math.isnan(
            result
        ), f"Результат не должен быть NaN для {num_threads} потоков"

        # Проверяем, что результат близок к аналитическому
        error = abs(result - analytical_sum)
        assert (
            error < 1e-5
        ), f"Слишком большая погрешность для {num_threads} потоков: {error}"


def test_threadpool_calculation() -> None:
    """Тест вычисления с ThreadPoolExecutor."""
    x = math.pi
    epsilon = 1e-7
    analytical_sum = -math.log(2)

    result = sc.calculate_with_threadpool(x, epsilon, 2)

    assert isinstance(result, float), "Результат должен быть float"
    assert not math.isnan(result), "Результат не должен быть NaN"

    error = abs(result - analytical_sum)
    assert error < 1e-5, f"Слишком большая погрешность: {error}"


def test_consistency_between_methods() -> None:
    """Тест согласованности результатов разных методов."""
    x = math.pi
    epsilon = 1e-6  # Немного меньшая точность для быстрых тестов

    single_result, _ = sc.calculate_series_sum_single_threaded(x, epsilon)
    multi_result = sc.calculate_series_sum_multi_threaded(x, epsilon, 2)
    pool_result = sc.calculate_with_threadpool(x, epsilon, 2)

    # Все результаты должны быть близки друг к другу
    assert math.isclose(
        single_result, multi_result, rel_tol=1e-5
    ), "Однопоточный и многопоточный результаты различаются"
    assert math.isclose(
        single_result, pool_result, rel_tol=1e-5
    ), "Однопоточный и ThreadPool результаты различаются"


def test_performance_comparison() -> None:
    """
    Тест сравнения производительности (не строгий тест,
    только проверка что функции выполняются).
    """
    import time

    x = math.pi
    epsilon = 1e-6

    # Измеряем время для однопоточного вычисления
    start = time.time()
    sc.calculate_series_sum_single_threaded(x, epsilon)
    single_time = time.time() - start

    # Измеряем время для многопоточного вычисления
    start = time.time()
    sc.calculate_series_sum_multi_threaded(x, epsilon, 4)
    multi_time = time.time() - start

    # Обе функции должны выполниться за конечное время
    assert single_time > 0, "Однопоточное вычисление должно занимать время"
    assert multi_time > 0, "Многопоточное вычисление должно занимать время"


@pytest.mark.parametrize(
    "x,expected",
    [
        (math.pi, -math.log(2)),
        (math.pi / 2, -math.log(2 * math.sin(math.pi / 4))),
        (math.pi / 3, -math.log(2 * math.sin(math.pi / 6))),
    ],
)
def test_control_value_parametrized(x: float, expected: float) -> None:
    """Параметризованный тест контрольных значений."""
    result = sc.get_control_value(x)
    assert math.isclose(
        result, expected, rel_tol=1e-10
    ), f"Для x={x}: ожидалось {expected}, получено {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
