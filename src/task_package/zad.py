"""
Модуль для вычисления суммы ряда с использованием многопоточности.
"""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple


def calculate_partial_sum(
    x: float,
    start_n: int,
    end_n: int,
    epsilon: float,
    results_list: List[float],
    idx: int,
) -> float:
    """
    Вычисляет частичную сумму ряда от start_n до end_n.

    Args:
        x: Значение аргумента
        start_n: Начальный индекс
        end_n: Конечный индекс
        epsilon: Точность вычислений
        results_list: Список для сохранения результатов
        idx: Индекс в списке результатов

    Returns:
        Частичная сумма ряда
    """
    partial_sum = 0.0
    n = start_n

    while n <= end_n:
        term = math.cos(n * x) / n

        # Проверяем точность (по абсолютному значению члена ряда)
        if abs(term) < epsilon:
            break

        partial_sum += term
        n += 1

    results_list[idx] = partial_sum
    return partial_sum


def calculate_series_sum_single_threaded(x: float, epsilon: float) -> Tuple[float, int]:
    """
    Однопоточное вычисление суммы ряда.

    Args:
        x: Значение аргумента
        epsilon: Точность вычислений

    Returns:
        Кортеж (сумма ряда, количество просуммированных членов)
    """
    total_sum = 0.0
    n = 1

    while True:
        term = math.cos(n * x) / n

        if abs(term) < epsilon:
            break

        total_sum += term
        n += 1

    return total_sum, n


def calculate_series_sum_multi_threaded(
    x: float, epsilon: float, num_threads: int = 4
) -> float:
    """
    Многопоточное вычисление суммы ряда.

    Args:
        x: Значение аргумента
        epsilon: Точность вычислений
        num_threads: Количество потоков

    Returns:
        Сумма ряда
    """
    # Определяем максимальное количество членов для анализа
    max_n = int(math.ceil(1 / epsilon)) + 100

    # Разделяем работу между потоками
    chunk_size = max_n // num_threads
    threads: List[threading.Thread] = []
    results = [0.0] * num_threads

    # Создаем и запускаем потоки
    for i in range(num_threads):
        start_n = i * chunk_size + 1
        end_n = (i + 1) * chunk_size if i < num_threads - 1 else max_n

        thread = threading.Thread(
            target=calculate_partial_sum, args=(x, start_n, end_n, epsilon, results, i)
        )
        threads.append(thread)
        thread.start()

    # Ожидаем завершения всех потоков
    for thread in threads:
        thread.join()

    # Суммируем результаты
    total_sum = sum(results)
    return total_sum


def calculate_with_threadpool(x: float, epsilon: float, num_threads: int = 4) -> float:
    """
    Вычисление суммы ряда с использованием ThreadPoolExecutor.

    Args:
        x: Значение аргумента
        epsilon: Точность вычислений
        num_threads: Количество потоков

    Returns:
        Сумма ряда
    """
    max_n = int(math.ceil(1 / epsilon)) + 100
    chunk_size = max_n // num_threads

    def worker(start_n: int, end_n: int) -> float:
        """Воркер-функция для выполнения в потоке."""
        partial_sum = 0.0
        n = start_n

        while n <= end_n:
            term = math.cos(n * x) / n
            if abs(term) < epsilon:
                break
            partial_sum += term
            n += 1

        return partial_sum

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []

        for i in range(num_threads):
            start_n = i * chunk_size + 1
            end_n = (i + 1) * chunk_size if i < num_threads - 1 else max_n
            futures.append(executor.submit(worker, start_n, end_n))

        # Собираем результаты
        total_sum = sum(future.result() for future in futures)

    return total_sum


def get_control_value(x: float) -> float:
    """
    Вычисляет контрольное значение функции.

    Args:
        x: Значение аргумента

    Returns:
        Контрольное значение y = -ln(2*sin(x/2))
    """
    return -math.log(2 * math.sin(x / 2))


def calculate_analytical_sum(x: float) -> float:
    """
    Вычисляет аналитическое значение суммы ряда для заданного x.

    Args:
        x: Значение аргумента

    Returns:
        Аналитическое значение суммы ряда или None, если не применимо
    """
    if math.isclose(x, math.pi):
        return -math.log(2)
    return get_control_value(x)


def main() -> None:
    """Основная функция программы."""
    # Параметры задачи
    x = math.pi
    epsilon = 1e-7

    print("=" * 60)
    print("ИНДИВИДУАЛЬНОЕ ЗАДАНИЕ")
    print("=" * 60)
    print("Ряд: S = Σ(cos(nx)/n), n=1..∞")
    print(f"x = {x}")
    print(f"Точность ε = {epsilon}")
    print("Контрольная функция: y = -ln(2*sin(x/2))")
    print("=" * 60)

    # Вычисляем контрольное значение
    y = get_control_value(x)
    print(f"\nКонтрольное значение y = {y:.15f}")

    # 1. Однопоточное вычисление (для сравнения)
    print("\n1. Однопоточное вычисление:")
    start_time = time.time()
    single_sum, terms_counted = calculate_series_sum_single_threaded(x, epsilon)
    single_time = time.time() - start_time
    print(f"   Сумма ряда S = {single_sum:.15f}")
    print(f"   Просуммировано членов: {terms_counted}")
    print(f"   Время выполнения: {single_time:.4f} сек")
    print(f"   Абсолютная погрешность: {abs(single_sum - y):.10e}")

    # 2. Многопоточное вычисление (базовый подход)
    print("\n2. Многопоточное вычисление (базовый подход):")
    for num_threads in [2, 4, 8]:
        print(f"\n   Число потоков: {num_threads}")
        start_time = time.time()
        multi_sum = calculate_series_sum_multi_threaded(x, epsilon, num_threads)
        multi_time = time.time() - start_time
        print(f"   Сумма ряда S = {multi_sum:.15f}")
        print(f"   Время выполнения: {multi_time:.4f} сек")
        if multi_time > 0:
            print(f"   Ускорение: {single_time/multi_time:.2f}x")
        print(f"   Абсолютная погрешность: {abs(multi_sum - y):.10e}")

    # 3. Многопоточное вычисление с ThreadPoolExecutor
    print("\n3. Многопоточное вычисление (ThreadPoolExecutor):")
    start_time = time.time()
    pool_sum = calculate_with_threadpool(x, epsilon, 4)
    pool_time = time.time() - start_time
    print(f"   Сумма ряда S = {pool_sum:.15f}")
    print(f"   Время выполнения: {pool_time:.4f} сек")
    print(f"   Абсолютная погрешность: {abs(pool_sum - y):.10e}")

    # 4. Аналитическое решение для x = π
    print("\n4. Аналитическое решение:")
    print("   При x = π ряд превращается в:")
    print("   S = -1 + 1/2 - 1/3 + 1/4 - ... = -ln(2)")
    analytic_sum = calculate_analytical_sum(x)
    print(f"   S = -ln(2) = {analytic_sum:.15f}")
    print(f"   Совпадение с контрольным значением: {abs(analytic_sum - y) < 1e-15}")

    # 5. Сравнение результатов
    print("\n" + "=" * 60)
    print("ИТОГИ:")
    print("=" * 60)
    print(f"Контрольное значение:      {y:.15f}")
    print(f"Однопоточный результат:    {single_sum:.15f}")
    print(f"Погрешность:               {abs(single_sum - y):.10e}")
    print(f"\nСходимость к -ln(2):       {abs(single_sum - (-math.log(2))):.10e}")

    # Проверка достижения требуемой точности
    print("\nПроверка точности:")
    print(f"Требуемая точность: |a_n| < {epsilon}")
    last_term = math.cos(terms_counted * x) / terms_counted
    print(f"Последний учтенный член: |a_{terms_counted}| = {abs(last_term):.10e}")

    if abs(last_term) < epsilon:
        print("✓ Требуемая точность достигнута!")
    else:
        print("✗ Требуемая точность не достигнута!")


if __name__ == "__main__":
    main()
