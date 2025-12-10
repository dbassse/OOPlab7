import math
import time
from task_package.zad import (get_control_value, 
                              calculate_series_sum_single_threaded,
                              calculate_series_sum_multi_threaded,
                              calculate_with_threadpool,
                              calculate_analytical_sum
)

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
