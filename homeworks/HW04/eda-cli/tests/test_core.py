from __future__ import annotations

import pandas as pd
import numpy as np
from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df)
    assert 0.0 <= flags["quality_score"] <= 1.0
    
    # Проверяем наличие новых флагов
    assert "has_constant_columns" in flags
    assert "has_high_cardinality_categoricals" in flags
    assert "has_many_zero_values" in flags


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2


# Новые тесты для проверки новых эвристик
def test_has_constant_columns():
    """Тест для проверки эвристики константных колонок."""
    # Создаем DataFrame с константной колонкой
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'constant_col': [7, 7, 7, 7, 7],  # Все значения одинаковые
        'normal_col': ['a', 'b', 'c', 'd', 'e']
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_constant_columns = True
    assert flags['has_constant_columns'] == True
    # Проверяем, что quality_score снизился из-за константной колонки
    assert flags['quality_score'] < 1.0


def test_has_high_cardinality_categoricals():
    """Тест для проверки высокой кардинальности категориальных признаков."""
    # Создаем DataFrame с высокой кардинальностью (>50 уникальных значений)
    df = pd.DataFrame({
        'id': range(60),
        'high_card_col': [f'value_{i}' for i in range(60)],  # 60 уникальных значений
        'low_card_col': ['A', 'B'] * 30  # Только 2 уникальных значения
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_high_cardinality_categoricals = True
    assert flags['has_high_cardinality_categoricals'] == True
    assert flags['high_cardinality_count'] >= 1
    # Проверяем, что quality_score снизился из-за высокой кардинальности
    assert flags['quality_score'] < 1.0


def test_no_high_cardinality():
    """Тест для проверки, когда нет высокой кардинальности."""
    df = pd.DataFrame({
        'id': range(10),
        'category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A']  # 3 уникальных значения
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что флаг has_high_cardinality_categoricals = False
    assert flags['has_high_cardinality_categoricals'] == False


def test_has_many_zero_values():
    """Тест для проверки множества нулевых значений в числовых колонках."""
    # Создаем DataFrame с многими нулевыми значениями
    df = pd.DataFrame({
        'id': range(10),
        'zeros_col': [0] * 8 + [1, 2],  # 80% нулей
        'normal_col': range(10)
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем наличие флага
    assert 'has_many_zero_values' in flags
    assert 'columns_with_many_zeros' in flags


def test_top_categories_with_parameter():
    """Тест для функции top_categories с параметром top_k."""
    df = pd.DataFrame({
        'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'E', 'F']
    })
    
    # Тестируем с top_k=3
    top_cats = top_categories(df, max_columns=5, top_k=3)
    assert 'category' in top_cats
    assert len(top_cats['category']) == 3  # Должно быть 3 записи
    
    # Тестируем с top_k=5
    top_cats = top_categories(df, max_columns=5, top_k=5)
    assert len(top_cats['category']) == 5  # Должно быть 5 записей


def test_empty_dataframe():
    """Тест для пустого DataFrame."""
    df = pd.DataFrame()
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    assert summary.n_rows == 0
    assert summary.n_cols == 0
    assert 0.0 <= flags['quality_score'] <= 1.0


def test_quality_score_calculation():
    """Тест для проверки правильности расчета quality_score."""
    # Создаем проблемный DataFrame
    df = pd.DataFrame({
        'const': [1] * 10,  # Константная колонка
        'many_missing': [None] * 5 + [1] * 5,  # 50% пропусков
        'high_card': [f'val_{i}' for i in range(10)],  # Высокая кардинальность
        'normal': range(10)
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df)
    
    # Проверяем, что все флаги установлены правильно
    assert flags['has_constant_columns'] == True
    assert flags['too_many_missing'] == False  # 50% = порогу, не больше
    assert flags['max_missing_share'] == 0.5
    assert 0 <= flags['quality_score'] <= 1.0
    # Проверяем, что quality_score меньше 1 из-за проблем
    assert flags['quality_score'] < 1.0