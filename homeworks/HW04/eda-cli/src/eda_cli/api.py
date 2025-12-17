from __future__ import annotations

from time import perf_counter
from typing import Dict, Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.3.0",  # Увеличиваем версию
    description=(
        "HTTP-сервис для оценки готовности датасета к обучению модели. "
        "Использует эвристики качества данных, включая новые из HW03."
    ),
    docs_url="/docs",
    redoc_url=None,
)


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ заглушки модели качества датасета."""

    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


class QualityFlagsResponse(BaseModel):
    """Ответ с полным набором флагов качества данных."""
    
    flags: Dict[str, Any] = Field(
        ...,
        description="Полный набор флагов качества, включая новые из HW03"
    )
    dataset_shape: Dict[str, int] = Field(
        ...,
        description="Размеры датасета"
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )


# ---------- Системный эндпоинт ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.3.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """

    start = perf_counter()

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков (если есть числовые и категориальные)
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7
    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Флаги, которые могут быть полезны для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Примитивный лог — на семинаре можно обсудить, как это превратить в нормальный logger
    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.
    """

    start = perf_counter()

    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        # content_type от браузера может быть разным, поэтому проверка мягкая
        # но для демонстрации оставим простую ветку 400
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # FastAPI даёт file.file как file-like объект, который можно читать pandas'ом
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)

    # Ожидаем, что compute_quality_flags вернёт quality_score в [0,1]
    score = float(flags_all.get("quality_score", 0.0))
    score = max(0.0, min(1.0, score))
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета берём из summary (если там есть поля n_rows/n_cols),
    # иначе — напрямую из DataFrame.
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv: полный набор флагов качества ----------

@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality", "hw03"],
    summary="Полный набор флагов качества данных из CSV-файла",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Новый эндпоинт, который принимает CSV-файл и возвращает полный набор 
    флагов качества данных, включая новые эвристики из HW03.
    
    Отличие от /quality-from-csv:
    - Возвращает ВСЕ флаги (включая небулевы)
    - Не фильтрует только булевы значения
    - Не преобразует в QualityResponse, а возвращает сырые флаги
    - Включает новые эвристики: has_constant_columns, has_high_cardinality_categoricals, etc.
    """
    
    start = perf_counter()

    # Проверка типа файла
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # Читаем CSV в DataFrame
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из S03 (с новыми эвристиками из HW03)
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вызываем compute_quality_flags, который теперь включает новые эвристики:
    # - has_constant_columns
    # - has_high_cardinality_categoricals
    # - has_many_zero_values
    # - и другие из HW03
    flags_all = compute_quality_flags(summary, missing_df)
    
    # Получаем размеры датасета
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])
    
    latency_ms = (perf_counter() - start) * 1000.0
    
    # Логируем вызов для отладки
    print(
        f"[quality-flags-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} "
        f"flags_count={len(flags_all)} latency_ms={latency_ms:.1f} ms"
    )
    
    # Проверяем, что новые флаги из HW03 присутствуют в ответе
    hw03_flags = ['has_constant_columns', 'has_high_cardinality_categoricals', 'has_many_zero_values']
    hw03_flags_present = [
        flag for flag in hw03_flags
        if flag in flags_all
    ]
    
    if hw03_flags_present:
        print(f"[quality-flags-from-csv] HW03 флаги присутствуют: {hw03_flags_present}")
    else:
        print("[quality-flags-from-csv] ВНИМАНИЕ: Флаги из HW03 не найдены в ответе!")

    return QualityFlagsResponse(
        flags=flags_all,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
        latency_ms=latency_ms,
    )


# ---------- Дополнительный эндпоинт для примера JSON-сводки ----------

@app.post(
    "/json-summary-from-csv",
    tags=["quality", "hw03-extra"],
    summary="JSON-сводка по датасету (дополнительная функциональность HW03)",
)
async def json_summary_from_csv(file: UploadFile = File(...)):
    """
    Дополнительный эндпоинт, который возвращает JSON-сводку по датасету,
    соответствующую функциональности из дополнительной части HW03 (Вариант B).
    
    Возвращает компактную сводку с:
    - размерами датасета
    - интегральным quality_score
    - списком проблемных колонок
    - и другой полезной информацией
    """
    
    start = perf_counter()

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных.")

    # Получаем данные через EDA-ядро
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(summary, missing_df)
    
    # Определяем проблемные колонки (пропуски > 30%)
    problematic_columns = []
    if not missing_df.empty:
        problematic_columns = missing_df[
            missing_df["missing_share"] > 0.3
        ].index.tolist()
    
    # Получаем константные колонки (новая эвристика из HW03)
    constant_columns = [
        col.name for col in summary.columns 
        if col.unique <= 1
    ]
    
    # Получаем колонки с высокой кардинальностью (новая эвристика из HW03)
    high_cardinality_columns = []
    for col in summary.columns:
        if not col.is_numeric and 'object' in col.dtype.lower():
            if col.unique > 50:  # порог из HW03
                high_cardinality_columns.append(col.name)
    
    # Формируем JSON-сводку
    json_summary = {
        "dataset_info": {
            "filename": file.filename,
            "n_rows": summary.n_rows,
            "n_cols": summary.n_cols,
            "numeric_columns": sum(1 for col in summary.columns if col.is_numeric),
            "categorical_columns": sum(1 for col in summary.columns if not col.is_numeric),
        },
        "quality_metrics": {
            "quality_score": flags_all.get("quality_score", 0.0),
            "max_missing_share": flags_all.get("max_missing_share", 0.0),
            "has_constant_columns": flags_all.get("has_constant_columns", False),
            "has_high_cardinality_categoricals": flags_all.get("has_high_cardinality_categoricals", False),
            "has_many_zero_values": flags_all.get("has_many_zero_values", False),
            "too_few_rows": flags_all.get("too_few_rows", False),
            "too_many_missing": flags_all.get("too_many_missing", False),
        },
        "problematic_columns": {
            "by_missing": problematic_columns,
            "constant_columns": constant_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "columns_with_many_zeros": flags_all.get("columns_with_many_zeros", []),
        },
        "execution_info": {
            "latency_ms": (perf_counter() - start) * 1000.0,
            "flags_count": len(flags_all),
        }
    }
    
    return json_summary


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)