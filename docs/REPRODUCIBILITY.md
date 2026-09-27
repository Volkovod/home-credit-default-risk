# Воспроизведение

## Что именно подтверждено

Обучение выполнено в Kaggle: `mrfirstik/home-credit-research-gpu-20260926`, версия 1,
2 × Tesla T4, статус COMPLETE. Метрики скачаны и пересчитаны по сохранённым прогнозам.
Затем одна полная отправка дала public 0.80088 / private 0.79968.
Код подтверждённого запуска — в `archive/executed/`; его SHA-256 и хэши публичных
скриптов — в [source_lineage.json](../reports/metrics/source_lineage.json).

Скрипты в `src/` отличаются от архивных только инфраструктурными изменениями:

1. Пути можно задать переменными окружения.
2. Выбор исходного кэша ограничен нужной версией признаков.
3. Недоступные опорные benchmark-модели можно пересоздать, а не требовать приватные outputs.

Последний путь добавлен при упаковке. Его код и контракты проверены, но полный
GPU-запуск этой публичной версии заново не выполнялся. Поэтому архивные метрики
не выдаются за новый результат после рефакторинга.

## Быстрый просмотр без GPU и исходных данных

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell вместо предыдущей строки:
# .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-report.txt
python -m unittest discover -s tests -v
python scripts/check_repository.py
python scripts/make_figures.py
```

Ноутбук `notebooks/00_research.ipynb` уже содержит выполненные выводы и графики.
Он использует только агрегированные CSV/JSON. Для интерактивного повторного
выполнения установите свой Jupyter frontend или откройте его в VS Code.

Чтобы заново собрать и выполнить этот обзор из скрипта:

```bash
python -m pip install -r requirements-notebook.txt
python scripts/build_research_notebook.py
```

CI выполняет проверки кода и артефактов, тесты и построение графиков на CPU.
Он не учит модели и не скачивает данные. Конфигурация CI включена, но её запуск
в вашем будущем GitHub-репозитории ещё не происходил.

## Данные

Получите исходные файлы на
[странице соревнования](https://www.kaggle.com/competitions/home-credit-default-risk/data)
с соблюдением условий Kaggle. Нужны десять CSV: `application_train`,
`application_test`, `bureau`, `bureau_balance`, `previous_application`,
`POS_CASH_balance`, `credit_card_balance`, `installments_payments`,
`sample_submission`, `HomeCredit_columns_description`.

Для локального запуска распакуйте их в `data/raw/`. Этап расширения ожидает
CSV, а не zip. Репозиторий не содержит этих данных и не требует доступа
к приватному dataset исходного автора.

## Три последовательных этапа на Kaggle

**1. Исходная витрина**

Импортируйте `notebooks/01_build_features.ipynb`, прикрепите официальный набор
CSV, отключите GPU. Выполните `Save Version → Run All`. Результат:
`/kaggle/working/home_credit_features/` с Parquet-матрицами, labels, ID,
`original_splits.npz`, словарём признаков и manifest.

**2. Новые признаки**

Импортируйте `notebooks/02_augment_features.ipynb`. Прикрепите те же CSV и output
первого notebook. GPU не требуется. Сохраните выполненную версию. Результат:
`/kaggle/working/home_credit_research_features/`, 1552 столбца.

**3. Модели**

Импортируйте `notebooks/03_train_gpu.ipynb`. Прикрепите output второго этапа.
Выберите **GPU T4 ×2**, затем выполните все ячейки. Убедитесь, что доступны
LightGBM с поддержкой GPU/OpenCL, XGBoost и CatBoost; в исходном успешном
Kaggle-окружении они уже были установлены.

При отсутствии прежних benchmark outputs код обучит три опорные модели заново.
После benchmark две выбранные семьи запускаются на разных GPU. Вариант с одним
GPU без изменения планировщика не поддерживается. Автоматической отправки
в соревнование внутри кода нет.

Результат: `/kaggle/working/home_credit_research/`. Основные файлы:

| Файл | Содержание |
|---|---|
| `run_summary.json` | Версии, выбранные модели, метрики, время |
| `benchmark*.csv/json` | Конфигурации и сравнение на одном split |
| `frozen_selection.json` | Решение по OOF, записанное до оценки holdout |
| `oof_predictions.csv`, `holdout_predictions.csv` | Прогнозы для независимого пересчёта |
| `split_manifest.csv` | Разделение клиентов |
| `*_schema.json`, `*_metadata.json` | Категории, peer-медианы, параметры и порядок столбцов |
| `*.ubj`, `*.cbm`, `*.txt` | Модели XGBoost, CatBoost, LightGBM |
| `submission.csv` | Основной прогноз: полный refit с двумя seed |
| `submission_cv.csv` | Альтернатива: ансамбль CV-моделей development |

Не все эти файлы входят в GitHub: клиентские строки и большие модели исключены.
Output Kaggle — место хранения полного результата запуска.

## Локальный/Linux-запуск

Требуются два совместимых GPU, драйверы CUDA и GPU-enabled LightGBM/OpenCL.
`pip install` сам по себе не гарантирует наличие драйверов или GPU-сборки LightGBM.
Python 3.11 подходит для указанных версий основных библиотек.

```bash
python -m pip install -r requirements.txt
HC_RAW=data/raw HC_OUTPUT=artifacts/base python src/build_features.py
HC_RAW=data/raw HC_CACHE=artifacts/base HC_OUTPUT=artifacts/augmented python src/augment_features.py
HC_INPUT_ROOT=artifacts HC_CACHE=artifacts/augmented HC_OUTPUT=artifacts/run python src/train_research.py
```

Это синтаксис Bash. В PowerShell задайте те же значения через `$env:HC_RAW`,
`$env:HC_CACHE`, `$env:HC_OUTPUT`, `$env:HC_INPUT_ROOT` перед каждым запуском.
Локальный полный GPU-прогон не проверялся; подтверждённая среда — Kaggle.

## Независимая проверка скачанного результата

```bash
python scripts/analyze_predictions.py \
  --result-dir artifacts/run \
  --sample-submission data/raw/sample_submission.csv \
  --output reports/metrics
python scripts/make_figures.py
```

Скрипт проверяет метрики, пять фолдов, фиксированную смесь, непересечение ID,
benchmark внутри development и точное совпадение submission с шаблоном.
Затем строит агрегированные ROC/PR/calibration-данные и парные bootstrap-интервалы.
Не переносите клиентские prediction CSV в публичный репозиторий.

Для нового обучения сначала замените относящиеся к нему метрики в `reports/metrics/`:
не смешивайте summary старого запуска с прогнозами нового. Исходный архивный
результат стоит сохранить отдельной версией репозитория.

## Время и воспроизводимость

Подтверждённый исследовательский запуск занял 3059,3 с (~51 мин).
Это benchmark новых кандидатов, CV, оценка и refit; подготовка кэшей и ранние
эксперименты в это время не входят. При пересоздании старых baseline время увеличится.
Пиковая RAM не измерялась; обещаний минимального объёма памяти нет.

GPU-обучение может быть недетерминированным даже с фиксированным seed —
[документация CatBoost](https://catboost.ai/docs/en/features/training-on-gpu).
Критерий повторения — корректность процедуры и близость оценок, а не побитовое
совпадение новых моделей. Для точного аудита сохранены прогнозы исходного запуска,
хэши, версии и неизменённые источники.
