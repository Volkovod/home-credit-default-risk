# Как загрузить проект

## В GitHub через браузер

1. Распакуйте обновлённый `home-credit-default-risk-github-v2.zip` в новую папку.
2. Откройте [создание репозитория](https://github.com/new). Введите имя
   `home-credit-default-risk`, выберите Public, чтобы проект могли видеть другие,
   и создайте репозиторий. Автоматически добавлять README и `.gitignore` не нужно:
   они уже находятся в проекте.
3. В пустом репозитории нажмите **uploading an existing file**. Если список
   файлов уже есть, выберите **Add file → Upload files**.
4. Откройте распакованную папку `home-credit-default-risk`. Перетащите в браузер
   её содержимое: `README.md`, файлы `requirements…`, `.gitignore` и папки
   `notebooks`, `src`, `configs`, `reports`, `docs`, `scripts`, `tests`, `archive`, `.github`.
5. Проверьте, что `README.md` находится прямо в корне репозитория.
   В поле сообщения напишите `Add Home Credit project` и подтвердите загрузку.

ZIP служит для передачи проекта. GitHub показывает README, ноутбуки и графики,
когда загружены распакованные файлы с сохранёнными папками.
Инструкция GitHub: [добавление файлов](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository).

Для загрузки используйте содержимое архива: из него исключены временные файлы,
данные, модели и ключи. Если проводите новые эксперименты локально, не добавляйте
в браузер всю рабочую папку без проверки: правила `.gitignore` применяются Git,
а при ручной загрузке нужно выбирать файлы самостоятельно.

После загрузки:

- главная страница репозитория покажет README;
- `notebooks/00_research.ipynb` откроется с сохранёнными таблицами и графиками;
- папка `reports/figures` содержит изображения, на которые ссылается описание;
- во вкладке Actions можно посмотреть результат автоматических проверок.

Для поля About подойдёт описание:

> Home Credit default prediction with XGBoost and CatBoost. Kaggle private ROC AUC: 0.79968.

Темы: `machine-learning`, `credit-risk`, `xgboost`, `catboost`, `kaggle`.

## Если нужен запуск на Kaggle

Для просмотра проекта достаточно GitHub. Для повторного обучения импортируйте
в Kaggle по очереди:

1. `notebooks/01_build_features.ipynb` — исходные CSV → таблица признаков;
2. `notebooks/02_augment_features.ipynb` — CSV и результат первого этапа → дополнительные признаки;
3. `notebooks/03_train_gpu.ipynb` — результат второго этапа → модели и прогнозы.

Данные подключаются отдельно. Для третьего этапа нужны два GPU T4.
Подробности — в [инструкции запуска](REPRODUCIBILITY.md).

`00_research.ipynb` — обзор результатов. Для его повторного выполнения нужны
папки `reports/metrics` и `reports/figures` из этого проекта.

`submission.csv` загружается на странице соревнования для получения оценки.
Текущий финальный файл уже отправлен; повторять отправку не требуется.
Он находится отдельно от GitHub-архива, в папке `outputs/home-credit`.

## Загрузка через Git

Если Git уже установлен, из корня распакованного проекта можно выполнить:

```bash
git init -b main
git add .
git diff --cached --stat
git commit -m "Add Home Credit project"
git remote add origin https://github.com/YOUR_USERNAME/home-credit-default-risk.git
git push -u origin main
```

Перед этим создайте пустой репозиторий и замените `YOUR_USERNAME` на имя
своего аккаунта. Для публикации через командную строку потребуется авторизация GitHub.
