# Traffic Jam Analysis with YOLOv8 and ByteTrack 🚦🚗

Модульная система анализа дорожных пробок с использованием **YOLOv8** для детекции транспорта и **ByteTrack** для трекинга объектов. Проект предоставляет автоматизированную оценку трафика по 10-балльной шкале на основе плотности, скорости, количества и других метрик. Поддерживается обработка изображений и видео, включая загрузку с YouTube через `yt-dlp`.

---

## 🔧 Возможности

- **Детекция и трекинг транспорта**
  - Поддержка автомобилей, мотоциклов, автобусов и грузовиков
  - Используются YOLOv8 и ByteTrack
- **Оценка пробок по 10-балльной шкале**:
  - 📏 Плотность транспорта в ROI (области интереса)
  - 🚗 Количество машин
  - 🕒 Средняя скорость
  - 📉 Дисперсия скоростей
  - ⛔ Доля неподвижных объектов
- **Сглаживание метрик** — скользящее среднее по 50 кадрам
- **Визуализация**:
  - ROI (жёлтая рамка, 90% кадра)
  - Bounding box'ы: зелёные (внутри ROI), серые (вне)
  - Наложение текста с метриками
- **Обработка видео и изображений**
- **Загрузка видео с YouTube** через `yt-dlp`
- **Модульная архитектура** для лёгкой поддержки и расширения

---

## 🗂 Структура проекта

```
traffic-jam-analysis/
├── src/
│   ├── detector.py         # YOLOv8 + ByteTrack
│   ├── traffic_analyzer.py # Логика оценки трафика
│   ├── visualizer.py       # Отрисовка меток и графиков
│   ├── processor.py        # Обработка видео/изображений
│   ├── main.py             # Точка входа, YouTube-загрузка
│   └── utils/              # Утилиты (пока пусто)
├── data/
│   └── sample_images/      # Входные изображения
├── output/                 # Обработанные видео, графики, логи
├── results/
│   ├── screenshots/        # Скриншоты результатов
│   ├── graphs/             # Графики анализа
│   └── logs/               # Текстовые логи
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## 🚀 Установка

```bash
git clone https://github.com/your-username/traffic-jam-analysis.git
cd traffic-jam-analysis
pip install -r requirements.txt
```

---

## 📥 Подготовка данных

- Добавьте изображения в папку `data/sample_images/`
- Убедитесь, что есть доступ к интернету для загрузки видео с YouTube

---

## ⚙️ Использование

Проверьте пути в `src/main.py`, например:

```python
image_dir = "data/sample_images"
image_output_path = "output/output_traffic_images.mp4"
```

Запустите скрипт:

```bash
python src/main.py
```

### 📈 Результаты

- 🎥 Видео: `output/output_traffic_images.mp4`, ...
- 📄 Логи: `output/output_traffic_images.txt`, ...
- 📊 Графики: `output/output_traffic_images_stats.png`, ...

---

## 🖼 Примеры

### 🔍 Скриншоты

Создайте скриншот с помощью ffmpeg:

```bash
ffmpeg -i output/output_traffic_1.mp4 -vf "select=eq(n\,100)" -vframes 1 results/screenshots/screenshot_1.png
```

Добавьте в Git:

```bash
git add results/screenshots/screenshot_1.png
git commit -m "Добавлен скриншот результата"
git push
```

### 📊 Графики

Автоматически сохраняются в `output/`.

```bash
cp output/output_traffic_1_stats.png results/graphs/traffic_stats_1.png
git add results/graphs/traffic_stats_1.png
git commit -m "Добавлен график метрик"
git push
```

### 📄 Логи

```bash
cp output/output_traffic_1.txt results/logs/traffic_log_1.txt
git add results/logs/traffic_log_1.txt
git commit -m "Добавлен лог анализа"
git push
```

---

## 🛠 Настройка

### ⚖️ Весовые коэффициенты (в `traffic_analyzer.py`):

```python
points = 10 * (
    0.4 * density_norm +
    0.2 * car_count_norm +
    0.2 * (1 - speed_norm) +
    0.1 * (1 - speed_var_norm) +
    0.1 * stationary_ratio
)
```

### 🪟 Сглаживание метрик:

```python
window_size = 100
```

### 📍 Область интереса (ROI):

```python
roi = (10, 10, 630, 350)  # В файле processor.py
```

### 🎬 Загрузка видео с YouTube:

```python
video_urls = [
    ("https://www.youtube.com/watch?v=zOq2XdwHGT0", "test_video_1.mp4"),
    ...
]
```

---

## 📦 Зависимости

См. `requirements.txt`:

- Python 3.8+
- `opencv-python`
- `ultralytics`
- `matplotlib`
- `lap`
- `numpy`
- `yt-dlp`

Установка:

```bash
pip install -r requirements.txt
```

---

## 📄 Лицензия

MIT License. Подробнее см. [LICENSE](./LICENSE).