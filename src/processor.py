import os
import cv2
import glob
from .detector import YoloDetector
from .traffic_analyzer import analyze_traffic
from .visualizer import visualize, plot_traffic_stats

def process_images(image_dir, output_video_path):
    """Обработка последовательности изображений и создание видео."""
    detector = YoloDetector()
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    if not image_paths:
        print(f"Ошибка: Изображения не найдены в {image_dir}")
        return

    sample_frame = cv2.imread(image_paths[0])
    if sample_frame is None:
        print("Ошибка: Не удалось загрузить образец изображения")
        return
    height, width = 640, 360
    frame_area = width * height
    roi = (int(width * 0.05), int(height * 0.05), int(width * 0.95), int(height * 0.95))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (width, height)
    )

    density_history = []
    speed_history = []
    score_history = []
    log_path = output_video_path.replace('.mp4', '.txt')
    with open(log_path, 'w') as log_file:
        for frame_id, img_path in enumerate(image_paths[:100]):
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            frame = cv2.resize(frame, (width, height))
            tracks, speeds = detector.track(frame)
            status, confidence, density, avg_speed, car_count, points = analyze_traffic(tracks, speeds, frame_area, roi)
            frame = visualize(frame, tracks, car_count, status, confidence, density, avg_speed, detector, roi)

            density_history.append(density)
            speed_history.append(avg_speed)
            score_history.append(points)

            out.write(frame)
            log_file.write(f"Кадр {frame_id}: Машин={car_count}, Статус={status}, Плотность={density:.2f}, Скорость={avg_speed:.2f}, Баллы={points}\n")

    out.release()
    plot_traffic_stats(density_history, speed_history, score_history, output_video_path.replace('.mp4', '_stats.png'))
    print(f"Результаты сохранены: {output_video_path}, {log_path}, {output_video_path.replace('.mp4', '_stats.png')}")

def process_video(video_path, output_video_path):
    """Обработка видеофайла для мониторинга трафика."""
    if not os.path.exists(video_path):
        print(f"Ошибка: Видео {video_path} не найдено")
        return

    detector = YoloDetector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео {video_path}")
        return

    width, height = 640, 360
    frame_area = width * height
    roi = (int(width * 0.05), int(height * 0.05), int(width * 0.95), int(height * 0.95))

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (width, height)
    )

    density_history = []
    speed_history = []
    score_history = []
    frame_id = 0
    log_path = output_video_path.replace('.mp4', '.txt')
    with open(log_path, 'w') as log_file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))
            tracks, speeds = detector.track(frame)
            status, confidence, density, avg_speed, car_count, points = analyze_traffic(tracks, speeds, frame_area, roi)
            frame = visualize(frame, tracks, car_count, status, confidence, density, avg_speed, detector, roi)

            density_history.append(density)
            speed_history.append(avg_speed)
            score_history.append(points)

            out.write(frame)
            log_file.write(f"Кадр {frame_id}: Машин={car_count}, Статус={status}, Плотность={density:.2f}, Скорость={avg_speed:.2f}, Баллы={points}\n")
            frame_id += 1

    cap.release()
    out.release()
    plot_traffic_stats(density_history, speed_history, score_history, output_video_path.replace('.mp4', '_stats.png'))
    print(f"Результаты сохранены: {output_video_path}, {log_path}, {output_video_path.replace('.mp4', '_stats.png')}")