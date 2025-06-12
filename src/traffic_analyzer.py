import numpy as np
from collections import deque

def is_in_roi(box, roi):
    """Проверка, находится ли объект в области интереса (ROI)."""
    x1, y1, x2, y2 = box[:4]
    rx1, ry1, rx2, ry2 = roi
    if rx1 >= rx2 or ry1 >= ry2:
        print(f"Ошибка: Неверные координаты ROI: ({rx1}, {ry1}, {rx2}, {ry2})")
        return False
    return x1 >= rx1 and y1 >= ry1 and x2 <= rx2 and y2 <= ry2

def analyze_traffic(tracks, speeds, frame_area, roi, window_size=50):
    """Анализ пробок с улучшенной оценкой и сглаживанием."""
    roi_tracks = [track for track in tracks if is_in_roi(track, roi)]
    total_box_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2, _, _, _ in roi_tracks)
    density = total_box_area / frame_area if frame_area > 0 else 0
    car_count = len(roi_tracks)
    avg_speed = np.mean(speeds) if speeds else 0
    speed_variance = np.var(speeds) if len(speeds) > 1 else 0
    stationary_count = sum(1 for speed in speeds if speed < 1)
    stationary_ratio = stationary_count / len(speeds) if speeds else 0


    density_norm = min(1.0, density / 0.5)  # Макс. плотность 0.5
    car_count_norm = min(1.0, car_count / 20)  # Макс. 20 машин
    speed_norm = min(1.0, avg_speed / 15)  # Макс. скорость 15 px/s
    speed_var_norm = min(1.0, speed_variance / 50)  # Макс. дисперсия 50


    points = 10 * (
        0.3 * density_norm +
        0.2 * car_count_norm +
        0.2 * (1 - speed_norm) +
        0.2 * (1 - speed_var_norm) +
        0.1 * stationary_ratio
    )
    points = min(10, max(0, int(points)))

    if not hasattr(analyze_traffic, 'density_deque'):
        analyze_traffic.density_deque = deque(maxlen=window_size)
        analyze_traffic.car_count_deque = deque(maxlen=window_size)
        analyze_traffic.speed_deque = deque(maxlen=window_size)
        analyze_traffic.points_deque = deque(maxlen=window_size)

    analyze_traffic.density_deque.append(density)
    analyze_traffic.car_count_deque.append(car_count)
    analyze_traffic.speed_deque.append(avg_speed)
    analyze_traffic.points_deque.append(points)

    smoothed_density = np.mean(analyze_traffic.density_deque)
    smoothed_car_count = int(np.mean(analyze_traffic.car_count_deque))
    smoothed_speed = np.mean(analyze_traffic.speed_deque)
    smoothed_points = int(np.mean(analyze_traffic.points_deque))

    if smoothed_points <= 2:
        status = f"{smoothed_points} points (Free Flow)"
        confidence = 1.0 - min(smoothed_density / 0.3, 1.0)
    elif smoothed_points <= 5:
        status = f"{smoothed_points} points (Congested)"
        confidence = min(1.0, (smoothed_density - 0.2) / 0.3 + (10 - smoothed_speed) / 10)
    elif smoothed_points <= 8:
        status = f"{smoothed_points} points (Heavy Traffic)"
        confidence = min(1.0, (smoothed_density - 0.4) / 0.3 + (5 - smoothed_speed) / 5)
    else:
        status = f"{smoothed_points} points (Gridlock)"
        confidence = min(1.0, (smoothed_density - 0.6) / 0.3 + (2 - smoothed_speed) / 2)

    return status, confidence, smoothed_density, smoothed_speed, smoothed_car_count, smoothed_points