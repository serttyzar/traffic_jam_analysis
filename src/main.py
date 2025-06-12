import os
import subprocess
from .processor import process_images, process_video

def download_youtube_videos():
    """Загрузка видео с YouTube с использованием yt-dlp."""
    video_urls = [
        ("https://www.youtube.com/watch?v=zOq2XdwHGT0", "test_video_1.mp4"),
        ("https://www.youtube.com/watch?v=bByNEd2jQ0k", "test_video_2.mp4"),
        ("https://www.youtube.com/watch?v=Y1jTEyb3wiI", "test_video_3.mp4"),
        ("https://www.youtube.com/watch?v=CftLBPI1Ga4", "test_video_4.mp4"),
        ("https://www.youtube.com/watch?v=IdSD3wNm1zQ", "test_video_5.mp4"),
        ("https://www.youtube.com/watch?v=do1MgKO5Jh4", "test_video_6.mp4"),
    ]
    
    os.makedirs("output", exist_ok=True)
    for url, output_name in video_urls:
        output_path = f"output/{output_name}"
        try:
            subprocess.run([
                "yt-dlp", "-f", "best[ext=mp4]", "-o", output_path, url
            ], check=True)
            print(f"Загружено: {output_name}")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при загрузке {url}: {e}")

def main():
    """Основная функция для запуска анализа пробок."""
    print("Установка зависимостей...")
    os.system("pip install lap ultralytics matplotlib opencv-python yt-dlp")

    # Обработка изображений
    image_dir = "data/sample_images"
    image_output_path = "output/output_traffic_images.mp4"
    process_images(image_dir, image_output_path)

    # Загрузка видео с YouTube
    print("Загрузка видео с YouTube...")
    download_youtube_videos()

    # Обработка видео
    for i in range(1, 7):
        input_path = f"output/test_video_{i}.mp4"
        output_path = f"output/output_traffic_{i}.mp4"
        if os.path.exists(input_path):
            process_video(input_path, output_path)
        else:
            print(f"Видео {input_path} не найдено, пропускаем")

if __name__ == "__main__":
    main()