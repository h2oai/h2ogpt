import argparse
import os
import random


def selenium(base_url, video_url):
    from selenium import webdriver
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.common.by import By
    import time

    # Set up Selenium browser (Chrome in this case)
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(options=options)

    # Visit site
    driver.get(base_url)

    # Simulate a human-like search
    search_bar = driver.find_element(By.NAME, "search_query")
    search_bar.send_keys(video_url)
    search_bar.send_keys(Keys.RETURN)

    # Wait for the page to load
    time.sleep(3)

    # Click on the first video result
    first_video = driver.find_element(By.CSS_SELECTOR, "a#video-title")
    first_video.click()

    # Let the video play for a few seconds (mimic human behavior)
    tsleep = random.randint(5, 15)
    time.sleep(tsleep)

    # Get video URL
    video_url_new = driver.current_url
    print(f"Video URL: {video_url_new}")

    return video_url, driver


def download_web_video(video_url, base_url="https://www.youtube.com", output_dir='.'):
    video_url, driver = selenium(base_url, video_url)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'restrictfilenames': True,
    }

    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

    # Close the browser
    driver.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Download a video from a given URL, e.g. https://www.youtube.com/watch?v=2Njmx-UuU3M")
    parser.add_argument("--video_url", type=str, required=True, help="The URL of the actual video to download")
    parser.add_argument("--base_url", type=str, required=False, default="https://www.youtube.com",
                        help="The base website URL that has the video to download, e.g. https://www.youtube.com")
    parser.add_argument("--output_dir", type=str, default=".", help="The directory to save the downloaded video")
    args = parser.parse_args()

    download_web_video(video_url=args.video_url, base_url=args.base_url, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
