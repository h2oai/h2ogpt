import argparse
import os


def download_video(url, output_dir='.'):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'restrictfilenames': True,
    }

    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])


def main():
    parser = argparse.ArgumentParser(description="Download a video from a given URL (e.g. https://www.youtube.com/watch?v=2Njmx-UuU3M)")
    parser.add_argument("--url", type=str, required=True, help="The URL of the video to download")
    parser.add_argument("--output", type=str, default=".", help="The directory to save the downloaded video")
    args = parser.parse_args()

    download_video(url=args.url, output_dir=args.output)


if __name__ == "__main__":
    main()
