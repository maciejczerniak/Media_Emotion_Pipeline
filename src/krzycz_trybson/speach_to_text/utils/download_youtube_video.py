import uuid
import asyncio
from pathlib import Path

from yt_dlp import YoutubeDL  # type: ignore


async def download_youtube_video(url: str, save_path: str) -> str:
    """
    Download a YouTube video and save it to save_path.
    Args:
        url: A YouTube video url.
        save_path: Path to save the video.

    Returns: Title of the video. If the download fails or the video is not available, returns a random UUID.
    """

    # More robust yt-dlp options to handle YouTube's anti-bot measures
    ydl_opts = {
        # Use more flexible format selection with fallbacks
        "format": "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best[height<=480]/best",
        "outtmpl": save_path,
        # Audio extraction settings
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        # Anti-detection measures
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "extractor_retries": 3,
        "retries": 3,
        # Bypass age restrictions and geo-blocking
        "age_limit": 99,
        "geo_bypass": True,
        # Less verbose output
        "quiet": True,
        "no_warnings": False,  # Keep warnings to debug issues
        # Use cookies if available (helps with some restricted content)
        "cookiefile": None,
        # Additional headers to avoid detection
        "http_headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
    }

    try:
        # Run in a thread to avoid blocking the async event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _download_sync, url, ydl_opts)
        return result

    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        # Try with even more permissive settings as fallback
        return await _fallback_download(url, save_path)


def _download_sync(url: str, ydl_opts: dict) -> str:
    """Synchronous download function to run in executor"""
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", str(uuid.uuid4()))
        return str(title)


async def _fallback_download(url: str, save_path: str) -> str:
    """Fallback download with even more permissive settings"""
    print(f"Attempting fallback download for {url}")

    fallback_opts = {
        # Very permissive format selection
        "format": "worst[ext=mp4]/worst",
        "outtmpl": save_path,
        # Skip audio extraction, just get any format
        "postprocessors": [],
        # Maximum bypass attempts
        "extractor_retries": 5,
        "retries": 5,
        "sleep_interval": 1,
        # Minimal processing
        "quiet": False,  # Show more info for debugging
        "verbose": True,
        # Ignore errors and continue
        "ignoreerrors": True,
        "no_abort_on_error": True,
    }

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _download_sync, url, fallback_opts)
        return result
    except Exception as e:
        print(f"Fallback download also failed for {url}: {str(e)}")
        # Return a UUID as specified in the original function
        return str(uuid.uuid4())


# Alternative function using different approach
async def download_youtube_video_alternative(url: str, save_path: str) -> str:
    """
    Alternative download method using yt-dlp's built-in format selection logic
    """
    ydl_opts = {
        # Let yt-dlp choose the best available format automatically
        "format": None,  # This forces yt-dlp to use its default best format selection
        "outtmpl": save_path,
        "extract_flat": False,
        # Simpler audio processing
        "writeinfojson": False,
        "writethumbnail": False,
        "writesubtitles": False,
        # Basic anti-detection
        "user_agent": "yt-dlp/2023.01.06",
        # Error handling
        "ignoreerrors": False,
        "quiet": True,
    }

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _download_sync, url, ydl_opts)
        return result
    except Exception as e:
        print(f"Alternative download failed for {url}: {str(e)}")
        return str(uuid.uuid4())


def check_available_formats(url: str):
    """Check what formats are available for a YouTube video"""
    print(f"\nChecking available formats for: {url}")

    ydl_opts = {
        "listformats": True,
        "quiet": False,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"✗ Failed to list formats: {e}")


def download(url: str, output_path: str = "test_download.%(ext)s"):
    """Test download with various format options"""

    test_formats = [
        "bestaudio[ext=m4a]",
        "bestaudio",
        "worst[ext=mp4]",
        "worst",
    ]

    for fmt in test_formats:
        print(f"\nTesting format: {fmt}")

        ydl_opts = {
            "format": fmt,
            "outtmpl": output_path,
            "quiet": True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)  # Don't actually download
                print(f"✓ Format '{fmt}' is available")
                print(f"  Title: {info.get('title', 'Unknown')}")
                return fmt
        except Exception as e:
            print(f"✗ Format '{fmt}' failed: {e}")

    print("No working formats found!")
    return None


def main():
    print("YouTube Download Troubleshooting Tool")
    print("=====================================")

    # Test URLs from the original error
    test_urls = [
        "https://www.youtube.com/watch?v=L72_ScVK_rY",  # From playlist
        "https://www.youtube.com/watch?v=yN85YyA0AIw",  # From playlist
    ]

    # Step 2: Check formats for each URL
    print("\n2. Checking available formats...")
    for url in test_urls:
        check_available_formats(url)
        working_format = download(url)
        if working_format:
            print(f"✓ Recommended format for this video: {working_format}")
        print("-" * 50)

    # Step 3: Provide recommendations
    print("\n3. Recommendations:")
    print("   • Use the updated download_youtube_video_fixed.py")
    print("   • If issues persist, try the simple fix version")
    print("   • Consider using cookies.txt file for restricted content")
    print("   • Run yt-dlp directly to test: yt-dlp --list-formats <URL>")


if __name__ == "__main__":
    links = [
    "https://www.youtube.com/watch?v=qy9f42zmSPI",
    "https://www.youtube.com/watch?v=-pNYvGjSDro"
]

    video_dir = "../../../../data/dataset_sadness_videos/videos/"
    video_dir_path = Path(video_dir)
    video_dir_path.mkdir(parents=True, exist_ok=True)

    async def main():
        download_jobs = [
            download_youtube_video(url, str(video_dir_path / f"{i}.wav"))
            for i, url in enumerate(links)
        ]
        results = await asyncio.gather(*download_jobs)
        print(results)

    asyncio.run(main())