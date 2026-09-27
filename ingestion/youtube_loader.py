from youtube_transcript_api import YouTubeTranscriptApi
from langchain.schema import Document
from utils.helpers import create_document, chunk_documents


def extract_video_id(url: str) -> str:
    """
    Extracts the video ID from a YouTube URL.
    
    Handles these formats:
    - https://www.youtube.com/watch?v=abc123
    - https://youtu.be/abc123
    """
    if "youtu.be/" in url:
        return url.split("youtu.be/")[-1].split("?")[0]
    elif "watch?v=" in url:
        return url.split("watch?v=")[-1.split("&")[0]
    else:
        raise ValueError(f"Could not extract video ID from URL: {url}")


def load_youtube(url: str) -> list[Document]:
    """
    Fetches the transcript of a YouTube video,
    combines all text into one document and returns chunks.
    """
    video_id = extract_video_id(url)

    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

    # transcript_list is a list of dicts like:
    # [{"text": "hello world", "start": 0.0, "duration": 1.5}, ...]
    # We join all text pieces into one full transcript
    full_transcript = " ".join(
        entry["text"] for entry in transcript_list
    )

    doc = create_document(
        text=full_transcript,
        metadata={
            "source": url,
            "video_id": video_id,
            "type": "youtube"
        }
    )

    return chunk_documents([doc])