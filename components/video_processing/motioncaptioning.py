from typing import List
from .utils import describe_video, get_model_strategy

def process_videos(video_path: List[str], model_name: str, api_key: str, info: str) -> str:
    """
    Process multiple videos using a specified model strategy and return the results.

    Args:
    - video_paths (List[str]): List of paths to the video files to be processed.
    - model_name (str): Name of the model strategy to use for processing.
    - api_key (str): API key for initializing the model if required.
    - additional_info: Additional information to pass to the model strategy for each video.

    Returns:
    - str: A formatted string with the results of processing each video.
    """
    
    print(f"Processing video now")

    # Get additional information for the current video
    info = info[0]
    video_path = video_path[0]

    # Initialize the model strategy
    model_strategy = get_model_strategy(model_name, api_key)

    # Describe the video to extract frames
    clip = describe_video(video_path)

    # Perform inference using the model strategy
    results = model_strategy.inference(clip, info)

    # Return the concatenated results
    return results