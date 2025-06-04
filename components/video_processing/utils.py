import av
import numpy as np

def describe_video(video_path: str) -> np.ndarray:
    """
    Extract frames from a video at specific intervals and return them as a stacked numpy array.

    Args:
    - video_path (str): Path to the video file.

    Returns:
    - np.ndarray: A numpy array containing the extracted frames.
    """
    # Open the video file
    container = av.open(video_path)
    
    # Get the total number of frames in the video
    total_frames = container.streams.video[0].frames

    # ✅ Prevent division by zero
    if total_frames == 0:
        raise ValueError(f"🚨 ERROR: The video '{video_path}' has zero frames. It may be empty or corrupted.")
    
    # Calculate indices for extracting frames at intervals
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    
    # Initialize list to store extracted frames
    frames = []
    
    # Seek to the beginning of the video
    container.seek(0)
    
    # Define the range of frames to process
    start_index = indices[0]
    end_index = indices[-1]
    
    # Iterate over frames and extract the ones at the specified indices
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    
    # Convert extracted frames to numpy array format
    clip = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    
    return clip

from .models import ChatGPT
def get_model_strategy(model_name: str, api_key: str = None):
    """
    Initialize and return the strategy object for the specified model.

    Args:
    - model_name (str): Name of the model ("ChatGPT").
    - api_key (str, optional): API key for initializing the model if required.

    Returns:
    - Strategy object for the specified model.
    """
    
    if model_name == "ChatGPT":
        strategy = ChatGPT()
        strategy.initialise_model(api_key)
    else:
        raise ValueError("Invalid model name")
    
    return strategy