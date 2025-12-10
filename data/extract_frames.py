import argparse
import os

from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from tqdm import tqdm


def extract_frames(video_path, output_folder):
    """
    Extract all frames from a video file and save them as JPG images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Output folder path where extracted frames will be saved.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    try:
        # Load the video file
        print(f"Loading video: {video_path}")
        clip = VideoFileClip(video_path)

        # Get the video's duration and an approximate total frame count (depends on FPS)
        duration = clip.duration
        fps = clip.fps
        total_frames_approx = int(duration * fps)
        print(f"Video duration: {duration:.2f} seconds")
        print(f"Frame rate (FPS): {fps:.2f}")
        print(f"Estimated total frames: {total_frames_approx}")

        # Extract and save frames one by one
        print("Starting frame extraction...")
        frame_count = 0
        for i, frame in tqdm(enumerate(clip.iter_frames(fps=fps, dtype="uint8"))):
            frame_filename = os.path.join(output_folder, f"{i:05d}.jpg")
            # Save using moviepy's save_frame or with PIL
            # Method 1: use moviepy (simpler)
            clip.save_frame(frame_filename, t=i / fps)  # t is time in seconds

            # Method 2: use PIL (if more image-processing control is needed)
            # img = Image.fromarray(frame)
            # img.save(frame_filename, "JPEG")

            frame_count += 1

        print(f"Frame extraction complete. Saved {frame_count} frames to {output_folder}")

        # Close the video file
        clip.close()

    except Exception as e:
        print(f"Error processing video: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract all frames from a video file and save as JPG images."
    )
    parser.add_argument(
        "--video_file", help="Path to the input video file.", type=str, default=""
    )
    parser.add_argument(
        "--output_dir", help="Output folder to save extracted frames.", type=str, default=""
    )

    args = parser.parse_args()

    args.video_file = "/data/videos/play_game_1.mp4"
    args.output_dir = "/data/videos/play_game_1"

    extract_frames(args.video_file, args.output_dir)
