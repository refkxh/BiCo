import os
import imageio.v2 as imageio

def trim_video_to_frames(input_path, output_path, max_frames=81):
    
    if not os.path.exists(input_path):
        print(f"Error: file not exit '{input_path}'")
        return False

    try:
        reader = imageio.get_reader(input_path)
    except Exception as e:
        print(f"Error: can't open file '{input_path}': {e}")
        return False

    try:
        meta = reader.get_meta_data() or {}
    except Exception:
        meta = {}

    fps = meta.get('fps') or meta.get('framerate') or 0.0

    try:
        first_frame = reader.get_data(0)
    except Exception as e:
        print(f"Error: cannot read first frame: {e}")
        reader.close()
        return False

    height, width = first_frame.shape[0], first_frame.shape[1]

    try:
        original_frame_count = reader.count_frames()
    except Exception:
        original_frame_count = meta.get('nframes') or meta.get('n_frames') or -1

    print(f"original video info: {width}x{height} @ {fps:.2f} FPS, total {original_frame_count} frames")

    try:
        writer_kwargs = {'codec': 'libx264'}
        if fps and fps > 0:
            writer_kwargs['fps'] = fps
        writer = imageio.get_writer(output_path, **writer_kwargs)
    except Exception as e:
        print(f"Error: can't write video '{output_path}': {e}")
        reader.close()
        return False

    frame_count = 0

    for idx in range(max_frames):
        try:
            frame = reader.get_data(idx)
        except Exception:
            if idx < max_frames:
                print(f"Warn: video end before {max_frames}frames.")
                print(f"output video only {frame_count} frames.")
            break
        writer.append_data(frame)
        frame_count += 1

    print(f"Finish Processing. total {frame_count} frames")

    reader.close()
    writer.close()

    print(f"video saved to: '{output_path}'")
    return True



if __name__ == "__main__":
    
    input_video_file = 'data/videos/play_game_1_original.mp4'
    output_video_file = 'data/videos/play_game_1.mp4'

    if not os.path.exists(input_video_file):
        print("="*50)
        print(f"Can;t find video file '{input_video_file}'")
        print("="*50)
    else:
        
        success = trim_video_to_frames(input_video_file, output_video_file, max_frames=81)

        if success:
            print("\nsuccessfully trim video to 81 frames.")
        else:
            print("\nfailed to trim video.")