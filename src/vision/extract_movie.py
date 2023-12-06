import os
import uuid

from src.utils import makedirs


def extract_unique_frames(urls=None, download_dir=None, export_dir=None):
    import fiftyone as fo
    import fiftyone.utils.youtube as fouy

    download_dir = download_dir or os.getenv('VID_DOWNLOADS', "viddownloads")
    if urls:
        fouy.download_youtube_videos(urls, download_dir=download_dir)

    # Create a FiftyOne Dataset
    dataset = fo.Dataset.from_videos_dir(download_dir)

    # Convert videos to images, sample 1 frame per second
    frame_view = dataset.to_frames(sample_frames=True, fps=1)

    import fiftyone.brain as fob

    # Index images by similarity
    results = fob.compute_similarity(frame_view, brain_key="frame_sim")

    # Find maximally unique frames
    num_unique = 50 # Scale this to whatever you want
    results.find_unique(num_unique)
    unique_view = frame_view.select(results.unique_ids)
    #print(unique_view)
    #return unique_view

    # Visualize in the App
    # session = fo.launch_app(frame_view)
    # session = fo.launch_app(unique_view)

    export_dir = export_dir or "/tmp/extraction_%s" % str(uuid.uuid4())
    makedirs(export_dir, exist_ok=True)
    unique_view.export(export_dir, dataset_type=fo.types.VideoDirectory)
    return export_dir
