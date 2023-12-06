#Installing collected packages: sseclient-py, pprintpp, kaleido, glob2, xmltodict, tifffile, taskgroup, retrying, rarfile, pyzstd, pyppmd, pybcj, priority, plotly, opencv-python-headless, multivolumefile, inflate64, imageio, hyperframe, humanize, hpack, graphql-core, ftfy, fiftyone-db, dnspython, dacite, argcomplete, strawberry-graphql, scikit-image, pymongo, py7zr, h2, voxel51-eta, universal-analytics-python3, sse-starlette, motor, mongoengine, hypercorn, fiftyone-brain, fiftyone
# Successfully installed argcomplete-3.1.6 dacite-1.7.0 dnspython-2.4.2 fiftyone-0.23.0 fiftyone-brain-0.14.0 fiftyone-db-1.0 ftfy-6.1.3 glob2-0.7 graphql-core-3.2.3 h2-4.1.0 hpack-4.0.0 humanize-4.9.0 hypercorn-0.15.0 hyperframe-6.0.1 imageio-2.33.0 inflate64-1.0.0 kaleido-0.2.1 mongoengine-0.24.2 motor-3.3.2 multivolumefile-0.2.3 opencv-python-headless-4.8.1.78 plotly-5.18.0 pprintpp-0.4.0 priority-2.0.0 py7zr-0.20.8 pybcj-1.0.2 pymongo-4.6.1 pyppmd-1.1.0 pyzstd-0.15.9 rarfile-4.1 retrying-1.3.4 scikit-image-0.22.0 sse-starlette-0.10.3 sseclient-py-1.8.0 strawberry-graphql-0.138.1 taskgroup-0.0.0a4 tifffile-2023.9.26 universal-analytics-python3-1.1.1 voxel51-eta-0.12.0 xmltodict-0.13.0
# Successfully installed pytube-15.0.0
import fiftyone as fo
import fiftyone.utils.youtube as fouy

download_dir = "viddownloads"
if False:
    urls = ["https://www.youtube.com/watch?v=cwjs1WAG9CM"]

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

# Visualize in the App
#session = fo.launch_app(frame_view)
session = fo.launch_app(unique_view)