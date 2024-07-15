import base64
import functools
import os
import tempfile
import time
import types
import uuid
from functools import partial
from io import BytesIO
import numpy as np
from PIL.Image import Resampling

from gradio_utils.grclient import check_job
from src.enums import valid_imagegen_models, valid_imagechange_models, valid_imagestyle_models, docs_joiner_default, \
    llava16_model_max_length, llava16_image_tokens, llava16_image_fudge
from src.image_utils import fix_image_file
from src.utils import is_gradio_version4, get_docs_tokens, get_limited_text, makedirs, call_subprocess_onetask, \
    have_fiftyone, sanitize_filename

IMAGE_EXTENSIONS = {'.png': 'PNG', '.apng': 'PNG', '.blp': 'BLP', '.bmp': 'BMP', '.dib': 'DIB', '.bufr': 'BUFR',
                    '.cur': 'CUR', '.pcx': 'PCX', '.dcx': 'DCX', '.dds': 'DDS', '.ps': 'EPS', '.eps': 'EPS',
                    '.fit': 'FITS', '.fits': 'FITS', '.fli': 'FLI', '.flc': 'FLI', '.fpx': 'FPX', '.ftc': 'FTEX',
                    '.ftu': 'FTEX', '.gbr': 'GBR', '.gif': 'GIF', '.grib': 'GRIB', '.h5': 'HDF5', '.hdf': 'HDF5',
                    '.jp2': 'JPEG2000', '.j2k': 'JPEG2000', '.jpc': 'JPEG2000', '.jpf': 'JPEG2000', '.jpx': 'JPEG2000',
                    '.j2c': 'JPEG2000', '.icns': 'ICNS', '.ico': 'ICO', '.im': 'IM', '.iim': 'IPTC', '.jfif': 'JPEG',
                    '.jpe': 'JPEG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.tif': 'TIFF', '.tiff': 'TIFF', '.mic': 'MIC',
                    '.mpg': 'MPEG', '.mpeg': 'MPEG', '.mpo': 'MPO', '.msp': 'MSP', '.palm': 'PALM', '.pcd': 'PCD',
                    '.pdf': 'PDF', '.pxr': 'PIXAR', '.pbm': 'PPM', '.pgm': 'PPM', '.ppm': 'PPM', '.pnm': 'PPM',
                    '.psd': 'PSD', '.qoi': 'QOI', '.bw': 'SGI', '.rgb': 'SGI', '.rgba': 'SGI', '.sgi': 'SGI',
                    '.ras': 'SUN', '.tga': 'TGA', '.icb': 'TGA', '.vda': 'TGA', '.vst': 'TGA', '.webp': 'WEBP',
                    '.wmf': 'WMF', '.emf': 'WMF', '.xbm': 'XBM', '.xpm': 'XPM'}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}


def is_animated_gif(file_path):
    if not file_path.endswith('.gif'):
        return False
    from PIL import Image, UnidentifiedImageError
    try:
        gif = Image.open(file_path)
    except (FileNotFoundError, UnidentifiedImageError):
        return False
    try:
        gif.seek(1)
    except EOFError:
        return False
    else:
        return True


def gif_to_mp4(gif_path):
    from moviepy.editor import VideoFileClip
    """
    Convert an animated GIF to an MP4 video.

    :param gif_path: Path to the input GIF file.
    :param mp4_path: Path to the output MP4 file.
    """
    clip = VideoFileClip(gif_path)
    mp4_path = gif_path.replace('.gif', '.mp4')
    clip.write_videofile(mp4_path, codec='libx264')
    return mp4_path


def is_video_file(file_path):
    """
    Determine if the file is a video by checking its extension, frame count, and frame rate.

    :param file_path: Path to the file.
    :return: True if the file is a video, False otherwise.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in VIDEO_EXTENSIONS:
        return False

    import cv2
    video = cv2.VideoCapture(file_path)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    video.release()

    # A valid video should have more than 0 frames and a positive frame rate
    return frame_count >= 1 and frame_rate > 0


def img_to_base64(image_file, resolution=None, output_format=None, str_bytes=True):
    # assert image_file.lower().endswith('jpg') or image_file.lower().endswith('jpeg')
    from PIL import Image

    from pathlib import Path
    ext = Path(image_file).suffix
    iformat = IMAGE_EXTENSIONS.get(ext)
    assert iformat is not None, "Invalid file extension %s for file %s" % (ext, image_file)

    image = Image.open(image_file)

    if resolution:
        image = image.resize(resolution, resample=Resampling.BICUBIC)

    if output_format:
        oformat = output_format.upper()
    elif iformat not in ['JPEG', 'PNG']:
        # use jpeg by default if nothing set, so most general format allowed
        oformat = 'JPEG'
    else:
        oformat = iformat

    buffered = BytesIO()
    image.save(buffered, format=oformat)
    img_str = base64.b64encode(buffered.getvalue())

    # FIXME: unsure about below
    if str_bytes:
        img_str = str(bytes("data:image/%s;base64," % oformat.lower(), encoding='utf-8') + img_str)
    else:
        img_str = f"data:image/{oformat.lower()};base64,{img_str.decode('utf-8')}"

    return img_str


def base64_to_img(img_str, output_path):
    """
    Convert a base64 string to an image or video file.

    :param img_str: The base64 encoded string with the image or video data.
    :param output_path: The path (without extension) where the output file will be saved.
    :return: The path to the saved file.
    """
    if img_str.startswith("b'"):
        # check if was a string of bytes joined like when str_bytes=True in above function
        img_str = img_str[2:-1]  # This removes the first b' and the last '

    # Split the string on "," to separate the metadata from the base64 data
    meta, base64_data = img_str.split(",", 1)
    # Extract the format from the metadata
    img_format = meta.split(';')[0].split('/')[-1]
    # Decode the base64 string to bytes
    img_bytes = base64.b64decode(base64_data)
    # Create output file path with the correct format extension
    output_file = f"{output_path}.{img_format}"
    # Write the bytes to a file
    with open(output_file, "wb") as f:
        f.write(img_bytes)
    print(f"Image saved to {output_file} with format {img_format}")
    return output_file


def video_to_base64frames(video_path):
    import cv2
    video = cv2.VideoCapture(video_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")
    return base64Frames


@functools.lru_cache(maxsize=10000, typed=False)
def video_to_frames(video_path, output_dir, resolution=None, image_format="jpg", video_frame_period=None,
                    extract_frames=None,
                    verbose=False):
    import cv2
    """
    Convert video to frames, save them as image files in the specified format, and return the list of file names.

    :param video_path: Path to the input video file.
    :param output_dir: Directory where the output frames will be saved.
    :param resolution: Tuple specifying the desired resolution (width, height) or None to keep the original resolution.
    :param image_format: String specifying the desired image format (e.g., "jpg", "png").
    :param video_frame_period: How often to sample frames from the video. If None, every 20th frame is saved.
      e.g. if pass non-real-time video, can set to 1 to save all frames, to mimic passing actual frames separately otherwise
    :param extract_frames: Number of frames to extract from the video. If None, all frames are saved.
    :param verbose: Boolean to control whether to print progress messages.
    :return: List of file names for the saved frames.

    Example usage:
    file_names = video_to_frames("input_video.mp4", "output_frames", resolution=(640, 480), image_format="png", verbose=True)
    print(file_names)
    """
    if output_dir is None:
        output_dir = os.path.join(tempfile.gettempdir(), 'image_path_%s' % sanitize_filename(video_path))

    enable_fiftyone = True  # optimal against issues if using function server
    if enable_fiftyone and \
            have_fiftyone and \
            (video_frame_period is not None and video_frame_period < 1 or not os.path.isfile(video_path)):
        # handles either automatic period or urls
        from src.vision.extract_movie import extract_unique_frames
        args = ()
        urls = [video_path] if not os.path.isfile(video_path) else None
        file = video_path if os.path.isfile(video_path) else None
        kwargs = {'urls': urls, 'file': file, 'download_dir': None, 'export_dir': output_dir,
                  'extract_frames': extract_frames}
        # fifty one is complex program and leaves around processes
        if False:  # NOTE: Assumes using function server to handle isolation if want production grade behavior
            func_new = partial(call_subprocess_onetask, extract_unique_frames, args, kwargs)
        else:
            func_new = functools.partial(extract_unique_frames, *args, **kwargs)
        export_dir = func_new()
        return [os.path.join(export_dir, x) for x in os.listdir(export_dir)]

    if video_frame_period and video_frame_period < 1:
        video_frame_period = None
    if video_frame_period in [None, 0]:
        # e.g. if no fiftyone and so can't do 0 case, then assume ok to do period based
        total_frames = count_frames(video_path)
        extract_frames = min(20, extract_frames or 20)  # no more than 20 frames total for now
        video_frame_period = total_frames // extract_frames

    video = cv2.VideoCapture(video_path)
    makedirs(output_dir)

    image_format = image_format or '.jpg'

    frame_count = 0
    file_names = []
    while True:
        success, frame = video.read()
        if not success:
            break

        # keep first frame, then keep a frame every video_frame_resolution frames
        if frame_count % video_frame_period != 0:
            frame_count += 1
            continue
        if resolution:
            frame = cv2.resize(frame, resolution)

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.{image_format}")
        cv2.imwrite(frame_filename, frame)
        file_names.append(frame_filename)
        frame_count += 1
    video.release()

    if verbose:
        print(f"{frame_count} frames saved to {output_dir}.")

    return file_names


def count_frames(video_path):
    import cv2
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        print("Error: Could not open video.")
        return -1

    # Get the total number of frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release the video capture object
    video.release()

    return total_frames


def process_file_list(file_list, output_dir, resolution=None, image_format="jpg",
                      rotate_align_resize_image=True,
                      video_frame_period=None,
                      extract_frames=None,
                      verbose=False):
    # FIXME: resolution is not used unless video, could use for every case, but resolution is set later when byte encoding for LLMs
    """
    Process a list of files, converting any videos to frames and updating the list to only contain image files.

    :param file_list: List of file paths to be processed.
    :param output_dir: Directory where the output frames will be saved.
    :param resolution: Tuple specifying the desired resolution (width, height) or None to keep the original resolution.
      Does not affect images as inputs, handled elsewhere when converting to base64 for LLM
    :param image_format: String specifying the desired image format (e.g., "jpg", "png").
    :param rotate_align_resize_image:  Whether to apply rotation, alignment, resize before giving to LLM
    :param video_frame_period: Period to save frames, if <1 then automatic
    :param extract_frames: how many frames to extract if automatic period mode
    :param verbose: Boolean to control whether to print progress messages.
    :return: Updated list of file names containing only image files.
    """
    if file_list is None:
        file_list = []
    if image_format is None:
        image_format = 'jpg'

    image_files = []

    for file in file_list:
        # i.e. if not file, then maybe youtube url
        is_maybe_video = os.path.isfile(file) and is_video_file(file) or not os.path.isfile(file) or is_animated_gif(
            file)
        if is_animated_gif(file):
            # FIXME: could convert gif -> mp4 with gif_to_mp4(gif_path)()
            # fiftyone can't handle animated gifs
            extract_frames = None
            if video_frame_period is not None and video_frame_period < 1:
                video_frame_period = None

        if is_maybe_video:
            # If it's a valid video, extract frames
            if verbose:
                print(f"Processing video file: {file}")
            # output_dir is None means only use file for location
            frame_files = video_to_frames(file, None, resolution, image_format, video_frame_period,
                                          extract_frames, verbose)
            image_files.extend(frame_files)
        else:
            # If it's not a valid video, add it to the image file list
            if rotate_align_resize_image:
                file_fixed = fix_image_file(file, do_align=True, do_rotate=True, do_pad=False, relaxed_resize=True)
            else:
                file_fixed = file
            image_files.append(file_fixed)

    return image_files


def fix_llava_prompt(file,
                     prompt=None,
                     allow_prompt_auto=True,
                     ):
    if prompt in ['auto', None] and allow_prompt_auto:
        prompt = "Describe the image and what does the image say?"
        # prompt = "According to the image, describe the image in full details with a well-structured response."
        if file in ['', None]:
            # let model handle if no prompt and no file
            prompt = ''
    # allow prompt = '', will describe image by default
    if prompt is None:
        if os.environ.get('HARD_ASSERTS'):
            raise ValueError('prompt is None')
        else:
            prompt = ''
    return prompt


def llava_prep(file_list,
               llava_model,
               image_model='llava-v1.6-vicuna-13b',
               client=None):
    assert client is not None or len(file_list) == 1

    file_list_new = []
    image_model_list_new = []
    for file in file_list:
        image_model_new, client, file_new = _llava_prep(file,
                                                        llava_model,
                                                        image_model=image_model,
                                                        client=client)
        file_list_new.append(file_new)
        image_model_list_new.append(image_model_new)
    assert len(image_model_list_new) >= 1
    assert len(file_list_new) >= 1
    return image_model_list_new[0], client, file_list_new


def _llava_prep(file,
                llava_model,
                image_model='llava-v1.6-vicuna-13b',
                client=None):
    prefix = ''
    if llava_model.startswith('http://'):
        prefix = 'http://'
    if llava_model.startswith('https://'):
        prefix = 'https://'
    llava_model = llava_model[len(prefix):]

    llava_model_split = llava_model.split(':')
    assert len(llava_model_split) >= 2
    # FIXME: Allow choose model in UI
    if len(llava_model_split) >= 2:
        pass
        # assume default model is ok
        # llava_ip = llava_model_split[0]
        # llava_port = llava_model_split[1]
    if len(llava_model_split) >= 3:
        image_model = llava_model_split[2]
        llava_model = ':'.join(llava_model_split[:2])
    # add back prefix
    llava_model = prefix + llava_model

    if client is None:
        from gradio_utils.grclient import GradioClient
        client = GradioClient(llava_model, check_hash=False, serialize=is_gradio_version4)
        client.setup()

    if not is_gradio_version4 and file and os.path.isfile(file):
        file = img_to_base64(file)

    assert image_model, "No image model specified"

    if isinstance(file, np.ndarray):
        from PIL import Image
        im = Image.fromarray(file)
        file = "%s.jpeg" % str(uuid.uuid4())
        im.save(file)

    return image_model, client, file


server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"


def get_prompt_with_texts(texts, prompt, max_new_tokens, min_max_new_tokens, tokenizer):
    if tokenizer is None:
        raise RuntimeError("Not setup for multi-image without tokenizer")
        # from transformers import AutoTokenizer
        # tokenizer = AutoTokenizer.from_pretrained(base_model)
    if hasattr(tokenizer, 'model_max_length'):
        model_max_length = tokenizer.model_max_length
    else:
        model_max_length = llava16_model_max_length

    user_part = '\n\nReduce the above information into single correct answer to the following question: ' + prompt
    user_part_tokens = len(tokenizer.encode(user_part))

    text_context_list = ['Answer #%s:\n\n%s' % (ii, text) for ii, text in enumerate(texts)]

    # see if too many tokens
    text_tokens_trial = len(tokenizer.encode(docs_joiner_default.join(text_context_list)))
    if user_part_tokens + text_tokens_trial + max_new_tokens >= model_max_length:
        max_new_tokens = min_max_new_tokens
    fudge = llava16_image_fudge
    max_input_tokens = model_max_length - max_new_tokens - fudge  # fudge for extra chars

    top_k_docs, one_doc_size, num_doc_tokens = \
        get_docs_tokens(tokenizer, text_context_list=text_context_list, max_input_tokens=max_input_tokens)
    text_context_list_cut = text_context_list[:top_k_docs]
    texts_joined = docs_joiner_default.join(text_context_list_cut)

    prompt_with_texts = '\n"""\n' + texts_joined + '\n"""\n'
    prompt_with_texts += user_part

    return prompt_with_texts.replace('image', 'document').replace('Image', 'Document')


def get_llava_response(file=None,
                       llava_model=None,
                       prompt=None,
                       chat_conversation=[],
                       allow_prompt_auto=False,
                       image_model='llava-v1.6-vicuna-13b', temperature=0.2,
                       top_p=0.7, max_new_tokens=512,
                       min_max_new_tokens=512,
                       tokenizer=None,
                       image_process_mode="Default",
                       include_image=False,
                       client=None,
                       max_time=None,
                       force_stream=True,
                       verbose=False,
                       ):
    max_new_tokens = min(max_new_tokens, 1024)  # for hard_cutoff to be easy to know

    kwargs = locals().copy()

    force_stream |= isinstance(file, list) and len(file) > 1
    if isinstance(file, str):
        file_list = [file]
    elif isinstance(file, list):
        file_list = file
        if len(file_list) == 0:
            file_list = [None]
    else:
        file_list = [None]

    if force_stream:
        text = ''
        for res in get_llava_stream(**kwargs):
            text = res
        return text, prompt

    image_model = os.path.basename(image_model)  # in case passed HF link
    prompt = fix_llava_prompt(file_list, prompt, allow_prompt_auto=allow_prompt_auto)
    max_new_tokens1 = max_new_tokens if len(file_list) <= 4 else min(max_new_tokens, min_max_new_tokens)
    if tokenizer:
        model_max_length = tokenizer.model_max_length
    else:
        model_max_length = llava16_model_max_length
    image_tokens = llava16_image_tokens if len(file_list) >= 1 and file_list[0] is not None else 0
    fudge = llava16_image_fudge
    hard_limit_tokens = model_max_length - max_new_tokens1 - fudge - image_tokens
    prompt = get_limited_text(hard_limit_tokens, prompt, tokenizer, verbose=False)

    image_model, client, file_list = \
        llava_prep(file_list, llava_model,
                   image_model=image_model,
                   client=client)

    reses = []
    for file in file_list:
        res = client.predict(prompt,
                             chat_conversation if len(file_list) == 1 else [],
                             file,
                             image_process_mode,
                             include_image,
                             image_model,
                             temperature,
                             top_p,
                             max_new_tokens1,
                             api_name='/textbox_api_submit')
        reses.append(res)

    if len(reses) > 1:
        reses = [x for x in reses if server_error_msg not in x]
        prompt_with_texts = get_prompt_with_texts(reses, prompt, max_new_tokens, min_max_new_tokens, tokenizer)
        res = client.predict(prompt_with_texts,
                             chat_conversation,
                             None,
                             image_process_mode,
                             include_image,
                             image_model,
                             temperature,
                             top_p,
                             max_new_tokens,
                             api_name='/textbox_api_submit')
    else:
        res = reses[0]

    return res, prompt


def get_llava_stream(file, llava_model,
                     prompt=None,
                     chat_conversation=[],
                     allow_prompt_auto=False,
                     image_model='llava-v1.6-vicuna-13b', temperature=0.2,
                     top_p=0.7, max_new_tokens=512,
                     min_max_new_tokens=512,
                     tokenizer=None,
                     image_process_mode="Default",
                     include_image=False,
                     client=None,
                     verbose_level=0,
                     max_time=None,
                     force_stream=True,  # dummy arg
                     verbose=False,
                     ):
    max_new_tokens = min(max_new_tokens, 1024)  # for hard_cutoff to be easy to know

    if isinstance(file, str):
        file_list = [file]
    elif isinstance(file, list):
        file_list = file
        if len(file_list) == 0:
            file_list = [None]
    else:
        file_list = [None]

    image_model = os.path.basename(image_model)  # in case passed HF link
    prompt = fix_llava_prompt(file_list, prompt, allow_prompt_auto=allow_prompt_auto)
    max_new_tokens1 = max_new_tokens if len(file_list) <= 4 else min(max_new_tokens, min_max_new_tokens)
    if tokenizer:
        model_max_length = tokenizer.model_max_length
    else:
        model_max_length = llava16_model_max_length
    image_tokens = llava16_image_tokens if len(file_list) >= 1 and file_list[0] is not None else 0
    fudge = llava16_image_fudge
    hard_limit_tokens = model_max_length - max_new_tokens1 - fudge - image_tokens
    prompt = get_limited_text(hard_limit_tokens, prompt, tokenizer)

    image_model, client, file_list = \
        llava_prep(file_list, llava_model,
                   image_model=image_model,
                   client=client)

    jobs = []
    for file in file_list:
        job = client.submit(prompt,
                            chat_conversation,
                            file,
                            image_process_mode,
                            include_image,
                            image_model,
                            temperature,
                            top_p,
                            max_new_tokens1,
                            api_name='/textbox_api_submit')
        jobs.append(job)

    t0 = time.time()
    job_outputs_nums = [0] * len(jobs)
    texts = [''] * len(jobs)
    done_all = False
    reses = [''] * len(jobs)
    while True:
        for ji, job in enumerate(jobs):
            if verbose_level == 2:
                print("Inside: %s" % llava_model, time.time() - t0, flush=True)
            e = check_job(job, timeout=0, raise_exception=False)
            if e is not None:
                continue
            if max_time is not None and time.time() - t0 > max_time:
                done_all = True
                break
            outputs_list = job.outputs().copy()
            job_outputs_num_new = len(outputs_list[job_outputs_nums[ji]:])
            for num in range(job_outputs_num_new):
                reses[ji] = outputs_list[job_outputs_nums[ji] + num]
                if verbose_level == 2:
                    print('Stream %d: %s' % (num, reses[ji]), flush=True)
                elif verbose_level == 1:
                    print('Stream %d' % (job_outputs_nums[ji] + num), flush=True)
                if reses[ji]:
                    texts[ji] = reses[ji]
                    if len(jobs) == 1:
                        yield texts[ji]
            job_outputs_nums[ji] += job_outputs_num_new
            time.sleep(0.005)
        if done_all or all([job.done() for job in jobs]):
            break

    for ji, job in enumerate(jobs):
        e = check_job(job, timeout=0, raise_exception=False)
        if e is not None:
            continue
        outputs_list = job.outputs().copy()
        job_outputs_num_new = len(outputs_list[job_outputs_nums[ji]:])
        for num in range(job_outputs_num_new):
            reses[ji] = outputs_list[job_outputs_nums[ji] + num]
            if verbose_level == 2:
                print('Final Stream %d: %s' % (num, reses[ji]), flush=True)
            elif verbose_level == 1:
                print('Final Stream %d' % (job_outputs_nums[ji] + num), flush=True)
            if reses[ji]:
                texts[ji] = reses[ji]
                if len(jobs) == 1:
                    yield texts[ji]
        job_outputs_nums[ji] += job_outputs_num_new
        if verbose_level == 1:
            print("total job_outputs_num=%d" % job_outputs_nums[ji], flush=True)

    if len(jobs) > 1:
        # recurse without image(s)
        ntexts_before = len(texts)
        texts = [x for x in texts if server_error_msg not in x]
        ntexts_after = len(texts)
        if ntexts_after != ntexts_before:
            print("texts: %s -> %s" % (ntexts_before, ntexts_after))
        prompt_with_texts = get_prompt_with_texts(texts, prompt, max_new_tokens, min_max_new_tokens, tokenizer)
        text = ''
        max_new_tokens = max_new_tokens if len(jobs) > 4 else min(max_new_tokens, min_max_new_tokens)
        for res in get_llava_stream(None,
                                    llava_model,
                                    prompt=prompt_with_texts,
                                    chat_conversation=chat_conversation,
                                    allow_prompt_auto=allow_prompt_auto,
                                    image_model=image_model,
                                    temperature=temperature,
                                    top_p=top_p,
                                    # avoid long outputs
                                    max_new_tokens=max_new_tokens,
                                    min_max_new_tokens=min_max_new_tokens,
                                    tokenizer=tokenizer,
                                    image_process_mode=image_process_mode,
                                    include_image=include_image,
                                    client=client,
                                    verbose_level=verbose_level,
                                    max_time=max_time,
                                    force_stream=force_stream,  # dummy arg
                                    verbose=verbose,
                                    ):
            text = res
            yield text
    else:
        assert len(texts) == 1
        text = texts[0]

    return text


def get_image_model_dict(enable_image,
                         image_models,
                         image_gpu_ids,
                         ):
    image_dict = {}
    if not enable_image:
        return image_dict

    if image_gpu_ids is None:
        image_gpu_ids = ['auto'] * len(image_models)
    if not image_gpu_ids:
        image_gpu_ids = ['auto'] * len(image_models)

    for image_model_name in valid_imagegen_models + valid_imagechange_models + valid_imagestyle_models:
        if image_model_name in image_models:
            imagegen_index = image_models.index(image_model_name)
            if image_model_name == 'sdxl_turbo':
                from src.vision.sdxl import get_pipe_make_image, make_image
            elif image_model_name == 'playv2':
                from src.vision.playv2 import get_pipe_make_image, make_image
            elif image_model_name == 'sdxl':
                from src.vision.stable_diffusion_xl import get_pipe_make_image, make_image
            elif image_model_name == 'sd3':
                from src.vision.stable_diffusion_xl import get_pipe_make_image, make_image
                get_pipe_make_image = functools.partial(get_pipe_make_image,
                                                        base_model='stabilityai/stable-diffusion-3-medium-diffusers',
                                                        refiner_model=None)
                make_image = functools.partial(make_image,
                                               base_model='stabilityai/stable-diffusion-3-medium-diffusers',
                                               refiner_model=None)
            elif image_model_name == 'sdxl_change':
                from src.vision.sdxl import get_pipe_change_image as get_pipe_make_image, change_image
                make_image = change_image
            # FIXME: style
            else:
                raise ValueError("Invalid image_model_name=%s" % image_model_name)
            pipe = get_pipe_make_image(gpu_id=image_gpu_ids[imagegen_index])
            image_dict[image_model_name] = dict(pipe=pipe, make_image=make_image)
    return image_dict


def pdf_to_base64_pngs(pdf_path, quality=75, max_size=(1024, 1024), ext='png', pages=None):
    """
    Define the function to convert a pdf slide deck to a list of images. Note that we need to ensure we resize images to keep them within Claude's size limits.
    """
    # https://github.com/anthropics/anthropic-cookbook/blob/main/multimodal/reading_charts_graphs_powerpoints.ipynb
    from PIL import Image
    import io
    import fitz
    import tempfile

    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Iterate through each page of the PDF
    images = []
    if pages is None:
        pages = list(range(doc.page_count))
    else:
        assert isinstance(pages, (list, tuple, types.GeneratorType))

    for page_num in pages:
        # Load the page
        page = doc.load_page(page_num)

        # Render the page as a PNG image
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))

        # Save the PNG image
        output_path = f"{tempfile.mkdtemp()}/page_{page_num + 1}.{ext}"
        pix.save(output_path)
        images.append(output_path)
    # Close the PDF document
    doc.close()

    if ext == 'png':
        iformat = 'PNG'
    elif ext in ['jpeg', 'jpg']:
        iformat = 'JPEG'
    else:
        raise ValueError("No such ext=%s" % ext)

    images = [Image.open(image) for image in images]
    base64_encoded_pngs = []
    for image in images:
        # Resize the image if it exceeds the maximum size
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        image_data = io.BytesIO()
        image.save(image_data, format=iformat, optimize=True, quality=quality)
        image_data.seek(0)
        base64_encoded = base64.b64encode(image_data.getvalue()).decode('utf-8')
        base64_encoded_pngs.append(base64_encoded)

    return base64_encoded_pngs
