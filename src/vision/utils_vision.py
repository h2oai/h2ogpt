import base64
import os
import time
import types
import uuid
from io import BytesIO
import numpy as np

from src.enums import valid_imagegen_models, valid_imagechange_models, valid_imagestyle_models


def img_to_base64(image_file, str_bytes=True):
    # assert image_file.lower().endswith('jpg') or image_file.lower().endswith('jpeg')
    from PIL import Image

    EXTENSIONS = {'.png': 'PNG', '.apng': 'PNG', '.blp': 'BLP', '.bmp': 'BMP', '.dib': 'DIB', '.bufr': 'BUFR',
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

    from pathlib import Path
    ext = Path(image_file).suffix
    if ext in EXTENSIONS:
        iformat = EXTENSIONS[ext]
    else:
        raise ValueError("Invalid file extension %s for file %s" % (ext, image_file))

    image = Image.open(image_file)
    buffered = BytesIO()
    image.save(buffered, format=iformat)
    img_str = base64.b64encode(buffered.getvalue())
    # FIXME: unsure about below
    if str_bytes:
        img_str = str(bytes("data:image/%s;base64," % iformat.lower(), encoding='utf-8') + img_str)
    else:
        img_str = f"data:image/{iformat.lower()};base64,{img_str.decode('utf-8')}"

    return img_str


def base64_to_img(img_str, output_path):
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


def llava_prep(file,
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
        client = GradioClient(llava_model, check_hash=False, serialize=True)
        client.setup()

    assert image_model, "No image model specified"

    if isinstance(file, np.ndarray):
        from PIL import Image
        im = Image.fromarray(file)
        file = "%s.jpeg" % str(uuid.uuid4())
        im.save(file)

    return image_model, client, file


def get_llava_response(file=None,
                       llava_model=None,
                       prompt=None,
                       chat_conversation=[],
                       allow_prompt_auto=False,
                       image_model='llava-v1.6-vicuna-13b', temperature=0.2,
                       top_p=0.7, max_new_tokens=512,
                       image_process_mode="Default",
                       include_image=False,
                       client=None,
                       max_time=None,
                       force_stream=True,
                       ):
    if isinstance(file, list) and len(file) >= 1:
        # llava only handles first image if list of images
        file = file[0]

    kwargs = locals()

    if force_stream:
        text = ''
        for res in get_llava_stream(**kwargs):
            text = res
        return text, prompt

    prompt = fix_llava_prompt(file, prompt, allow_prompt_auto=allow_prompt_auto)

    image_model, client, file = \
        llava_prep(file, llava_model,
                   image_model=image_model,
                   client=client)

    res = client.predict(prompt,
                         chat_conversation,
                         file,
                         image_process_mode,
                         include_image,
                         image_model,
                         temperature,
                         top_p,
                         max_new_tokens,
                         api_name='/textbox_api_submit')
    res = res[-1][-1]
    return res, prompt


def get_llava_stream(file, llava_model,
                     prompt=None,
                     chat_conversation=[],
                     allow_prompt_auto=False,
                     image_model='llava-v1.6-vicuna-13b', temperature=0.2,
                     top_p=0.7, max_new_tokens=512,
                     image_process_mode="Default",
                     include_image=False,
                     client=None,
                     verbose_level=0,
                     max_time=None,
                     force_stream=True,  # dummy arg
                     ):
    if isinstance(file, list) and len(file) >= 1:
        # llava only handles first image if list of images
        file = file[0]

    image_model = os.path.basename(image_model)  # in case passed HF link
    prompt = fix_llava_prompt(file, prompt, allow_prompt_auto=allow_prompt_auto)

    image_model, client, file = \
        llava_prep(file, llava_model,
                   image_model=image_model,
                   client=client)

    job = client.submit(prompt,
                        chat_conversation,
                        file,
                        image_process_mode,
                        include_image,
                        image_model,
                        temperature,
                        top_p,
                        max_new_tokens,
                        api_name='/textbox_api_submit')

    t0 = time.time()
    job_outputs_num = 0
    text = ''
    while not job.done():
        if verbose_level == 2:
            print("Inside: %s" % llava_model, time.time() - t0, flush=True)
        if max_time is not None and time.time() - t0 > max_time:
            return text
        outputs_list = job.outputs().copy()
        job_outputs_num_new = len(outputs_list[job_outputs_num:])
        for num in range(job_outputs_num_new):
            res = outputs_list[job_outputs_num + num]
            if verbose_level == 2:
                print('Stream %d: %s' % (num, res), flush=True)
            elif verbose_level == 1:
                print('Stream %d' % (job_outputs_num + num), flush=True)
            if res and len(res[0]) > 0:
                text = res[-1][-1]
                yield text
        job_outputs_num += job_outputs_num_new
        time.sleep(0.01)

    outputs_list = job.outputs().copy()
    job_outputs_num_new = len(outputs_list[job_outputs_num:])
    for num in range(job_outputs_num_new):
        if max_time is not None and time.time() - t0 > max_time:
            return text
        res = outputs_list[job_outputs_num + num]
        if verbose_level == 2:
            print('Final Stream %d: %s' % (num, res), flush=True)
        elif verbose_level == 1:
            print('Final Stream %d' % (job_outputs_num + num), flush=True)
        if res and len(res[0]) > 0:
            text = res[-1][-1]
            yield text
    job_outputs_num += job_outputs_num_new
    if verbose_level == 1:
        print("total job_outputs_num=%d" % job_outputs_num, flush=True)
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

    for image_model_name in valid_imagegen_models + valid_imagechange_models + valid_imagestyle_models:
        if image_model_name in image_models:
            imagegen_index = image_models.index(image_model_name)
            if image_model_name == 'sdxl_turbo':
                from src.vision.sdxl import get_pipe_make_image, make_image
            elif image_model_name == 'playv2':
                from src.vision.playv2 import get_pipe_make_image, make_image
            elif image_model_name == 'sdxl':
                from src.vision.stable_diffusion_xl import get_pipe_make_image, make_image
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
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))

        # Save the PNG image
        output_path = f"{tempfile.mkdtemp()}/page_{page_num+1}.{ext}"
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
