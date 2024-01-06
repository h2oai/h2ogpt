import base64
from io import BytesIO


def png_to_base64(image_file):
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
    img_str = str(bytes("data:image/%s;base64," % iformat.lower(), encoding='utf-8') + img_str)

    return img_str


def get_llava_response(file, llava_model,
                       prompt=None,
                       image_model='llava-v1.5-13b', temperature=0.2,
                       top_p=0.7, max_new_tokens=512):
    if prompt in ['auto', None]:
        prompt = "Describe the image and what does the image say?"
    # prompt = "According to the image, describe the image in full details with a well-structured response."

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

    img_str = png_to_base64(file)

    from gradio_client import Client
    client = Client(llava_model, serialize=False)
    load_res = client.predict(api_name='/demo_load')
    model_options = [x[1] for x in load_res['choices']]
    assert len(model_options), "LLaVa endpoint has no models: %s" % str(load_res)

    # if no default choice or default choice not there, choose first
    if not image_model or image_model not in model_options:
        image_model = model_options[0]

    # test_file_local, test_file_server = client.predict(file_to_upload, api_name='/upload_api')

    image_process_mode = "Default"
    include_image = False
    res1 = client.predict(prompt, img_str, image_process_mode, include_image, api_name='/textbox_api_btn')

    model_selector, temperature, top_p, max_output_tokens = image_model, temperature, top_p, max_new_tokens
    res = client.predict(model_selector, temperature, top_p, max_output_tokens, include_image,
                         api_name='/textbox_api_submit')
    res = res[-1][-1]
    return res, prompt
