import base64
from io import BytesIO


def png_to_base64(image_file):
    assert image_file.lower().endswith('jpg') or image_file.lower().endswith('jpeg')
    from PIL import Image

    image = Image.open(image_file)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    img_str = str(bytes("data:image/jpeg;base64,", encoding='utf-8') + img_str)

    return img_str
