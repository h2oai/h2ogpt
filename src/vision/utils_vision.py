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


def get_llava_response(file, llava_model, prompt="Describe the image", image_model='llava-v1.5-13b', temperature=0.2,
                       top_p=0.7, max_new_tokens=512):
    # prompt = "According to the image, describe the image in full details with a well-structured response."

    img_str = png_to_base64(file)

    from gradio_client import Client
    client = Client(llava_model, serialize=False)
    client.predict(api_name='/demo_load')

    # test_file_local, test_file_server = client.predict(file_to_upload, api_name='/upload_api')

    image_process_mode = "Default"
    include_image = False
    res1 = client.predict(prompt, img_str, image_process_mode, include_image, api_name='/textbox_api_btn')

    model_selector, temperature, top_p, max_output_tokens = image_model, temperature, top_p, max_new_tokens
    res = client.predict(model_selector, temperature, top_p, max_output_tokens, include_image,
                         api_name='/textbox_api_submit')
    res = res[-1][-1]
    return res
