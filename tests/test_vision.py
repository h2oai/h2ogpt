import os
import time
from tests.utils import wrap_test_forked


@wrap_test_forked
def test_llava_client():
    file = "models/wizard.jpg"
    llava_model = os.getenv('H2OGPT_LLAVA_MODEL', 'http://192.168.1.46:7861')
    # prompt = "According to the image, describe the image in full details with a well-structured response."
    prompt = "Describe the image"

    from src.vision.utils_vision import png_to_base64
    img_str = png_to_base64(file)

    from gradio_client import Client
    client = Client(llava_model, serialize=False)
    client.predict(api_name='/demo_load')

    # test_file_local, test_file_server = client.predict(file_to_upload, api_name='/upload_api')

    image_process_mode = "Default"
    include_image = False
    res1 = client.predict(prompt, img_str, image_process_mode, include_image, api_name='/textbox_api_btn')

    model_selector, temperature, top_p, max_output_tokens = 'llava-v1.5-13b', 0.2, 0.7, 512
    res = client.predict(model_selector, temperature, top_p, max_output_tokens, include_image,
                         api_name='/textbox_api_submit')
    res = res[-1][-1]
    print(res)
    assert 'pumpkins' in res

    model_selector, temperature, top_p, max_output_tokens = 'Nous-Hermes-2-Vision', 0.2, 0.7, 512
    res = client.predict(model_selector, temperature, top_p, max_output_tokens, include_image,
                         api_name='/textbox_api_submit')
    res = res[-1][-1]
    print(res)
    assert 'headband' in res or 'woman' in res or 'orange' in res


@wrap_test_forked
def test_llava_client2():
    file = "models/wizard.jpg"
    llava_model = os.getenv('H2OGPT_LLAVA_MODEL', 'http://192.168.1.46:7861')
    from src.vision.utils_vision import get_llava_response
    res, llava_prompt = get_llava_response(file, llava_model)
    print(res)
    assert 'pumpkins' in res


@wrap_test_forked
def test_llava_client_stream():
    from src.vision.utils_vision import png_to_base64
    img_str = png_to_base64("models/wizard.jpg")

    from gradio_client import Client
    client = Client(os.getenv('H2OGPT_LLAVA_MODEL', 'http://192.168.1.46:7861'), serialize=False)
    client.predict(api_name='/demo_load')
    # prompt = "According to the image, describe the image in full details with a well-structured response."
    prompt = "Describe the image"

    # test_file_local, test_file_server = client.predict(file_to_upload, api_name='/upload_api')

    image_process_mode = "Default"
    include_image = False
    res1 = client.predict(prompt, img_str, image_process_mode, include_image, api_name='/textbox_api_btn')

    #model_selector, temperature, top_p, max_output_tokens = 'Nous-Hermes-2-Vision', 0.2, 0.7, 512
    model_selector, temperature, top_p, max_output_tokens = 'llava-v1.5-13b', 0.2, 0.7, 512
    job = client.submit(model_selector, temperature, top_p, max_output_tokens, include_image,
                        api_name='/textbox_api_submit')

    job_outputs_num = 0
    while not job.done():
        outputs_list = job.outputs().copy()
        job_outputs_num_new = len(outputs_list[job_outputs_num:])
        for num in range(job_outputs_num_new):
            res = outputs_list[job_outputs_num + num]
            print('Stream %d: %s\n' % (job_outputs_num + num, res[-1][-1]), flush=True)
        job_outputs_num += job_outputs_num_new
        time.sleep(0.01)

    outputs_list = job.outputs().copy()
    job_outputs_num_new = len(outputs_list[job_outputs_num:])
    for num in range(job_outputs_num_new):
        res = outputs_list[job_outputs_num + num]
        print('Final Stream %d: %s\n' % (job_outputs_num + num, res[-1][-1]), flush=True)
    job_outputs_num += job_outputs_num_new
    print("total job_outputs_num=%d" % job_outputs_num, flush=True)

    assert 'The image features' in res[-1][-1]


@wrap_test_forked
def test_make_image():
    from src.vision.sdxl import make_image
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    make_image(prompt, filename="output_p2i.png")


@wrap_test_forked
def test_change_image():
    from src.vision.sdxl import change_image
    init_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    change_image(init_file=init_file,
                 prompt="cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k",
                 filename="output_i2i.png")


@wrap_test_forked
def test_video_extraction():
    urls = ["https://www.youtube.com/shorts/fRkZCriQQNU"]
    from src.vision.extract_movie import extract_unique_frames
    export_dir = extract_unique_frames(urls=urls, download_dir=None)
    image_files = [f for f in os.listdir(export_dir) if os.path.isfile(os.path.join(export_dir, f))]
    assert len(image_files) > 9
    assert image_files[0].endswith('.jpg')
    print(export_dir)
    # feh -rF -D 1000 export_dir


@wrap_test_forked
def test_make_image_playv2():
    from src.vision.playv2 import make_image
    prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
    make_image(prompt, filename="output_p2i_v2.png")
