import os
import time
from tests.utils import wrap_test_forked


@wrap_test_forked
def test_llava_client2():
    file = "models/wizard.jpg"
    llava_model = os.getenv('H2OGPT_LLAVA_MODEL', 'http://192.168.1.46:7861')
    from src.vision.utils_vision import get_llava_response
    res, llava_prompt = get_llava_response(file, llava_model, allow_prompt_auto=True)
    print(res)
    assert 'pumpkins' in res or 'glowing' in res


@wrap_test_forked
def test_llava_client_stream():
    file = "models/wizard.jpg"
    llava_model = os.getenv('H2OGPT_LLAVA_MODEL', 'http://192.168.1.46:7861')
    from src.vision.utils_vision import get_llava_stream
    text = ''
    for res in get_llava_stream(file, llava_model, allow_prompt_auto=True):
        text = res
    print(text)

    assert 'The image features' in text or 'The image is an illustration' in text


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


@wrap_test_forked
def test_fastfood():
    from src.image_utils import align_image
    assert os.path.isfile(align_image("tests/fastfood.jpg"))
    # can't find box for receipt
    assert align_image("tests/receipt.jpg") == "tests/receipt.jpg"
    assert os.path.isfile(align_image("tests/rotate-ex2.png"))

    from src.image_utils import correct_rotation
    assert os.path.isfile(correct_rotation("tests/fastfood.jpg"))
    assert os.path.isfile(correct_rotation("tests/receipt.jpg"))
    assert os.path.isfile(correct_rotation("tests/rotate-ex2.png"))

    # new
    assert align_image("tests/revenue.png") == "tests/revenue.png"
    assert align_image("tests/dental.png") == "tests/dental.png"
    assert align_image("tests/jon.png") == "tests/jon.png"

    assert os.path.isfile(correct_rotation("tests/revenue.png"))
    assert os.path.isfile(correct_rotation("tests/dental.png"))
    assert os.path.isfile(correct_rotation("tests/jon.png"))
