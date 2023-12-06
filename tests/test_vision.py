from src.vision.utils_vision import png_to_base64


def test_llava_client():
    img_str = png_to_base64("models/wizard.jpg")

    from gradio_client import Client
    client = Client('http://192.168.1.46:7861', serialize=False)
    client.predict(api_name='/demo_load')
    #prompt = "According to the image, describe the image in full details with a well-structured response."
    prompt = "Describe the image"

    #test_file_local, test_file_server = client.predict(file_to_upload, api_name='/upload_api')

    image_process_mode = "Default"
    include_image = False
    res1 = client.predict(prompt, img_str, image_process_mode, include_image, api_name='/textbox_api_btn')

    model_selector, temperature, top_p, max_output_tokens = 'llava-v1.5-13b', 0.2, 0.7, 512
    res = client.predict(model_selector, temperature, top_p, max_output_tokens, include_image, api_name='/textbox_api_submit')
    res = res[-1][1]
    print(res)

    model_selector, temperature, top_p, max_output_tokens = 'Nous-Hermes-2-Vision', 0.2, 0.7, 512
    res = client.predict(model_selector, temperature, top_p, max_output_tokens, include_image, api_name='/textbox_api_submit')
    res = res[-1][1]
    print(res)
