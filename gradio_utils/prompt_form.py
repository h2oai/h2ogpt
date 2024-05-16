import functools
import os
import math
import csv
import datetime

import filelock
import gradio as gr

from src.utils import is_gradio_version4


def get_chatbot_name(base_model, model_path_llama, inference_server='', prompt_type='', model_label_prefix='', debug=False):
    #have_inference_server = inference_server not in [no_server_str, None, '']
    #if not have_inference_server and prompt_type in [None, '', 'plain']:
    #    label_postfix = '   [Please select prompt_type in Models tab or on CLI for chat models]'
    #else:
    # pass
    label_postfix = ''
    if not debug:
        inference_server = ''
    else:
        inference_server = ' : ' + inference_server
    if base_model == 'llama':
        model_path_llama = os.path.basename(model_path_llama)
        if model_path_llama.endswith('?download=true'):
            model_path_llama = model_path_llama.replace('?download=true', '')
        label = f'{model_label_prefix} [Model: {model_path_llama}{inference_server}]'
    else:
        if base_model == 'mixtral-8x7b-32768':
            base_model = 'groq:mixtral-8x7b-32768'
        label = f'{model_label_prefix} [Model: {base_model}{inference_server}]'
    label += label_postfix
    return label


def get_avatars(base_model, model_path_llama, inference_server=''):
    if base_model == 'llama':
        base_model = model_path_llama
    if inference_server is None:
        inference_server = ''

    model_base = os.getenv('H2OGPT_MODEL_BASE', 'models/')
    human_avatar = "human.jpg"
    if 'h2ogpt-gm'.lower() in base_model.lower():
        bot_avatar = "h2oai.png"
    elif 'llava-' in base_model.lower():
        bot_avatar = "llava.png"
    elif 'mistralai'.lower() in base_model.lower() or \
            'mistral'.lower() in base_model.lower() or \
            'mixtral'.lower() in base_model.lower():
        bot_avatar = "mistralai.png"
    elif '01-ai/Yi-'.lower() in base_model.lower():
        bot_avatar = "yi.svg"
    elif 'wizard' in base_model.lower():
        bot_avatar = "wizard.jpg"
    elif 'openchat' in base_model.lower():
        bot_avatar = "openchat.png"
    elif 'vicuna' in base_model.lower():
        bot_avatar = "vicuna.jpeg"
    elif 'longalpaca' in base_model.lower():
        bot_avatar = "longalpaca.png"
    elif 'llama2-70b-chat' in base_model.lower():
        bot_avatar = "meta.png"
    elif 'llama2-13b-chat' in base_model.lower():
        bot_avatar = "meta.png"
    elif 'llama2-7b-chat' in base_model.lower():
        bot_avatar = "meta.png"
    elif 'llama2' in base_model.lower():
        bot_avatar = "lama2.jpeg"
    elif 'llama-2' in base_model.lower():
        bot_avatar = "lama2.jpeg"
    elif 'llama' in base_model.lower():
        bot_avatar = "lama.jpeg"
    elif 'openai' in base_model.lower() or 'openai' in inference_server.lower():
        bot_avatar = "openai.png"
    elif 'hugging' in base_model.lower():
        bot_avatar = "hf-logo.png"
    elif 'claude' in base_model.lower():
        bot_avatar = "anthropic.jpeg"
    elif 'gemini' in base_model.lower():
        bot_avatar = "google.png"
    else:
        bot_avatar = "h2oai.png"

    bot_avatar = os.path.join(model_base, bot_avatar)
    human_avatar = os.path.join(model_base, human_avatar)

    human_avatar = human_avatar if os.path.isfile(human_avatar) else None
    bot_avatar = bot_avatar if os.path.isfile(bot_avatar) else None
    return human_avatar, bot_avatar


def ratingfn1():
    return 1


def ratingfn2():
    return 2


def ratingfn3():
    return 3


def ratingfn4():
    return 4


def ratingfn5():
    return 5


def submit_review(review_text, text_output, text_output2, *text_outputs1, reviews_file=None, num_model_lock=None,
                  do_info=True):
    if reviews_file is None:
        if do_info:
            gr.Info('No review file')
        return ''

    chatbots = [text_output, text_output2] + list(text_outputs1)
    last_chatbots = [x[-1] for x in chatbots if x]

    now = datetime.datetime.now()
    with filelock.FileLock(reviews_file + '.lock'):
        with open(reviews_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([review_text, *last_chatbots, now])
            if do_info:
                gr.Info('Review submitted!')
    return ''


def make_chatbots(output_label0, output_label0_model2, **kwargs):
    visible_models = kwargs['visible_models']
    all_models = kwargs['all_possible_display_names']
    visible_ratings = kwargs['visible_ratings']
    reviews_file = kwargs['reviews_file'] or 'reviews.csv'

    text_outputs = []
    chat_kwargs = []
    min_width = 250 if kwargs['gradio_size'] in ['small', 'large', 'medium'] else 160
    for model_state_locki, model_state_lock in enumerate(kwargs['model_states']):
        output_label = get_chatbot_name(model_state_lock["base_model"],
                                        model_state_lock['llamacpp_dict']["model_path_llama"],
                                        model_state_lock["inference_server"],
                                        model_state_lock["prompt_type"],
                                        model_label_prefix=kwargs['model_label_prefix'],
                                        debug=bool(os.environ.get('DEBUG_MODEL_LOCK', 0)))
        if kwargs['avatars']:
            avatar_images = get_avatars(model_state_lock["base_model"],
                                        model_state_lock['llamacpp_dict']["model_path_llama"],
                                        model_state_lock["inference_server"])
        else:
            avatar_images = None
        chat_kwargs.append(dict(render_markdown=kwargs.get('render_markdown', True),
                                label=output_label,
                                show_label=kwargs.get('visible_chatbot_label', True),
                                elem_classes='chatsmall',
                                height=kwargs['height'] or 400,
                                min_width=min_width,
                                avatar_images=avatar_images,
                                likeable=True,
                                latex_delimiters=[],
                                show_copy_button=kwargs['show_copy_button'],
                                visible=kwargs['model_lock'] and (visible_models is None or
                                                                  model_state_locki in visible_models or
                                                                  all_models[model_state_locki] in visible_models
                                                                  )))

    # base view on initial visible choice
    if visible_models and kwargs['model_lock_layout_based_upon_initial_visible']:
        len_visible = len(visible_models)
    else:
        len_visible = len(kwargs['model_states'])
    if kwargs['model_lock_columns'] == -1:
        kwargs['model_lock_columns'] = len_visible
    if kwargs['model_lock_columns'] is None:
        kwargs['model_lock_columns'] = 3

    ncols = kwargs['model_lock_columns']
    if kwargs['model_states'] == 0:
        nrows = 0
    else:
        nrows = math.ceil(len_visible / kwargs['model_lock_columns'])

    if kwargs['model_lock_columns'] == 0:
        # not using model_lock
        pass
    elif nrows <= 1:
        with gr.Row():
            for chat_kwargs1, model_state_lock in zip(chat_kwargs, kwargs['model_states']):
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
    elif nrows == kwargs['model_states']:
        with gr.Row():
            for chat_kwargs1, model_state_lock in zip(chat_kwargs, kwargs['model_states']):
                text_outputs.append(gr.Chatbot(**chat_kwargs1))
    elif nrows > 0:
        len_chatbots = len(kwargs['model_states'])
        nrows = math.ceil(len_chatbots / kwargs['model_lock_columns'])
        for nrowi in range(nrows):
            with gr.Row():
                for mii, (chat_kwargs1, model_state_lock) in enumerate(zip(chat_kwargs, kwargs['model_states'])):
                    if mii < nrowi * len_chatbots / nrows or mii >= (1 + nrowi) * len_chatbots / nrows:
                        continue
                    text_outputs.append(gr.Chatbot(**chat_kwargs1))
    if len(kwargs['model_states']) > 0:
        assert len(text_outputs) == len(kwargs['model_states'])

    if kwargs['avatars']:
        avatar_images = get_avatars(kwargs["base_model"], kwargs['llamacpp_dict']["model_path_llama"],
                                    kwargs["inference_server"])
    else:
        avatar_images = None
    no_model_lock_chat_kwargs = dict(render_markdown=kwargs.get('render_markdown', True),
                                     show_label=kwargs.get('visible_chatbot_label', True),
                                     elem_classes='chatsmall',
                                     height=kwargs['height'] or 400,
                                     min_width=min_width,
                                     show_copy_button=kwargs['show_copy_button'],
                                     avatar_images=avatar_images,
                                     latex_delimiters=[],
                                     )
    with gr.Row():
        text_output = gr.Chatbot(label=output_label0,
                                 visible=not kwargs['model_lock'],
                                 **no_model_lock_chat_kwargs,
                                 likeable=True,
                                 )
        text_output2 = gr.Chatbot(label=output_label0_model2,
                                  visible=False and not kwargs['model_lock'],
                                  **no_model_lock_chat_kwargs,
                                  likeable=True,
                                  )

    chatbots = [text_output, text_output2] + text_outputs

    with gr.Row(visible=visible_ratings):
        review_textbox = gr.Textbox(visible=True, label="Review", placeholder="Type your review...", scale=4)
        rating_text_output = gr.Textbox(elem_id="text_output", visible=False)
        with gr.Column():
            with gr.Row():
                rating1 = gr.Button(value='⭑', variant='outline-primary', scale=1, elem_id="rating1", size="sm")
                rating2 = gr.Button(value='⭑', variant='outline-primary', scale=1, elem_id="rating2", size="sm")
                rating3 = gr.Button(value='⭑', variant='outline-primary', scale=1, elem_id="rating3", size="sm")
                rating4 = gr.Button(value='⭑', variant='outline-primary', scale=1, elem_id="rating4", size="sm")
                rating5 = gr.Button(value='⭑', variant='outline-primary', scale=1, elem_id="rating5", size="sm")

            review_js1 = """
            function highlightButtons() {
                var element = document.getElementById("rating1");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating2");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 

                var element = document.getElementById("rating3");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 

                var element = document.getElementById("rating4");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 

                var element = document.getElementById("rating5");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 
            }
            """

            review_js2 = """
            function highlightButtons() {
                var element = document.getElementById("rating1");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating2");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating3");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 

                var element = document.getElementById("rating4");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 

                var element = document.getElementById("rating5");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 
            }
            """
            review_js3 = """
            function highlightButtons() {
                var element = document.getElementById("rating1");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating2");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating3");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating4");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 

                var element = document.getElementById("rating5");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 
            }
            """
            review_js4 = """
            function highlightButtons() {
                var element = document.getElementById("rating1");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating2");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating3");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating4");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating5");
                // element.style.backgroundColor = "rgba(173, 181, 189, 0.5)"; 
                element.style.color = "rgba(173, 181, 189, 0.5)"; 
            }
            """
            review_js5 = """
            function highlightButtons() {
                var element = document.getElementById("rating1");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating2");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating3");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating4");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 

                var element = document.getElementById("rating5");
                // element.style.backgroundColor = "#ffa41c"; 
                element.style.color = "#ffa41c"; 
            }
            """
            if is_gradio_version4:
                rating1.click(ratingfn1, outputs=rating_text_output, js=review_js1)
                rating2.click(ratingfn2, outputs=rating_text_output, js=review_js2)
                rating3.click(ratingfn3, outputs=rating_text_output, js=review_js3)
                rating4.click(ratingfn4, outputs=rating_text_output, js=review_js4)
                rating5.click(ratingfn5, outputs=rating_text_output, js=review_js5)
            else:
                rating1.click(ratingfn1, outputs=rating_text_output, _js=review_js1)
                rating2.click(ratingfn2, outputs=rating_text_output, _js=review_js2)
                rating3.click(ratingfn3, outputs=rating_text_output, _js=review_js3)
                rating4.click(ratingfn4, outputs=rating_text_output, _js=review_js4)
                rating5.click(ratingfn5, outputs=rating_text_output, _js=review_js5)

            submit_review_btn = gr.Button("Submit Review", scale=1)
            submit_review_func = functools.partial(submit_review,
                                                   reviews_file=reviews_file if reviews_file else None,
                                                   num_model_lock=len(chatbots))
            submit_review_btn.click(submit_review_func,
                                    inputs=[review_textbox, rating_text_output,
                                            text_output, text_output2] + text_outputs,
                                    outputs=review_textbox)

    # set likeable method
    def on_like(like_data: gr.LikeData):
        submit_review(str(like_data.liked) + "," + str(like_data.target.label), *tuple([['', like_data.value], []]),
                      reviews_file=reviews_file, num_model_lock=len(chatbots), do_info=False)

    for chatbot in chatbots:
        chatbot.like(on_like)

    return text_output, text_output2, text_outputs
