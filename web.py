#!/usr/bin/env python
# encoding: utf-8
import gradio as gr
from search import search_

model = ''
form_radio = {
    'choices': ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50'],
    'value': 'ViT-B-16',
    'interactive': True,
    'label': '模型选择'
}

pic_num = {
    'minimum': 0,
    'maximum': 1000,
    'value': 10,
    'step': 5,
    'interactive': True,
    'label': '返回图片数(可能被过滤部分)'
}

top_p_slider = {
    'minimum': 0,
    'maximum': 1,
    'value': 0.1,
    'step': 0.05,
    'interactive': True,
    'label': 'Top P'
}

small_pic = {
    'choices': ['是', '否'],
    'value': '是',
    'interactive': True,
    'label': '是否返回缩略图'
}


def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(
            minimum=params['minimum'],
            maximum=params['maximum'],
            value=params['value'],
            step=params['step'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Radio':
        return gr.Radio(
            choices=params['choices'],
            value=params['value'],
            interactive=params['interactive'],
            label=params['label']
        )
    elif comp == 'Button':
        return gr.Button(
            value=params['value'],
            interactive=True
        )


def text_search_button_clicked(_question, top_k, models, top_p):
    file_list = search_(_question, top_k, models, top_p, search_text=True)
    file_list = [(file, f"{i + 1}") for i, file in enumerate(file_list)]
    return gr.Gallery(file_list, columns=5)


def pic_search_button_clicked(_question, top_k, models, top_p):
    file_list = search_(_question, top_k, models, top_p, search_img=True)
    file_list = [(file, f"{i + 1}") for i, file in enumerate(file_list)]
    return gr.Gallery(file_list, columns=5)


with gr.Blocks() as demo:
    with gr.Tab("文到图搜索"):
        gr.Markdown('<h1 style="text-align: center; font-size: 40px;">中文Clip文到图搜索应用</h1>')

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                txt_message = gr.Textbox(label="请填写文本")

                pic_nums = create_component(pic_num)
                models = create_component(form_radio, comp='Radio')
                top_p = create_component(top_p_slider)
                search = create_component({'value': 'search'}, comp='Button')

            with gr.Column(scale=1):
                bt_pic = gr.Gallery(label="检索结果为:", columns=5)

            search.click(
                text_search_button_clicked,
                [txt_message, pic_nums, models, top_p],
                [bt_pic]
            )

    with gr.Tab("图到图搜索"):
        gr.Markdown('<h1 style="text-align: center; font-size: 40px;">中文Clip图到图搜索应用</h1>')
        with gr.Row():
            with gr.Column(scale=1, min_width=50):
                input_pic = gr.Image(label="图片", sources=['upload'], type="filepath")
                pic_nums = create_component(pic_num)
                models = create_component(form_radio, comp='Radio')
                top_p = create_component(top_p_slider)
                search = create_component({'value': 'search'}, comp='Button')

            with gr.Column(scale=1):
                bt_pic = gr.Gallery(label="检索结果为:", columns=5)

            search.click(
                pic_search_button_clicked,
                [input_pic, pic_nums, models, top_p],
                [bt_pic]
            )

# launch
demo.launch(share=False, debug=True, show_api=False, server_port=8089, server_name="0.0.0.0")
