import re
import sys
import pathlib
import csv
import gradio as gr

sys.path.append("CLIP_explainability/Transformer-MM-Explainability/")

import torch
import CLIP.clip as clip

import spacy
from PIL import Image, ImageFont, ImageDraw, ImageOps

from clip_grounding.utils.image import pad_to_square
from clip_grounding.datasets.png import (
    overlay_relevance_map_on_image,
)
from CLIP_explainability.utils import interpret, show_img_heatmap, show_heatmap_on_text

clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

def iter_file(filename):
    with pathlib.Path(filename).open("r") as fh:
        header = next(fh)
        for line in fh:
            yield line

colour_map = {
        "N": "#f77189",
        "CARDINAL": "#f7764a",
        "DATE": "#d98a32",
        "EVENT": "#bf9632",
        "FAC": "#a99e31",
        "GPE": "#90a531",
        "LANGUAGE": "#68ad31",
        "LAW": "#32b25e",
        "LOC": "#34af86",
        "MONEY": "#35ae9c",
        "NORP": "#36acac",
        "ORDINAL": "#37aabd",
        "ORG": "#39a7d4",
        "PERCENT": "#539ff4",
        "PERSON": "#9890f4",
        "PRODUCT": "#c47ef4",
        "QUANTITY": "#ef5ff4",
        "TIME": "#f565d0",
        "WORK_OF_ART": "#f66baf",
    }

device = "cuda" if torch.cuda.is_available() else "cpu"

# nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()

# Gradio Section:
def update_slider(model):
    if model == "ViT-L/14":
        return gr.update(maximum=23, value=23)
    else:
        return gr.update(maximum=11, value=11)

def run_demo(*args):
    if len(args) == 4:
        image, text, model_name, vision_layer = args
    elif len(args) == 2:
        image, text = args
        model_name = "ViT-B/32"
        vision_layer = 11
    else:
        raise ValueError("Unexpected number of parameters")

    vision_layer = int(vision_layer)
    model, preprocess = clip.load(model_name, device=device, jit=False)
    orig_image = pad_to_square(image)
    img = preprocess(orig_image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)

    R_text, R_image = interpret(model=model, image=img, texts=text_input, device=device, start_layer=vision_layer)

    image_relevance = show_img_heatmap(R_image[0], img, orig_image=orig_image, device=device)
    overlapped = overlay_relevance_map_on_image(image, image_relevance)

    text_scores, text_tokens_decoded = show_heatmap_on_text(text, text_input, R_text[0])

    highlighted_text = []
    for i, token in enumerate(text_tokens_decoded):
        highlighted_text.append((str(token), float(text_scores[i])))

    return overlapped, highlighted_text


# Default demo:

examples = list(csv.reader(iter_file("examples.csv")))
with gr.Blocks(title="CLIP Grounding Explainability") as iface_default:
    gr.Markdown(pathlib.Path("description.md").read_text)
    with gr.Row():
        with gr.Column() as inputs:
            orig = gr.components.Image(type='pil', label="Original Image")
            description = gr.components.Textbox(label="Image description")
            default_model = gr.Dropdown(label="CLIP Model", choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14'], value="ViT-B/32")
            default_layer = gr.Slider(label="Vision start layer", minimum=0, maximum=11, step=1, value=11)
            submit = gr.Button("Submit")
        with gr.Column() as outputs:
            image = gr.components.Image(type='pil', label="Output Image")
            text = gr.components.HighlightedText(label="Text importance")
    gr.Examples(examples=examples, inputs=[orig, description])
    default_model.change(update_slider, inputs=default_model, outputs=default_layer)
    submit.click(run_demo, inputs=[orig, description, default_model, default_layer], outputs=[image, text])


# NER demo:
def add_label_to_img(img, label, add_entity_label=True):
    img = ImageOps.expand(img, border=45, fill=(255,255,255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 24)
    m = re.match(r".*\((\w+)\)", label)
    if add_entity_label and m is not None:
        cat = m.group(1)
        colours = tuple(map(lambda l: int(''.join(l),16), zip(*[iter(colour_map[cat][1:])]*2)))

        draw.text((5,5), label , align="center", fill=colours, font=font)
    else:
        draw.text((5,5), label, align="center", fill=(0, 0, 0), font=font)

    return img

def NER_demo(image, text, model_name):

    # As the default image, we run the default demo on the input image and text:
    overlapped, highlighted_text = run_demo(image, text, model_name)

    gallery_images = [add_label_to_img(overlapped, "Complete sentence", add_entity_label=False)]

    labeled_text = dict(
            text=text,
            entities=[],
        )

    # Then, we run the demo for each of the noun chunks in the text:
    for chunk in nlp(text).noun_chunks:
        if len(chunk) == 1 and chunk[0].pos_ == "PRON":
            continue
        chunk_text = chunk.text
        chunk_label = None
        for t in chunk:
            if t.ent_type_ != '':
                chunk_label = t.ent_type_
                break
        if chunk_label is None:
            chunk_label = "N"

        labeled_text['entities'].append({'entity': chunk_label, 'start': chunk.start_char, 'end': chunk.end_char})
        overlapped, highlighted_text = run_demo(image, chunk_text, model_name)
        overlapped_labelled = add_label_to_img(overlapped, f"{chunk_text} ({chunk_label})")
        gallery_images.append(overlapped_labelled)

    return labeled_text, gallery_images


entity_examples = list(csv.reader(iter_file("entity_examples.csv")))
with gr.Blocks(title="Entity Grounding explainability using CLIP") as iface_NER:
    gr.Markdown(pathlib.Path("entity_description.md").read_text)
    with gr.Row():
        with gr.Column() as inputs:
            img = gr.Image(type='pil', label="Original Image")
            intext = gr.components.Textbox(label="Descriptive text")
            ner_model = gr.Dropdown(label="CLIP Model", choices=['ViT-B/16', 'ViT-B/32', 'ViT-L/14'], value="ViT-B/32")
            ner_layer = gr.Slider(label="Vision start layer", minimum=0, maximum=11, step=1, value=11)
            submit = gr.Button("Submit")
        with gr.Column() as outputs:
            text = gr.components.HighlightedText(show_legend=True, color_map=colour_map, label="Noun chunks")
            gallery = gr.components.Gallery(type='pil', label="NER Entity explanations")

    gr.Examples(examples=entity_examples, inputs=[img, text])
    ner_model.change(update_slider, inputs=ner_model, outputs=ner_layer)
    submit.click(run_demo, inputs=[img, intext, ner_model, ner_layer], outputs=[text, gallery])

demo_tabs = gr.TabbedInterface([iface_default, iface_NER], ["Default", "Entities"])
with demo_tabs:
    gr.Markdown(pathlib.Path("footer.md").read_text)
demo_tabs.launch(show_error=True)
