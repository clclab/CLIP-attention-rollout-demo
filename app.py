import sys
import gradio as gr

# sys.path.append("../")
sys.path.append("CLIP_explainability/Transformer-MM-Explainability/")

import torch
import CLIP.clip as clip

import spacy
from PIL import Image, ImageFont, ImageDraw, ImageOps

import os
os.system('python -m spacy download en_core_web_sm')


from clip_grounding.utils.image import pad_to_square
from clip_grounding.datasets.png import (
    overlay_relevance_map_on_image,
)
from CLIP_explainability.utils import interpret, show_img_heatmap, show_heatmap_on_text

clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# nlp = spacy.load("en_core_web_sm")
import en_core_web_sm
nlp = en_core_web_sm.load()

# Gradio Section:
def run_demo(image, text):
    orig_image = pad_to_square(image)
    img = preprocess(orig_image).unsqueeze(0).to(device)
    text_input = clip.tokenize([text]).to(device)

    R_text, R_image = interpret(model=model, image=img, texts=text_input, device=device)

    image_relevance = show_img_heatmap(R_image[0], img, orig_image=orig_image, device=device, show=False)
    overlapped = overlay_relevance_map_on_image(image, image_relevance)

    text_scores, text_tokens_decoded = show_heatmap_on_text(text, text_input, R_text[0], show=False)

    highlighted_text = []
    for i, token in enumerate(text_tokens_decoded):
        highlighted_text.append((str(token), float(text_scores[i])))

    return overlapped, highlighted_text


# Default demo:
input_img = gr.inputs.Image(type='pil', label="Original Image")
input_txt = "text"
inputs = [input_img, input_txt]

outputs = [gr.inputs.Image(type='pil', label="Output Image"), "highlight"]


iface = gr.Interface(fn=run_demo,
                     inputs=inputs,
                     outputs=outputs,
                     title="CLIP Grounding Explainability",
                     description="A demonstration based on the Generic Attention-model Explainability method for Interpreting Bi-Modal Transformers by Chefer et al. (2021): https://github.com/hila-chefer/Transformer-MM-Explainability.",
                     examples=[["example_images/London.png", "London Eye"],
                               ["example_images/London.png", "Big Ben"],
                               ["example_images/harrypotter.png", "Harry"],
                               ["example_images/harrypotter.png", "Hermione"],
                               ["example_images/harrypotter.png", "Ron"],
                               ["example_images/Amsterdam.png", "Amsterdam canal"],
                               ["example_images/Amsterdam.png", "Old buildings"],
                               ["example_images/Amsterdam.png", "Pink flowers"],
                               ["example_images/dogs_on_bed.png", "Two dogs"],
                               ["example_images/dogs_on_bed.png", "Book"],
                               ["example_images/dogs_on_bed.png", "Cat"]])

# NER demo:
def add_label_to_img(img, label, add_entity_label=True):
    img = ImageOps.expand(img, border=45, fill=(255,255,255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 24)
    if add_entity_label:
        draw.text((0,0), f"Entity: {str(label)}" , align="center", fill=(0, 0, 0), font=font)
    else:
        draw.text((0,0), str(label), align="center", fill=(0, 0, 0), font=font)

    return img

def NER_demo(image, text):
    # Apply NER to extract named entities, and run the explainability method
    # for each named entity.
    highlighed_entities = []
    for ent in nlp(text).ents:
        ent_text = ent.text
        ent_label = ent.label_
        highlighed_entities.append((ent_text, ent_label))

    # As the default image, we run the default demo on the input image and text:
    overlapped, highlighted_text = run_demo(image, text)

    # Then, we run the demo for each of the named entities:
    gallery_images = [add_label_to_img(overlapped, "Full explanation", add_entity_label=False)]
    for ent_text, ent_label in highlighed_entities:
        overlapped_ent, highlighted_text_ent = run_demo(image, ent_text)
        overlapped_ent_labelled = add_label_to_img(overlapped_ent, f"{str(ent_text)} ({str(ent_label)})")

        gallery_images.append(overlapped_ent_labelled)

    return highlighed_entities, gallery_images

input_img_NER = gr.inputs.Image(type='pil', label="Original Image")
input_txt_NER = "text"
inputs_NER = [input_img_NER, input_txt_NER]

outputs_NER = ["highlight", gr.Gallery(type='pil', label="NER Entity explanations")]


iface_NER = gr.Interface(fn=NER_demo,
                         inputs=inputs_NER,
                         outputs=outputs_NER,
                         examples=[["example_images/London.png", "In this image we see Big Ben and the London Eye, on both sides of the river Thames."]],
                         cache_examples=False)

demo_tabs = gr.TabbedInterface([iface, iface_NER], ["Default", "NER"])
demo_tabs.launch(debug=True)