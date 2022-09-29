import re
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

default_inputs = [
        gr.components.Image(type='pil', label="Original Image"),
        gr.components.Textbox(label="Image description"),
    ]

default_outputs = [
        gr.components.Image(type='pil', label="Output Image"),
        gr.components.HighlightedText(label="Text importance"),
    ]


description = """This demo is a copy of the demo CLIPGroundingExlainability built by Paul Hilders, Danilo de Goede and Piyush Bagad, as part of the course Interpretability and Explainability in AI (MSc AI, UvA, June 2022).
<br> <br>
                 This demo shows attributions scores on both the image and the text input when presenting CLIP with a
                 <text,image> pair. Attributions are computed as Gradient-weighted Attention Rollout (Chefer et al.,
                 2021), and can be thought of as an estimate of the effective attention CLIP pays to its input when
                 computing a multimodal representation. <span style="color:red">Warning:</span> Note that attribution
                 methods such as the one from this demo can only give an estimate of the real underlying behavior
                 of the model."""

iface = gr.Interface(fn=run_demo,
                     inputs=default_inputs,
                     outputs=default_outputs,
                     title="CLIP Grounding Explainability",
                     description=description,
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
    m = re.match(r".*\((\w+)\)", label)
    if add_entity_label and m is not None:
        cat = m.group(1)
        colours = tuple(map(lambda l: int(''.join(l),16), zip(*[iter(colour_map[cat][1:])]*2)))

        draw.text((5,5), label , align="center", fill=colours, font=font)
    else:
        draw.text((5,5), label, align="center", fill=(0, 0, 0), font=font)

    return img

def NER_demo(image, text):
    # As the default image, we run the default demo on the input image and text:
    overlapped, highlighted_text = run_demo(image, text)

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
        overlapped, highlighted_text = run_demo(image, chunk_text)
        overlapped_labelled = add_label_to_img(overlapped, f"{chunk_text} ({chunk_label})")
        gallery_images.append(overlapped_labelled)

    return labeled_text, gallery_images

inputs_NER = [
        gr.Image(type='pil', label="Original Image"),
        gr.components.Textbox(label="Descriptive text"),
    ]

#colours = highlighter._style["color_map"]
outputs_NER = [
        gr.components.HighlightedText(show_legend=True, color_map=colour_map, label="Noun chunks"),
        gr.components.Gallery(type='pil', label="NER Entity explanations")
    ]

description_NER = """Automatically generated CLIP grounding explanations for
                     named entities, retrieved from the spacy NER model. <span style="color:red">Warning:</span> Note
                     that attribution methods such as the one from this demo can only give an estimate of the real
                     underlying behavior of the model."""

iface_NER = gr.Interface(fn=NER_demo,
                         inputs=inputs_NER,
                         outputs=outputs_NER,
                         title="Named Entity Grounding explainability using CLIP",
                         description=description_NER,
                         examples=[
                             ["example_images/London.png", "In this image we see Big Ben and the London Eye, on both sides of the river Thames."],
                             ["example_images/harrypotter.png", "Hermione, Harry and Ron in their school uniform"],
                             ],
                         cache_examples=False)

demo_tabs = gr.TabbedInterface([iface, iface_NER], ["Default", "NER"])

with demo_tabs:
    gr.Markdown("""
                ### Acknowledgements
                This demo was developed for the Interpretability & Explainability in AI course at the University of
                Amsterdam. We would like to express our thanks to Jelle Zuidema, Jaap Jumelet, Tom Kersten, Christos
                Athanasiadis, Peter Heemskerk, Zhi Zhang, and all the other TAs who helped us during this course.

                ---
                ### References
                \[1\]: Chefer, H., Gur, S., & Wolf, L. (2021). Generic attention-model explainability for interpreting bi-modal and encoder-decoder transformers. <br>
                \[2\]: Abnar, S., & Zuidema, W. (2020). Quantifying attention flow in transformers. arXiv preprint arXiv:2005.00928. <br>
                \[3\]: [https://samiraabnar.github.io/articles/2020-04/attention_flow](https://samiraabnar.github.io/articles/2020-04/attention_flow) <br>
                """)
demo_tabs.launch(show_error=True)
