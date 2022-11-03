This demo is a copy of the demo CLIPGroundingExlainability built by Paul Hilders, Danilo de Goede and Piyush Bagad, as part of the course Interpretability and Explainability in AI (MSc AI, UvA, June 2022).


This demo shows attributions scores on both the image and the text input when presenting CLIP with a
<text,image> pair. Attributions are computed as Gradient-weighted Attention Rollout (Chefer et al.,
2021), and can be thought of as an estimate of the effective attention CLIP pays to its input when
computing a multimodal representation. <span style="color:red">Warning:</span> Note that attribution
methods such as the one from this demo can only give an estimate of the real underlying behavior
of the model.
