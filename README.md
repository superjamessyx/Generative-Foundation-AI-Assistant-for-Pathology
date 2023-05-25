# PathAsst: Generative Foundation AI Assistant for Pathology



<div align=center> <img src="./img/logo.png"/> </div>



## Abstract

As advances in large language models (LLMs) and multimodal techniques continue to mature, the development of general-purpose multimodal large language models (MLLMs) has surged, with significant applications in natural image interpretation. However, the field of pathology has largely remained untapped in this regard, despite the growing need for accurate, timely, and personalized diagnostics. To bridge the gap in pathology MLLMs, we present the PathAsst in this study, which is a generative foundation AI assistant to revolutionize diagnostic and predictive analytics in pathology. To develop PathAsst, we collect over 142K high-quality pathology image-text pairs from a variety of reliable sources, including PubMed, comprehensive pathology textbooks, reputable pathology websites, and private data annotated by pathologists. Leveraging the advanced capabilities of ChatGPT/GPT-4, we generate over 180k instruction-following samples. Furthermore, we devise additional instruction-following data, specifically tailored for the invocation of the pathology-specific models, allowing the PathAsst to effectively interact with these models based on the input image and user intent, consequently enhancing the model's diagnostic capabilities. Subsequently, our PathAsst is trained based on Vicuna-13B language model in coordination with the CLIP vision encoder. The results of PathAsst show the potential of harnessing the AI-powered generative foundation model to improve pathology diagnosis and treatment processes. We are committed to open-sourcing our meticulously curated dataset, as well as a comprehensive toolkit designed to aid researchers in the extensive collection and preprocessing of their own datasets.



### Pathology Dataset Construction

<img src="./img/data_construction.png" alt="image-20230525002624933" style="zoom: 67%;" />



### PathAsst network architecture

<img src="./img/framework.png" alt="image-20230525002917910" style="zoom: 67%;" />



### Examples

<img src="./img/example_pdl1.png" alt="image-20230525003340750" style="zoom: 80%;" />





We will continue to open source the following content as soon as possible:

- [ ] A trained ConvNext-Tiny model specifically designed for selecting pathology images.

- [ ] An annotated 2K bounding box data for subfigure detection, alongside the trained YOLOv7 model.

- [ ] Scripts for automated extraction of image-text pairs from PDF books.

- [ ] A fine-tuned LLaMA-7B model intended for sub-caption splitting and caption refining.

- [ ] A collection of 100K processed PubMed image-text pairs.

In addition, we plan to train and release four versions of the CLIP model, which will be fine-tuned using more than 200K pathology samples, including clip-vit-base-patch16, clip-vit-base-patch32, clip-vit-large-patch14, and clip-vit-large-patch14-336.



