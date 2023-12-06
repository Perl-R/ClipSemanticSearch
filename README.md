# ClipSemanticSearch

CAP 5415 Fall 2023

Members:
* Charlee Mione
* Robin Perlman
* Jay Doshi
* Raj Doshi
* Isaac Tuckey
* Zachary Hull

## Project Goals

CLIP is a revolutionary model architecture which uses Contrastive Loss to embed image and text features into a shared dimensional space. This has many use cases, and in this project we use CLIP for semantic search tasks. Since images and text are embedded into a shared space, CLIP shows promise for performing Text-to-Image and Image-to-Image Search. We perform benchmarks such as Recall@k, Zero Shot Classification, and Search Speed/Space Complexity.

We also include a Gradio Demo as well to show how CLIP is used for image search and text search

![CLIP Architecture](https://miro.medium.com/v2/resize:fit:3662/1*tg7akErlMSyCLQxrMtQIYw.png)

## Gradio Demo Information

Link to Demonstration Video: https://drive.google.com/file/d/1Bh4ta_mJqRucW6fbHFfDH-yKee6k-AMc/view?usp=drive_link

TODO: Add Info here

## Dataset: COCO Captions

For much of the training and validation we use the COCO Captions dataset, which can be downloaded from [here](https://cocodataset.org/#home)

We specifically use the train2017 and val2017 datasets.
