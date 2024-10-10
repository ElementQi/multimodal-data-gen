# multimodal-data-gen

This repo aims to generate multimodal tuning data

## Methods

There are 2 methods to generate data:

- Just re-write the conversations based on already formatted json files, but this may causes inconsistency.

- From captions and boxes for a image to generate conversations, like what [llava paper](https://arxiv.org/pdf/2304.08485) did:

> Table 1: the visual image is not used to prompt GPT, we only show it here as a reference

This requires a deeper dive into the core code of LLaVA.

### Re-write

In root directory, modify and run the code listed in `example/rewrite.sh`.

## Data

The demo data [llava_instruct_10.json](./data/llava_instruct_10.json) is the first 10 data from [liuhaotian/LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K?row=15).

The generated demo data using re-write method is stored in `saves/rewritten_llava_instruct.json`
