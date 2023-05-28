# @title Import
import os

print(os.getcwd())
import sys

from utils import create_logger, set_seed

import time
import argparse
import json
from PIL import Image
import torch
import numpy as np
from clip.clip import CLIP
from gen_utils import generate_caption
from transformers import AutoModelForMaskedLM, AutoTokenizer


# @title Define parameters
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1, help="Only supports batch_size=1 currently.")
    parser.add_argument("--device", type=str,
                        default='cuda', choices=['cuda', 'cpu'])

    ## Generation and Controllable Type
    parser.add_argument('--run_type',
                        default='caption',
                        nargs='?',
                        choices=['caption', 'controllable'])
    parser.add_argument('--prompt',
                        default='image of ', type=str)
    parser.add_argument('--order',
                        default='shuffle',
                        nargs='?',
                        choices=['sequential', 'shuffle', 'span', 'random', 'parallel'],
                        help="Generation order of text")
    parser.add_argument('--control_type',
                        default='sentiment',
                        nargs='?',
                        choices=["sentiment", "pos"],
                        help="which controllable task to conduct")
    parser.add_argument('--pos_type', type=list,
                        default=[['DET'], ['ADJ', 'NOUN'], ['NOUN'],
                                 ['VERB'], ['VERB'], ['ADV'], ['ADP'],
                                 ['DET', 'NOUN'], ['NOUN'], ['NOUN', '.'],
                                 ['.', 'NOUN'], ['.', 'NOUN']],
                        help="predefined part-of-speech templete")
    parser.add_argument('--sentiment_type',
                        default="positive",
                        nargs='?',
                        choices=["positive", "negative"])
    parser.add_argument('--samples_num',
                        default=2, type=int)

    ## Hyperparameters
    parser.add_argument("--sentence_len", type=int, default=10)
    parser.add_argument("--candidate_k", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.02, help="weight for fluency")
    parser.add_argument("--beta", type=float, default=2.0, help="weight for image-matching degree")
    parser.add_argument("--gamma", type=float, default=0, help="weight for controllable degree")
    parser.add_argument("--lm_temperature", type=float, default=0.1)
    parser.add_argument("--num_iterations", type=int, default=10, help="predefined iterations for Gibbs Sampling")

    ## Models and Paths
    parser.add_argument("--lm_model", type=str, default='bert-base-uncased',
                        help="Path to language model")  # bert,roberta
    parser.add_argument("--match_model", type=str, default='openai/clip-vit-base-patch32',
                        help="Path to Image-Text model")  # clip,align
    parser.add_argument("--caption_img_path", type=str, default='./examples/girl.jpg',
                        help="file path of the image for captioning")
    parser.add_argument("--stop_words_path", type=str, default='stop_words.txt',
                        help="Path to stop_words.txt")
    parser.add_argument("--add_extra_stopwords", type=list, default=[],
                        help="you can add some extra stop words")

    args = parser.parse_args(args=[])
    return args


# @title Select types and parameters
args = get_args()


import pandas as pd
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np

import matplotlib.pyplot as plt

# Define the path to your COCO dataset annotation JSON file
annotation_file = 'captions_val2014.json'

with open(annotation_file, 'r') as json_file:
  json_data = json.load(json_file)
  images = json_data.get('images')
  annotations = json_data.get('annotations')

annotations = pd.DataFrame(annotations)
images = pd.DataFrame(images)
images.rename(columns={'id':'image_id'}, inplace=True)
data = pd.merge(left = annotations , right = images, how = "inner", on = "image_id")[["caption", "coco_url", "image_id"]].drop_duplicates("image_id")
# data.to_csv("/content/COCOcaption.csv")


class CocoCaptionDataset(Dataset):
    def __init__(self, dataframe, size):
        self.dataframe = dataframe
        self.transform = transforms.ToTensor()
        self.size = size
        os.makedirs("data/", exist_ok=True)

    def __len__(self):
        # return len(self.dataframe)
        return self.size

    # def __getitem__(self, index):
    #     row = self.dataframe.iloc[index]
    #     caption = row['caption']
    #     coco_url = row['coco_url']
    #
    #     # Get image filename from coco_url
    #     image_filename = f"{index:0>8}.jpg"
    #     image_path = os.path.join("data/", image_filename)
    #
    #     # Download image if not already downloaded
    #     if not os.path.exists(image_path):
    #         Image.open(requests.get(coco_url, stream=True).raw).convert("RGB").save(image_path)
    #
    #     # Load image from file
    #     image = Image.open(image_path).convert("RGB")
    #     image = self.transform(image)
    #
    #     return image, caption, coco_url

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        caption = row['caption']
        coco_url = row['coco_url']
        image = Image.open(requests.get(coco_url, stream=True).raw).convert("RGB")
        return image, caption, coco_url

    def collate_fn(self, batch):
        images, captions, coco_url = zip(*batch)
        return images, captions,coco_url




# for imgs, caps in train_loader:
#     for img, cap in zip(imgs, caps):
#       print(cap)
#       plt.imshow(img.swapaxes(0,2))
#       plt.show()

# len(coco_dataset)


# @title Image captioning
def run_caption(args, name_batch_list, url, img_batch_pil_list, lm_model, lm_tokenizer, clip, token_mask, logger,
                all_results, model_type, generate_order):
    gen_texts, clip_scores = generate_caption(img_name=url,
                                              model_type=model_type,
                                              model=lm_model,
                                              clip=clip,
                                              tokenizer=lm_tokenizer,
                                              image_instance=img_batch_pil_list,
                                              token_mask=token_mask,
                                              logger=logger,
                                              prompt=args.prompt,
                                              batch_size=args.batch_size,
                                              max_len=args.sentence_len,
                                              top_k=args.candidate_k,
                                              temperature=args.lm_temperature,
                                              max_iter=args.num_iterations,
                                              alpha=args.alpha,
                                              beta=args.beta,
                                              generate_order=generate_order)

    all_results += [dict() for _ in range(args.batch_size)]
    for iter, (gen_texts_iters, clip_scores_iters) in enumerate(zip(gen_texts, clip_scores)):
        for idx, (gen_text_iters, clip_score_iters) in enumerate(zip(gen_texts_iters, clip_scores_iters)):
            all_results[-(idx+1)][f"{iter}_clip"] = clip_score_iters
            all_results[-(idx+1)][f"{iter}_text"] = gen_text_iters

    return all_results


from transformers import ViltProcessor, ViltForMaskedLM, AutoModelForMaskedLM, AutoTokenizer


def stopandmask(tokenizer, model_type):
    if model_type == "ViLT":
        tokenizer = tokenizer.tokenizer
    with open(args.stop_words_path, 'r', encoding='utf-8') as stop_words_file:
        stop_words = stop_words_file.readlines()
        stop_words_ = [stop_word.rstrip('\n') for stop_word in stop_words]
        stop_words_ += args.add_extra_stopwords
        stop_ids = tokenizer.convert_tokens_to_ids(stop_words_)
        token_mask = torch.ones((1, tokenizer.vocab_size))
        for stop_id in stop_ids:
            token_mask[0, stop_id] = 0
        token_mask = token_mask.to(args.device)
    return stop_words, stop_ids, token_mask


def run(batch_size=2, alpha=0.02, beta=2, num_candidate=200, num_iter=20, len_sentence=20, model_type="ViLT",
        generate_order="sequential", size=256, korean=True):
    args = get_args()
    args.alpha = alpha
    args.beta = beta
    args.candidate_k = num_candidate
    args.num_iterations = num_iter
    args.sentence_len = len_sentence
    args.batch_size = batch_size
    args.order = generate_order
    args.samples_num = 1

    set_seed(args.seed)
    run_type = "caption" if args.run_type == "caption" else args.control_type
    if run_type == "sentiment":
        run_type = args.sentiment_type

    logger = create_logger(
        "logger", '{}_{}_len{}_topk{}_alpha{}_beta{}_gamma{}_lmtemp{}_{}.log'.format(
            run_type, args.order, args.sentence_len,
            args.candidate_k, args.alpha, args.beta, args.gamma, args.lm_temperature,
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())))

    logger.info(f"Generating order:{args.order}")
    logger.info(f"Run type:{run_type}")
    logger.info(args)


    # Load pre-trained model (weights)

    if korean:
        args.match_model = "koclip/koclip-base"
    if model_type == "ViLT":
        print("ViLT")
        lm_model = ViltForMaskedLM.from_pretrained("dandelin/vilt-b32-mlm")
        lm_tokenizer = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
        lm_model.eval()


    elif korean:
        print("Korean")
        lm_model = AutoModelForMaskedLM.from_pretrained("snunlp/KR-BERT-char16424")
        lm_tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-BERT-char16424")
        lm_model.eval()

    else:
        print("BERT")
        lm_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        lm_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        lm_model.eval()

    clip = CLIP(args.match_model,korean)
    clip.eval()
    lm_model = lm_model.to(args.device)
    clip = clip.to(args.device)

    ## Remove stop words, token mask
    stop_words, stop_ids, token_mask = stopandmask(lm_tokenizer, model_type)

    img_dir = args.caption_img_path

    coco_dataset = CocoCaptionDataset(data, size)
    coco_dataset = coco_dataset
    train_loader = DataLoader(coco_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True,
                              collate_fn=coco_dataset.collate_fn)

    for sample_id in range(args.samples_num):
        all_results = []
        logger.info(f"Sample {sample_id + 1}: ")
        for batch_idx, (img_batch_pil_list, name_batch_list, coco_url) in enumerate(train_loader):
            logger.info(f"The {batch_idx + 1}-th batch:")
            with torch.no_grad():
                all_results = run_caption(args=args,
                                          name_batch_list=name_batch_list,
                                          url=coco_url,
                                          img_batch_pil_list=img_batch_pil_list,
                                          lm_model=lm_model,
                                          lm_tokenizer=lm_tokenizer,
                                          clip=clip,
                                          token_mask=token_mask,
                                          logger=logger,
                                          all_results=all_results,
                                          model_type=model_type,
                                          generate_order=generate_order)

            save_dir = f"results/model_type_{model_type}_order_{args.order}_len{args.sentence_len}_alpha{args.alpha}_iters{args.num_iterations}" \
                       f"_topk{args.candidate_k}_alpha{args.alpha}/sample_{sample_id}"

            if os.path.exists(save_dir) == False:
                os.makedirs(save_dir)

            pd.DataFrame(all_results).to_csv(os.path.join(save_dir, "result.csv"))

import gc
gc.collect()
torch.cuda.empty_cache()
#
# gc.collect()
# torch.cuda.empty_cache()
# # run(batch_size=32,num_iter=15, num_candidate=50, len_sentence=12, alpha=0.02, generate_order="shuffle", model_type="BERT", korean=False, size=1000)
# gc.collect()
# torch.cuda.empty_cache()
# run(batch_size=16,num_iter=15, num_candidate=100, len_sentence=12, alpha=0.02, generate_order="shuffle", model_type="ViLT", korean=False, size=1000)
# gc.collect()
# torch.cuda.empty_cache()
# run(batch_size=16,num_iter=15, num_candidate=100, len_sentence=12, alpha=0.02, generate_order="shuffle", model_type="BERT", korean=False, size=1000)
# gc.collect()
# torch.cuda.empty_cache()
# run(batch_size=4,num_iter=15, num_candidate=200, len_sentence=12, alpha=0.02, generate_order="shuffle", model_type="ViLT", korean=False, size=1000)
gc.collect()
torch.cuda.empty_cache()
run(batch_size=8, num_iter=15, num_candidate=200, len_sentence=12, alpha=0.02, generate_order="shuffle", model_type="BERT", korean=False, size=1000)
