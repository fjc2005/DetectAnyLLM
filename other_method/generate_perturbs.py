# Revise from https://github.com/baoguangsheng/fast-detect-gpt
import os.path
import time
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re
import torch
import tqdm
import argparse
import json
import torch.nn as nn
import accelerate
from accelerate.utils import gather_object
from torch.utils.data import DataLoader
from dataset import CustomDataset

parser = argparse.ArgumentParser()
parser.add_argument("--perturb_model_name", type=str, default="t5-small")
parser.add_argument("--cache_dir", type=str, default="./model/")
parser.add_argument('--data_path', type=str, default='./data/MIRAGE_BENCH/DIG/rewrite.json', help='The path to the evaluation data. Default: ./data/DIG/rewrite.json.')
parser.add_argument('--data_format', type=str, default='MIRAGE', help='The format of the evaluation data. Should be MIRAGE or ImBD. Default: MIRAGE.')
parser.add_argument('--save_path', type=str, default='./results/perturbations')
parser.add_argument('--save_file', type=str, default='perturbations.json')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--n_perturbations', type=int, default=100)
parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
parser.add_argument('--mask_top_p', type=float, default=1.0)
parser.add_argument('--span_length', type=int, default=2)

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

def from_pretrained(cls, model_name, kwargs, cache_dir):
    # use local model if it exists
    if "/" in model_name:
        local_path = os.path.join(cache_dir, model_name.split("/")[-1])
    else:
        local_path = os.path.join(cache_dir, model_name)

    if os.path.exists(local_path):
        return cls.from_pretrained(local_path, **kwargs)
    return cls.from_pretrained(model_name, **kwargs, cache_dir=cache_dir, device_map='auto')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    buffer_size = 1
    tokens = text.split(' ')
    mask_string = '<<<mask>>>'

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f'<extra_id_{num_filled}>'
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = ' '.join(tokens)
    return text


def count_masks(texts):
    return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(args, mask_model, mask_tokenizer, texts):
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(mask_model.device)
    outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p,
                                  num_return_sequences=1, eos_token_id=stop_id)
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(' ') for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    span_length = args.span_length
    pct = args.pct_words_masked
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        if attempts > 100: perturbed_texts[idxs[0]] = perturbed_texts[0]
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts


def perturb_texts(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    chunk_size = 10
    outputs = []
    for i in range(0, len(texts), chunk_size):
        outputs.extend(perturb_texts_(args, mask_model, mask_tokenizer, texts[i:i + chunk_size], ceil_pct=ceil_pct))
    return outputs


class PerturbationGenerator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.mask_model = from_pretrained(AutoModelForSeq2SeqLM, args.perturb_model_name, {'torch_dtype': torch.float16}, args.cache_dir)
        self.mask_model.eval()
        self.mask_tokenizer = from_pretrained(AutoTokenizer, args.perturb_model_name, {'model_max_length': 512}, args.cache_dir)
        try:
            n_positions = self.mask_model.config.n_positions
        except AttributeError:
            n_positions = 512

    def forward(self, original_texts, rewritten_texts):
        all_results = []
        for i in range(len(original_texts)):
            original_text = original_texts[i]
            rewritten_text = rewritten_texts[i]
            # perturb
            p_rewritten_text = perturb_texts(self.args, self.mask_model, self.mask_tokenizer,
                                             [rewritten_text for _ in range(self.args.n_perturbations)])
            p_original_text = perturb_texts(self.args, self.mask_model, self.mask_tokenizer,
                                            [original_text for _ in range(self.args.n_perturbations)])
            assert len(
                p_rewritten_text) == self.args.n_perturbations, f"Expected {self.args.n_perturbations} perturbed samples, got {len(p_rewritten_text)}"
            assert len(
                p_original_text) == self.args.n_perturbations, f"Expected {self.args.n_perturbations} perturbed samples, got {len(p_original_text)}"
            # result
            all_results.append({
                "original": original_text,
                "rewritten": rewritten_text,
                "perturbed_original": p_original_text,
                "perturbed_rewritten": p_rewritten_text,
            })
        return all_results


if __name__ == '__main__':
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    accelerator = accelerate.Accelerator()
    model = PerturbationGenerator(args)
    # load data
    dataset = CustomDataset(data_path=args.data_path, data_format=args.data_format)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
    model, data_loader = accelerator.prepare(model, data_loader)
    # generate perturb samples
    perturbs = []
    start_time = time.time()
    for item in tqdm.tqdm(data_loader, desc=f"Perturb text", disable=not accelerator.is_main_process):
        results = model(item['original'], item['rewritten'])
        perturbs.extend(results)
    accelerator.wait_for_everyone()
    gathered_perturbs = gather_object(perturbs)

    if accelerator.is_main_process:
        final_perturbs = gathered_perturbs

        print(f"Total time: {time.time() - start_time:.4f}s")
        n_perturbations = args.n_perturbations
        name = f'perturbation_{n_perturbations}'
        saver = {
            'perturb_model': args.perturb_model_name,
            'n_perturbations': args.n_perturbations,
            'data': final_perturbs
        }
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        with open(os.path.join(args.save_path, f'{args.save_file}.json'), 'w', encoding='utf-8') as f:
            f.write(json.dumps(saver, indent=4, ensure_ascii=False))