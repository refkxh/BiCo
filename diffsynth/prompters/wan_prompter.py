from .base_prompter import BasePrompter
from ..models.wan_video_text_encoder import WanTextEncoder
from transformers import AutoTokenizer
import os, torch
import ftfy
import html
import string
import regex as re


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def canonicalize(text, keep_punctuation_exact_string=None):
    text = text.replace('_', ' ')
    if keep_punctuation_exact_string:
        text = keep_punctuation_exact_string.join(
            part.translate(str.maketrans('', '', string.punctuation))
            for part in text.split(keep_punctuation_exact_string))
    else:
        text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class HuggingfaceTokenizer:

    def __init__(self, name, seq_len=None, clean=None, **kwargs):
        assert clean in (None, 'whitespace', 'lower', 'canonicalize')
        self.name = name
        self.seq_len = seq_len
        self.clean = clean

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
        self.vocab_size = self.tokenizer.vocab_size

    def __call__(self, sequence, **kwargs):
        return_mask = kwargs.pop('return_mask', False)

        # arguments
        _kwargs = {'return_tensors': 'pt'}
        if self.seq_len is not None:
            _kwargs.update({
                'padding': 'max_length',
                'truncation': True,
                'max_length': self.seq_len
            })
        _kwargs.update(**kwargs)

        # tokenization
        if isinstance(sequence, str):
            sequence = [sequence]
        if self.clean:
            sequence = [self._clean(u) for u in sequence]
        ids = self.tokenizer(sequence, **_kwargs)

        # output
        if return_mask:
            return ids.input_ids, ids.attention_mask
        else:
            return ids.input_ids

    def _clean(self, text):
        if self.clean == 'whitespace':
            text = whitespace_clean(basic_clean(text))
        elif self.clean == 'lower':
            text = whitespace_clean(basic_clean(text)).lower()
        elif self.clean == 'canonicalize':
            text = canonicalize(basic_clean(text))
        return text


class WanPrompter(BasePrompter):

    def __init__(self, tokenizer_path=None, text_len=512):
        super().__init__()
        self.text_len = text_len
        self.text_encoder = None
        self.fetch_tokenizer(tokenizer_path)

    def fetch_tokenizer(self, tokenizer_path=None):
        if tokenizer_path is not None:
            self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=self.text_len, clean='whitespace')

    def fetch_models(self, text_encoder: WanTextEncoder = None):
        self.text_encoder = text_encoder

    def encode_prompt(self, prompt, positive=True, device="cuda", multi_concept_prompt=False):
        if multi_concept_prompt:
            main_prompt = prompt[0].split("#")
            concepts = prompt[1:]

            main_prompt = [p.strip() for p in main_prompt]
            main_prompt_ids = [
                (
                    self.tokenizer(p, return_mask=False, add_special_tokens=False)
                    if len(p) > 0
                    else None
                )
                for p in main_prompt
            ]
            main_prompt_ids.append(
                self.tokenizer("", return_mask=False, add_special_tokens=True)
            )
            concepts_ids = [
                self.tokenizer(c.strip(), return_mask=False, add_special_tokens=False)
                for c in concepts
            ]
            
            ids = torch.zeros((1, self.text_len), dtype=torch.long)
            mask = torch.zeros((1, self.text_len), dtype=torch.long)
            concept_mask = torch.zeros((1, self.text_len), dtype=torch.long)
            pos = 0
            for i, p in enumerate(main_prompt_ids):
                if p is not None:
                    p_length = torch.sum(p.gt(0)).item()
                    ids[0, pos : pos + p_length] = p[0, :p_length]
                    mask[0, pos : pos + p_length] = 1
                    pos += p_length
                if i < len(concepts_ids):
                    c = concepts_ids[i]
                    c_length = torch.sum(c.gt(0)).item()
                    ids[0, pos : pos + c_length] = c[0, :c_length]
                    mask[0, pos : pos + c_length] = 1
                    concept_mask[0, pos : pos + c_length] = i + 1
                    pos += c_length
        else:
            prompt = self.process_prompt(prompt, positive=positive)
            ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)

        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(ids, mask)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0

        if multi_concept_prompt:
            return prompt_emb, concept_mask.to(device)
        else:
            return prompt_emb
