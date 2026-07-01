import re
import json
import uuid
import random
import numpy as np
import wonderwords
from nltk.tokenize import sent_tokenize

import random
from utils.logging import logger
from dataloaders.BaseDataGenerator import BaseDataGenerator 
from utils.build_model import get_tokenizer
import torch

class NeedleInHaystackDataset(BaseDataGenerator):
    def __init__(
        self,
        tokenizer_name: str,
        max_seq_len: int = 512,
        tokens_to_generate: int = 128,
        template: str = "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text? The special magic {type_needle_v} for {query} mentioned in the provided text are",
        type_haystack: str = 'essay',
        type_needle_k: str = 'words',
        type_needle_v: str = 'numbers',
        num_needle_k: int = 1,
        num_needle_v: int = 1,
        num_needle_q: int = 1,
        random_seed: int = 42,
        remove_newline_tab: bool = False,
        local_files_only: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.tokens_to_generate = tokens_to_generate
        self.tokenizer = get_tokenizer(tokenizer_name, local_files_only=local_files_only)

        self.pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        assert (
            self.pad_token_id is not None
        ), "Either pad_token_id or eos_token_id should be defined"

        # Initialize parameters
        self.template = template
        self.type_haystack = type_haystack
        self.type_needle_k = type_needle_k
        self.type_needle_v = type_needle_v
        self.num_needle_k = max(num_needle_k, num_needle_q)
        self.num_needle_v = num_needle_v
        self.num_needle_q = num_needle_q
        self.random_seed = random_seed
        self.remove_newline_tab = remove_newline_tab

        # Set random seeds
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Load resources
        self.load_resources()

        # Set incremental values based on haystack type and max sequence length
        if self.type_haystack == 'essay':
            self.incremental = 50
        elif self.type_haystack in ['repeat', 'needle']:
            self.incremental = 25
        else:
            self.incremental = 25

        if self.type_haystack != 'essay' and self.max_seq_len < 4096:
            self.incremental = 5

        self.num_haystack = self.incremental

    def load_resources(self):
        # Load haystack data
        if self.type_haystack == 'essay':
            import os
            # Allow override via environment variable
            essay_path = os.environ.get("PAUL_GRAHAM_ESSAYS_PATH")
            if essay_path is None or not os.path.exists(essay_path):
                raise FileNotFoundError(
                    f"Essay file not found. Please set PAUL_GRAHAM_ESSAYS_PATH environment variable "
                    f"to point to a JSON file containing essay text, or use a different haystack type "
                    f"(e.g., 'repeat', 'needle', 'none')."
                )
            with open(essay_path, 'r') as f:
                essay = json.load(f)['text']
            self.haystack = re.sub(r'\s+', " ", essay).split(" ")
        elif self.type_haystack == 'repeat':
            self.haystack = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
        elif self.type_haystack == 'needle':
            self.haystack = "One of the special magic {type_needle_v} for {key} is: {value}."
        elif self.type_haystack == 'none':
            self.haystack = ""
        else:
            raise NotImplementedError(f'{self.type_haystack} is not implemented.')

        # Load words
        nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        self.words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
        self.words = sorted(list(set(self.words)))

        # Positions
        self.DEPTHS = list(
            np.round(np.linspace(0, 100, num=40, endpoint=True)).astype(int)
        )

    def generate_random_number(self, num_digits=7):
        lower_bound = 10 ** (num_digits - 1)
        upper_bound = 10 ** num_digits - 1
        return str(random.randint(lower_bound, upper_bound))

    def generate_random_word(self):
        word = random.choice(self.words)
        return word

    def generate_random_uuid(self):
        return str(uuid.UUID(int=random.getrandbits(128), version=4))

    def generate_random(self, type_needle):
        if type_needle == 'numbers':
            return self.generate_random_number()
        elif type_needle == 'words':
            return self.generate_random_word()
        elif type_needle == 'uuids':
            return self.generate_random_uuid()
        else:
            raise NotImplementedError(f'{type_needle} is not implemented.')
        

    def create_context(self, num_haystack, needles):
        if self.type_haystack == 'essay':
            text = " ".join(self.haystack[:num_haystack])
            document_sents = sent_tokenize(text.strip())
            random_positions = sorted([int(len(document_sents) * (depth / 100)) 
                                       for depth in random.sample(self.DEPTHS, len(needles))])
            insertion_positions = [0] + random_positions + [len(document_sents)]
            document_sents_list = []
            for i in range(1, len(insertion_positions)):
                last_pos = insertion_positions[i - 1]
                next_pos = insertion_positions[i]
                document_sents_list.append(" ".join(document_sents[last_pos:next_pos]))
                if i - 1 < len(needles):
                    document_sents_list.append(needles[i - 1])
            context = " ".join(document_sents_list)
        elif self.type_haystack in ['repeat', 'needle']:
            if self.type_haystack == 'repeat':
                sentences = [self.haystack] * num_haystack
            elif self.type_haystack == 'needle':
                sentences = [self.haystack.format(
                    type_needle_v=self.type_needle_v,
                    key=self.generate_random(self.type_needle_k),
                    value=self.generate_random(self.type_needle_v),
                    ) for _ in range(num_haystack)
                ]

            indexes = sorted(random.sample(range(num_haystack), len(needles)), reverse=True)
            for index, element in zip(indexes, needles):
                sentences.insert(index, element)
            context = "\n".join(sentences)
        else:
            context = "\n".join(needles)
        return context

    def generate_input_output(self, num_haystack):
        keys, values, needles = [], [], []
        for _ in range(self.num_needle_k):
            key = self.generate_random(self.type_needle_k)
            keys.append(key)
            value_list = []
            for _ in range(self.num_needle_v):
                v = self.generate_random(self.type_needle_v)
                value_list.append(v)
                needles.append(
                    f"One of the special magic {self.type_needle_v} for {key} is: {v}."
                )
            values.append(value_list)
        random.shuffle(needles)

        # Generate context
        context = self.create_context(num_haystack, needles)

        # Query and Answer
        indices = random.sample(range(self.num_needle_k), self.num_needle_q)
        queries = [keys[i] for i in indices]
        answers = [a for i in indices for a in values[i]]
        if len(queries) > 1:
            query = ', '.join(queries[:-1]) + ', and ' + queries[-1]
        else:
            query = queries[0]

        template = self.template
        type_needle_v = self.type_needle_v
        if self.num_needle_q * self.num_needle_v == 1:
            template = template.replace('Some', 'A')
            template = template.replace('are all', 'is')
            template = template.replace('are', 'is')
            template = template.replace('answers', 'answer')
            type_needle_v = type_needle_v.rstrip('s')  # remove plural 's'

        input_text = template.format(
            type_needle_v=type_needle_v,
            context=context,
            query=query,
        )

        return input_text, answers

    def __iter__(self):
        while True:
            num_haystack = self.num_haystack
            while num_haystack > 0:
                input_text, answers = self.generate_input_output(num_haystack)
                total_length = len(self.tokenizer.encode(input_text + ' '.join(answers))) + self.tokens_to_generate

                if total_length <= self.max_seq_len:
                    break
                else:
                    num_haystack -= self.incremental

            if num_haystack <= 0:
                raise Exception("Cannot fit sample within max_seq_len")
            
            if self.remove_newline_tab:
                input_text = ' '.join(input_text.replace('\n', ' ').replace('\t', ' ').split())

            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).squeeze()
            input_ids = torch.cat([input_ids, torch.tensor([self.pad_token_id] * (self.max_seq_len - input_ids.size(0)))])
            response_ids = self.tokenizer.encode(' '.join(answers), return_tensors="pt", add_special_tokens=False).squeeze()
            
            yield {"input_ids": input_ids, "response_ids": response_ids}
