# https://github.com/michaelfeil/infinity/issues/125#issuecomment-2101696124
# This script is based on the modifications from https://github.com/UKPLab/sentence-transformers
import importlib
import json
import os
from collections import OrderedDict
from typing import Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer
from torch import Tensor, nn
from tqdm.autonotebook import trange
from transformers import AutoConfig, AutoTokenizer
from sentence_transformers.util import disabled_tqdm
from huggingface_hub import snapshot_download


def batch_to_device(batch, target_device: str):
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class INSTRUCTORPooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding.
    This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max/cls. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but divide by sqrt(input_length).
    :param pooling_mode_weightedmean_tokens: Perform (position) weighted mean pooling,
    see https://arxiv.org/abs/2202.08904
    :param pooling_mode_lasttoken: Perform last token pooling,
    see https://arxiv.org/abs/2202.08904 & https://arxiv.org/abs/2201.10005
    """

    def __init__(
        self,
        word_embedding_dimension: int,
        pooling_mode: Union[str, None] = None,
        pooling_mode_cls_token: bool = False,
        pooling_mode_max_tokens: bool = False,
        pooling_mode_mean_tokens: bool = True,
        pooling_mode_mean_sqrt_len_tokens: bool = False,
        pooling_mode_weightedmean_tokens: bool = False,
        pooling_mode_lasttoken: bool = False,
    ):
        super().__init__()

        self.config_keys = [
            "word_embedding_dimension",
            "pooling_mode_cls_token",
            "pooling_mode_mean_tokens",
            "pooling_mode_max_tokens",
            "pooling_mode_mean_sqrt_len_tokens",
            "pooling_mode_weightedmean_tokens",
            "pooling_mode_lasttoken",
        ]

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ["mean", "max", "cls", "weightedmean", "lasttoken"]
            pooling_mode_cls_token = pooling_mode == "cls"
            pooling_mode_max_tokens = pooling_mode == "max"
            pooling_mode_mean_tokens = pooling_mode == "mean"
            pooling_mode_weightedmean_tokens = pooling_mode == "weightedmean"
            pooling_mode_lasttoken = pooling_mode == "lasttoken"

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weightedmean_tokens = pooling_mode_weightedmean_tokens
        self.pooling_mode_lasttoken = pooling_mode_lasttoken

        pooling_mode_multiplier = sum(
            [
                pooling_mode_cls_token,
                pooling_mode_max_tokens,
                pooling_mode_mean_tokens,
                pooling_mode_mean_sqrt_len_tokens,
                pooling_mode_weightedmean_tokens,
                pooling_mode_lasttoken,
            ]
        )
        self.pooling_output_dimension = (
            pooling_mode_multiplier * word_embedding_dimension
        )

    def __repr__(self):
        return f"Pooling({self.get_config_dict()})"

    def get_pooling_mode_str(self) -> str:
        """
        Returns the pooling mode as string
        """
        modes = []
        if self.pooling_mode_cls_token:
            modes.append("cls")
        if self.pooling_mode_mean_tokens:
            modes.append("mean")
        if self.pooling_mode_max_tokens:
            modes.append("max")
        if self.pooling_mode_mean_sqrt_len_tokens:
            modes.append("mean_sqrt_len_tokens")
        if self.pooling_mode_weightedmean_tokens:
            modes.append("weightedmean")
        if self.pooling_mode_lasttoken:
            modes.append("lasttoken")

        return "+".join(modes)

    def forward(self, features):
        # print(features.keys())
        token_embeddings = features["token_embeddings"]
        attention_mask = features["attention_mask"]

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get(
                "cls_token_embeddings", token_embeddings[:, 0]
            )  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            token_embeddings[
                input_mask_expanded == 0
            ] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = (
                    features["token_weights_sum"]
                    .unsqueeze(-1)
                    .expand(sum_embeddings.size())
                )
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
        if self.pooling_mode_weightedmean_tokens:
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            # token_embeddings shape: bs, seq, hidden_dim
            weights = (
                torch.arange(start=1, end=token_embeddings.shape[1] + 1)
                .unsqueeze(0)
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
                .to(token_embeddings.device)
            )
            assert weights.shape == token_embeddings.shape == input_mask_expanded.shape
            input_mask_expanded = input_mask_expanded * weights

            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            # If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if "token_weights_sum" in features:
                sum_mask = (
                    features["token_weights_sum"]
                    .unsqueeze(-1)
                    .expand(sum_embeddings.size())
                )
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)
            output_vectors.append(sum_embeddings / sum_mask)
        if self.pooling_mode_lasttoken:
            batch_size, _, hidden_dim = token_embeddings.shape
            # attention_mask shape: (bs, seq_len)
            # Get shape [bs] indices of the last token (i.e. the last token for each batch item)
            # argmin gives us the index of the first 0 in the attention mask;
            # We get the last 1 index by subtracting 1
            gather_indices = (
                torch.argmin(attention_mask, 1, keepdim=False) - 1
            )  # Shape [bs]

            # There are empty sequences, where the index would become -1 which will crash
            gather_indices = torch.clamp(gather_indices, min=0)

            # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
            gather_indices = gather_indices.unsqueeze(1)
            assert gather_indices.shape == (batch_size, 1, hidden_dim)

            # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
            # Actually no need for the attention mask as we gather the last token where attn_mask = 1
            # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
            # use the attention mask to ignore them again
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            embedding = torch.gather(
                token_embeddings * input_mask_expanded, 1, gather_indices
            ).squeeze(dim=1)
            output_vectors.append(embedding)

        output_vector = torch.cat(output_vectors, 1)
        features.update({"sentence_embedding": output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(
            os.path.join(output_path, "config.json"), "w", encoding="UTF-8"
        ) as config_file:
            json.dump(self.get_config_dict(), config_file, indent=2)

    @staticmethod
    def load(input_path):
        with open(
            os.path.join(input_path, "config.json"), encoding="UTF-8"
        ) as config_file:
            config = json.load(config_file)

        return INSTRUCTORPooling(**config)


def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError:
        msg = f"{dotted_path} doesn't look like a module path"
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = f"Module {module_path} does not define a {class_name} attribute/class"
        raise ImportError(msg)


class INSTRUCTORTransformer(Transformer):
    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length=None,
        model_args=None,
        cache_dir=None,
        tokenizer_args=None,
        do_lower_case: bool = False,
        tokenizer_name_or_path: Union[str, None] = None,
        load_model: bool = True,
    ):
        super().__init__(model_name_or_path)
        if model_args is None:
            model_args = {}
        if tokenizer_args is None:
            tokenizer_args = {}
        self.config_keys = ["max_seq_length", "do_lower_case"]
        self.do_lower_case = do_lower_case
        self.model_name_or_path = model_name_or_path
        if model_name_or_path == "bi-contriever":
            model_name_or_path = "facebook/contriever"
        if model_name_or_path.startswith("bigtr"):
            model_name_or_path = model_name_or_path.split("#")[1]
        if "bigtr" in model_name_or_path and os.path.isdir(model_name_or_path):
            config = AutoConfig.from_pretrained(
                os.path.join(model_name_or_path, "with_prompt"),
                **model_args,
                cache_dir=cache_dir,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_name_or_path, **model_args, cache_dir=cache_dir
            )

        if load_model:
            self._load_model(self.model_name_or_path, config, cache_dir, **model_args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path
            if tokenizer_name_or_path is not None
            else model_name_or_path,
            cache_dir=cache_dir,
            **tokenizer_args,
        )

        if max_seq_length is None:
            if (
                hasattr(self.auto_model, "config")
                and hasattr(self.auto_model.config, "max_position_embeddings")
                and hasattr(self.tokenizer, "model_max_length")
            ):
                max_seq_length = min(
                    self.auto_model.config.max_position_embeddings,
                    self.tokenizer.model_max_length,
                )

        self.max_seq_length = max_seq_length
        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def forward(self, features):
        input_features = {
            "input_ids": features["input_ids"],
            "attention_mask": features["attention_mask"],
        }
        if "token_type_ids" in features:
            input_features["token_type_ids"] = features["token_type_ids"]

        instruction_mask = features["instruction_mask"]
        output_states = self.auto_model(**input_features, return_dict=False)
        output_tokens = output_states[0]
        attention_mask = features["attention_mask"]
        instruction_mask = features["instruction_mask"]
        attention_mask = attention_mask * instruction_mask
        features.update(
            {"token_embeddings": output_tokens, "attention_mask": attention_mask}
        )

        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if (
                len(output_states) < 3
            ):  # Some models only output last_hidden_states and all_hidden_states
                all_layer_idx = 1
            hidden_states = output_states[all_layer_idx]
            features.update({"all_layer_embeddings": hidden_states})

        return features

    @staticmethod
    def load(input_path: str):
        # Old classes used other config names than 'sentence_bert_config.json'
        for config_name in [
            "sentence_bert_config.json",
            "sentence_roberta_config.json",
            "sentence_distilbert_config.json",
            "sentence_camembert_config.json",
            "sentence_albert_config.json",
            "sentence_xlm-roberta_config.json",
            "sentence_xlnet_config.json",
        ]:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break

        with open(sbert_config_path, encoding="UTF-8") as config_file:
            config = json.load(config_file)
        return INSTRUCTORTransformer(model_name_or_path=input_path, **config)

    def tokenize(self, texts):
        """
        Tokenizes a text and maps tokens to token-ids
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
            to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

            # Lowercase
            if self.do_lower_case:
                to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

            input_features = self.tokenizer(
                *to_tokenize,
                padding="max_length",
                truncation="longest_first",
                return_tensors="pt",
                max_length=self.max_seq_length,
            )

        elif isinstance(texts[0], list):
            assert isinstance(texts[0][1], str)
            assert (
                len(texts[0]) == 2
            ), "The input should have both instruction and input text"

            instructions = []
            instruction_prepended_input_texts = []
            for pair in texts:
                instruction = pair[0].strip()
                text = pair[1].strip()
                if self.do_lower_case:
                    instruction = instruction.lower()
                    text = text.lower()
                instructions.append(instruction)
                instruction_prepended_input_texts.append("".join([instruction, text]))

            input_features = self.tokenize(instruction_prepended_input_texts)
            instruction_features = self.tokenize(instructions)
            input_features = INSTRUCTOR.prepare_input_features(
                input_features, instruction_features
            )
        else:
            raise ValueError("not support other modes")

        output.update(input_features)
        return output


class INSTRUCTOR(SentenceTransformer):
    @staticmethod
    def prepare_input_features(
        input_features, instruction_features, return_data_type: str = "pt"
    ):
        if return_data_type == "np":
            input_features["attention_mask"] = torch.from_numpy(
                input_features["attention_mask"]
            )
            instruction_features["attention_mask"] = torch.from_numpy(
                instruction_features["attention_mask"]
            )

        input_attention_mask_shape = input_features["attention_mask"].shape
        instruction_attention_mask = instruction_features["attention_mask"]

        # reducing the attention length by 1 in order to omit the attention corresponding to the end_token
        instruction_attention_mask = instruction_attention_mask[:, 1:]

        # creating instruction attention matrix equivalent to the size of the input attention matrix
        expanded_instruction_attention_mask = torch.zeros(
            input_attention_mask_shape, dtype=torch.int64
        )
        # assigning the the actual instruction attention matrix to the expanded_instruction_attention_mask
        # eg:
        # instruction_attention_mask: 3x3
        #  [[1,1,1],
        #   [1,1,0],
        #   [1,0,0]]
        # expanded_instruction_attention_mask: 3x4
        #  [[1,1,1,0],
        #   [1,1,0,0],
        #   [1,0,0,0]]
        expanded_instruction_attention_mask[
            : instruction_attention_mask.size(0), : instruction_attention_mask.size(1)
        ] = instruction_attention_mask

        # In the pooling layer we want to consider only the tokens corresponding to the input text
        # and not the instruction. This is achieved by inverting the
        # attention_mask corresponding to the instruction.
        expanded_instruction_attention_mask = 1 - expanded_instruction_attention_mask
        input_features["instruction_mask"] = expanded_instruction_attention_mask
        if return_data_type == "np":
            input_features["attention_mask"] = input_features["attention_mask"].numpy()
            instruction_features["attention_mask"] = instruction_features[
                "attention_mask"
            ].numpy()
        return input_features

    def smart_batching_collate(self, batch):
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
            labels.append(example.label)

        labels = torch.tensor(labels)
        batched_input_features = []

        for idx in range(num_texts):
            assert isinstance(texts[idx][0], list)
            assert (
                len(texts[idx][0]) == 2
            ), "The input should have both instruction and input text"

            num = len(texts[idx])
            instructions = []
            instruction_prepended_input_texts = []
            for local_idx in range(num):
                assert len(texts[idx][local_idx]) == 2
                instructions.append(texts[idx][local_idx][0])
                instruction_prepended_input_texts.append("".join(texts[idx][local_idx]))
                assert isinstance(instructions[-1], str)
                assert isinstance(instruction_prepended_input_texts[-1], str)

            input_features = self.tokenize(instruction_prepended_input_texts)
            instruction_features = self.tokenize(instructions)
            input_features = INSTRUCTOR.prepare_input_features(
                input_features, instruction_features
            )
            batched_input_features.append(input_features)

        return batched_input_features, labels

    def _load_sbert_model(self, model_path, token=None, cache_folder=None, revision=None, trust_remote_code=False, local_files_only=False, model_kwargs={}, tokenizer_kwargs={}, config_kwargs={}):
        """
        Loads a full sentence-transformers model
        """
        if os.path.isdir(model_path):
            # If model_path is a local directory, load the model directly
            model_path = str(model_path)
        else:
            # If model_path is a Hugging Face repository ID, download the model
            download_kwargs = {
                "repo_id": model_path,
                "revision": revision,
                "library_name": "sentence-transformers",
                "token": token,
                "cache_dir": cache_folder,
                "tqdm_class": disabled_tqdm,
            }
            model_path = snapshot_download(**download_kwargs)

        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = os.path.join(
            model_path, "config_sentence_transformers.json"
        )
        if os.path.exists(config_sentence_transformers_json_path):
            with open(
                config_sentence_transformers_json_path, encoding="UTF-8"
            ) as config_file:
                self._model_config = json.load(config_file)

        # Check if a readme exists
        model_card_path = os.path.join(model_path, "README.md")
        if os.path.exists(model_card_path):
            try:
                with open(model_card_path, encoding="utf8") as config_file:
                    self._model_card_text = config_file.read()
            except:
                pass

        # Load the modules of sentence transformer
        modules_json_path = os.path.join(model_path, "modules.json")
        with open(modules_json_path, encoding="UTF-8") as config_file:
            modules_config = json.load(config_file)

        modules = OrderedDict()
        for module_config in modules_config:
            if module_config["idx"] == 0:
                module_class = INSTRUCTORTransformer
            elif module_config["idx"] == 1:
                module_class = INSTRUCTORPooling
            else:
                module_class = import_from_string(module_config["type"])
            module = module_class.load(os.path.join(model_path, module_config["path"]))
            modules[module_config["name"]] = module

        return modules

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        show_progress_bar: Union[bool, None] = None,
        output_value: str = "sentence_embedding",
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        device: Union[str, None] = None,
        normalize_embeddings: bool = False,
    ):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings.
        Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors.
        Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return.
        Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
        In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor,
           a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = False

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != "sentence_embedding":
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(
            sentences, "__len__"
        ):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self._target_device

        self.to(device)

        all_embeddings = []
        if isinstance(sentences[0], list):
            lengths = []
            for sen in sentences:
                lengths.append(-self._text_length(sen[1]))
            length_sorted_idx = np.argsort(lengths)
        else:
            length_sorted_idx = np.argsort(
                [-self._text_length(sen) for sen in sentences]
            )
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar
        ):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == "token_embeddings":
                    embeddings = []
                    for token_emb, attention in zip(
                        out_features[output_value], out_features["attention_mask"]
                    ):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0 : last_mask_id + 1])
                elif output_value is None:  # Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features["sentence_embedding"])):
                        row = {
                            name: out_features[name][sent_idx] for name in out_features
                        }
                        embeddings.append(row)
                else:  # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(
                            embeddings, p=2, dim=1
                        )

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings