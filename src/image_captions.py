"""
Based upon ImageCaptionLoader in LangChain version: langchain/document_loaders/image_captions.py
But accepts preloaded model to avoid slowness in use and CUDA forking issues

Loader that loads image captions
By default, the loader utilizes the pre-trained image captioning model.
https://huggingface.co/microsoft/Florence-2-base

"""
from typing import List, Union, Any, Tuple

import requests
from langchain.docstore.document import Document
from langchain_community.document_loaders import ImageCaptionLoader

from utils import get_device, NullContext, clear_torch_cache

from importlib.metadata import distribution, PackageNotFoundError

try:
    assert distribution('bitsandbytes') is not None
    have_bitsandbytes = True
except (PackageNotFoundError, AssertionError):
    have_bitsandbytes = False


from io import BytesIO
from pathlib import Path
from typing import Any, List, Tuple, Union

import requests
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader


class ImageCaptionLoader(BaseLoader):
    """Load image captions.

    By default, the loader utilizes the pre-trained
    Salesforce BLIP image captioning model.
    https://huggingface.co/Salesforce/blip-image-captioning-base
    """

    def __init__(
        self,
        images: Union[str, Path, bytes, List[Union[str, bytes, Path]]],
        caption_processor: str = "Salesforce/blip-image-captioning-base",
        caption_model: str = "Salesforce/blip-image-captioning-base",
    ):
        """Initialize with a list of image data (bytes) or file paths

        Args:
            images: Either a single image or a list of images. Accepts
                    image data (bytes) or file paths to images.
            caption_processor: The name of the pre-trained BLIP processor.
            caption_model: The name of the pre-trained BLIP model.
        """
        if isinstance(images, (str, Path, bytes)):
            self.images = [images]
        else:
            self.images = images

        self.caption_processor = caption_processor
        self.caption_model = caption_model

    def load(self) -> List[Document]:
        """Load from a list of image data or file paths"""
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            raise ImportError(
                "`transformers` package not found, please install with "
                "`pip install transformers`."
            )

        processor = BlipProcessor.from_pretrained(self.caption_processor)
        model = BlipForConditionalGeneration.from_pretrained(self.caption_model)

        results = []
        for image in self.images:
            caption, metadata = self._get_captions_and_metadata(
                model=model, processor=processor, image=image
            )
            doc = Document(page_content=caption, metadata=metadata)
            results.append(doc)

        return results

    def _get_captions_and_metadata(
        self, model: Any, processor: Any, image: Union[str, Path, bytes]
    ) -> Tuple[str, dict]:
        """Helper function for getting the captions and metadata of an image."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "`PIL` package not found, please install with `pip install pillow`"
            )

        image_source = image  # Save the original source for later reference

        try:
            if isinstance(image, bytes):
                image = Image.open(BytesIO(image)).convert("RGB")
            elif isinstance(image, str) and (
                image.startswith("http://") or image.startswith("https://")
            ):
                image = Image.open(requests.get(image, stream=True).raw).convert("RGB")
            else:
                image = Image.open(image).convert("RGB")
        except Exception:
            if isinstance(image_source, bytes):
                msg = "Could not get image data from bytes"
            else:
                msg = f"Could not get image data for {image_source}"
            raise ValueError(msg)

        inputs = processor(image, "an image of", return_tensors="pt")
        output = model.generate(**inputs)

        caption: str = processor.decode(output[0])
        if isinstance(image_source, bytes):
            metadata: dict = {"image_source": "Image bytes provided"}
        else:
            metadata = {"image_path": str(image_source)}

        return caption, metadata


class H2OImageCaptionLoader(ImageCaptionLoader):
    """Loader that loads the captions of an image"""

    def __init__(self, path_images: Union[str, List[str]] = None,
                 caption_processor: str = None,
                 caption_model: str = None,
                 caption_gpu=True,
                 load_in_8bit=True,
                 # True doesn't seem to work, even though https://huggingface.co/Salesforce/blip2-flan-t5-xxl#in-8-bit-precision-int8
                 load_half=False,
                 load_gptq='',
                 load_awq='',
                 load_exllama=False,
                 use_safetensors=False,
                 revision=None,
                 min_new_tokens=512,
                 max_tokens=50,
                 gpu_id='auto'):
        if caption_model is None or caption_model is None:
            caption_processor = "microsoft/Florence-2-base"
            caption_model = "microsoft/Florence-2-base"

        super().__init__(path_images, caption_processor, caption_model)
        self.caption_processor = caption_processor
        self.caption_model = caption_model
        self.processor = None
        self.model = None
        self.caption_gpu = caption_gpu
        self.context_class = NullContext
        self.load_in_8bit = load_in_8bit and have_bitsandbytes  # only for blip2
        self.load_half = load_half
        self.load_gptq = load_gptq
        self.load_awq = load_awq
        self.load_exllama = load_exllama
        self.use_safetensors = use_safetensors
        self.revision = revision
        self.gpu_id = gpu_id
        # default prompt
        self.prompt = "image of"
        self.min_new_tokens = min_new_tokens
        self.max_tokens = max_tokens

        self.device = 'cpu'
        self.device_map = {"": 'cpu'}
        self.set_context()

    def set_context(self):
        if get_device() == 'cuda' and self.caption_gpu:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpus > 0:
                self.context_class = torch.device
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'
        if self.caption_gpu:
            if self.gpu_id == 'auto':
                # blip2 has issues with multi-GPU.  Error says need to somehow set language model in device map
                # device_map = 'auto'
                self.device_map = {"": 0}
            else:
                if self.device == 'cuda':
                    self.device_map = {"": 'cuda:%d' % self.gpu_id}
                else:
                    self.device_map = {"": 'cpu'}
        else:
            self.device_map = {"": 'cpu'}

    def load_model(self):
        try:
            import transformers
        except ImportError:
            raise ValueError(
                "`transformers` package not found, please install with "
                "`pip install transformers`."
            )
        self.set_context()
        if self.model:
            if not self.load_in_8bit and str(self.model.device) != self.device_map['']:
                self.model.to(self.device)
            return self
        import torch
        with torch.no_grad():
            with self.context_class(self.device):
                context_class_cast = NullContext if self.device == 'cpu' else torch.autocast
                with context_class_cast(self.device):
                    if 'blip2' in self.caption_processor.lower():
                        from transformers import Blip2Processor, Blip2ForConditionalGeneration
                        if self.load_half and not self.load_in_8bit:
                            self.processor = Blip2Processor.from_pretrained(self.caption_processor,
                                                                            device_map=self.device_map).half()
                            self.model = Blip2ForConditionalGeneration.from_pretrained(self.caption_model,
                                                                                       device_map=self.device_map).half()
                        else:
                            self.processor = Blip2Processor.from_pretrained(self.caption_processor,
                                                                            load_in_8bit=self.load_in_8bit,
                                                                            device_map=self.device_map,
                                                                            )
                            self.model = Blip2ForConditionalGeneration.from_pretrained(self.caption_model,
                                                                                       load_in_8bit=self.load_in_8bit,
                                                                                       device_map=self.device_map)
                    elif 'blip' in self.caption_processor.lower():
                        from transformers import BlipForConditionalGeneration, BlipProcessor
                        self.load_half = False  # not supported
                        self.processor = BlipProcessor.from_pretrained(self.caption_processor, device_map=self.device_map)
                        self.model = BlipForConditionalGeneration.from_pretrained(self.caption_model,
                                                                                  device_map=self.device_map)
                    else:
                        from transformers import AutoModelForCausalLM, AutoProcessor
                        self.load_half = False  # not supported
                        self.processor = AutoProcessor.from_pretrained(self.caption_processor, device_map=self.device_map,
                        trust_remote_code=True)
                        self.model = AutoModelForCausalLM.from_pretrained(self.caption_model, device_map=self.device_map,
                        trust_remote_code=True)
        return self

    def set_image_paths(self, path_images: Union[str, List[str]]):
        """
        Load from a list of image files
        """
        if isinstance(path_images, str):
            self.image_paths = [path_images]
        else:
            self.image_paths = path_images

    def load(self, prompt=None) -> List[Document]:
        if self.processor is None or self.model is None:
            self.load_model()
        results = []
        for path_image in self.image_paths:
            caption, metadata = self._get_captions_and_metadata(
                model=self.model, processor=self.processor, path_image=path_image,
                prompt=prompt,
            )
            doc = Document(page_content=caption, metadata=metadata)
            results.append(doc)

        return results

    def unload_model(self):
        if hasattr(self, 'model') and hasattr(self.model, 'cpu'):
            self.model.cpu()
            clear_torch_cache()

    def _get_captions_and_metadata(
            self, model: Any, processor: Any, path_image: str,
            prompt=None) -> Tuple[str, dict]:
        """
        Helper function for getting the captions and metadata of an image
        """
        if prompt is None:
            prompt = self.prompt
        try:
            from PIL import Image
        except ImportError:
            raise ValueError(
                "`PIL` package not found, please install with `pip install pillow`"
            )

        try:
            if path_image.startswith("http://") or path_image.startswith("https://"):
                image = Image.open(requests.get(path_image, stream=True).raw).convert(
                    "RGB"
                )
            else:
                image = Image.open(path_image).convert("RGB")
        except Exception:
            raise ValueError(f"Could not get image data for {path_image}")

        import torch
        with torch.no_grad():
            with self.context_class(self.device):
                context_class_cast = NullContext if self.device == 'cpu' else torch.autocast
                with context_class_cast(self.device):
                    extra_kwargs = {}

                    if isinstance(self.caption_model, str) and 'florence' in self.caption_model.lower():
                        caption_detail_task_map = {
                            "low": "<CAPTION>",
                            "medium": "<DETAILED_CAPTION>",
                            "high": "<MORE_DETAILED_CAPTION>",
                        }
                        task_prompt = caption_detail_task_map[
                           'high' if 'large' in self.caption_model else 'medium'
                        ]
                        num_beams = 3 if 'large' in self.caption_model else 1
                        extra_kwargs.update(dict(num_beams=num_beams))
                        if prompt and False:
                            prompt = task_prompt + prompt
                        else:
                            prompt = task_prompt

                    if isinstance(self.caption_model, str) and 'blip' in self.caption_model:
                        min_length = len(prompt) // 4 + self.min_new_tokens
                        self.max_tokens = max(self.max_tokens, min_length)
                        extra_kwargs.update(dict(min_length=min_length))
                        if self.load_half:
                            # FIXME: RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
                            inputs = processor(image, prompt, return_tensors="pt")  # .half()
                        else:
                            inputs = processor(image, prompt, return_tensors="pt")
                    else:
                        inputs = processor(text=prompt, images=image, return_tensors="pt")
                    inputs.to(model.device)
                    output = model.generate(**inputs, max_length=self.max_tokens, **extra_kwargs)

                    caption: str = processor.decode(output[0], skip_special_tokens=True)
                    if isinstance(self.caption_model, str) and 'blip' in self.caption_model:
                        prompti = caption.find(prompt)
                        if prompti >= 0:
                            caption = caption[prompti + len(prompt):]
                    elif isinstance(self.caption_model, str) and 'florence' in self.caption_model.lower():
                        parsed_answer = processor.post_process_generation(
                            caption, task=task_prompt, image_size=(image.width, image.height)
                        )
                        caption: str = parsed_answer[task_prompt].strip()

                    metadata: dict = {"image_path": path_image}

        return caption, metadata
