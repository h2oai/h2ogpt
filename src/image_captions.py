"""
Based upon ImageCaptionLoader in LangChain version: langchain/document_loaders/image_captions.py
But accepts preloaded model to avoid slowness in use and CUDA forking issues

Loader that loads image captions
By default, the loader utilizes the pre-trained BLIP image captioning model.
https://huggingface.co/Salesforce/blip-image-captioning-base

"""
from typing import List, Union, Any, Tuple

import requests
from langchain.docstore.document import Document
from langchain.document_loaders import ImageCaptionLoader

from utils import get_device, NullContext

import pkg_resources

try:
    assert pkg_resources.get_distribution('bitsandbytes') is not None
    have_bitsandbytes = True
except (pkg_resources.DistributionNotFound, AssertionError):
    have_bitsandbytes = False


class H2OImageCaptionLoader(ImageCaptionLoader):
    """Loader that loads the captions of an image"""

    def __init__(self, path_images: Union[str, List[str]] = None,
                 blip_processor: str = None,
                 blip_model: str = None,
                 caption_gpu=True,
                 load_in_8bit=True,
                 # True doesn't seem to work, even though https://huggingface.co/Salesforce/blip2-flan-t5-xxl#in-8-bit-precision-int8
                 load_half=False,
                 load_gptq='',
                 load_exllama=False,
                 use_safetensors=False,
                 revision=None,
                 min_new_tokens=20,
                 max_tokens=50):
        if blip_model is None or blip_model is None:
            blip_processor = "Salesforce/blip-image-captioning-base"
            blip_model = "Salesforce/blip-image-captioning-base"

        super().__init__(path_images, blip_processor, blip_model)
        self.blip_processor = blip_processor
        self.blip_model = blip_model
        self.processor = None
        self.model = None
        self.caption_gpu = caption_gpu
        self.context_class = NullContext
        self.device = 'cpu'
        self.load_in_8bit = load_in_8bit and have_bitsandbytes  # only for blip2
        self.load_half = load_half
        self.load_gptq = load_gptq
        self.load_exllama = load_exllama
        self.use_safetensors = use_safetensors
        self.revision = revision
        self.gpu_id = 'auto'
        # default prompt
        self.prompt = "image of"
        self.min_new_tokens = min_new_tokens
        self.max_tokens = max_tokens

    def set_context(self):
        if get_device() == 'cuda' and self.caption_gpu:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
            if n_gpus > 0:
                self.context_class = torch.device
                self.device = 'cuda'

    def load_model(self):
        try:
            import transformers
        except ImportError:
            raise ValueError(
                "`transformers` package not found, please install with "
                "`pip install transformers`."
            )
        self.set_context()
        if self.caption_gpu:
            if self.gpu_id == 'auto':
                # blip2 has issues with multi-GPU.  Error says need to somehow set language model in device map
                # device_map = 'auto'
                device_map = {"": 0}
            else:
                if self.device == 'cuda':
                    device_map = {"": self.gpu_id}
                else:
                    device_map = {"": 'cpu'}
        else:
            device_map = {"": 'cpu'}
        import torch
        with torch.no_grad():
            with self.context_class(self.device):
                context_class_cast = NullContext if self.device == 'cpu' else torch.autocast
                with context_class_cast(self.device):
                    if 'blip2' in self.blip_processor.lower():
                        from transformers import Blip2Processor, Blip2ForConditionalGeneration
                        if self.load_half and not self.load_in_8bit:
                            self.processor = Blip2Processor.from_pretrained(self.blip_processor,
                                                                            device_map=device_map).half()
                            self.model = Blip2ForConditionalGeneration.from_pretrained(self.blip_model,
                                                                                       device_map=device_map).half()
                        else:
                            self.processor = Blip2Processor.from_pretrained(self.blip_processor,
                                                                            load_in_8bit=self.load_in_8bit,
                                                                            device_map=device_map,
                                                                            )
                            self.model = Blip2ForConditionalGeneration.from_pretrained(self.blip_model,
                                                                                       load_in_8bit=self.load_in_8bit,
                                                                                       device_map=device_map)
                    else:
                        from transformers import BlipForConditionalGeneration, BlipProcessor
                        self.load_half = False  # not supported
                        if self.caption_gpu:
                            if device_map == 'auto':
                                # Blip doesn't support device_map='auto'
                                if self.device == 'cuda':
                                    if self.gpu_id == 'auto':
                                        device_map = {"": 0}
                                    else:
                                        device_map = {"": self.gpu_id}
                                else:
                                    device_map = {"": 'cpu'}
                        else:
                            device_map = {"": 'cpu'}
                        self.processor = BlipProcessor.from_pretrained(self.blip_processor, device_map=device_map)
                        self.model = BlipForConditionalGeneration.from_pretrained(self.blip_model,
                                                                                  device_map=device_map)
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
                    if self.load_half:
                        inputs = processor(image, prompt, return_tensors="pt").half()
                    else:
                        inputs = processor(image, prompt, return_tensors="pt")
                    min_length = len(prompt) // 4 + self.min_new_tokens
                    self.max_tokens = max(self.max_tokens, min_length)
                    output = model.generate(**inputs, min_length=min_length, max_length=self.max_tokens)

                    caption: str = processor.decode(output[0], skip_special_tokens=True)
                    prompti = caption.find(prompt)
                    if prompti >= 0:
                        caption = caption[prompti + len(prompt):]
                    metadata: dict = {"image_path": path_image}

        return caption, metadata
