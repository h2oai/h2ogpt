"""
Based upon ImageCaptionLoader in LangChain version: langchain/document_loaders/image_captions.py
But accepts preloaded model to avoid slowness in use and CUDA forking issues

Loader that loads image captions
By default, the loader utilizes the pre-trained BLIP image captioning model.
https://huggingface.co/Salesforce/blip-image-captioning-base

"""
from typing import List, Union

from langchain.docstore.document import Document
from langchain.document_loaders import ImageCaptionLoader

from utils import get_device, NullContext


class H2OImageCaptionLoader(ImageCaptionLoader):
    """Loader that loads the captions of an image"""

    def __init__(self, path_images: Union[str, List[str]] = None,
                 blip_processor: str = "Salesforce/blip-image-captioning-base",
                 blip_model: str = "Salesforce/blip-image-captioning-base",
                 caption_gpu=True):
        super().__init__(path_images, blip_processor, blip_model)
        self.blip_processor = blip_processor
        self.blip_model = blip_model
        self.processor = None
        self.model = None
        self.caption_gpu = caption_gpu
        self.context_class = NullContext
        self.device = 'cpu'

    def set_context(self):
        if get_device() == 'cuda' and self.caption_gpu:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
            if n_gpus > 0:
                self.context_class = torch.device
                self.device = 'cuda'

    def load_model(self):
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError:
            raise ValueError(
                "`transformers` package not found, please install with "
                "`pip install transformers`."
            )
        with self.context_class(self.device):
            self.processor = BlipProcessor.from_pretrained(self.blip_processor)
            self.model = BlipForConditionalGeneration.from_pretrained(self.blip_model)

    def set_image_paths(self, path_images: Union[str, List[str]]):
        """
        Load from a list of image files
        """
        if isinstance(path_images, str):
            self.image_paths = [path_images]
        else:
            self.image_paths = path_images

    def load(self) -> List[Document]:
        if self.processor is None or self.model is None:
            self.load_model()
        results = []
        for path_image in self.image_paths:
            caption, metadata = self._get_captions_and_metadata(
                model=self.model, processor=self.processor, path_image=path_image
            )
            doc = Document(page_content=caption, metadata=metadata)
            results.append(doc)

        return results
