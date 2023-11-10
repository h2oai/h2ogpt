"""
Based upon ImageCaptionLoader in LangChain version: langchain/document_loaders/image_captions.py
But accepts preloaded model to avoid slowness in use and CUDA forking issues

Loader that uses Pix2Struct models to image caption

"""
from typing import List, Union, Any, Tuple

from langchain.docstore.document import Document
from langchain.document_loaders import ImageCaptionLoader
from utils import get_device, clear_torch_cache
from PIL import Image


class H2OPix2StructLoader(ImageCaptionLoader):
    """Loader that extracts text from images"""

    def __init__(self, path_images: Union[str, List[str]] = None, model_type="google/pix2struct-textcaps-base",
                 max_new_tokens=50):
        super().__init__(path_images)
        self._pix2struct_model = None
        self._model_type = model_type
        self._max_new_tokens = max_new_tokens

    def set_context(self):
        if get_device() == 'cuda':
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpus > 0:
                self.context_class = torch.device
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'

    def load_model(self):
        try:
            from transformers import AutoProcessor, Pix2StructForConditionalGeneration
        except ImportError:
            raise ValueError(
                "`transformers` package not found, please install with "
                "`pip install transformers`."
            )
        if self._pix2struct_model:
            self._pix2struct_model = self._pix2struct_model.to(self.device)
            return self
        self.set_context()
        self._pix2struct_processor = AutoProcessor.from_pretrained(self._model_type)
        self._pix2struct_model = Pix2StructForConditionalGeneration.from_pretrained(self._model_type).to(self.device)
        return self

    def unload_model(self):
        if hasattr(self._pix2struct_model, 'cpu'):
            self._pix2struct_model.cpu()
            clear_torch_cache()

    def set_image_paths(self, path_images: Union[str, List[str]]):
        """
        Load from a list of image files
        """
        if isinstance(path_images, str):
            self.image_paths = [path_images]
        else:
            self.image_paths = path_images

    def load(self, prompt=None) -> List[Document]:
        if self._pix2struct_model is None:
            self.load_model()
        results = []
        for path_image in self.image_paths:
            caption, metadata = self._get_captions_and_metadata(
                processor=self._pix2struct_processor, model=self._pix2struct_model, path_image=path_image
            )
            doc = Document(page_content=caption, metadata=metadata)
            results.append(doc)

        return results

    def _get_captions_and_metadata(
            self, processor: Any, model: Any, path_image: str) -> Tuple[str, dict]:
        """
        Helper function for getting the captions and metadata of an image
        """
        try:
            image = Image.open(path_image)
        except Exception:
            raise ValueError(f"Could not get image data for {path_image}")
        inputs = self._pix2struct_processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)
        generated_ids = self._pix2struct_model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        generated_text = self._pix2struct_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        metadata: dict = {"image_path": path_image}
        return generated_text, metadata
