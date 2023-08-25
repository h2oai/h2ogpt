"""
Based upon ImageCaptionLoader in LangChain version: langchain/document_loaders/image_captions.py
But accepts preloaded model to avoid slowness in use and CUDA forking issues

Loader that uses H2O DocTR OCR models to extract text from images

"""
from typing import List, Union, Any, Tuple

import requests
from langchain.docstore.document import Document
from langchain.document_loaders import ImageCaptionLoader

from utils import get_device, NullContext


class H2OOCRLoader(ImageCaptionLoader):
    """Loader that extracts text from images"""

    def __init__(self, path_images: Union[str, List[str]] = None):
        super().__init__(path_images)
        self._ocr_model = None

    def set_context(self):
        if get_device() == 'cuda':
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
            if n_gpus > 0:
                self.context_class = torch.device
                self.device = 'cuda'

    def load_model(self):
        try:
            from doctr.models.zoo import ocr_predictor
        except ImportError:
            raise ValueError(
                "`doctr` package not found, please install with "
                "`pip install git+https://github.com/h2oai/doctr.git[torch]`."
            )
        self.set_context()
        self._ocr_model = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_efficientnetv2_mV2", pretrained=True).to(self.device)
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
        if self._ocr_model is None:
            self.load_model()
        results = []
        for path_image in self.image_paths:
            caption, metadata = self._get_captions_and_metadata(
                model=self._ocr_model, path_image=path_image
            )
            doc = Document(page_content=caption, metadata=metadata)
            results.append(doc)

        return results

    def _get_captions_and_metadata(
            self, model: Any, path_image: str) -> Tuple[str, dict]:
        """
        Helper function for getting the captions and metadata of an image
        """
        try:
            from doctr.io import DocumentFile
        except ImportError:
            raise ValueError(
                "`doctr` package not found, please install with "
                "`pip install git+https://github.com/h2oai/doctr.git`."
            )
        try:
            image = DocumentFile.from_images(path_image)[0]
        except Exception:
            raise ValueError(f"Could not get image data for {path_image}")
        ocr_output = model([image])
        words = []
        for block_num, block in enumerate(ocr_output.pages[0].blocks):
            for line_num, line in enumerate(block.lines):
                for word_num, word in enumerate(line.words):
                    if not (word.value or "").strip():
                        continue
                    words.append(word.value)
        metadata: dict = {"image_path": path_image}
        return " ".join(words), metadata
