"""
Based upon ImageCaptionLoader in LangChain version: langchain/document_loaders/image_captions.py
But accepts preloaded model to avoid slowness in use and CUDA forking issues

Loader that uses H2O DocTR OCR models to extract text from images

"""
from typing import List, Union, Any, Tuple

import requests
from langchain.docstore.document import Document
from langchain.document_loaders import ImageCaptionLoader
import numpy as np
from utils import get_device, NullContext

class H2OOCRLoader(ImageCaptionLoader):
    """Loader that extracts text from images"""

    def __init__(self, path_images: Union[str, List[str]] = None, layout_aware = False):
        super().__init__(path_images)
        self._ocr_model = None
        self.layout_aware = layout_aware

    def set_context(self):
        if get_device() == 'cuda':
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
            if n_gpus > 0:
                self.context_class = torch.device
                self.device = 'cuda'

    def load_model(self):
        try:
            from weasyprint import HTML  # to avoid warning
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
        boxes = []
        for block_num, block in enumerate(ocr_output.pages[0].blocks):
            for line_num, line in enumerate(block.lines):
                for word_num, word in enumerate(line.words):
                    if not (word.value or "").strip():
                        continue
                    words.append(word.value)
                    boxes.append([word.geometry[0][0], word.geometry[0][1], word.geometry[1][0], word.geometry[1][1]])
        if self.layout_aware:
            ids = boxes_sort(boxes)
            texts = [words[i] for i in ids]
            text_boxes = [boxes[i] for i in ids]
            words = space_layout(texts=texts, boxes=text_boxes)
        else:
            words = " ".join(words)
        metadata: dict = {"image_path": path_image}
        return words, metadata
    
def boxes_sort(boxes):
    """ From left top to right bottom
    Params:
        boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    """
    sorted_id = sorted(range(len(boxes)), key=lambda x: (boxes[x][1]))

    # sorted_boxes = [boxes[id] for id in sorted_id]


    return sorted_id

def is_same_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    
    box1_midy = (box1[1] + box1[3]) / 2
    box2_midy = (box2[1] + box2[3]) / 2

    if box1_midy < box2[3] and box1_midy > box2[1] and box2_midy < box1[3] and box2_midy > box1[1]:
        return True
    else:
        return False
    
def union_box(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]

def space_layout(texts, boxes):
    line_boxes = []
    line_texts = []
    max_line_char_num = 0
    line_width = 0
    # print(f"len_boxes: {len(boxes)}")
    boxes = np.array(boxes)
    texts = np.array(texts)
    while len(boxes) > 0:
        box = boxes[0]
        mid = (boxes[:, 3] + boxes[:, 1])/2
        inline_boxes = np.logical_and(mid > box[1], mid < box[3])
        sorted_xs = np.argsort(boxes[inline_boxes][:, 0], axis = 0)
        line_box = boxes[inline_boxes][sorted_xs]
        line_text = texts[inline_boxes][sorted_xs]
        boxes = boxes[~inline_boxes]
        texts = texts[~inline_boxes]
        
        line_boxes.append(line_box.tolist())
        line_texts.append(line_text.tolist())
        if len(" ".join(line_texts[-1])) > max_line_char_num:
            max_line_char_num = len(" ".join(line_texts[-1]))
            line_width = np.array(line_boxes[-1])
            line_width = line_width[:, 2].max() - line_width[:, 0].min()

    char_width = (line_width / max_line_char_num)
    if char_width == 0:
        char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            left_char_num = int(box[0] / char_width)
            left_char_num = max((left_char_num - len(space_line_text)), 1)
            
            #verbose layout
            space_line_text += " " * left_char_num
            
            #minified layout
            # if left_char_num > 1:
            #     space_line_text += f" <{left_char_num}> " 
            # else:
            #     space_line_text += " "
            
            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text + "\n")

    return "".join(space_line_texts)
