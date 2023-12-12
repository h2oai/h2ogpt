import logging
import os
import tempfile
import time
from typing import Dict, Iterator, Optional, Tuple

from langchain.document_loaders.base import BaseBlobParser
from langchain.document_loaders.blob_loaders import Blob
from langchain.document_loaders.generic import GenericLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)


class OpenAIWhisperParser(BaseBlobParser):
    """Transcribe and parse audio files.
    Audio transcription is with OpenAI Whisper model."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io

        try:
            from openai import OpenAI
            if self.api_key:
                client = OpenAI(api_key=self.api_key)
            else:
                client = OpenAI()
        except ImportError:
            raise ImportError(
                "openai package not found, please install it with "
                "`pip install openai`"
            )
        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub package not found, please install it with " "`pip install pydub`"
            )

        # Audio file from disk
        audio = AudioSegment.from_file(blob.path)

        # Define the duration of each chunk in minutes
        # Need to meet 25MB size limit for Whisper API
        chunk_duration = 20
        chunk_duration_ms = chunk_duration * 60 * 1000

        # Split the audio into chunk_duration_ms chunks
        for split_number, i in enumerate(range(0, len(audio), chunk_duration_ms)):
            # Audio chunk
            chunk = audio[i: i + chunk_duration_ms]
            file_obj = io.BytesIO(chunk.export(format="mp3").read())
            if blob.source is not None:
                file_obj.name = blob.source + f"_part_{split_number}.mp3"
            else:
                file_obj.name = f"part_{split_number}.mp3"

            # Transcribe
            print(f"Transcribing part {split_number + 1}!")
            attempts = 0
            while attempts < 3:
                try:
                    transcript = client.audio.transcribe("whisper-1", file_obj)
                    break
                except Exception as e:
                    attempts += 1
                    print(f"Attempt {attempts} failed. Exception: {str(e)}")
                    time.sleep(5)
            else:
                print("Failed to transcribe after 3 attempts.")
                continue

            yield Document(
                page_content=transcript.text,
                metadata={"source": blob.source, "chunk": split_number},
            )


class OpenAIWhisperParserLocal(BaseBlobParser):
    """Transcribe and parse audio files with OpenAI Whisper model.

    Audio transcription with OpenAI Whisper model locally from transformers.

    Parameters:
    device - device to use
        NOTE: By default uses the gpu if available,
        if you want to use cpu, please set device = "cpu"
    lang_model - whisper model to use, for example "openai/whisper-medium"
    forced_decoder_ids - id states for decoder in multilanguage model,
        usage example:
        from transformers import WhisperProcessor
        processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="french",
          task="transcribe")
        forced_decoder_ids = WhisperProcessor.get_decoder_prompt_ids(language="french",
        task="translate")



    """

    def __init__(
            self,
            device: str = 'gpu',
            device_id: int = 0,
            lang_model: Optional[str] = None,
            forced_decoder_ids: Optional[Tuple[Dict]] = None,
            use_better=True,
            use_faster=False,
    ):
        """Initialize the parser.

        Args:
            device: device to use.
            lang_model: whisper model to use, for example "openai/whisper-medium".
              Defaults to None.
            forced_decoder_ids: id states for decoder in a multilanguage model.
              Defaults to None.
        """
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError(
                "transformers package not found, please install it with "
                "`pip install transformers`"
            )
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch package not found, please install it with " "`pip install torch`"
            )

        # set device, cpu by default check if there is a GPU available
        if device == "cpu":
            self.device = "cpu"
            if lang_model is not None:
                self.lang_model = lang_model
                print("WARNING! Model override. Using model: ", self.lang_model)
            else:
                # unless overridden, use the small base model on cpu
                self.lang_model = "openai/whisper-base"
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
                # check GPU memory and select automatically the model
                mem = torch.cuda.get_device_properties(self.device).total_memory / (
                        1024 ** 2
                )
                if mem < 5000:
                    rec_model = "openai/whisper-base"
                elif mem < 7000:
                    rec_model = "openai/whisper-small"
                elif mem < 12000:
                    rec_model = "openai/whisper-medium"
                else:
                    rec_model = "openai/whisper-large-v3"

                # check if model is overridden
                if lang_model is not None:
                    self.lang_model = lang_model
                    print("WARNING! Model override. Might not fit in your GPU")
                else:
                    self.lang_model = rec_model
            else:
                "cpu"

        print("Using the following model: ", self.lang_model)

        # load model for inference
        if self.device == 'cpu':
            device_map = {"", 'cpu'}
        else:
            device_map = {"": 'cuda:%d' % device_id} if device_id >= 0 else {'': 'cuda'}

        # https://huggingface.co/blog/asr-chunking
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.lang_model,
            chunk_length_s=30,
            stride_length_s=5,
            batch_size=8,
            device_map=device_map,
        )
        if use_better:
            # even faster if not doing real time ASR
            # stride_length_s=5,  batch_size=8
            try:
                from optimum.bettertransformer import BetterTransformer
                self.pipe.model = BetterTransformer.transform(self.pipe.model, use_flash_attention_2=True)
            except Exception as e:
                print("No optimum, not using BetterTransformer: %s" % str(e), flush=True)

        if use_faster and have_use_faster and self.lang_model in ['openai/whisper-large-v2',
                                                                  'openai/whisper-large-v3']:
            self.pipe.model.to('cpu')
            del self.pipe.model
            clear_torch_cache()
            print("Using faster_whisper", flush=True)
            # has to come here, no framework and no config for model
            # pip install git+https://github.com/SYSTRAN/faster-whisper.git
            from faster_whisper import WhisperModel
            model_size = "large-v3" if self.lang_model == 'openai/whisper-large-v3' else "large-v2"
            # Run on GPU with FP16
            model = WhisperModel(model_size, device=self.device, compute_type="float16")
            # or run on GPU with INT8
            # model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
            # or run on CPU with INT8
            # model = WhisperModel(model_size, device="cpu", compute_type="int8")
            self.pipe.model = model

        if forced_decoder_ids is not None:
            try:
                self.pipe.model.config.forced_decoder_ids = forced_decoder_ids
            except Exception as exception_text:
                logger.info(
                    "Unable to set forced_decoder_ids parameter for whisper model"
                    f"Text of exception: {exception_text}"
                    "Therefore whisper model will use default mode for decoder"
                )

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Lazily parse the blob."""

        import io

        try:
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "pydub package not found, please install it with `pip install pydub`"
            )

        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa package not found, please install it with "
                "`pip install librosa`"
            )

        file = str(blob.path)
        if any([file.endswith(x) for x in ['.mp4', '.mpeg', '.mpg']]):
            import audioread.ffdec  # Use ffmpeg decoder
            aro = audioread.ffdec.FFmpegAudioFile(blob.path)
            y, sr = librosa.load(aro, sr=16000)
        else:

            # Audio file from disk
            audio = AudioSegment.from_file(blob.path)

            file_obj = io.BytesIO(audio.export(format="mp3").read())

            # Transcribe
            print(f"Transcribing part {blob.path}!")

            y, sr = librosa.load(file_obj, sr=16000)

        prediction = self.pipe(y.copy(), batch_size=8)["text"]

        yield Document(
            page_content=prediction,
            metadata={"source": blob.source},
        )


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

from utils import get_device, NullContext, clear_torch_cache, have_use_faster, makedirs

from importlib.metadata import distribution, PackageNotFoundError

try:
    assert distribution('bitsandbytes') is not None
    have_bitsandbytes = True
except (PackageNotFoundError, AssertionError):
    have_bitsandbytes = False


class H2OAudioCaptionLoader(ImageCaptionLoader):
    """Loader that loads the transcriptions of audio"""

    def __init__(self, path_audios: Union[str, List[str]] = None,
                 asr_model='openai/whisper-medium',
                 asr_gpu=True,
                 gpu_id='auto',
                 use_better=True,
                 use_faster=False,
                 ):
        super().__init__(path_audios)
        self.audio_paths = path_audios
        self.model = None
        self.asr_model = asr_model
        self.asr_gpu = asr_gpu
        self.context_class = NullContext
        self.gpu_id = gpu_id if isinstance(gpu_id, int) else 0
        self.device = 'cpu'
        self.device_map = {"": 'cpu'}
        self.set_context()
        self.use_better = use_better
        self.use_faster = use_faster
        self.files_out = []

    def set_context(self):
        if get_device() == 'cuda' and self.asr_gpu:
            import torch
            n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_gpus > 0:
                self.context_class = torch.device
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = 'cpu'
        if get_device() == 'cuda' and self.asr_gpu:
            if self.gpu_id == 'auto':
                # blip2 has issues with multi-GPU.  Error says need to somehow set language model in device map
                # device_map = 'auto'
                self.gpu_id = 0
            self.device_map = {"": 'cuda:%d' % self.gpu_id}
        else:
            self.gpu_id = -1
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
            if str(self.model.pipe.model.device) != self.device_map['']:
                self.model.pipe.model.to(self.device_map[''])
            return self
        import torch
        with torch.no_grad():
            with self.context_class(self.device):
                context_class_cast = NullContext if self.device == 'cpu' else torch.autocast
                with context_class_cast(self.device):
                    self.model = OpenAIWhisperParserLocal(device=self.device,
                                                          device_id=self.gpu_id,
                                                          lang_model=self.asr_model,
                                                          use_better=self.use_better,
                                                          use_faster=self.use_faster,
                                                          )
        return self

    def set_audio_paths(self, path_audios: Union[str, List[str]]):
        """
        Load from a list of audio files
        """
        if isinstance(path_audios, str):
            self.audio_paths = [path_audios]
        else:
            self.audio_paths = path_audios

    def load(self, from_youtube=False) -> List[Document]:
        if self.model is None:
            self.load_model()

        # https://librosa.org/doc/main/generated/librosa.load.html
        if from_youtube:
            save_dir = tempfile.mkdtemp()
            makedirs(save_dir, exist_ok=True)
            youtube_loader = YoutubeAudioLoader(self.audio_paths, save_dir)
            loader = GenericLoader(youtube_loader, self.model)
            docs = loader.load()
            self.files_out = youtube_loader.files_out
            return docs
        else:
            docs = []
            for fil in self.audio_paths:
                loader = GenericLoader.from_filesystem(
                    os.path.dirname(fil),
                    glob=os.path.basename(fil),
                    parser=self.model)
                docs += loader.load()
            return docs

    def unload_model(self):
        if hasattr(self, 'model') and hasattr(self.model, 'pipe') and hasattr(self.model.pipe.model, 'cpu'):
            self.model.pipe.model.cpu()
            clear_torch_cache()


from typing import Iterable, List

from langchain.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain.document_loaders.blob_loaders.schema import Blob, BlobLoader


class YoutubeAudioLoader(BlobLoader):

    """Load YouTube urls as audio file(s)."""

    def __init__(self, urls: List[str], save_dir: str):
        if not isinstance(urls, list):
            raise TypeError("urls must be a list")

        self.urls = urls
        self.save_dir = save_dir
        self.files_out = []

    def yield_blobs(self) -> Iterable[Blob]:
        """Yield audio blobs for each url."""

        try:
            import yt_dlp
        except ImportError:
            raise ImportError(
                "yt_dlp package not found, please install it with "
                "`pip install yt_dlp`"
            )

        # Use yt_dlp to download audio given a YouTube url
        ydl_opts = {
            "format": "m4a/bestaudio/best",
            "noplaylist": True,
            "outtmpl": self.save_dir + "/%(title)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "m4a",
                }
            ],
        }

        for url in self.urls:
            # Download file
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download(url)

        # Yield the written blobs
        loader = FileSystemBlobLoader(self.save_dir, glob="*.m4a")
        self.files_out = [os.path.join(self.save_dir, f) for f in os.listdir(self.save_dir)]
        for blob in loader.yield_blobs():
            yield blob
