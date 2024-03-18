import os
os.environ["COQUI_TOS_AGREED"] = "1"


import pytest
from tests.utils import wrap_test_forked

from TTS.api import TTS

@pytest.mark.parametrize(
    "model_name",
    TTS().list_models()
)
@wrap_test_forked
def test_get_models(model_name):
    import torch
    from TTS.api import TTS

    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Init TTS
    tts = TTS(model_name).to(device)

    # Run TTS
    # ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
    # Text to speech list of amplitude values as output
    try:
        wav = tts.tts(text="Hello world!", speaker_wav="./models/male.wav", language="en")
        # Text to speech to a file
        tts.tts_to_file(text="Hello world!", speaker_wav="./models/male.wav", language="en", file_path="output.wav")
    except ValueError:
        wav = tts.tts(text="Hello world!", speaker_wav="./models/male.wav")
        # Text to speech to a file
        tts.tts_to_file(text="Hello world!", speaker_wav="./models/male.wav", file_path="output.wav")

    # files are located in e.g. /home/jon/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v1.1
    # downloaded from e.g. https://coqui.gateway.scarf.sh/v0.6.1_models/tts_models--en--ljspeech--glow-tts.zip
    # all stored in https://h2o-release.s3.amazonaws.com/h2ogpt/tts_in_.local_share_tts.tgz