from typing import Optional
from TTS.api import TTS
import tempfile
import os
import platform
import subprocess


class TTSAgent:
    def __init__(
        self,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        device: str = "auto",
    ):
        self.tts = TTS(model_name=model_name, progress_bar=False, gpu=device)
        self.last_wav_path = None

    def set_voice(self, voice_name: str):
        # Coqui TTS는 다중 화자 모델에서만 speaker_name을 지원
        self.speaker = voice_name

    def speak(self, text: str):
        # 임시 wav 파일로 음성 저장 후 재생
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            wav_path = tmp.name
        # speaker 옵션은 다중화자 모델에서만 사용
        kwargs = {}
        if hasattr(self, "speaker"):
            kwargs["speaker"] = self.speaker
        self.tts.tts_to_file(text=text, file_path=wav_path, **kwargs)
        self.last_wav_path = wav_path
        # OS별 wav 재생
        if platform.system() == "Windows":
            subprocess.run(
                [
                    "powershell",
                    "-c",
                    f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync(); Remove-Item '{wav_path}'",
                ],
                shell=True,
            )
        else:
            os.system(f'aplay "{wav_path}" && rm "{wav_path}"')


# 사용 예시
if __name__ == "__main__":
    tts = TTSAgent()
    tts.speak("Hello, this is a test of the Coqui TTS agent.")
