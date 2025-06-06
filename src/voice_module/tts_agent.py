import os
import platform
import subprocess
import re
from typing import Optional
import json
import tempfile
import warnings
from TTS.api import TTS

warnings.filterwarnings("ignore", category=DeprecationWarning)


class TTSAgent:
    """
    Coqui TTS 기반 감정(이모지) 인식 음성합성 에이전트
    config dict를 생성자에서 직접 받아 모델, 화자, 언어, 감정 매핑을 관리
    """

    def __init__(self, config):
        tts_config = config["TTS"] if "TTS" in config else config
        self.model_name = tts_config.get(
            "model_name", "tts_models/multilingual/multi-dataset/your_tts"
        )
        self.cache_dir = tts_config.get("cache_dir")
        self.speaker = tts_config.get("default_speaker")
        self.language = tts_config.get("default_language")
        self.emoji_emotion_map = tts_config.get("emotion_map", {})
        global EMOJI_EMOTION_MAP
        EMOJI_EMOTION_MAP = self.emoji_emotion_map
        if not self.cache_dir:
            home = os.path.expanduser("~")
            self.cache_dir = os.path.join(home, ".local", "share", "tts")
        os.environ["TTS_CACHE_PATH"] = self.cache_dir
        try:
            self.tts = TTS(model_name=self.model_name, progress_bar=True)
        except Exception as e:
            print(f"[TTSAgent] Error initializing TTS model: {e}")
            self.tts = None
        self.last_wav_path = None

    def speak(self, text: str):
        if not self.tts:
            print("[TTSAgent] TTS model not initialized.")
            return
        emotion = self._extract_emotion(text)
        clean_text = self._remove_emoji(text)
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                wav_path = tmp.name
            kwargs = {}
            # speaker: 멀티스피커 모델은 speaker 목록에서 첫 번째 화자를 기본값으로 사용
            if not self.speaker:
                try:
                    speakers = self.tts.speakers
                    if speakers:
                        self.speaker = speakers[0]
                        print(
                            f"[TTSAgent] No speaker specified. Using default speaker: {self.speaker}"
                        )
                except Exception:
                    pass
            # language: 멀티링구얼 모델은 languages에서 첫 번째 언어를 기본값으로 사용
            if not self.language:
                try:
                    languages = getattr(self.tts, "languages", None)
                    if languages:
                        self.language = languages[0]
                        print(
                            f"[TTSAgent] No language specified. Using default language: {self.language}"
                        )
                except Exception:
                    self.language = None
            if self.speaker:
                kwargs["speaker"] = self.speaker
            if self.language:
                kwargs["language"] = self.language
            if emotion:
                kwargs["emotion"] = emotion
            self.tts.tts_to_file(text=clean_text, file_path=wav_path, **kwargs)
            self.last_wav_path = wav_path
            self._play_wav(wav_path)
        except Exception as e:
            print(f"[TTSAgent] Error during TTS synthesis or playback: {e}")
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    def _extract_emotion(self, text: str) -> Optional[str]:
        for emoji, emotion in EMOJI_EMOTION_MAP.items():
            if emoji in text:
                return emotion
        pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
        m = re.search(pattern, text)
        if m:
            return "with emotion"
        return None

    def _remove_emoji(self, text: str) -> str:
        pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
        return re.sub(pattern, "", text)

    def set_voice(self, voice_name: str):
        self.speaker = voice_name

    def _play_wav(self, wav_path: str):
        try:
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
        except Exception as e:
            print(f"[TTSAgent] Error playing audio: {e}")


# 예제
if __name__ == "__main__":
    ROOT = os.getcwd()
    with open(f"{ROOT}/config/config.json", encoding="utf-8") as f:
        config = json.load(f)

    tts_agent = TTSAgent(config)
    tts_agent.speak("Hello, world! 😊")
    tts_agent.speak("I am happy today! 😄")
    tts_agent.speak("Let's have a great day! 🌞")
