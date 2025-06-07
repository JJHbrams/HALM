import os
import re
import json
import warnings
from typing import Optional
from bark import SAMPLE_RATE, generate_audio, preload_models
import simpleaudio as sa

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


class TTSAgent:
    """
    Suno Bark 기반 텍스트-음성 합성 에이전트 (이모지 제거 및 감정 프롬프트 지원)
    config dict를 생성자에서 받아, remove_emoji(text)로 이모지 없는 텍스트 반환 및 speak(text)로 음성합성
    """

    def __init__(self, config=None):
        self.voice_preset = None
        self.emotion_map = {}
        if config:
            tts_config = config["TTS"] if "TTS" in config else config
            self.voice_preset = tts_config.get("voice_preset")
            self.emotion_map = tts_config.get("emotion_map", {})
        preload_models()

    def remove_emoji(self, text: str) -> str:
        pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]"
        return re.sub(pattern, "", text)

    def convert_emoji_to_bark_prompt(self, text: str) -> str:
        """
        텍스트 내 이모지를 Bark 프롬프트(예: [happy], [sad])로 변환
        emotion_map에 등록된 이모지만 변환
        """
        for emoji, prompt in self.emotion_map.items():
            if emoji in text:
                text = text.replace(emoji, f"[{prompt}]")
        return text

    def speak(self, text: str):
        """
        텍스트를 Bark로 음성 합성하여 mp3로 저장하고, play=True면 바로 재생합니다.
        :param text: 음성으로 변환할 텍스트
        """
        if not text:
            return
        # 이모지 제거 및 감정 프롬프트 변환
        text = self.remove_emoji(text)
        text = self.convert_emoji_to_bark_prompt(text)
        # Bark로 음성 생성
        audio_array = generate_audio(text, history_prompt=self.voice_preset)

        num_channels = 1
        bytes_per_sample = 4  # float32
        sample_rate = SAMPLE_RATE
        play_obj = sa.play_buffer(audio_array, num_channels, bytes_per_sample, sample_rate)

        # Wait for playback to finish before exiting
        play_obj.wait_done()


# 예제 (Suno Bark 공식 예제 스타일)
if __name__ == "__main__":
    ROOT = os.getcwd()
    with open(f"{ROOT}/config/config.json", encoding="utf-8") as f:
        config = json.load(f)

    tts_agent = TTSAgent(config)

    # 예제 텍스트
    text_prompt = """    
    Hello😊, my name is Suno😉. And, uh — and I like pizza. [laughs] 
    But I also have other interests such as playing tic tac toe.
    """
    tts_agent.speak(text_prompt)
