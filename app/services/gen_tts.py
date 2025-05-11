import tempfile
from tempfile import NamedTemporaryFile

from gtts import gTTS


def text_to_speech(text: str, lang="en", slow=False) -> str:
    """
    Convert text to speech using gTTS
    """
    try:
        # Find OS's temp directory
        temp_dir = tempfile.gettempdir()

        # Create a temp file to store the audio stream
        temp_file = NamedTemporaryFile(
            delete=False, suffix=".mp3", dir=temp_dir)

        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(temp_file.name)

        return temp_file.name
    except Exception as e:
        raise RuntimeError(f"TTS Conversion failed: {str(e)}")


if __name__ == "__main__":
    input_text = """
    This is a test of the text-to-speech conversion.
    The quick brown fox jumps over the lazy dog.
    """
    audio_file = text_to_speech(input_text)
    print(f"Audio file saved at: {audio_file}")
