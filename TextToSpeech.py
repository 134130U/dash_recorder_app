import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor, SpeechT5HifiGan
from datasets import load_dataset
from IPython.display import Audio, display

def load_speech_model(checkpoint="bilalfaye/speecht5_tts-wolof-v0.2", vocoder_checkpoint="microsoft/speecht5_hifigan"):
    """ Load the SpeechT5 model, processor, and vocoder for text-to-speech. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = SpeechT5Processor.from_pretrained(checkpoint)
    model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint).to(device)
    vocoder = SpeechT5HifiGan.from_pretrained(vocoder_checkpoint).to(device)

    return processor, model, vocoder, device

# Load the model
processor, model, vocoder, device = load_speech_model()
# Load speaker embeddings (pretrained from CMU Arctic dataset)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

def generate_speech_from_text(text, speaker_embedding=speaker_embedding, processor=processor, model=model, vocoder=vocoder):
    """ Generates speech from input text using SpeechT5 and HiFi-GAN vocoder. """

    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_text_positions)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    speech = model.generate(
        inputs["input_ids"],
        speaker_embeddings=speaker_embedding.to(model.device),
        vocoder=vocoder,
        num_beams=7,
        temperature=0.6,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
    )

    speech = speech.detach().cpu().numpy()
    print(type(speech))
    display(Audio(speech, rate=16000))
    return speech