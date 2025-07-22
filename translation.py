#%%
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_load_name = 'bilalfaye/nllb-200-distilled-600M-wolof-french'

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_name).to(device)
tokenizer = NllbTokenizer.from_pretrained(model_load_name)

def translate(
    text, src_lang='wol_Latn', tgt_lang='french_Latn',
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    """Turn a text or a list of texts into a list of translations"""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True,
        max_length=max_input_length
    )
    model.eval()
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)