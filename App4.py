import uuid
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_recording_components import AudioRecorder
import soundfile as sf
import numpy as np
import io
import os
import base64
import tempfile
from transformers import (
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from dotenv import find_dotenv, load_dotenv
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import TextToSpeech
from TextToSpeech import load_speech_model, generate_speech_from_text
import translation
from translation import translate

GROQ_API_KEY = os.getenv("GROQ-API-KEY")

chat = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",        #  "llama-3.1-70b-versatile" "Mixtral-8x7b-32768" "deepseek-r1-distill-llama-70b",
    api_key=GROQ_API_KEY
)

system = "Tu es un assistant trés utilis. Ton role est de répondre aux question. Il faut dire 'Je ne sais quand tu ne connais pas la réponse."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat

# Load ASR model
model = AutoModelForSpeechSeq2Seq.from_pretrained("Alwaly/whisper-medium-wolof")
processor = AutoProcessor.from_pretrained("Alwaly/whisper-medium-wolof")
model.generation_config.forced_decoder_ids = None
model.generation_config.suppress_tokens = None
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor)

# Load translation model
tokenizer2 = AutoTokenizer.from_pretrained("Alwaly/wolofToFrenchTranslator")
model2 = AutoModelForSeq2SeqLM.from_pretrained("Alwaly/wolofToFrenchTranslator")
pipe2 = pipeline("text2text-generation", model=model2, tokenizer=tokenizer2)

# Init Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
audio_samples = []

# Layout
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("🎙️ Wolof Audio Transcriber", className="text-center text-primary mb-4"))]),

    dbc.Row([
        dbc.Col([dbc.Button("🔴 Record", id="record-button", color="danger")], width="auto"),
        dbc.Col(html.Div(id="waveform", className="wave-container"), width=3),
        dbc.Col([
            dbc.Button("⏹️ Stop", id="stop-button", color="secondary", className="me-2", n_clicks=0),
            dbc.Button("▶️ Play", id="play-button", color="success", className="me-2"),
            dbc.Button("📝 Translate", id="translate-button", color="primary")
        ], width="auto")
    ], className="mb-3 align-items-center"),

    dbc.Row([dbc.Col(AudioRecorder(id="audio-recorder"))]),

    dbc.Row([dbc.Col(html.Div(id="audio-output"))]),

    dbc.Row([
        dbc.Col(html.Div(id='transcription_display'), width=6),
        dbc.Col(html.Div(id='translation-output'), width=6)
    ]),
    dbc.Row([html.Button('Submit', id='my-button')]),
    dbc.Row(
         dcc.Loading([
             dcc.Markdown(id='content',children='')
         ],
         type='cube')
     ),
    dbc.Row([dbc.Col([dbc.Button("📝 Translate to wolof", id="translate-wolof_button_id", color="primary"),
                      dbc.Button("▶️ Ecouter Wolof", id="play-button2", color="success", className="me-2"),]),
             dbc.Col(html.Div(id="audio-output2"))
             ]),
    dbc.Row([html.Div(id='wolof-output')]),

    # Hidden values
    html.Div(id="transcription_id", style={"display": "none"}),
    html.Div(id="dummy-output", style={"display": "none"}),
    html.Div(id="translated_text_id", style={"display": "none"}),
    html.Div(id="translate2wolof_id", style={"display": "none"}),

], fluid=True, className="p-4")


@app.callback(
    Output("audio-recorder", "recording"),
    Input("record-button", "n_clicks"),
    Input("stop-button", "n_clicks"),
    State("audio-recorder", "recording"),
    prevent_initial_call=True
)
def control_recording(record_clicks, stop_clicks, recording):
    global audio_samples
    if record_clicks and (record_clicks > stop_clicks):
        audio_samples = []
        return True
    return False

@app.callback(
    Output("waveform", "children"),
    Input("audio-recorder", "recording"),
    prevent_initial_call=True
)
def show_waveform(recording):
    heights = [30, 55, 20, 45, 60, 25, 50, 40, 35, 20, 60, 30, 50, 25, 45, 30, 55, 40, 30, 35]
    if recording:
        return html.Div(
            [html.Div(style={"height": f"{h}px"}, className="wave-segment animate") for h in heights],
            className="waveform"
        )
    return html.Div([html.Div(className="wave-bar2") for _ in range(5)])

@app.callback(
    Output("dummy-output", "children"),
    Input("audio-recorder", "audio"),
    prevent_initial_call=True
)
def update_audio(audio):
    global audio_samples
    if audio is not None:
        audio_samples += list(audio.values())
    return ""

@app.callback(
    Output("audio-output", "children"),
    Output("transcription_id", "children"),
    Input("play-button", "n_clicks"),
    prevent_initial_call=True
)
def play_audio(play_clicks):
    if play_clicks and audio_samples:
        audio_array = np.array(audio_samples)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_array, 16000, format="WAV")
            wav_bytes = wav_buffer.getvalue()
            wav_base64 = base64.b64encode(wav_bytes).decode()
            audio_src = f"data:audio/wav;base64,{wav_base64}"

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav.write(wav_bytes)
                tmp_wav_path = tmp_wav.name

        result = pipe(tmp_wav_path)
        raw_text = result["text"]
        audio_component = html.Audio(src=audio_src, controls=True)
        return audio_component, raw_text
    return "", ""


@app.callback(
    Output("transcription_display", "children"),
    Input("transcription_id", "children")
)
def show_transcription(text):
    if text:
        return html.H4(f"Transcription: {text}")
    return ""

@app.callback(
    Output("translation-output", "children"),
    Output("translated_text_id", "children"),
    Input("translate-button", "n_clicks"),
    State("transcription_id", "children"),
    prevent_initial_call=True
)
def translate_audio(n, transcribed_text):
    if transcribed_text:
        result = pipe2(transcribed_text)
        translated_text = result[0]['generated_text']
        return html.H4(f" Translation: {translated_text}"), translated_text
    return "", ""

@app.callback(
    Output('content','children'),
    Input('my-button','n_clicks'),
    State('translated_text_id','children')
)

def get_infos(_,user_input):
    res = chain.invoke({"text": user_input})
    return res.content

@app.callback(
    Output('translate2wolof_id','children'),
    Output('wolof-output','children'),
    Input('translate-wolof_button_id','n_clicks'),
    State('content','children'))
def get_infos(_,text_content):
    if text_content:
        translated_text = translate(text_content, src_lang="fra_Latn", tgt_lang="wol_Latn")[0]
        return translated_text,html.H4(f"{translated_text}")
    return "",""

@app.callback(
    Output("audio-output2", "children"),
    Input('play-button2','n_clicks'),
    State('translate2wolof_id','children'),
    prevent_initial_call=True
)

def play_audio(clicks,text):
    if clicks and text:
        audio_samples2 = [generate_speech_from_text(t) for t in text]
        audio_array = np.array(audio_samples2)

        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_array, 16000, format="WAV")
            wav_bytes = wav_buffer.getvalue()
            wav_base64 = base64.b64encode(wav_bytes).decode()
            audio_src = f"data:audio/wav;base64,{wav_base64}"

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav.write(wav_bytes)
                tmp_wav_path = tmp_wav.name

        audio_component = html.Audio(src=audio_src, controls=True)
        return audio_component

if __name__ == "__main__":
    app.run_server(debug=True)
