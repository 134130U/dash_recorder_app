import uuid
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash_recording_components import AudioRecorder
import soundfile as sf
import numpy as np
import io
import os
import TextToSpeech
from TextToSpeech import load_speech_model, generate_speech_from_text
import base64
import tempfile

app = Dash(__name__, external_scripts=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [dbc.Row(html.A([html.Img(src=app.get_asset_url('DataBeez-logo.png'),
                                 id='oolu-logo',
                                 style={
                                     "height": "50px",
                                     "width": "auto",
                                     "margin-bottom": "25px",
                                 }, ),])),html.H1('Wolof Text to Speech', style={'textAlign':'center'}),
     dbc.Row(
         [dcc.Textarea(id='user_input', style={'width':'50%', 'height':50, 'margin_top':20}),
         html.Br(),
         html.Button('Submit', id='my-button')]
     ),
     dbc.Row(
         dcc.Loading([
             html.Div(id="audio-output")
         ],
         type='cube')
     )
     ]
)

@app.callback(
    Output("audio-output", "children"),
    Input('my-button', 'n_clicks'),
    State('user_input', 'value'),
    prevent_initial_call=True
)
def play_audio(clicks,text):
    if clicks and text:
        audio_samples = generate_speech_from_text(text)
        audio_array = np.array(audio_samples)

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
    app.run_server(debug=True,port=8082)