# from openai import OpenAI
# import os
# from TTS.api import TTS
# client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
# tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

# mensagem_usuario ='Tell me about yourself'
# completion = client.chat.completions.create(
# model="model-identifier",
# messages=[
#     {"role": "user", "content": mensagem_usuario}  
# ],
# temperature=0.7,
# )

# print(completion.choices[0].message.content)

# tts.tts_to_file(completion.choices[0].message.content, file_path="output.wav")

# os.system("mpv output.wav")
import ollama
# from TTS.api import TTS
import subprocess
import os 
import re
from piper.voice import PiperVoice
import wave
import json

def chat():
    # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    model = "src/whisper_ros/whisper_demos/whisper_demos/models/en_US-lessac-medium.onnx"
    voice = PiperVoice.load(model)
    print("Modelo carregado. Digite sua mensagem (ou 'sair' para encerrar).")

    conversation = [] 
    
    while True:
        print("You:")
        user_input = input()
        messages = [
            {"role": "user", "content": user_input},
        ]
        completion = ollama.chat(
            model="saller",
            messages=messages, 
        )
        response = completion["message"]["content"]
        position = response.find("</think>")
        print(response)
        if position != -1:  # Se encontrado
            response = response[position + len("</think>"):]
        clear_response = re.sub(r'[^\w\s,.\-:();""]', '', response)  # Remove caracteres especiais
        clear_response = re.sub(r'[\*\+\-]', '', clear_response)  # Remove os marcadores de lista (asterisco, mais, menos)
        clear_response = " ".join(clear_response.split())

        conversation.append({"user": user_input, "response": clear_response})

        wav_file = wave.open("audio.wav", "w")
        audio = voice.synthesize(clear_response, wav_file)
        # os.system("mpv audio.wav")
        if user_input.lower() in ["sair", "exit", "quit"]:
            break
    with open("conversation.json", "w") as json_file:
        json.dump(conversation, json_file, ensure_ascii=False, indent=4)
if __name__ == "__main__":
    chat()

