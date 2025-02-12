#!/usr/bin/env python3

# MIT License

# Copyright (c) 2023 Miguel Ángel González Santamarta

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from whisper_msgs.action import STT
from openai import OpenAI
from TTS.api import TTS
import os
import re
from piper.voice import PiperVoice
import wave

class WhisperDemoNode(Node):

    def __init__(self) -> None:
        super().__init__("whisper_demo_node")

        self._action_client = ActionClient(self, STT, "/whisper/listen")
        # self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
        self.model = "src/whisper_ros/whisper_demos/whisper_demos/models/en_US-lessac-medium.onnx"
        self.voice = PiperVoice.load(self.model)
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

        self.saving = False
        self.history = []  

    def listen(self) -> None:
        while True:
            self.saving = False
            goal = STT.Goal()

            self._action_client.wait_for_server()
            send_goal_future = self._action_client.send_goal_async(goal)

            rclpy.spin_until_future_complete(self, send_goal_future)
            get_result_future = send_goal_future.result().get_result_async()
            self.get_logger().info("SPEAK")

            rclpy.spin_until_future_complete(self, get_result_future)
            self.get_logger().info("Generate results")
            result: STT.Result = get_result_future.result().result
            self.get_logger().info(f"I hear: {result.transcription.text}")
            self.get_logger().info(f"Audio time: {result.transcription.audio_time}")
            self.get_logger().info(
                f"Transcription time: {result.transcription.transcription_time}"
            )
            
            mensagem_usuario = result.transcription.text
            self.get_logger().info(f"User: {mensagem_usuario}")
        
            self.history.append({"role": "user", "content": mensagem_usuario})

            messages = self.history

            completion = self.client.chat.completions.create(
                model="model-identifier",
                messages=messages, 
                temperature=0.7,
            )
            cleaned_message = completion.choices[0].message.content
            position = cleaned_message.find("</thing>")

            if position != -1:  # Se encontrado
                cleaned_message = cleaned_message[position + len("</thing>"):]
            clear_response = re.sub(r'[^\w\s,.\-:();""]', '', cleaned_message)  # Remove caracteres especiais
            clear_response = re.sub(r'[\*\+\-]', '', clear_response)  # Remove os marcadores de lista (asterisco, mais, menos)
            clear_response = " ".join(clear_response.split())
            self.get_logger().info(f"Assistant: {cleaned_message}")


            self.history.append({"role": "assistant", "content": cleaned_message})

            # self.get_logger().info("Save")
            # self.tts.tts_to_file(cleaned_message, file_path="output.wav")
            # self.get_logger().info("Speaking")
            # os.system("mpv output.wav")

            self.get_logger().info("Save")
            wav_file = wave.open("audio.wav", "w")
            audio = self.voice.synthesize(clear_response, wav_file)
            self.get_logger().info("Speaking")
            os.system("mpv audio.wav")
            self.get_logger().info(f"{mensagem_usuario.lower(),'finish' in mensagem_usuario.lower().replace('.', ''), 'conversation'  in mensagem_usuario.lower().replace('.', '')}")
            if "finish" in mensagem_usuario.lower().replace('.', '') and "conversation"  in mensagem_usuario.lower().replace('.', ''):
                self.get_logger().info("Exit...")
                break

def main():

    rclpy.init()
    node = WhisperDemoNode()
    node.listen()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

