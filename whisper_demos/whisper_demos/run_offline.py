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
from llama_cpp import Llama
from llama_cpp_agent.llm_agent import LlamaCppAgent
from llama_cpp_agent.messages_formatter import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppPythonProvider
# from TTS.api import TTS
import subprocess
import os 
import re
from piper.voice import PiperVoice
import wave

def load_model(
    model_path: str,  # Caminho para o arquivo do modelo LLama que será carregado.
    threads: int = 4,  # Número de threads a serem usadas para a inferência.
    context: int = 4096,  # Tamanho do contexto considerado pelo modelo.
    temperature: float = 0.6,  # Define a aleatoriedade das respostas.
    chat_format: str = "CHATML",  # Formato do prompt a ser utilizado.
    sysprompt: str = "Just answer the users question please." # Prompt do sistema inicial.
):
    """Carrega o modelo LLama e configura os parâmetros."""
    model = Llama(
        model_path=model_path,  # Caminho para o arquivo do modelo LLama.
        n_gpu_layers=0,  # Número de camadas a serem executadas na GPU (0 para CPU apenas).
        f16_kv=True,  # Usa valores-chave em float16 para economia de memória.
        repeat_last_n=64,  # Número de últimos tokens a serem levados em consideração para penalidade de repetição.
        use_mmap=True,  # Usa mmap para carregar o modelo de forma mais eficiente na RAM.
        use_mlock=False,  # Se True, bloqueia o modelo na RAM para evitar swaps.
        embedding=False,  # Se True, ativa a geração de embeddings em vez de texto.
        n_threads=threads,  # Número de threads da CPU usadas na inferência.
        n_batch=128,  # Número de tokens processados por lote para eficiência.
        n_ctx=context,  # Tamanho do contexto do modelo (quantos tokens ele pode lembrar).
        offload_kqv=True,  # Se True, descarrega algumas operações da memória RAM para GPU.
        last_n_tokens_size=1024,  # Buffer de tokens recentes para evitar repetições excessivas.
        verbose=False,  # Desativa mensagens desnecessárias.
        seed=0,  # Define a semente para reprodução de resultados (-1 para aleatório).
    )
    provider = LlamaCppPythonProvider(model)
    settings = provider.get_provider_default_settings()
    settings.max_tokens = -1  # Número máximo de tokens gerados (-1 significa ilimitado).
    settings.temperature = temperature  # Controla a aleatoriedade da saída (0 = determinístico, >1 = mais criativo).
    settings.top_k = 40  # Considera apenas as 40 palavras mais prováveis na geração.
    settings.top_p = 0.4  # Usa amostragem nuclear para limitar escolhas a palavras com 40% de probabilidade cumulativa.
    settings.repeat_penalty = 1.18  # Penaliza repetições para tornar o texto mais variado.
    settings.stream = True  # Define se a saída será transmitida em tempo real.

    formatter = next((fmt for fmt in MessagesFormatterType if fmt.name == chat_format), None)
    if formatter is None:
        raise ValueError("Formato de chat inválido")
    
    agent = LlamaCppAgent(provider, debug_output=False, system_prompt=sysprompt, predefined_messages_formatter_type=formatter)
    return agent, settings

def chat():
    # tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
    agent, settings = load_model("DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf")
    model = "src/whisper_ros/whisper_demos/whisper_demos/models/en_US-lessac-medium.onnx"
    voice = PiperVoice.load(model)
    print("Modelo carregado. Digite sua mensagem (ou 'sair' para encerrar).")
    while True:
        print("You:")
        user_input = input()
        
        response = agent.get_chat_response(user_input, llm_sampling_settings=settings)
        
        print(f"LLM: {response}")
        position = response.find("</think>")
        print(position)
        if position != -1:  # Se encontrado
            response = response[position + len("</think>"):]
        clear_response = re.sub(r'[^\w\s,.\-:();""]', '', response)  # Remove caracteres especiais
        clear_response = re.sub(r'[\*\+\-]', '', clear_response)  # Remove os marcadores de lista (asterisco, mais, menos)
        clear_response = " ".join(clear_response.split())
        wav_file = wave.open("audio.wav", "w")
        audio = voice.synthesize(clear_response, wav_file)
        os.system("mpv audio.wav")
        if user_input.lower() in ["sair", "exit", "quit"]:
            break
if __name__ == "__main__":
    chat()



