# local-llm-chat
This is the project implementing the chatbot using LLM.

## Development Environment
This below command makes the local environment for only this project.
```aiignore
python -m venv .venv
.venv\Scripts\activate
```
And you should install the pytorch CUDA version for GPU working.
```aiignore
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
You can check the torch library is working or not using below command.
```aiignore
> python check_gpu.py

below is the result if torch loaded successfully
>> torch version: 2.5.1+cu121
>> cuda available: True
>> device name: NVIDIA RTX A2000 12GB
>> device count: 1
```
