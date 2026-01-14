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
You should install transformers using bellow command
```aiignore
pip install -U transformers accelerate peft bitsandbytes datasets sentencepiece
```

You should make config file using below command
```aiignore
accelerate config

Compute environment:  single machine
Distributed training: no
Do you want to use GPU?  yes
Mixed precision:  fp16 (recommend bf16)
```

check the environment using below command if it is set well or not
```aiignore
python -c "import transformers, accelerate, peft, bitsandbytes; print('ready for 4bit')"
```