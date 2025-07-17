# If you cannot access the huggingface website in your region, you should uncomment the following line
# export HF_ENDPOINT="https://hf-mirror.com"


# If you only want to reproduce our method, you only need to download gpt-neo-2.7B
huggingface-cli download --resume-download EleutherAI/gpt-neo-2.7B --local-dir ./model/gpt-neo-2.7B


# If you want to reproduce all experiments, you should uncomment the following lines
# # DetectGPT
# huggingface-cli download --resume-download google-t5/t5-small --local-dir ./model/t5-small
# # FastDetectGPT
# huggingface-cli download --resume-download EleutherAI/gpt-j-6B --local-dir ./model/gpt-j-6B
# # TextFluoroscopy
# huggingface-cli download --resume-download Alibaba-NLP/gte-Qwen1.5-7B-instruct --local-dir ./model/gte-Qwen1.5-7B-instruct
# huggingface-cli download --resume-download FishAndSheep/TextFluoroscopy --local-dir ./model/TextFluoroscopy
# # Binoculars
# huggingface-cli download --resume-download tiiuae/falcon-7b --local-dir ./model/falcon-7b
# huggingface-cli download --resume-download tiiuae/falcon-7b-instruct --local-dir ./model/falcon-7b-instruct
# # Ablation
# huggingface-cli download --resume-download Qwen/Qwen2-0.5B --local-dir ./model/Qwen2-0.5B