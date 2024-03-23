Write-Output "1. Check for Existing Processes on ports"

# if this is not working for you, check your task manager for python.exe processes running on ports 7861, 7862, etc since h2ogpt/gradio will try to bind to those ports if 7860 is taken
if(Get-NetTCPConnection -LocalPort "7860" -ErrorAction SilentlyContinue) { 
    Write-Output "Process running on port 7680. Ending Process.... (wait 10 seconds to end)"
    Get-Process -Id (Get-NetTCPConnection -LocalPort "7860").OwningProcess | Stop-Process
    Start-Sleep -Seconds 10
    if(Get-NetTCPConnection -LocalPort "7860") { Write-Error "Failed to stop process."}
}else{
    Write-Output "No processes running on port 7680"
}

Write-Output "2. Set env variables"
$Env:CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=all"
$Env:LLAMA_CUBLAS=1
$Env:FORCE_CMAKE=1

Write-Output "3. Start h2o Llama2 server"
cd C:\code\h2ogpt # Change to your h2ogpt directory
# the model below should already be downloaded in order for this to work. check to make sure this .gguf file is in this location ...\h2ogpt\llamacpp-path\llama-2-7b-chat.Q6_K.gguf
python generate.py --base_model=llama --model_path_llama=llama-2-7b-chat.Q6_K.gguf --prompt_type=llama2 --openai_server=True --openai_port=5000 --concurrency_count=1 --add_disk_models_to_ui=False --enable_tts=False --enable_stt=False --max_seq_len=4096 --save_dir=saveinf