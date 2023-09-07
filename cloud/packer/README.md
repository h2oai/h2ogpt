# h2oGPT Packer Templates

These scripts help create images in public clouds that can then submitted to Azure/GCP Marketplace for commercial use.

### Packer Scripts 
- Azure - `h2ogpt-azure.json`
- GCP - `h2ogpt-gcp.json`

### Provisioning Scripts
 - `setup_environment.sh`
    - Responsible for setting up CUDA, GCC, Nginx, Python
- `install_h2ogpt.sh`
    - Responsible for setting up h2oGPT with its dependencies
- `h2oai-h2ogpt-4096-llama2-13b-chat.sh`
    - Responsible for setting up default model h2oai-h2ogpt-4096-llama2-13b-chat with vLLM in port 80 via Nginx
    - vLLM, h2oGPT and Nginx are executed through services
    - Model is downloaded at the runtime

__Jenkins Pipeline__: http://jenkins.h2o.local:8080/job/build-h2ogpt-cloud-images/

### Notes:
 - Since model is downloaded at the runtime after VM is provisioned it takes around 5 - 10 min start h2oGPT correctly
