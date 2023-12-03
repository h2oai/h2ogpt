pip download openai==1.3.7 --no-deps
mkdir -p openai_wheel
mv openai-1.3.7-py3-none-any.whl openai_wheel
cd openai_wheel
unzip openai-1.3.7-py3-none-any.whl
rm -rf openai-1.3.7-py3-none-any.whl

mv openai-1.3.7.dist-info openvllm-1.3.7.dist-info
mv openai openvllm

find . -name '*.py' | xargs sed -i 's/from openai /from openvllm /g'
find . -name '*.py' | xargs sed -i 's/openai\./openvllm./g'
find . -name '*.py' | xargs sed -i 's/from openai\./from openvllm./g'
find . -name '*.py' | xargs sed -i 's/import openai/import openvllm/g'
find . -name '*.py' | xargs sed -i 's/OpenAI/vLLM/g'
find . -type f | xargs sed -i 's/ openai/ openvllm/g'
find . -type f | xargs sed -i 's/openai /openvllm /g'
find . -type f | xargs sed -i 's/OpenAI/vLLM/g'
find . -type f | xargs sed -i 's/\/openai/\/vllm/g'
find . -type f | xargs sed -i 's/openai\./openvllm\./g'
find . -type f | xargs sed -i 's/OPENAI/OPENVLLM/g'
find . -type f | xargs sed -i 's/openai\//openvllm\//g'
find . -type f | xargs sed -i 's/"openai"/"openvllm"/g'
find . -type f | xargs sed -i 's/_has_openai_credentials/_has_openvllm_credentials/g'
find . -type f | xargs sed -i 's/openai-/openvllm-/g'
find . -type f | xargs sed -i 's/:openai:/:openavllm:/g'

# add stop_token_ids everywhere frequency_penalty exists.

rm -rf openvllm-1.3.7-py3-none-any.whl
zip -r openvllm-1.3.7-py3-none-any.whl openvllm-1.3.7.dist-info openvllm
