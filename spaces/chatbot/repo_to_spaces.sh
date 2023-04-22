#!/bin/bash

# NOTE: start in h2ogpt repo base directory
# i.e. can run below to update both spaces (assumes repos already existed, else will have to login HF for each)
# (h2ollm) jon@pseudotensor:~/h2ogpt$ ./spaces/chatbot/repo_to_spaces.sh h2ogpt-chatbot ; ./spaces/chatbot/repo_to_spaces.sh h2ogpt-chatbot2

spacename=${1:-h2ogpt-chatbot}
echo "Space name: $spacename"

# NOTE: start in h2ogpt repo base directory

h2ogpt_hash=`git rev-parse HEAD`

ln -sr generate.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py requirements.txt spaces/chatbot/
cd ..

git clone https://huggingface.co/spaces/h2oai/"${spacename}"
cd "${spacename}"
rm -rf app.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py requirements.txt
cd ../h2ogpt/spaces/chatbot/
cp generate.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py requirements.txt ../../../"${spacename}"/
cd ../../../"${spacename}"/

mv generate.py app.py

git add app.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py requirements.txt
git commit -m "Update with h2oGPT hash ${h2ogpt_hash}"
# ensure write token used and login with git control: huggingface-cli login --token <HUGGINGFACE_API_TOKEN> --add-to-git-credential
git push