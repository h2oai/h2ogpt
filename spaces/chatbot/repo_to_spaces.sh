#!/bin/bash

# NOTE: start in h2ogpt repo base directory

h2ogpt_hash=`git rev-parse HEAD`

ln -sr generate.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py requirements.txt spaces/chatbot/
cd ..

git clone https://huggingface.co/spaces/h2oai/h2ogpt-chatbot
cd h2ogpt-chatbot
rm -rf app.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py requirements.txt
cd ../h2ogpt/spaces/chatbot/
cp generate.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py requirements.txt ../../../h2ogpt-chatbot/
cd ../../../h2ogpt-chatbot/

mv generate.py app.py

git add app.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py requirements.txt
git commit -m "Update with h2ogpt hash ${h2ogpt_hash}"
# ensure write token used and login with git control: huggingface-cli login --token <HUGGINGFACE_API_TOKEN> --add-to-git-credential
git push