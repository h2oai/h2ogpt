#!/bin/bash

# NOTE: start in h2ogpt repo base directory
# i.e. can run below to update both spaces (assumes repos already existed, else will have to login HF for each)
# (h2ollm) jon@pseudotensor:~/h2ogpt$ ./spaces/chatbot/repo_to_spaces.sh h2ogpt-chatbot ; ./spaces/chatbot/repo_to_spaces.sh h2ogpt-chatbot2

spacename=${1:-h2ogpt-chatbot}
echo "Space name: $spacename"

# NOTE: start in h2ogpt repo base directory

h2ogpt_hash=`git rev-parse HEAD`

ln -sr generate.py gradio_runner.py gradio_themes.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py pdf_langchain.py create_data.py requirements.txt requirements_optional_pdf.txt spaces/chatbot/
cd ..

git clone https://huggingface.co/spaces/h2oai/"${spacename}"
cd "${spacename}"
git pull --rebase
rm -rf app.py generate.py gradio_runner.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py pdf_langchain.py create_data.py requirements.txt requirements_optional_pdf.txt
cd ../h2ogpt/spaces/chatbot/
cp generate.py gradio_runner.py gradio_themes.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py client_test.py pdf_langchain.py create_data.py requirements.txt requirements_optional_pdf.txt ../../../"${spacename}"/
cd ../../../"${spacename}"/

ln -s generate.py app.py

# for langchain support
mv requirements.txt requirements.txt.001
cat requirements.txt.001 requirements_optional_pdf.txt >> requirements.txt
rm -rf requirements.txt.001

git add app.py generate.py gradio_runner.py gradio_themes.py h2o-logo.svg LICENSE stopping.py prompter.py finetune.py utils.py pdf_langchain.py create_data.py client_test.py requirements.txt
git commit -m "Update with h2oGPT hash ${h2ogpt_hash}"
# ensure write token used and login with git control: huggingface-cli login --token <HUGGINGFACE_API_TOKEN> --add-to-git-credential
git push