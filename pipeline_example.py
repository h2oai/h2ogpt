from transformers import AutoModelForCausalLM, AutoTokenizer
from h2oai_pipeline import H2OTextGenerationPipeline
import textwrap as tr

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", device_map="auto")

generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

input = "Explain to me the difference between nuclear fission and fusion."
outputs = generate_text(input, max_new_tokens=200)
for output in outputs:
    print(tr.fill(output['generated_text'], width=40))

# Generated text should be similar to below.
"""
Nuclear fusion and fission are two ways
that energy can be released from atoms.
While both processes involve the
splitting of atoms, they differ in how
they do so. In nuclear fission, atoms
are split apart, or fission, when they
are hit by neutrons that have been given
additional energy. The splitting of the
atom releases a lot of energy, as well
as other particles, including other
neutrons. In nuclear fusion, two or more
atoms are combined together to form a
single larger atom, releasing a lot of
energy in the process. The fusion of two
atoms into one is called a nuclear
reaction.
"""