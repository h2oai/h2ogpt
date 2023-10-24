import pytest
from tests.utils import wrap_test_forked
from src.enums import LangChainAction

from importlib.metadata import version

transformers_version = version('transformers')
# pip install packaging
from packaging import version

sufficient_transformers_version = version.parse(transformers_version) >= version.parse("4.31.0")

encoding = None


def num_tokens_from_string(string: str, model_name=None) -> int:
    """Returns the number of tokens in a text string."""
    global encoding
    if encoding is None:
        from transformers import AutoTokenizer
        encoding = AutoTokenizer.from_pretrained(model_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


import uuid


def make_key():
    return str(uuid.uuid4())[:8]


def make_value():
    return str(uuid.uuid4())[:4]


SECRET_KEY = make_key()
SECRET_VALUE = make_value()

ANSWER_LEN = 256  # allow space for answer (same as


def get_prompt(before, after):
    return f"[INST] {before}'{SECRET_KEY}' = '{SECRET_VALUE}'\n{after}\n\n What is the value of the key '{SECRET_KEY}'? [/INST]"


def create_long_prompt_with_secret(prompt_len=None, secret_pos=None, model_name=None):
    import time
    t0 = time.time()
    before = "## UUID key/value pairs to remember:\n\n"
    while num_tokens_from_string(before, model_name) < secret_pos:
        before += f"'{make_key()}' = '{make_value()}'\n"
    after = ""
    while num_tokens_from_string(after, model_name) < (prompt_len - secret_pos - ANSWER_LEN):
        after += f"'{make_key()}' = '{make_value()}'\n"
    prompt = get_prompt(before, after)
    assert SECRET_VALUE in prompt
    assert num_tokens_from_string(prompt, model_name) <= prompt_len
    t1 = time.time()
    print("time to create long prompt: %.4f" % (t1 - t0))
    return prompt


@pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-4096-llama2-13b-chat'])
@pytest.mark.parametrize("rope_scaling", [
    # None,
    # "{'type':'linear', 'factor':2}",
    "{'type':'dynamic', 'factor':2}",
    # "{'type':'dynamic', 'factor':4}"
])
@pytest.mark.parametrize("prompt_len", [
    # 2000, 4000,
    5000, 6000,
    # 7000, 8000, # OOM
])
@pytest.mark.parametrize("rel_secret_pos", [
    0.2,
    # 0.5,
    # 0.8
])
@pytest.mark.parametrize("client", [
    False,
    True
])
@pytest.mark.skipif(not sufficient_transformers_version, reason="Insufficient transformers version")
@wrap_test_forked
def test_gradio_long_context_uuid_key_value_retrieval(base_model, rope_scaling, prompt_len, rel_secret_pos, client):
    import ast
    rope_scaling_factor = 1
    if rope_scaling:
        rope_scaling = ast.literal_eval(rope_scaling)
        rope_scaling_factor = rope_scaling.get("factor")
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model, token=True,
                                        trust_remote_code=True)
    max_len = 4096
    if hasattr(config, 'max_position_embeddings'):
        max_len = config.max_position_embeddings
    if prompt_len > max_len * rope_scaling_factor:
        pytest.xfail("no chance")
    secret_pos = int(prompt_len * rel_secret_pos)
    prompt = create_long_prompt_with_secret(prompt_len=prompt_len, secret_pos=secret_pos, model_name=base_model)

    if client:
        main_kwargs = dict(base_model=base_model,
                           chat=True, stream_output=False,
                           gradio=True, num_beams=1,
                           prompt_type='plain',  # prompting done explicitly above, so can use with generate() below
                           block_gradio_exit=False,
                           rope_scaling=rope_scaling,
                           use_auth_token=True,
                           save_dir="long_context")
        from src.gen import main
        main(**main_kwargs)
        from src.client_test import run_client_chat
        res_dict, client = run_client_chat(
            prompt=prompt,
            stream_output=False, max_new_tokens=16384,
            langchain_mode='Disabled',
            langchain_action=LangChainAction.QUERY.value,
            langchain_agents=[]
        )
        assert res_dict['prompt'] == prompt
        assert res_dict['iinput'] == ''
        response = res_dict['response']
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map='auto',
            rope_scaling=rope_scaling,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        print(inputs.input_ids.shape)
        gen_out = model.generate(**inputs, max_new_tokens=300)
        response = tokenizer.batch_decode(gen_out)[0]
        response = response.split("</s>")[0]
        print(response)
        response = response.replace(prompt, "").replace("<s> ", "")  # only keep response

    print(f"\nLLM response (expected value is '{SECRET_VALUE}'):", flush=True)
    print(response)
    assert SECRET_VALUE in response
    print("DONE", flush=True)


@pytest.mark.parametrize("type", [
    None,
    # 'linear',
    'dynamic',
])
@pytest.mark.parametrize("factor", [
    1.0, 2.0, 4.0
])
@pytest.mark.parametrize("base_model", [
    "huggyllama/llama-7b",
    "meta-llama/Llama-2-7b-chat-hf"
])
@wrap_test_forked
@pytest.mark.skipif(not sufficient_transformers_version, reason="Insufficient transformers version")
def test_huggyllama_transformers_pr(base_model, type, factor):
    if type is None and factor > 1.0:
        pytest.xfail('no point')
    if type and factor == 1.0:
        pytest.xfail('no point')
    rope_scaling = {'type': type, 'factor': factor} if type else None

    # https://github.com/huggingface/transformers/pull/24653#issue-1788278122
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map='auto',
        rope_scaling=rope_scaling,
    )

    prompt = '''You are given this machine learning research paper, please read it carefully and answer the follow up question.

=== BEGIN ===

2306.15595v2 [cs.CL] 28 Jun 2023

arXiv

EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION

Shouyuan Chen Sherman Wong Liangjian Chen  Yuandong Tian
Meta Platforms Inc.
{chenshouyuan, shermanwong, cli, yuandong}@meta . com

1 INTRODUCTION

Large language models (LLMs) typically come with a pre-defined context window size. For exam-
ple, inputs to LLaMA models (Touvron et al., 2023) must be fewer than 2048 tokens. This pre-set
context window limit is frequently exceeded in applications such as conducting long conversations,
summarizing long documents, or executing long-term planning. For these applications, LLMs with
longer context windows are preferred. However, training an LLM from scratch with long context
windows requires significant investments. This naturally leads to a question: Can we extend the
context window of an existing pre-trained LLM?

One straightforward approach is to fine-tune an existing pre-trained Transformer with a longer con-
text window. However, empirically, we found that models trained this way adapt to long context
windows very slowly. After training for more than 10000 batches, the effective context window
saw a minimal increase, moving from 2048 to 2560 (Table 4). This suggests that such method is
inefficient for extending to substantially longer context windows.

While certain techniques such as ALiBi (Press et al., 2022) and LeX (Sun et al., 2022) enable length
extrapolation of Transformers, i.e. train on short context windows and inference on longer ones,
many existing pre-trained LLMs, including LLaMA (Touvron et al., 2023), use positional encodings
that have weak extrapolation properties (e.g., RoPE (Su et al., 2021)). Therefore, the applicability
of these techniques for extending the context window sizes of such LLMs remains limited.

In this work, we introduce Position Interpolation to enable context window extensions for certain
existing pre-trained LLMs, including LLaMA. The key idea is, instead of extrapolation, we directly
down-scale the position indices so that the maximum position index matches the previous context
window limit in the pre-training stage. See Figure 1 for an illustration. In other words, to accom-
modate more input tokens, we interpolate the position encodings at neighboring integer positions,
utilizing the fact that position encodings can be applied on non-integer positions, as opposed to
extrapolating outside the trained positions, which may lead to catastrophic values. We verify our
approach theoretically, by showing that the interpolated attention score has a much smaller upper

bound (~ 600x smaller in LLaMA 7B setting) than the extrapolated one, and is thus much more
stable. Therefore, interpolated position encodings are easier for the model to adapt.

Empirically, we found that Position Interpolation is highly effective and efficient, requiring only a
very short period of fine-tuning for the model to fully adapt to greatly extended context windows.
We present experimental results for extending the context window to up to 32768 from the initial
2048 across 7B to 65B LLaMA models using Position Interpolation. Our results show that

1. Position Interpolation can easily enable very long context windows (e.g. 32768), requiring
only fine-tuning for 1000 steps on the Pile (Gao et al., 2020) to achieve a good quality.
The cost of fine-tuning is negligible compared to the pre-training costs. This confirms
our hypothesis that it is relatively easy for the models to adapt to interpolated position
encodings.

2. Position Interpolation generates strong models that can effectively make use of much ex-
tended context window. We show that models extended by Position Interpolation enjoy
significant perplexity gains from greatly extended context windows for text modeling, and
we show that the perplexity reduces graceful with the enlargement of context windows.
We also applied Position Interpolation in a long text summarization task, and demonstrate
competitive performances.

3. Position Interpolation preserves model quality relatively well for tasks within its original
context window sizes. We present a variety of evaluation results for the extended LLaMA
models on the original LLaMA benchmark. Compared with original LLaMA models, the
extended LLLaM A models saw a minor degradation on several standard benchmarks within
a 2048 token limit.

Our results highlight the innate ability of Transformer models to “extrapolate to sequence lengths
longer than the ones encountered during training” as hypothesized in the seminal work of Vaswani
et al. (2017). We reaffirm this hypothesis and suggest that the previously known weakness of ex-
trapolating to longer sequences for language modeling (Press et al., 2022) may be due to direct

extrapolation of positional encodings and it can be largely mitigated by interpolating position en-
codings instead.

Concurrent work. Right before our release, we are informed with a concurrent blogpost (Super-
HOT kaiokendev (2023)) that also interpolates positional encoding in RoPE to extend the context
window from 2K to 8K. Recently, open source community picks it up in Reddit post ! and Github
Issues 2, which shows that fine-tuning with LoRA (Hu et al., 2021) also seems to work well. Our
paper shows a full fine-tuning with up to 65B model work well with Position Interpolation, and we
also give theoretical explanations why interpolation achieves much more stable results than extrap-
olation, by showing that the upper bound of interplated attention score is much lower than that of
extrapolated ones.

2 METHOD

2.1 BACKGROUND: ROTARY POSITION EMBEDDING (ROPE)

Transformer models require explicit positional information to be injected, typically in the form of
positional encodings, to represent the order of inputs. We consider Rotary Position Embedding
(ROPE) (Su et al., 2021), which is the position encoding used in the LLLaMA model (Touvron et al.,
2023). Given a position index m € [0, ¢) and an embedding vector x := [zg, 71,..., 241], Where
d is the dimension of the attention head, RoPE defines a vector-valued complex function f{x, m) as
follows

Using RoPE, the self-attention score
is only dependent on relative position m — 7 through trigonometric functions. Here q and k are the
query and key vector for a specific attention head. At each layer, RoPE is applied on both query and
key embeddings for computing attention scores.

2.2 DIRECT EXTRAPOLATION

While the attention score in RoPE only depends on the relative positions, which is what we want,
its extrapolation performance is not great . In particular, when directly extending to larger context
windows unseen in the training, the perplexity may shoot up to very high numbers (i.e., > 10%),
comparable to untrained models.

Ideally, we want to see the model trained on a context window of size L = 2048 to still work
reasonably well on longer context window, but may not have the capability to leverage information
that appears beyond L. For example, to answer a question located at 3000, the model trained on
maximal window size of I = 2048 cannot leverage evidences provided at location 0, but still
can leverage the evidences provided at location 2900. In contrast, in reality we see catastrophic
behaviors, i.e., question at location 3000 cannot be answered correctly, even if the evidences are
located at location 2900.

What is the reason behind? How could this happen if the attention score a,,,—,, decays as the relative
distance |m — n/| increases, according to Section 3.4.3 of (Su et al., 2021), and content from very
far distances should not matter that much? It turns out that the upper bound derived in Section 3.4.3
of (Su et al., 2021) may be too loose: while it indeed decays with respect to |m — nl, the bound
can still be quite large (i.e., the bound can be critically depends on the magnitude of v;) and thus
vacuous. In fact, if we treat all trigonometric functions as basis functions (i.e, ¢;(s) := #93), and
think about Eqn. 2 as basis expansion as the following:

where s is the positional span between a query and a key and h; := (ga; + igaj+1){k2j — tk2j+1)
are complex coefficients depending on q and k (here the definition of h; is exactly the same as the
definition of k; in Sec 3.4.3 in RoPE (Su et al., 2021)). Now the the issue becomes clear: as shown
in Fig. 2, a, can be small in magnitude in the range of [0, 2048], but gives huge values out of the
region. The underlying reason is that the trigonometric family {¢;} (with sufficiently large d) is
a universal approximator and can fit any arbitrary functions. Therefore, for a, there always exist
coefficients {h;} (i.e. key and query) that corresponds to small function values in [0, 2048] but

much larger in regions beyond.

2.3 PROPOSED APPROACH: POSITION INTERPOLATION (PI)

In Fig. 2, thanks to the smoothness of bases functions ¢; interpolation is much more stable and will
not lead to wild values. Therefore, instead of extrapolate the attention score in Eqn. 3 to s > L,
how about we define an attention score a{s) = a(Ls/L’) where L’ is the longer context window?
Formally, we replace RoPE f by {’ defined as follows

We call this transformation on the position encoding Position Interpolation. In this step, we reduce
position indices from [0, L') to [0, L) to match the original range of indices before computing RoPE.
Consequently, as inputs to RoPE, the maximum relative distance between any two tokens has been
reduced from I’ to L. Since we align the ranges of position indices and relative distances before
and after extension, we mitigate the effect on attention score computation due to context window
extensions, which can allow the model easier to adapt. To further demonstrate this is the case, in the
following theorem, we show that the interpolated attention score is well-behaved:

While there is no close form for B(s) := 4/21 |Ag41(s)|, numerically it is at least larger than d, and for many positional difference s, B(s) is much larger than d
(check Appendix B for the plot). Therefore, the interpolation bound is at least 2 - 294.73 ~ 600 x
smaller than the extrapolation bound, and thus the interpolated attention score is much more stable
than extrapolated one.

Notably, our method of rescaling of position indices does not introduce extra weight, or modify
the model architecture in any way. This makes it attractive in practical applications, since most
infrastructure and optimization for the original model can be reused after the extension.

Fine-tuning. We can further fine-tune the interpolated model using the next token prediction task
with interpolated position encodings on the extended context window size using a pre-training cor-
pus such as the Pile (Gao et al., 2020). In the next section, we show that our fine-tuning process
only needs tens to hundreds thousands of examples. We also find that the result of the fine-tuning
is not sensitive to the choice of examples. The reason may be that the model is only adapting to the
new context window during the fine-tuning phase, starting from a good initialization, as opposed to
acquiring new knowledge.

Other ways to reduce interpolation/extrapolation bound. From the expression of the interpola-
tion (Eqn. 5) and extrapolation bound (Eqn. 8), a common term is max; ||, which is the maximal
magnitude of query/key products. If we enforce a regularization on || during LLM training, it is
possible that the catastrophic extrapolation error can be mitigated or even resolved. In fact, if we
apply ridge regression with proper regularization to fit a curve in Fig. 2, the magnitude of extrapo-
lated a(s) when s > L can be comparable to that within [0, L]. To our knowledge, we are not aware
of existing LLM pre-training techniques that leverage this regularization and will leave it for future
work.

3 EXPERIMENTS

We show Position Interpolation can effectively extend context window up to 32 times of the original
size, and such extension can be done with only several hundreds of training steps. We show the
resulting models are strong LLMs with fully effective long context windows. We demonstrate its
performance in a number of tasks including language modeling, passkey retrieval, and long doc-
ument summarization. We also present benchmark results of the extended models on the original
LLaMA evaluation benchmarks.
3.1 SETUP

Model Variants. We extended the pre-trained 7B, 13B, 33B and 65B LLaMA models (Touvron
et al., 2023) to various context window of sizes up to 32768, using either direct fine-tuning or
Position Interpoloation method. Except for rescaling the position indices for models extended with
Position Interpolation, we did not modify LLaMA model architectures (Touvron et al., 2023) in any
ways.

Training Procedure. We fine-tune all model variants using the next token prediction objective. We
use AdamW (Loshchilov & Hutter, 2019) with 5; = 0.9 and 2 = 0.95. We use a linear learning
rate warmup of 20 steps starting from 10% of the maximum learning rate. For 7B and 13B models,
we set the learning rate to 2 x 1075 and for 33B and 65B models we set the learning rate to 1072. We
set the weight decay to zero. For extending 7B, 13B and 33B models to the 8192 context window
size, we use 32 A100 GPUs and 64 global batch size. For all other cases we use 128 A100 GPUs and
128 global batch size. We note that the main need of using more GPUs is memory limitation during
fine-tuning, and it is possible to use fewer GPUs in certain cases. We train all models using PyTorch
(Paszke et al., 2019) with Fully Sharded Data Parallel (Zhao et al., 2023) and Flash Attention (Dao
et al., 2022).

If not specified otherwise, for the Position Interpolation method, we fine-tune the models for 1000
steps. For the direct fine-tuning method, we use 10000 steps. We primarily fine-tune using the Pile
training dataset (Gao et al., 2020). In Section 3.4 we also compared fine-tuning performance on the
RedPajama dataset (Computer, 2023).

3.2 LONG SEQUENCE LANGUAGE MODELING

We evaluate the long sequence language modeling performance of our extended models and base-
lines on two datasets: book corpus (PG-19) (Rae et al., 2020) and cleaned Arxiv Math proof-pile
dataset (Azerbayev et al., 2022).

We use the test splits of PG19 (Rae et al., 2020) and proof-pile (Azerbayev et al., 2022). For PG19,
we use the whole test split consisting of 100 documents. For the proof-pile dataset, we use a random
subsample of 128 documents with at least 32768 SentencePiece (Kudo & Richardson, 2018) tokens
and truncate to the first 32768 tokens for each test document. We evaluate perplexity at various
context window size by using a sliding window approach following Press et al. (2022) with stride
S = 256.

In Table 1 and Table 2, we report the perplexity results for our models and baselines on the datasets.
From the results, we found that models extended with our method enjoy a significantly improved
perplexity from longer context window sizes. By increasing the context window size from 2048 to
16384, we observed -0.28 and -0.5 reductions of perplexity for extending LLaMA 7B models on
both datasets, -0.27 and -0.48 reductions for extending LL.aMA 13B models, and -0.14 and -0.42
reductions for extending LLaMA 33B models. For LLaMA 65B models, we observed -0.12 and
-0.3 reductions of perplexity by extending to the 8192 context window size.

In general, we observed a consistent trend of our models achieving better perplexity with longer
context windows. This indicates our models can effectively make use of the longer context windows
to better predict next tokens in language modeling tasks. Moreover, we found this trend extends to
32768 window size without diminishing on the PG19 dataset for LLaMA 7B and 13B models. This
indicates that our method may enable extension to even longer context windows.

In contrast, we observed that models extended via the direct fine-tuning method has shown regres-
sion (up to +0.48) or minor improvement (up to -0.12) on the perplexity at longer context windows.
This indicates that models extended this way have limited capability of making use of context win-
dows longer than their pre-trained settings.

We saw a minor degradation of the perplexity on the original context window of 2048 for our ex-
tended models in some cases. For example, on the Proof-pile dataset, we saw a degradation ranging
from 0.01 to 0.05 across all models with extended with Position Interpolation. A small degradation
of performance within original evaluation context window is expected since Position Interpolation
forces position encodings in original context window to reside in a much narrower region, which
may negatively affect the language model’s performance. We present more benchmark results on
the original context window size in Section 3.4.

In Table 3 we report the relationship between perplexity and the number of fine-tuning steps for
LLaMA 7B model extending to 8192 and 16384 context window sizes using Position Interpolation
evaluated on the PG19 dataset. We can see without fine-tuning (at step 0) the model can exhibit
certain language modeling capability, as indicated by < 20 perplexity for extending to 8192 context
window (in contrast, the direct extrapolation method leads to > 10% perplexity). With fine-tuning,
we observed that the perplexity improves quickly. At 200 steps the models surpassed the original
model’s perplexity on 2048 context window size, indicating the models gaining ability of effectively
using sequences longer than the pre-training settings for language modeling. At 1000 steps, we can
see the models have improved steadily and achieve a significantly better perplexity.

3.3 MEASURING EFFECTIVE CONTEXT WINDOW SIZE THROUGH PASSKEY RETRIEVAL

We study the effective context window size, i.e. the maximum distance of a token can effectively
attend to during inference, of our models after extension. To measure this, we follow a synthetic
evaluation task of passkey retrieval proposed by Mohtashami & Jaggi (2023). In this task, the models
are asked to recover a random passkey hidden in a long document. See Figure 3 for the format of
the document.

Given a language model, we estimate the upper and lower bounds of effective context windows as
follows. Suppose the random passkey is k tokens away from the end of the input. When a model
persistently fails to retrieve the correct passkey value across several independent attempts, it suggests
that the effective context window size of the model is less than k. Conversely, if a model consistently
succeeds in retrieving the correct passkey value, we deduce that the effective context window size
of the model is at least k.

We evaluate the 7B and 33B LLaMA model variants that are extended via Position Interpolation or
direct fine-tuning. For each model, we use 32 different &£ uniformly spaced in the targeted context
window L’ and run the above tests for 10 times for each k, where each time a random passkey of 5
random digits is used. In Table 4, we report kyax as a function of the number of fine-tuning steps,

We can see that models extended via Position Interpolation all successfully attain their desired ex-
tension objectives in terms of effective context window sizes, indicating by the effective context
window size reaching maximum kp, = L/, after merely fine-tuning for 200 steps, consistently
across both 7B and 33B model sizes and up to 32768 context windows. In contrast, LLLaMA models
that are extended via direct fine-tuning only saw a minimal increase of the effective context win-
dow size kay from 2048 to 2560, even after fine-tuning for more than 10000 steps, with no clear
indication of an acceleration in the increase of window size.

3.4 BENCHMARKS ON ORIGINAL CONTEXT WINDOW SIZE

We evaluate the models extended by Position Interpolation on several standard benchmark tasks
within the original context window size of 2048. The evaluation results are listed in Table 5. From
the results, we saw that models extended to 8192 produce comparable results on the original bench-
mark which is designed for a much smaller context window, with a degradation of up to 2% on
the benchmark tasks, for both 7B and 33B model sizes. Models extended to longer context win-
dows regressed more on the benchmarks, but still in reasonable ranges for most tasks. We also note
that the choice of fine-tuning datasets does not seem to lead significant difference in the benchmark
performances, which may be due to the limited number of fine-tuning steps used in our method.
The regression on benchmark tasks is consistent with our observation on perplexity regression in
Section 3.2.

3.5 LONG DOCUMENT SUMMARIZATION

In this task, we evaluate our models’ performance on the long document summarization task. In
particular, we consider the GovReport (Huang et al., 2021) dataset, which contains 17457 documents
for training and 972 documents for evaluation. Each document comes with a human generated
summary. We truncate all input documents to their first 15000 tokens.

We fine-tune the LL.aMA models extended with Position Interpolation with a context window of
16384. Note the rescaling of position indices are still required during this fine-tuning step. We first
Model Size Context Window Fine-tune on  BoolQ PIQA Race-M Race-H WinoGrande

format the raw document using the prompt template in Figure 4, and then concatenate the prompt
with the ground-truth summary (truncate to 1000 tokens) associated with each document. We fine-
tune the model using the next token prediction task with the above setup for 10 epochs. The losses
from the input prompt proportion of training examples are excluded during our fine-tuning.

We use a generation temperature of 0.5 and top, = 0.95 as our inference parameter to generate a
summarization of each document in the test set. The final output is truncated at 1000 tokens. We
used the ROUGE-1/ROUGE-2/ROUGE-L scores (Lin, 2004) as the evaluation metrics to evaluate
the models’ outputs vs the ground-truth summaries.

In Table 6 we report our evaluation results. We have also included results from two baselines in
existing SCROLLS Leaderboard (Shaham et al., 2022; Ainslie et al., 2023). In general, we have
obtained competitive R1 score among other models with minimal tuning of hyper-parameters. This
result suggests our models with 16384 context window can effectively handle the long document
summarization task.

=== END OF FILE ===

'''
    question = "Question: What's the title of this paper?"  # Something from the beginning

    inputs = tokenizer(prompt + question, return_tensors="pt").to("cuda")

    print(inputs.input_ids.shape)
    assert inputs.input_ids.shape[1] > 6200, "input not long enough"

    gen_out = model.generate(**inputs, max_new_tokens=100)
    response = tokenizer.batch_decode(gen_out)[0]
    response = response.replace(prompt + question, "")  # only keep response
    assert len(response) < 500, "response must be less than 100 tokens"
    print(response)
    if rope_scaling is None:
        assert 'Extending Context Window of Large' not in response
        assert 'Extending Context Window of Large'.upper() not in response
    else:
        assert ('Extending Context Window of Large' in response or
                'Extending Context Window of Large'.upper() in response)
