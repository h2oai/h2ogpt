import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import quant

from gptq import GPTQ, Observer
from utils import (
    find_layers,
    DEV,
    get_loaders,
)


def get_neox(model, seqlen=-1):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import GPTNeoXForCausalLM

    model = GPTNeoXForCausalLM.from_pretrained(model, torch_dtype=torch.float16)
    model.seqlen = seqlen if seqlen != -1 else model.config.max_position_embeddings
    return model


@torch.no_grad()
def neox_sequential(model, dataloader, dev):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.gpt_neox.layers

    model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.gpt_neox.embed_in = model.gpt_neox.embed_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    print("Ready.")

    quantizers = {}
    observer = Observer()
    for i in range(len(layers)):

        print(f"Quantizing layer {i+1}/{len(layers)}..")
        print("+------------------+--------------+------------+-----------+-------+")
        print("|       name       | weight_error | fp_inp_SNR | q_inp_SNR | time  |")
        print("+==================+==============+============+===========+=======+")

        layer = layers[i].to(dev)
        full = find_layers(layer)
        sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name], observe=False)
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=args.act_order,
                    name=name,
                )
                quantizers["gpt_neox.layers.%d.%s" % (i, name)] = (
                    gptq[name].quantizer.cpu(),
                    scale.cpu(),
                    zero.cpu(),
                    g_idx.cpu(),
                    args.wbits,
                    args.groupsize,
                )
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print("+------------------+--------------+------------+-----------+-------+")
        print("\n")

    model.config.use_cache = use_cache

    return quantizers


@torch.no_grad()
def neox_eval(model, testenc, dev):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.gpt_neox.layers

    model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.gpt_neox.embed_in = model.gpt_neox.embed_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = quant.Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantizer.quantize(W).to(
                    next(iter(layer.parameters())).dtype
                )

        for j in range(nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.to(dev)
    model.embed_out = model.embed_out.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.gpt_neox.final_layer_norm(hidden_states)
        lm_logits = model.embed_out(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


# TODO: perform packing on GPU
def neox_pack(model, quantizers, wbits, groupsize):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    quant.make_quant_linear(model, quantizers, wbits, groupsize)
    qlayers = find_layers(model, [quant.QuantLinear])
    print("Packing ...")
    for name in qlayers:
        print(name)
        quantizers[name], scale, zero, g_idx, _, _ = quantizers[name]
        qlayers[name].pack(layers[name], scale, zero, g_idx)
    print("Done.")
    return model


def load_quant(model, checkpoint, wbits, groupsize=-1, eval=True, warmup_autotune=True):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, modeling_utils

    config = GPTNeoXConfig.from_pretrained(model)

    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = GPTNeoXForCausalLM(config)
    torch.set_default_dtype(torch.float)
    if eval:
        model = model.eval()
    layers = find_layers(model)
    for name in ["embed_in", "embed_out"]:
        if name in layers:
            del layers[name]
    quant.make_quant_linear(model, layers, wbits, groupsize)

    del layers

    print("Loading model ...")
    if checkpoint.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load

        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))

    if warmup_autotune:
        quant.autotune_warmup_linear(model, transpose=not (eval))

    model.seqlen = model.config.max_position_embeddings
    print("Done.")

    return model


def neox_multigpu(model, gpus):
    model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(gpus[0])
    model.gpt_neox.final_layer_norm = model.gpt_neox.final_layer_norm.to(gpus[-1])
    import copy

    model.embed_out = copy.deepcopy(model.embed_out).to(gpus[-1])

    cache = {"mask": None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device

        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache["mask"] is None or cache["mask"].device != self.dev:
                cache["mask"] = kwargs["attention_mask"].to(self.dev)
            kwargs["attention_mask"] = cache["mask"]
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.gpt_neox.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus


def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else DEV)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(model.gpt_neox.layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i : i + 1],
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(
                    out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)
                ).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()
        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
            print("max memory(MiB):", max_memory)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="llama model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--nearest", action="store_true", help="Whether to run the RTN baseline."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--seqlen",
        type=int,
        default=-1,
        help="seqlen to use for quantization; default uses full seqlen",
    )
    parser.add_argument(
        "--trits", action="store_true", help="Whether to use trits for quantization."
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=-1,
        help="Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument("--eval", action="store_true", help="evaluate quantized model.")
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Save quantized checkpoint under this name.",
    )
    parser.add_argument(
        "--save_safetensors",
        type=str,
        default="",
        help="Save quantized `.safetensors` checkpoint under this name.",
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Number of tokens to use for benchmarking.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to compute perplexity during benchmarking for verification.",
    )
    parser.add_argument(
        "--sym", action="store_true", help="Whether to perform symmetric quantization."
    )
    parser.add_argument(
        "--act-order",
        action="store_true",
        help="Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--new-eval", action="store_true", help="Whether to use the new PTB and C4 eval"
    )
    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()

    if args.load:
        model = load_quant(args.model, args.load, args.wbits, args.groupsize)
    else:
        model = get_neox(args.model)
        model.eval()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if not args.load and args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = neox_sequential(model, dataloader, DEV)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            neox_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, : args.benchmark]
            benchmark(model, input_ids, check=args.check)

    if args.eval:
        datasets = ["wikitext2", "ptb", "c4"]
        if args.new_eval:
            datasets = ["wikitext2", "ptb-new", "c4-new"]
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            print(dataset)
            neox_eval(model, testloader, DEV)

    if args.save:
        neox_pack(model, quantizers, args.wbits, args.groupsize)
        torch.save(model.state_dict(), args.save)

    if args.save_safetensors:
        neox_pack(model, quantizers, args.wbits, args.groupsize)
        from safetensors.torch import save_file as safe_save

        state_dict = model.state_dict()
        state_dict = {k: v.clone().contiguous() for k, v in state_dict.items()}
        safe_save(state_dict, args.save_safetensors)
