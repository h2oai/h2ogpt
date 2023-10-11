

# %%
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# %%
# Read the json file
# This file processes the llm_gpu_benchmark.json file in the tmp/inputs folder
# File is generated using the command
# curl  -sSL https://raw.githubusercontent.com/h2oai/h2ogpt/main/benchmarks/perf.json | jq -s '.' > llm_gpu_benchmarks.json
with open('llm_gpu_benchmarks.json') as f:
    data = json.load(f)
del f

# %%
# Read the json file into a dataframe
df = pd.json_normalize(data)
del data

# %%
# Process the dataframe
# Drop columns that are not needed
df.drop(columns=['task', 'ngpus', 'reps', 'date', 'git_sha', 'transformers', 'bitsandbytes', 'cuda', 'hostname',
                 'summarize_input_len_bytes'], inplace=True)
# Rename columns
df.rename(columns={'n_gpus': 'gpu_count'}, inplace=True)
# Split the gpu column into gpu and gpu_memory
df["gpu_name"] = df.gpus.str.extract(r'[1-9] x ([\w\- ]+) .+')
df["gpu_memory_gb"] = round(
    pd.to_numeric(df.gpus.str.extract(r'[\w ]+ \(([\d]+) .+', expand=False), errors='coerce') / 1024)
df["gpu_memory_gb"] = df["gpu_memory_gb"].astype('Int64')
df.drop(columns=['gpus'], inplace=True)
# Manage gpu_names
df.gpu_name = df.gpu_name.str.replace('NVIDIA ', '')
df.gpu_name = df.gpu_name.str.replace('GeForce ', '')
df.gpu_name = df.gpu_name.str.replace('A100-SXM4-80GB', 'A100 SXM4')
df.gpu_name = df.gpu_memory_gb.astype(str) + "-" + df.gpu_name
# Remove CPUs
df.drop(df[df.gpu_name.isnull()].index, inplace=True)

# %%
# Remove duplicate rows
df.drop_duplicates(['backend', 'base_model', 'bits', 'gpu_count', 'gpu_name'], inplace=True)

# %% Add baseline comparison columns
# Looking at the CPU data for 4, 8, and 16 bit quantization values for the benchmark we are simplifying it to a single
# value
cpu_summary_out_throughput = 1353 / 1216  # bytes/second  (calculated from summarize_output_len_bytes / summarize_time)
cpu_generate_out_throughput = 849 / 180  # bytes/second   (calculated from generate_output_len_bytes / generate_time)

# add GPU throughput columns
df["summary_out_throughput"] = df.summarize_output_len_bytes / df.summarize_time
df["generate_out_throughput"] = df.generate_output_len_bytes / df.generate_time
# add GPU throughput boost columns
df["summary_out_throughput_normalize"] = df.summary_out_throughput / cpu_summary_out_throughput
df["generate_out_throughput_normalize"] = df.generate_out_throughput / cpu_generate_out_throughput

# %%
# df.to_excel('tmp/scratchpad/output/llm_gpu_benchmarks.xlsx', index=False)

# %%
pio.renderers.default = "browser"

# %%
bits_bar_colors = {'4': px.colors.qualitative.D3[0],
                   '8': px.colors.qualitative.D3[1],
                   '16': px.colors.qualitative.D3[2]}

backends = list(df.backend.unique())
base_models = list(df.base_model.unique())
n_gpus = list(df.gpu_count.unique())

# %%
for backend in backends:
    # for backend in ['transformers']:
    fig_bar = make_subplots(rows=len(n_gpus),
                            cols=len(base_models) * 2,
                            shared_xaxes='all',
                            shared_yaxes='columns',
                            start_cell="top-left",
                            vertical_spacing=0.1,
                            print_grid=False,
                            row_titles=[f'{gpu_count} GPUs' for gpu_count in n_gpus],
                            column_titles=['llama2-7b-chat Summarization', 'llama2-7b-chat Generation',
                                           'llama2-13b-chat Summarization', 'llama2-13b-chat Generation',
                                           'llama2-70b-chat Summarization', 'llama2-70b-chat Generation'],)

    # for base_model in ['h2oai/h2ogpt-4096-llama2-7b-chat']:
    for base_model in base_models:
        for gpu_count in n_gpus:
            for bits in sorted(df.bits.unique()):
                sub_df = df[(df.backend == backend) &
                            (df.base_model == base_model) &
                            (df.gpu_count == gpu_count) &
                            (df.bits == bits)].sort_values(by='gpu_name')
                fig_bar.add_trace(go.Bar(x=sub_df.summary_out_throughput_normalize,
                                         y=sub_df.gpu_name,
                                         name=f'sum-{bits} bits',
                                         legendgroup=f'sum-{bits} bits',
                                         marker=dict(color=bits_bar_colors[f'{bits}']),
                                         orientation='h'),
                                  row=n_gpus.index(gpu_count) + 1,
                                  col=base_models.index(base_model) * 2 + 1)
                fig_bar.add_trace(go.Bar(x=sub_df.generate_out_throughput_normalize,
                                         y=sub_df.gpu_name,
                                         name=f'gen-{bits} bits',
                                         legendgroup=f'gen-{bits} bits',
                                         marker=dict(color=bits_bar_colors[f'{bits}']),
                                         orientation='h'),
                                  row=list(n_gpus).index(gpu_count) + 1,
                                  col=list(base_models).index(base_model) * 2 + 2)

    fig_bar.update_layout(plot_bgcolor='rgb(250,250,250)',
                          showlegend=True,
                          barmode="group")
    # fig_bar.show()
    fig_bar.write_html(f'llm_gpu_benchmark_{backend}.html', include_plotlyjs='cdn')