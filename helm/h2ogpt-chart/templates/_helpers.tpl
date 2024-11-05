{{/*
Expand the name of the chart.
*/}}
{{- define "h2ogpt.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "h2ogpt.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Allow the release namespace to be overridden.
*/}}
{{- define "h2ogpt.namespace" -}}
{{- default .Release.Namespace .Values.namespaceOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}


{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "h2ogpt.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "h2ogpt.labels" -}}
helm.sh/chart: {{ include "h2ogpt.chart" . }}
{{ include "h2ogpt.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "h2ogpt.selectorLabels" -}}
app.kubernetes.io/name: {{ include "h2ogpt.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "h2ogpt.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "h2ogpt.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Config for h2oGPT
*/}}

{{- define "h2ogpt.config" -}}
{{- with .Values.h2ogpt }}
verbose: {{ default "True" .overrideConfig.verbose }}
{{- if .overrideConfig.heap_app_id }}
heap_app_id: {{ .overrideConfig.heap_app_id }}
{{- end }}
num_async: {{ default 10 .overrideConfig.num_async }}
save_dir: {{ default "/docker_logs" .overrideConfig.save_dir }}
score_model: {{ default "None" .overrideConfig.score_model }}
share: {{ default "False" .overrideConfig.share }}
enforce_h2ogpt_api_key: {{ default "False" .overrideConfig.enforce_h2ogpt_api_key }}
enforce_h2ogpt_ui_key: {{ default "False" .overrideConfig.enforce_h2ogpt_ui_key }}
{{- if .overrideConfig.h2ogpt_api_keys }}
h2ogpt_api_keys: {{ .overrideConfig.h2ogpt_api_keys }}
{{- end }}
{{- if .overrideConfig.use_auth_token }}
use_auth_token: {{ .overrideConfig.use_auth_token }}
{{- end }}
visible_models: {{ default "['meta-llama/Meta-Llama-3.1-8B-Instruct']" .overrideConfig.visible_models }}
visible_vision_models: {{ default "['mistralai/Pixtral-12B-2409']" .overrideConfig.visible_vision_models }}
top_k_docs_max_show: {{ default 100 .overrideConfig.top_k_docs_max_show }}
{{- if .overrideConfig.admin_pass }}
admin_pass: {{ .overrideConfig.admin_pass }}
{{- end }}
{{- if .openai.enabled }}
openai_server: "True"
openai_port: 5000
openai_workers: {{ default 5 .openai.openai_workers }}
{{- end }}
{{- if .agents.enabled }}
agent_server: "True"
agent_port: 5004
agent_workers: {{ .agents.agent_workers }}
{{- end }}
function_server: {{ default "True" .overrideConfig.function_server }}
function_port: 5002
function_server_workers: {{ default 1 .overrideConfig.function_server_workers }}
multiple_workers_gunicorn: {{ default "True" .overrideConfig.multiple_workers_gunicorn }}
llava_model: {{ default "openai:mistralai/Pixtral-12B-2409" .overrideConfig.llava_model }}
enable_llava: {{ default "True" .overrideConfig.enable_llava }}
{{- if ge (int (index .resources.requests "nvidia.com/gpu") ) (int 1) }}
enable_tts: {{ default "False" .overrideConfig.enable_tts }}
enable_stt: {{ default "True" .overrideConfig.enable_stt }}
enable_transcriptions: {{ default "True" .overrideConfig.enable_transcriptions }}
asr_model: {{ default "distil-whisper/distil-large-v3" .overrideConfig.asr_model }}
pre_load_embedding_model: {{ default "True" .overrideConfig.pre_load_embedding_model }}
pre_load_image_audio_models: {{ default "True" .overrideConfig.pre_load_image_audio_models }}
cut_distance: {{ default 10000 .overrideConfig.cut_distance }}
hf_embedding_model: {{ default "BAAI/bge-large-en-v1.5" .overrideConfig.hf_embedding_model }}
enable_captions: {{ default "False" .overrideConfig.enable_captions }}
enable_doctr: {{ default "True" .overrideConfig.enable_doctr }}
{{- else }}
enable_tts: {{ default "False" .overrideConfig.enable_tts }}
enable_stt: {{ default "False" .overrideConfig.enable_stt }}
enable_transcriptions: {{ default "False" .overrideConfig.enable_transcriptions }}
embedding_gpu_id: {{ default "cpu" .overrideConfig.embedding_gpu_id }}
hf_embedding_model: {{ default "fake" .overrideConfig.hf_embedding_model }}
pre_load_embedding_model: {{ default "False" .overrideConfig.pre_load_embedding_model }}
pre_load_image_audio_models:  {{ default "False" .overrideConfig.pre_load_image_audio_models }}
enable_captions: {{ default "False" .overrideConfig.enable_captions }}
enable_doctr: {{ default "False" .overrideConfig.enable_doctr }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Config for agents
*/}}

{{- define "agents.config" -}}
{{- with .Values.agents }}
verbose: {{ default "True" .overrideConfig.verbose }}
{{- if .overrideConfig.heap_app_id }}
heap_app_id: {{ .overrideConfig.heap_app_id }}
{{- end }}
num_async: {{ default 10 .overrideConfig.num_async }}
save_dir: {{ default "/docker_logs" .overrideConfig.save_dir }}
score_model: {{ default "None" .overrideConfig.score_model }}
share: {{ default "False" .overrideConfig.share }}
enforce_h2ogpt_api_key: {{ default "False" .overrideConfig.enforce_h2ogpt_api_key }}
enforce_h2ogpt_ui_key: {{ default "False" .overrideConfig.enforce_h2ogpt_ui_key }}
{{- if .overrideConfig.h2ogpt_api_keys }}
h2ogpt_api_keys: {{ .overrideConfig.h2ogpt_api_keys }}
{{- end }}
{{- if .overrideConfig.use_auth_token }}
use_auth_token: {{ .overrideConfig.use_auth_token }}
{{- end }}
visible_models: {{ default "['meta-llama/Meta-Llama-3.1-8B-Instruct']" .overrideConfig.visible_models }}
visible_vision_models: {{ default "['mistralai/Pixtral-12B-2409']" .overrideConfig.visible_vision_models }}
top_k_docs_max_show: {{ default 100 .overrideConfig.top_k_docs_max_show }}
{{- if .overrideConfig.admin_pass }}
admin_pass: {{ .overrideConfig.admin_pass }}
{{- end }}
agent_server: "True"
agent_port: 5004
agent_workers: {{ default 5 .agent_workers }}
multiple_workers_gunicorn: {{ default "True" .overrideConfig.multiple_workers_gunicorn }}
llava_model: {{ default "openai:mistralai/Pixtral-12B-2409" .overrideConfig.llava_model }}
enable_llava: {{ default "True" .overrideConfig.enable_llava }}
{{- if ge (int (index .resources.requests "nvidia.com/gpu") ) (int 1) }}
enable_tts: {{ default "False" .overrideConfig.enable_tts }}
enable_stt: {{ default "True" .overrideConfig.enable_stt }}
enable_transcriptions: {{ default "True" .overrideConfig.enable_transcriptions }}
asr_model: {{ default "distil-whisper/distil-large-v3" .overrideConfig.asr_model }}
pre_load_embedding_model: {{ default "True" .overrideConfig.pre_load_embedding_model }}
pre_load_image_audio_models: {{ default "True" .overrideConfig.pre_load_image_audio_models }}
cut_distance: {{ default 10000 .overrideConfig.cut_distance }}
hf_embedding_model: {{ default "BAAI/bge-large-en-v1.5" .overrideConfig.hf_embedding_model }}
enable_captions: {{ default "False" .overrideConfig.enable_captions }}
enable_doctr: {{ default "True" .overrideConfig.enable_doctr }}
{{- else }}
enable_tts: {{ default "False" .overrideConfig.enable_tts }}
enable_stt: {{ default "False" .overrideConfig.enable_stt }}
enable_transcriptions: {{ default "False" .overrideConfig.enable_transcriptions }}
embedding_gpu_id: {{ default "cpu" .overrideConfig.embedding_gpu_id }}
hf_embedding_model: {{ default "fake" .overrideConfig.hf_embedding_model }}
pre_load_embedding_model: {{ default "False" .overrideConfig.pre_load_embedding_model }}
pre_load_image_audio_models:  {{ default "False" .overrideConfig.pre_load_image_audio_models }}
enable_captions: {{ default "False" .overrideConfig.enable_captions }}
enable_doctr: {{ default "False" .overrideConfig.enable_doctr }}
{{- end }}
{{- end }}
{{- end }}