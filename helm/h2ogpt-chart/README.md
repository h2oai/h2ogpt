# h2ogpt

![Version: 0.2.1-1254](https://img.shields.io/badge/Version-0.2.1--1254-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 0.2.1-1254](https://img.shields.io/badge/AppVersion-0.2.1--1254-informational?style=flat-square)

A Helm chart for h2oGPT

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| agents.autoscaling.enabled | bool | `false` |  |
| agents.autoscaling.maxReplicas | int | `2` |  |
| agents.autoscaling.minReplicas | int | `1` |  |
| agents.autoscaling.targetCPU | int | `80` |  |
| agents.autoscaling.targetMemory | string | `"32Gi"` |  |
| agents.enabled | bool | `false` | Enable agents, this must be `false` if `h2ogpt.agents.enabled` is `true` |
| agents.env | object | `{}` |  |
| agents.extraVolumeMounts | list | `[]` | Extra volume mounts |
| agents.extraVolumes | list | `[]` | Extra volumes, for more certs, mount under /etc/ssl/more-certs |
| agents.image.pullPolicy | string | `"IfNotPresent"` |  |
| agents.image.repository | string | `"gcr.io/vorvan/h2oai/h2ogpt-runtime"` |  |
| agents.image.tag | string | `nil` |  |
| agents.imagePullSecrets | string | `nil` |  |
| agents.initImage.pullPolicy | string | `nil` |  |
| agents.initImage.repository | string | `nil` |  |
| agents.initImage.tag | string | `nil` |  |
| agents.nodeSelector | string | `nil` |  |
| agents.overrideConfig.agent_workers | int | `5` |  |
| agents.overrideConfig.concurrency_count | int | `100` |  |
| agents.overrideConfig.embedding_gpu_id | string | `"cpu"` |  |
| agents.overrideConfig.enable_stt | bool | `false` |  |
| agents.overrideConfig.enable_transcriptions | bool | `false` |  |
| agents.overrideConfig.enable_tts | bool | `false` |  |
| agents.overrideConfig.enforce_h2ogpt_api_key | bool | `true` |  |
| agents.overrideConfig.enforce_h2ogpt_ui_key | bool | `false` |  |
| agents.overrideConfig.hf_embedding_model | string | `"fake"` |  |
| agents.overrideConfig.metadata_in_context | string | `""` |  |
| agents.overrideConfig.num_async | int | `10` |  |
| agents.overrideConfig.rotate_align_resize_image | bool | `false` |  |
| agents.overrideConfig.score_model | string | `"None"` |  |
| agents.overrideConfig.share | bool | `false` |  |
| agents.overrideConfig.top_k_docs_max_show | int | `100` |  |
| agents.overrideConfig.visible_hosts_tab | bool | `false` |  |
| agents.overrideConfig.visible_login_tab | bool | `false` |  |
| agents.overrideConfig.visible_models_tab | bool | `false` |  |
| agents.overrideConfig.visible_system_tab | bool | `false` |  |
| agents.podAffinity | string | `nil` | Set hostname and zone to true for pod affinity rules based on hostname and zone. |
| agents.podAnnotations | object | `{}` |  |
| agents.podLabels | object | `{}` |  |
| agents.podSecurityContext.fsGroup | string | `nil` |  |
| agents.podSecurityContext.runAsGroup | string | `nil` |  |
| agents.podSecurityContext.runAsNonRoot | bool | `true` |  |
| agents.podSecurityContext.runAsUser | string | `nil` |  |
| agents.replicaCount | int | `1` |  |
| agents.resources.limits.memory | string | `"64Gi"` |  |
| agents.resources.requests.memory | string | `"32Gi"` |  |
| agents.securityContext.allowPrivilegeEscalation | bool | `false` |  |
| agents.securityContext.capabilities.drop[0] | string | `"ALL"` |  |
| agents.securityContext.runAsNonRoot | bool | `true` |  |
| agents.securityContext.seccompProfile.type | string | `"RuntimeDefault"` |  |
| agents.service.agentsPort | int | `5004` |  |
| agents.service.annotations | object | `{}` |  |
| agents.service.type | string | `"NodePort"` |  |
| agents.storage.class | string | `nil` |  |
| agents.storage.size | string | `"128Gi"` |  |
| agents.storage.useEphemeral | bool | `true` |  |
| agents.tolerations | string | `nil` |  |
| agents.updateStrategy.type | string | `"RollingUpdate"` |  |
| caCertificates | string | `""` | CA certs |
| fullnameOverride | string | `""` |  |
| global.externalLLM.enabled | bool | `false` |  |
| global.externalLLM.modelLock | string | `nil` |  |
| global.externalLLM.secret | object | `{}` | list of secrets for h2ogpt and agents env |
| global.visionModels.enabled | bool | `false` | Enable vision models |
| global.visionModels.rotateAlignResizeImage | bool | `false` |  |
| global.visionModels.visibleModels | list | `[]` | Visible vision models, the vision model itslef needs to be set via modeLock or base_model. Ex: visibleModels: ['OpenGVLab/InternVL-Chat-V1-5'] |
| h2ogpt.agents | object | `{"agent_workers":5,"enabled":false}` | Enable agents |
| h2ogpt.agents.enabled | bool | `false` | Run agents with h2oGPT container |
| h2ogpt.enabled | bool | `true` | Enable h2oGPT |
| h2ogpt.env | object | `{}` |  |
| h2ogpt.extraVolumeMounts | list | `[]` | Extra volume mounts |
| h2ogpt.extraVolumes | list | `[]` | Extra volumes, for more certs, mount under /etc/ssl/more-certs |
| h2ogpt.image.pullPolicy | string | `"IfNotPresent"` |  |
| h2ogpt.image.repository | string | `"gcr.io/vorvan/h2oai/h2ogpt-runtime"` |  |
| h2ogpt.image.tag | string | `nil` |  |
| h2ogpt.imagePullSecrets | string | `nil` |  |
| h2ogpt.initImage.pullPolicy | string | `nil` |  |
| h2ogpt.initImage.repository | string | `nil` |  |
| h2ogpt.initImage.tag | string | `nil` |  |
| h2ogpt.nodeSelector | string | `nil` |  |
| h2ogpt.overrideConfig.concurrency_count | int | `100` |  |
| h2ogpt.overrideConfig.embedding_gpu_id | string | `"cpu"` |  |
| h2ogpt.overrideConfig.enable_stt | bool | `false` |  |
| h2ogpt.overrideConfig.enable_transcriptions | bool | `false` |  |
| h2ogpt.overrideConfig.enable_tts | bool | `false` |  |
| h2ogpt.overrideConfig.enforce_h2ogpt_api_key | bool | `true` |  |
| h2ogpt.overrideConfig.enforce_h2ogpt_ui_key | bool | `false` |  |
| h2ogpt.overrideConfig.hf_embedding_model | string | `"fake"` |  |
| h2ogpt.overrideConfig.metadata_in_context | string | `""` |  |
| h2ogpt.overrideConfig.num_async | int | `10` |  |
| h2ogpt.overrideConfig.openai_server | bool | `true` |  |
| h2ogpt.overrideConfig.openai_workers | int | `5` |  |
| h2ogpt.overrideConfig.rotate_align_resize_image | bool | `false` |  |
| h2ogpt.overrideConfig.score_model | string | `"None"` |  |
| h2ogpt.overrideConfig.share | bool | `false` |  |
| h2ogpt.overrideConfig.top_k_docs_max_show | int | `100` |  |
| h2ogpt.overrideConfig.visible_hosts_tab | bool | `false` |  |
| h2ogpt.overrideConfig.visible_login_tab | bool | `false` |  |
| h2ogpt.overrideConfig.visible_models_tab | bool | `false` |  |
| h2ogpt.overrideConfig.visible_system_tab | bool | `false` |  |
| h2ogpt.podAffinity | string | `nil` | Set hostname and zone to true for pod affinity rules based on hostname and zone. |
| h2ogpt.podAnnotations | object | `{}` |  |
| h2ogpt.podLabels | object | `{}` |  |
| h2ogpt.podSecurityContext.fsGroup | string | `nil` |  |
| h2ogpt.podSecurityContext.runAsGroup | string | `nil` |  |
| h2ogpt.podSecurityContext.runAsNonRoot | bool | `true` |  |
| h2ogpt.podSecurityContext.runAsUser | string | `nil` |  |
| h2ogpt.replicaCount | int | `1` |  |
| h2ogpt.resources.limits.memory | string | `"64Gi"` |  |
| h2ogpt.resources.requests.memory | string | `"32Gi"` |  |
| h2ogpt.securityContext.allowPrivilegeEscalation | bool | `false` |  |
| h2ogpt.securityContext.capabilities.drop[0] | string | `"ALL"` |  |
| h2ogpt.securityContext.runAsNonRoot | bool | `true` |  |
| h2ogpt.securityContext.seccompProfile.type | string | `"RuntimeDefault"` |  |
| h2ogpt.service.agentsPort | int | `5004` |  |
| h2ogpt.service.functionPort | int | `5002` |  |
| h2ogpt.service.openaiPort | int | `5000` |  |
| h2ogpt.service.type | string | `"NodePort"` |  |
| h2ogpt.service.webPort | int | `80` |  |
| h2ogpt.service.webServiceAnnotations | object | `{}` |  |
| h2ogpt.storage.class | string | `nil` |  |
| h2ogpt.storage.size | string | `"128Gi"` |  |
| h2ogpt.storage.useEphemeral | bool | `true` |  |
| h2ogpt.tolerations | string | `nil` |  |
| h2ogpt.updateStrategy.type | string | `"RollingUpdate"` |  |
| lmdeploy.containerArgs[0] | string | `"OpenGVLab/InternVL-Chat-V1-5"` |  |
| lmdeploy.enabled | bool | `false` | Enable lmdeploy |
| lmdeploy.env | object | `{}` |  |
| lmdeploy.hfSecret | string | `nil` |  |
| lmdeploy.image.pullPolicy | string | `"IfNotPresent"` |  |
| lmdeploy.image.repository | string | `"gcr.io/vorvan/h2oai/h2oai-h2ogpt-lmdeploy"` |  |
| lmdeploy.image.tag | string | `nil` |  |
| lmdeploy.nodeSelector | string | `nil` |  |
| lmdeploy.overrideConfig | string | `nil` |  |
| lmdeploy.podAffinity | string | `nil` | Set hostname and zone to true for pod affinity rules based on hostname and zone. |
| lmdeploy.podAnnotations | object | `{}` |  |
| lmdeploy.podLabels | object | `{}` |  |
| lmdeploy.podSecurityContext | string | `nil` |  |
| lmdeploy.replicaCount | int | `1` |  |
| lmdeploy.resources | string | `nil` |  |
| lmdeploy.securityContext | string | `nil` |  |
| lmdeploy.service.port | int | `23333` |  |
| lmdeploy.service.type | string | `"ClusterIP"` |  |
| lmdeploy.storage.class | string | `nil` |  |
| lmdeploy.storage.size | string | `"512Gi"` |  |
| lmdeploy.storage.useEphemeral | bool | `true` |  |
| lmdeploy.tolerations | string | `nil` |  |
| lmdeploy.updateStrategy.type | string | `"RollingUpdate"` |  |
| nameOverride | string | `""` |  |
| namespaceOverride | string | `""` |  |
| tgi.containerArgs | string | `nil` |  |
| tgi.enabled | bool | `false` | Enable tgi |
| tgi.env | object | `{}` |  |
| tgi.hfSecret | string | `nil` |  |
| tgi.image.pullPolicy | string | `"IfNotPresent"` |  |
| tgi.image.repository | string | `"ghcr.io/huggingface/text-generation-inference"` |  |
| tgi.image.tag | string | `"0.9.3"` |  |
| tgi.nodeSelector | string | `nil` |  |
| tgi.overrideConfig | string | `nil` |  |
| tgi.podAffinity | string | `nil` | Set hostname and zone to true for pod affinity rules based on hostname and zone. |
| tgi.podAnnotations | object | `{}` |  |
| tgi.podLabels | object | `{}` |  |
| tgi.podSecurityContext | string | `nil` |  |
| tgi.replicaCount | int | `1` |  |
| tgi.resources | string | `nil` |  |
| tgi.securityContext | string | `nil` |  |
| tgi.service.port | int | `8080` |  |
| tgi.service.type | string | `"ClusterIP"` |  |
| tgi.storage.class | string | `nil` |  |
| tgi.storage.size | string | `"512Gi"` |  |
| tgi.storage.useEphemeral | bool | `true` |  |
| tgi.tolerations | string | `nil` |  |
| tgi.updateStrategy.type | string | `"RollingUpdate"` |  |
| vllm.containerArgs[0] | string | `"--model"` |  |
| vllm.containerArgs[1] | string | `"h2oai/h2ogpt-4096-llama2-7b-chat"` |  |
| vllm.containerArgs[2] | string | `"--tokenizer"` |  |
| vllm.containerArgs[3] | string | `"hf-internal-testing/llama-tokenizer"` |  |
| vllm.containerArgs[4] | string | `"--tensor-parallel-size"` |  |
| vllm.containerArgs[5] | int | `2` |  |
| vllm.containerArgs[6] | string | `"--seed"` |  |
| vllm.containerArgs[7] | int | `1234` |  |
| vllm.containerArgs[8] | string | `"--trust-remote-code"` |  |
| vllm.enabled | bool | `false` | Enable vllm |
| vllm.env.DO_NOT_TRACK | string | `"1"` |  |
| vllm.env.VLLM_NO_USAGE_STATS | string | `"1"` |  |
| vllm.image.pullPolicy | string | `"IfNotPresent"` |  |
| vllm.image.repository | string | `"vllm/vllm-openai"` |  |
| vllm.image.tag | string | `"latest"` |  |
| vllm.imagePullSecrets | string | `nil` |  |
| vllm.nodeSelector | string | `nil` |  |
| vllm.overrideConfig | string | `nil` |  |
| vllm.podAffinity | string | `nil` | Set hostname and zone to true for pod affinity rules based on hostname and zone. |
| vllm.podAnnotations | object | `{}` |  |
| vllm.podLabels | object | `{}` |  |
| vllm.podSecurityContext.fsGroup | string | `nil` |  |
| vllm.podSecurityContext.runAsGroup | string | `nil` |  |
| vllm.podSecurityContext.runAsNonRoot | bool | `true` |  |
| vllm.podSecurityContext.runAsUser | string | `nil` |  |
| vllm.replicaCount | int | `1` |  |
| vllm.resources | string | `nil` |  |
| vllm.securityContext.allowPrivilegeEscalation | bool | `false` |  |
| vllm.securityContext.capabilities.drop[0] | string | `"ALL"` |  |
| vllm.securityContext.runAsNonRoot | bool | `true` |  |
| vllm.securityContext.seccompProfile | string | `nil` |  |
| vllm.service.port | int | `5000` |  |
| vllm.service.type | string | `"ClusterIP"` |  |
| vllm.storage.class | string | `nil` |  |
| vllm.storage.size | string | `"512Gi"` |  |
| vllm.storage.useEphemeral | bool | `true` |  |
| vllm.tolerations | string | `nil` |  |
| vllm.updateStrategy.type | string | `"RollingUpdate"` |  |

