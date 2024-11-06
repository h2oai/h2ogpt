# h2ogpt

![Version: 0.2.1-1254](https://img.shields.io/badge/Version-0.2.1--1254-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 0.2.1-1254](https://img.shields.io/badge/AppVersion-0.2.1--1254-informational?style=flat-square)

A Helm chart for h2oGPT

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| agent.additionalConfig | object | `{}` | You can pass additional config here if overrideConfig does not have it. |
| agent.agent_workers | int | `5` |  |
| agent.autoscaling.enabled | bool | `false` |  |
| agent.autoscaling.maxReplicas | int | `2` |  |
| agent.autoscaling.minReplicas | int | `1` |  |
| agent.autoscaling.targetCPU | int | `80` |  |
| agent.autoscaling.targetMemory | string | `"32Gi"` |  |
| agent.enabled | bool | `true` | Enable agent, this must be `false` if `h2ogpt.agent.enabled` is `true` |
| agent.env | object | `{}` |  |
| agent.extraVolumeMounts | list | `[]` | Extra volume mounts |
| agent.extraVolumes | list | `[]` | Extra volumes, for more certs, mount under /etc/ssl/more-certs |
| agent.image.pullPolicy | string | `"IfNotPresent"` |  |
| agent.image.repository | string | `"gcr.io/vorvan/h2oai/h2ogpt-runtime"` |  |
| agent.image.tag | string | `nil` |  |
| agent.imagePullSecrets | string | `nil` |  |
| agent.initImage.pullPolicy | string | `nil` |  |
| agent.initImage.repository | string | `nil` |  |
| agent.initImage.tag | string | `nil` |  |
| agent.nodeSelector | object | `{}` | Node selector for the agent pods. |
| agent.overrideConfig | object | `{}` | Supported configs are commented. If you don't pass any value, keep {} |
| agent.podAffinity | string | `nil` | Set hostname and zone to true for pod affinity rules based on hostname and zone. |
| agent.podAnnotations | object | `{}` |  |
| agent.podLabels | object | `{}` |  |
| agent.podSecurityContext.fsGroup | string | `nil` |  |
| agent.podSecurityContext.runAsGroup | string | `nil` |  |
| agent.podSecurityContext.runAsNonRoot | bool | `true` |  |
| agent.podSecurityContext.runAsUser | string | `nil` |  |
| agent.replicaCount | int | `1` |  |
| agent.resources.limits."nvidia.com/gpu" | int | `1` |  |
| agent.resources.limits.memory | string | `"64Gi"` |  |
| agent.resources.requests."nvidia.com/gpu" | int | `1` |  |
| agent.resources.requests.memory | string | `"32Gi"` |  |
| agent.securityContext.allowPrivilegeEscalation | bool | `false` |  |
| agent.securityContext.capabilities.drop[0] | string | `"ALL"` |  |
| agent.securityContext.runAsNonRoot | bool | `true` |  |
| agent.securityContext.seccompProfile.type | string | `"RuntimeDefault"` |  |
| agent.service.agentPort | int | `5004` |  |
| agent.service.annotations | object | `{}` |  |
| agent.service.type | string | `"NodePort"` |  |
| agent.storage.class | string | `nil` |  |
| agent.storage.size | string | `"128Gi"` |  |
| agent.storage.useEphemeral | bool | `true` |  |
| agent.tolerations | list | `[]` | Node taints to tolerate by the agent pods. |
| agent.updateStrategy.type | string | `"RollingUpdate"` |  |
| caCertificates | string | `""` | CA certs |
| fullnameOverride | string | `""` |  |
| global.externalLLM.enabled | bool | `false` |  |
| global.externalLLM.modelLock | string | `nil` |  |
| global.externalLLM.secret | object | `{}` | list of secrets for h2ogpt and agent env |
| global.visionModels.enabled | bool | `false` | Enable vision models |
| global.visionModels.rotateAlignResizeImage | bool | `false` |  |
| global.visionModels.visibleModels | list | `[]` | Visible vision models, the vision model itslef needs to be set via modeLock or base_model. Ex: visibleModels: ['OpenGVLab/InternVL-Chat-V1-5'] |
| h2ogpt.additionalConfig | object | `{}` | You can pass additional config here if overrideConfig does not have it. |
| h2ogpt.agent | object | `{"agent_workers":5,"enabled":false}` | Enable agent |
| h2ogpt.agent.enabled | bool | `false` | Run agent with h2oGPT container |
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
| h2ogpt.nodeSelector | object | `{}` | Node selector for the h2ogpt pods. |
| h2ogpt.openai.enabled | bool | `true` |  |
| h2ogpt.openai.openai_workers | int | `5` |  |
| h2ogpt.overrideConfig | object | `{}` | Supported configs are commented. If you don't pass any value, keep {} |
| h2ogpt.podAffinity | string | `nil` | Set hostname and zone to true for pod affinity rules based on hostname and zone. |
| h2ogpt.podAnnotations | object | `{}` |  |
| h2ogpt.podLabels | object | `{}` |  |
| h2ogpt.podSecurityContext.fsGroup | string | `nil` |  |
| h2ogpt.podSecurityContext.runAsGroup | string | `nil` |  |
| h2ogpt.podSecurityContext.runAsNonRoot | bool | `true` |  |
| h2ogpt.podSecurityContext.runAsUser | string | `nil` |  |
| h2ogpt.replicaCount | int | `1` |  |
| h2ogpt.resources.limits."nvidia.com/gpu" | int | `0` |  |
| h2ogpt.resources.limits.memory | string | `"64Gi"` |  |
| h2ogpt.resources.requests."nvidia.com/gpu" | int | `0` |  |
| h2ogpt.resources.requests.memory | string | `"32Gi"` |  |
| h2ogpt.securityContext.allowPrivilegeEscalation | bool | `false` |  |
| h2ogpt.securityContext.capabilities.drop[0] | string | `"ALL"` |  |
| h2ogpt.securityContext.runAsNonRoot | bool | `true` |  |
| h2ogpt.securityContext.seccompProfile.type | string | `"RuntimeDefault"` |  |
| h2ogpt.service.agentPort | int | `5004` |  |
| h2ogpt.service.functionPort | int | `5002` |  |
| h2ogpt.service.openaiPort | int | `5000` |  |
| h2ogpt.service.type | string | `"NodePort"` |  |
| h2ogpt.service.webPort | int | `80` |  |
| h2ogpt.service.webServiceAnnotations | object | `{}` |  |
| h2ogpt.storage.class | string | `nil` |  |
| h2ogpt.storage.size | string | `"128Gi"` |  |
| h2ogpt.storage.useEphemeral | bool | `true` |  |
| h2ogpt.tolerations | list | `[]` | Node taints to tolerate by the h2ogpt pods. |
| h2ogpt.updateStrategy.type | string | `"RollingUpdate"` |  |
| nameOverride | string | `""` |  |
| namespaceOverride | string | `""` |  |
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

