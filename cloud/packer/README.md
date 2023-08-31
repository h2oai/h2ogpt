# h2oGPT Packer Templates

These scripts help create images in public clouds that can then submitted to AWS/Azure/GCP Marketplace for commercial use.

### Prerequisites
The templates require packer software, that can be downloaded from packer.io. Each cloud is unique, so there are three templates that captures the differences.
Follow the instructions specified [here](https://developer.hashicorp.com/packer/tutorials/docker-get-started/get-started-install-cli) to install & setup Packer cli.

### Creating an image

## GCP

```
packer build --force -var "account_file=/Users/admin/gcp-packer-eng-llm-credentials.json" h2ogpt-gcp.json
```

## Azure

```
packer build h2ogpt-azure.json
```

