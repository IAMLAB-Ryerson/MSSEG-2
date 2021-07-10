## IAMLAB MSSEG-2 Docker

[![Docker image](https://img.shields.io/badge/docker-1.0.0-blue)](https://hub.docker.com/r/samirmitha/iamlab_msseg2)

# Building From Source
To build docker from source download the source code here and run the following docker build command:
```
docker build -t samirmitha/iamlab_msseg2:1.0.0 .
```

# Pulling Docker Image Directly
The docker image can be pulled directly from dockerhub using the following command:
```
docker pull samirmitha/iamlab_msseg2:1.0.0
```

# Using Boutiques Descriptor
The boutiques descriptor can be used to autogenerate bash commands to run on the docker. Boutiques can be pip installed using:
```
pip install boutiques
```

The docker can then be run using boutiques with the following command:
```
bosh exec launch iamlab_msseg2.json example_invocation.json
```
