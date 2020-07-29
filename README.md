# Triton inference server and test client for EdgeNetWithCategories

This repo is heavily based on https://github.com/lgray/triton-torchgeo-gat-example/ .

First create a new image for the server based on `nvcr.io/nvidia/tritonserver:20.06-py3`,
but with some `.so` files and the jit model already in place:

```
. build-server-image
# Or directly: docker build -t local/tritonserver_with_torchso -f Dockerfile.build .
```

Then spin up the server and client containers, and have the client send inference requests
to the server:

```
. start_containers
```
