# Create a docker network in which the server and client containers will talk to each other
NETWORKNAME="edgenetwithcats-network"
if docker network inspect "$NETWORKNAME" > /dev/null 2>&1 ; then
    echo "Network $NETWORKNAME already exists"
else
    echo "Creating network $NETWORKNAME"
    docker network create "$NETWORKNAME"
fi

# Spin up the server container
docker run -dt --name triton-edgenetwithcats-server \
    --network "$NETWORKNAME" \
    --shm-size=1g --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p8000:8000 -p8001:8001 -p8002:8002 \
    -e LD_LIBRARY_PATH="/opt/tritonserver/lib/pytorch:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib" \
    -e LD_PRELOAD="/usr/lib/libtorchscatter.so /usr/lib/libtorchsparse.so" \
    local/tritonserver_with_torchso

# Start the triton server in it
docker exec -dti triton-edgenetwithcats-server tritonserver --model-repository=/models

# Spin up the client container; mount the directory with the data
docker run -dt --name triton-edgenetwithcats-client \
    --network $NETWORKNAME \
    -v`pwd`/hgcal_testdata:/hgcal_testdata \
    -v`pwd`/client_script:/run_inference \
    nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk

# Start sending the inference requests from the client to the server
docker exec -ti triton-edgenetwithcats-client \
    python /run_inference/client.py -m edgenetwithcats -u triton-edgenetwithcats-server:8001
