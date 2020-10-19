# Create a docker network in which the server and client containers will talk to each other
if docker network inspect edgenetwithcats-network > /dev/null 2>&1 ; then
    echo "Network edgenetwithcats-network already exists"
else
    echo "Creating network edgenetwithcats-network"
    docker network create edgenetwithcats-network
fi

# Spin up the server container
docker run -dt --name triton-edgenetwithcats-server \
    --network edgenetwithcats-network \
    --shm-size=1g --ulimit memlock=-1 \
    --gpus all \
    --ulimit stack=67108864 \
    -p8000:8000 -p8001:8001 -p8002:8002 \
    -e LD_LIBRARY_PATH="/opt/tritonserver/lib/pytorch:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib" \
    -e LD_PRELOAD="/usr/lib/libtorchscatter.so /usr/lib/libtorchsparse.so" \
    local/tritonserver_with_torchso

# Start the triton server in it
docker exec -dti triton-edgenetwithcats-server tritonserver --model-repository=/models

# Spin up the client container; mount the directory with the data and the directory with client.py in it
docker run -dt --name triton-edgenetwithcats-client \
    --network edgenetwithcats-network \
    -v`pwd`/hgcal_testdata:/hgcal_testdata \
    -v`pwd`/client_script:/run_inference \
    nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk

# May take some time for the server to spin up
sleep 15

# Start sending the inference requests from the client to the server
docker exec -ti triton-edgenetwithcats-client \
    python /run_inference/client.py -m edgenetwithcats -u triton-edgenetwithcats-server:8001
