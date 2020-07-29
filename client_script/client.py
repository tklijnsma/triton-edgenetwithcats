#!/usr/bin/env python
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import numpy as np
import sys
import random
import glob

# ############# HACK #############
# tritongrpcclient python api is currently limiting
# the default sending size to be ~4Mb (other api's
# seem to not suffer from this problem). This hack
# manually sets the grpc options to allow more data
# to be sent. This should be solved in future versions
# of the python api.
# See: https://github.com/NVIDIA/triton-inference-server/issues/1776#issuecomment-655894276
with open('/usr/local/lib/python3.6/dist-packages/tritongrpcclient/__init__.py', 'r') as f:
    init_file = f.read()
fixed_init_file = init_file.replace(
    'self._channel = grpc.insecure_channel(url, options=None)',
    'self._channel = grpc.insecure_channel(url, options=[(\'grpc.max_send_message_length\', 512 * 1024 * 1024), (\'grpc.max_receive_message_length\', 512 * 1024 * 1024)])'
    )
if fixed_init_file != init_file:
    print('WARNING: Hacking tritongrpcclient to allow larger requests to be sent')
    with open('/usr/local/lib/python3.6/dist-packages/tritongrpcclient/__init__.py', 'w') as f:
        f.write(fixed_init_file)
# ############# HACK #############

import tritongrpcclient

def build_edge_index(n_nodes, Ri_rows, Ri_cols, Ro_rows, Ro_cols):
    # Warning: not a very well optimized function
    n_edges = Ri_rows.shape[0]
    spRi_idxs = np.stack([Ri_rows.astype(np.int64), Ri_cols.astype(np.int64)])
    spRi_vals = np.ones((Ri_rows.shape[0],), dtype=np.float32)
    spRi = (spRi_idxs,spRi_vals,n_nodes,n_edges)

    spRo_idxs = np.stack([Ro_rows.astype(np.int64), Ro_cols.astype(np.int64)])
    spRo_vals = np.ones((Ro_rows.shape[0],), dtype=np.float32)
    spRo = (spRo_idxs,spRo_vals,n_nodes,n_edges)

    Ro = spRo[0].T.astype(np.int64)
    Ri = spRi[0].T.astype(np.int64)
    
    i_out = Ro[Ro[:,1].argsort(kind='stable')][:,0]
    i_in  = Ri[Ri[:,1].argsort(kind='stable')][:,0]
    edge_index = np.stack((i_out,i_in))
    return edge_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-m',
                        '--model_name',
                        type=str,
                        required=True,
                        help='Model name')
    parser.add_argument('-u',
                        '--url',
                        type=str,
                        required=False,
                        default='localhost:8001',
                        help='Inference server URL. Default is localhost:8001.')

    FLAGS = parser.parse_args()
    try:
        triton_client = tritongrpcclient.InferenceServerClient(url=FLAGS.url,
                                                               verbose=FLAGS.verbose)
    except Exception as e:
        print("channel creation failed: " + str(e))
        sys.exit()

    print(dir(triton_client))
    print(triton_client.get_model_repository_index())
    print(triton_client.get_server_metadata())

    print('Trying get_model_config now...')

    model_name = FLAGS.model_name 

    mconf = triton_client.get_model_config(model_name, as_json=True)
    print('config:\n', mconf)
    
    # Loop over testdata and send inferences
    for npzfile in glob.glob('/hgcal_testdata/*.npz'):
        inputs = []
        outputs = []

        with np.load(npzfile) as data:
            x = data['X'].astype(np.float32)
            edge_index = build_edge_index(
                x.shape[0],
                data['Ri_rows'], data['Ri_cols'], data['Ro_rows'], data['Ro_cols']
                )
            print(x.shape, edge_index.shape)

        nnodes = x.shape[0]
        nedges = edge_index.shape[1]

        inputs.append(tritongrpcclient.InferInput('x__0', [nnodes, 5], 'FP32'))
        inputs.append(tritongrpcclient.InferInput('edge_index__1', [2, nedges], "INT64"))

        inputs[0].set_data_from_numpy(x)
        inputs[1].set_data_from_numpy(edge_index)

        outputs.append(tritongrpcclient.InferRequestedOutput('output__0'))

        results = triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs
            )
        output0_data = results.as_numpy('output__0')
        print(output0_data)

    statistics = triton_client.get_inference_statistics(model_name=model_name)
    print(statistics)
    if len(statistics.model_stats) != 1:
        print("FAILED: Inference Statistics")
        sys.exit(1)
    print('PASS: infer')
