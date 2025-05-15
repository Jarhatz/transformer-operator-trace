# Visualizing Transformer Layer Operations - Perfetto
This repository contains two separate Perfetto visualization scripts for visualizing the underlying operations that occur within a transformer block.
The first, is a real-time trace of a decoder layer from any HuggingFace autoregressive LM (e.g. `meta-llama/Llama-3.2-1B`).
The second is a simulated trace for performing transformer operations on a hardware resource with an adjustable number of compute engines (threads).
Both scripts will attempt to automatically open a browser window with the Perfetto Chrome trace.

## Setup
1. Make sure PyTorch is installed on your environment. Follow their [instructions](https://pytorch.org/get-started/locally/) to install and build on your specific platform.
2. `pip install transformers`

## Real-Time Trace
```bash
python src/realtime/trace.py --model_id meta-llama/Llama-3.2-1B --n_tokens 512
```
_*This script traces real operators/kernels running on physical hardware for the prompt evaluation phase of the layer._

## Simulated Trace
```bash
python src/simulate/trace.py --model_id meta-llama/Llama-3.2-1B --seq_len 256 --n_engines 3
```
_*This script traces simulated operation executions during the autoregressive decoding stage using the cached KV of seq_len._


The `src/simulate/trace.py` utilizes a `ComputeScheduler` which uses a naive thread scheduling algorithm based on the earliest available compute engine.
All threads are synchronized when a preceding operation is required to finish prior to launching a new operation.
Otherwise, a parallel execution is launched whenever there is an available engine.
The duration of each operation `(ie. MatMul, MatAdd, Norm, ROPE, Cache (Memory Access), SoftMax(1,2,3 Stages), etc.)` is approximated using a simple formula based on input size and operation attributes.
All of this logic can be found in the `TransformerOpSimulator` class in `src/simulate/simulator.py` and `ComputeScheduler` class in `src/simulate/scheduler.py`.

![Visual Trace](trace/visual_trace.png)

**BONUS**: For the bonus task, I have implemented a simple scheduling algorithm which can adapt to a hardware resource with any number of compute engines. Under the current implementation and the assumed kernel definitions, the optimal number of threads concurrently utilized during a transformer decoder layer is `3`. As shown in the figure above, the highest parallelization that can be done is the computation of `Q`, `K`, and `V` in parallel assuming the `MatMul` operator is the core kernel operation.
