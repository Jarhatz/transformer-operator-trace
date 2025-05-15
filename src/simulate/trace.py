import os
BASE_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import subprocess
import argparse
import torch
from transformers import AutoModelForCausalLM
import json

from scheduler import ComputeScheduler


def load(model_id: str):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    decoder_layer = model.model.layers[0]

    # Prepare inputs
    n_tokens = 1
    input_ids = torch.randint(0, model.config.vocab_size, (1, n_tokens))
    input_embeds = model.model.embed_tokens(input_ids)
    cache_position = torch.arange(0, n_tokens, dtype=torch.long)
    position_ids = cache_position.unsqueeze(0)
    causal_mask = model.model._update_causal_mask(
        torch.ones_like(input_ids), input_embeds, cache_position, None, False
    )
    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)

    return decoder_layer, {
        "input_embeds": input_embeds,
        "cache_position": cache_position,
        "position_ids": position_ids,
        "causal_mask": causal_mask,
        "position_embeddings": position_embeddings
    }


# def trace(
#     n_engines,
#     model_id,
#     decoder_layer,
#     input_embeds,
#     cache_position,
#     position_ids,
#     causal_mask,
#     position_embeddings
# ):
#     # Create trace
#     trace = {
#         "traceEvents": [],
#         "displayTimeUnit": "ns"
#     }
    
#     current_ts = 100000000000

#     # Add events for each computation step
#     trace["traceEvents"].extend(generate_transformer_block_events(
#         n_engines,
#         current_ts,
#         {
#             "hidden_size": decoder_layer.hidden_size,
#             "n_heads": decoder_layer.self_attn.config.num_attention_heads,
#             "n_kv_heads": decoder_layer.self_attn.config.num_key_value_heads,
#             "head_dim": decoder_layer.self_attn.head_dim,
#             "intermediate_size": decoder_layer.mlp.gate_proj.out_features,
#             "seq_len": input_embeds.shape[1],
#         }
#     ))
    
#     # Write trace to file
#     trace_path = f"{BASE_PATH}/trace/{model_id.split('/')[-1]}_block_trace.json"
#     with open(trace_path, 'w') as f:
#         json.dump(trace, f)
    
#     return trace_path


# def generate_transformer_block_events(n_engines, start_ts, model_dims):
#     events = []

#     def schedule_event(engine, name, duration, args=None):
#         nonlocal events
#         start_time = engine_times[engine]
#         events.append({
#             "name": name,
#             "ph": "X",
#             "ts": start_time / 1000, # μs
#             "dur": duration / 1000, # μs
#             "pid": 1,
#             "tid": engine,
#             "args": args or {}
#         })
#         overhead = 1000 # 1μs overhead for scheduling
#         engine_times[engine] = start_time + duration + overhead
#         return start_time + duration

#     # Thread IDs for our compute engines
#     compute_engines = list(range(1, n_engines + 1))

#     # Track current time for each engine
#     engine_times = {engine: start_ts for engine in compute_engines}

#     # 1. RMSNorm (pre-normalization)
#     schedule_event(1, "RMSNorm (pre)", 5000, {
#         "input_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]",
#         "compute_type": "normalization"
#     })
    
#     # 2. QKV Projections (can run in parallel)
#     schedule_event(1, "Q Projection", 15000, {
#         "input_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]",
#         "output_size": f"[1, {model_dims['seq_len']}, {model_dims['n_heads']}, {model_dims['head_dim']}]",
#         "compute_type": "dot_product"
#     })
    
#     schedule_event(2, "K Projection", 15000, {
#         "input_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]",
#         "output_size": f"[1, {model_dims['seq_len']}, {model_dims['n_kv_heads']}, {model_dims['head_dim']}]",
#         "compute_type": "dot_product"
#     })
    
#     schedule_event(3, "V Projection", 15000, {
#         "input_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]",
#         "output_size": f"[1, {model_dims['seq_len']}, {model_dims['n_kv_heads']}, {model_dims['head_dim']}]",
#         "compute_type": "dot_product"
#     })
    
#     # Wait for all projections to complete
#     max_projection_time = max(engine_times[1], engine_times[2], engine_times[3])
#     for engine in [1, 2, 3]:
#         engine_times[engine] = max_projection_time
    
#     # 3. Apply rotary position embeddings
#     schedule_event(1, "Apply RoPE to Q", 8000, {
#         "compute_type": "positional_encoding",
#         "tensor_size": f"[1, {model_dims['seq_len']}, {model_dims['n_heads']}, {model_dims['head_dim']}]"
#     })
    
#     schedule_event(2, "Apply RoPE to K", 8000, {
#         "compute_type": "positional_encoding",
#         "tensor_size": f"[1, {model_dims['seq_len']}, {model_dims['n_kv_heads']}, {model_dims['head_dim']}]"
#     })
    
#     # 4. Attention computation (3 steps of softmax)
#     # First synchronize engines 1 and 2
#     max_rope_time = max(engine_times[1], engine_times[2])
#     engine_times[1] = engine_times[2] = max_rope_time
    
#     schedule_event(1, "QK Dot Product", 10000, {
#         "compute_type": "dot_product",
#         "output_size": f"[1, {model_dims['n_heads']}, {model_dims['seq_len']}, {model_dims['seq_len']}]"
#     })
    
#     schedule_event(1, "Softmax - Find Max", 5000, {
#         "compute_type": "softmax_pass1",
#         "tensor_size": f"[1, {model_dims['n_heads']}, {model_dims['seq_len']}, {model_dims['seq_len']}]"
#     })
    
#     schedule_event(1, "Softmax - Exponentiation", 5000, {
#         "compute_type": "softmax_pass2",
#         "tensor_size": f"[1, {model_dims['n_heads']}, {model_dims['seq_len']}, {model_dims['seq_len']}]"
#     })
    
#     schedule_event(1, "Softmax - Normalization", 5000, {
#         "compute_type": "softmax_pass3",
#         "tensor_size": f"[1, {model_dims['n_heads']}, {model_dims['seq_len']}, {model_dims['seq_len']}]"
#     })
    
#     # 5. Attention output computation
#     schedule_event(1, "Attention-Value Product", 10000, {
#         "compute_type": "dot_product",
#         "input_sizes": [
#             f"[1, {model_dims['n_heads']}, {model_dims['seq_len']}, {model_dims['seq_len']}]",
#             f"[1, {model_dims['seq_len']}, {model_dims['n_kv_heads']}, {model_dims['head_dim']}]"
#         ],
#         "output_size": f"[1, {model_dims['seq_len']}, {model_dims['n_heads']}, {model_dims['head_dim']}]"
#     })
    
#     # 6. Output projection
#     schedule_event(1, "Output Projection", 15000, {
#         "compute_type": "dot_product",
#         "input_size": f"[1, {model_dims['seq_len']}, {model_dims['n_heads']}, {model_dims['head_dim']}]",
#         "output_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]"
#     })
    
#     # 7. Residual connection (can use engine 4 while others are busy)
#     schedule_event(4, "Residual Connection 1", 3000, {
#         "compute_type": "memory_transfer",
#         "tensor_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]"
#     })
    
#     # 8. Second RMSNorm
#     schedule_event(2, "RMSNorm (post)", 5000, {
#         "compute_type": "normalization",
#         "tensor_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]"
#     })
    
#     # 9. FFN computations (can be parallelized)
#     schedule_event(1, "Gate Projection", 20000, {
#         "compute_type": "dot_product",
#         "input_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]",
#         "output_size": f"[1, {model_dims['seq_len']}, {model_dims['intermediate_size']}]"
#     })
    
#     schedule_event(2, "Up Projection", 20000, {
#         "compute_type": "dot_product",
#         "input_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]",
#         "output_size": f"[1, {model_dims['seq_len']}, {model_dims['intermediate_size']}]"
#     })
    
#     # Synchronize for SiLU activation
#     max_proj_time = max(engine_times[1], engine_times[2])
#     engine_times[1] = engine_times[2] = max_proj_time
    
#     # 10. SiLU activation and element-wise multiplication
#     schedule_event(1, "SiLU & Element-wise Mult", 10000, {
#         "compute_type": "activation",
#         "tensor_size": f"[1, {model_dims['seq_len']}, {model_dims['intermediate_size']}]"
#     })
    
#     # 11. Down projection
#     schedule_event(1, "Down Projection", 20000, {
#         "compute_type": "dot_product",
#         "input_size": f"[1, {model_dims['seq_len']}, {model_dims['intermediate_size']}]",
#         "output_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]"
#     })
    
#     # 12. Final residual connection
#     schedule_event(4, "Residual Connection 2", 3000, {
#         "compute_type": "memory_transfer",
#         "tensor_size": f"[1, {model_dims['seq_len']}, {model_dims['hidden_size']}]"
#     })
    
#     return events


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trace Transformer Block")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model to use for tracing a transformer block"
    )
    parser.add_argument(
        "--seq_len",
        "-s",
        type=int,
        default=256,
        help="Sequence length to use for tracing the transformer block"
    )
    parser.add_argument(
        "--n_engines",
        "-e",
        type=int,
        default=3,
        help="Number of engines to simulate when tracing the transformer block"
    )
    args = parser.parse_args()

    # Load model and sample inputs
    model, inputs = load(args.model_id)

    tracer = ComputeScheduler(args.n_engines, model, inputs, args.seq_len)

    tracer.run()
    # # Trace the transformer block
    # trace_path = trace(args.n_engines, args.model_id, *inputs)

    # Save trace to file
    trace_path = f"{BASE_PATH}/trace/{args.model_id.split('/')[-1]}_sim_trace.json"
    tracer.save(trace_path)

    # Open trace in Perfetto UI
    subprocess.run(f"{BASE_PATH}/src/open_trace_in_ui -i {trace_path}", shell=True)
    
