import os
BASE_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))

import subprocess
import argparse
import torch
from transformers import AutoModelForCausalLM


def load(model_id: str, n_tokens: int, device: str):
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_id)
    decoder_layer = model.model.layers[0].to(device)

    # Prepare inputs
    input_ids = torch.randint(0, model.config.vocab_size, (1, n_tokens))
    input_embeds = model.model.embed_tokens(input_ids).to(device)
    cache_position = torch.arange(0, input_ids.shape[1], dtype=torch.long).to(device)
    position_ids = cache_position.unsqueeze(0).to(device)
    causal_mask = model.model._update_causal_mask(
        torch.ones_like(input_ids), input_embeds, cache_position, None, False
    )
    position_embeddings = model.model.rotary_emb(input_embeds, position_ids)

    return decoder_layer, input_embeds, cache_position, position_ids, causal_mask, position_embeddings


def trace(
    model_id,
    decoder_layer,
    input_embeds,
    cache_position,
    position_ids,
    causal_mask,
    position_embeddings
):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        
        # Forward-pass through transformer block
        hidden_states = decoder_layer(
            input_embeds,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )[0]
            
    # Save trace
    trace_path = f"{BASE_PATH}/trace/{model_id.split('/')[-1]}_block_trace.json"
    prof.export_chrome_trace(trace_path)
    print("Saved trace to: ", trace_path)
    return trace_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trace Transformer Block")
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model to use for tracing a transformer block"
    )
    parser.add_argument(
        "--n_tokens",
        "-n",
        type=int,
        default=4,
        help="Number of tokens to use for tracing the transformer block"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for tracing the transformer block"
    )
    args = parser.parse_args()

    # Load model and sample inputs
    inputs = load(args.model_id, args.n_tokens, args.device)

    # Trace the transformer block
    trace_path = trace(args.model_id, *inputs)

    # Open trace in Perfetto UI
    subprocess.run(f"{BASE_PATH}/src/open_trace_in_ui -i {trace_path}", shell=True)
