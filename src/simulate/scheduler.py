import json
import torch
from simulator import (
    TransformerOpSimulator,
    MatMul,
    MatAdd,
    ROPE,
    KVCache,
    SoftMax_Max,
    SoftMax_Exp,
    SoftMax_Norm,
)


class ComputeScheduler:
    def __init__(
        self,
        n_engines,
        model,
        inputs,
        seq_len
    ):
        """
        Initializes the computation scheduler and simulator with a model and number of engines.
        Args:
            model (torch.nn.Module): The model to simulate.
            num_engines (int): The number of engines to use for simulation.
            inputs (dict): Dictionary containing all input arguments to model.
        """
        # Model and input tensors
        self.model = model
        self.inputs = inputs
        self.seq_len = seq_len

        # Compute engine thread IDS and initial timestamps
        self.current_ts = 100000000000
        self.compute_engines = list(range(1, n_engines + 1))
        self.engine_times = {engine: self.current_ts for engine in self.compute_engines}

        self.trace = {
            "traceEvents": [],
            "displayTimeUnit": "ns"
        }

    def schedule_thread(self):
        """
        Schedules a thread for computation using the earliest available timestamp.
        """
        engine = min(self.engine_times, key=self.engine_times.get)
        start_time = self.engine_times[engine]
        return engine, start_time

    def synchronize_threads(self):
        max_time_engine_id = max(self.engine_times, key=self.engine_times.get)
        self.engine_times = {engine: self.engine_times[max_time_engine_id] for engine in self.compute_engines}
    
    def add_event(self, name, duration, args=None):
        engine_id, start_time = self.schedule_thread()

        self.trace["traceEvents"].append({
            "name": name,
            "ph": "X",
            "ts": self.engine_times[engine_id],
            "dur": duration,
            "pid": 1,
            "tid": engine_id,
            "args": args or {}
        })
        overhead = 100
        self.engine_times[engine_id] = start_time + duration + overhead

    def run(self):
        print("RMSNorm (pre)")
        pre_layernorm = TransformerOpSimulator(self.model.input_layernorm)
        duration, hidden_states = pre_layernorm(self.inputs['input_embeds'])
        self.add_event("RMSNorm (pre)", duration, {
            "op_type": "Norm",
            "input_size": list(self.inputs['input_embeds'].shape),
            "output_size": list(hidden_states.shape),
        })
        self.synchronize_threads()
        print()

        print("QKV PROJECTIONS (PARALLEL)")
        hidden_shape = (hidden_states.shape[0], hidden_states.shape[1], -1, self.model.self_attn.head_dim)
        q_proj = TransformerOpSimulator(self.model.self_attn.q_proj)
        k_proj = TransformerOpSimulator(self.model.self_attn.k_proj)
        v_proj = TransformerOpSimulator(self.model.self_attn.v_proj)

        duration, q = q_proj(hidden_states)
        self.add_event("Q Projection", duration, {
            "op_type": "MatMul",
            "input_size": list(hidden_states.shape),
            "output_size": list(q.shape),
        })
        q = q.view(hidden_shape).transpose(1, 2)

        duration, k = k_proj(hidden_states)
        self.add_event("K Projection", duration, {
            "op_type": "MatMul",
            "input_size": list(hidden_states.shape),
            "output_size": list(k.shape),
        })
        k = k.view(hidden_shape).transpose(1, 2)

        duration, v = v_proj(hidden_states)
        self.add_event("V Projection", duration, {
            "op_type": "MatMul",
            "input_size": list(hidden_states.shape),
            "output_size": list(v.shape),
        })
        v = v.view(hidden_shape).transpose(1, 2)
        print()

        # Wait for all projections to complete
        self.synchronize_threads()

        print("APPLY ROPE")
        rope = TransformerOpSimulator(ROPE(*self.inputs['position_embeddings']))
        duration, q_rope = rope(q)
        self.add_event("ROPE (Q)", duration, {
            "op_type": "ROPE",
            "input_size": list(q.shape),
            "output_size": list(q_rope.shape),
        })
        duration, k_rope = rope(k)
        self.add_event("ROPE (K)", duration, {
            "op_type": "ROPE",
            "input_size": list(k.shape),
            "output_size": list(k_rope.shape),
        })
        print()

        print("MEMORY ACCESS (KV Cache)")
        update_kv = TransformerOpSimulator(KVCache(self.seq_len))
        self.synchronize_threads()
        duration, k_cache = update_kv(k_rope, type="K")
        k_cache = torch.repeat_interleave(k_cache, dim=1, repeats=self.model.self_attn.num_key_value_groups)
        self.add_event("KV Cache (K)", duration, {
            "op_type": "Cache",
            "input_size": list(k.shape),
            "output_size": list(k_cache.shape),
        })

        duration, v_cache = update_kv(v, type="V")
        v_cache = torch.repeat_interleave(v_cache, dim=1, repeats=self.model.self_attn.num_key_value_groups)
        self.add_event("KV Cache (V)", duration, {
            "op_type": "Cache",
            "input_size": list(v.shape),
            "output_size": list(v_cache.shape),
        })
        print()

        # Wait for all ROPE and KV cache updates to complete
        self.synchronize_threads()

        print("QK^T DOT PRODUCT")
        qk_dot = TransformerOpSimulator(MatMul())
        duration, attn_weights = qk_dot(q_rope, k_cache.transpose(2, 3))
        attn_weights *= self.model.self_attn.scaling
        self.add_event("Attention Weights (QK^T)", duration, {
            "op_type": "MatMul",
            "input_size": list(q_rope.shape),
            "output_size": list(attn_weights.shape),
        })
        self.synchronize_threads()
        print()

        print("SOFTMAX (3 steps)")
        print('  1. Find Max')
        softmax_max = TransformerOpSimulator(SoftMax_Max())
        duration, max = softmax_max(attn_weights)
        self.add_event("Softmax - Find Max", duration, {
            "op_type": "Softmax.1",
            "input_size": list(attn_weights.shape),
            "output_size": list(max.shape),
        })
        self.synchronize_threads()

        print('  2. Exponentiate')
        softmax_exp = TransformerOpSimulator(SoftMax_Exp())
        duration, exp = softmax_exp(attn_weights, max)
        self.add_event("Softmax - Exponentiation", duration, {
            "op_type": "Softmax.2",
            "input_size": list(attn_weights.shape),
            "output_size": list(exp.shape),
        })
        self.synchronize_threads()

        print('  3. Normalize')
        softmax_norm = TransformerOpSimulator(SoftMax_Norm())
        duration, attn_scores = softmax_norm(attn_weights, exp)
        self.add_event("Softmax - Normalize", duration, {
            "op_type": "Softmax.3",
            "input_size": list(attn_weights.shape),
            "output_size": list(attn_scores.shape),
        })
        self.synchronize_threads()
        print()

        print("ATTENTION OUTPUT")
        attn_output = TransformerOpSimulator(MatMul())
        duration, attn_output = attn_output(attn_scores, v_cache)
        attn_output = attn_output.transpose(1, 2)
        self.add_event("Attention Output", duration, {
            "op_type": "MatMul",
            "input_size": list(attn_scores.shape),
            "output_size": list(attn_output.shape),
        })
        self.synchronize_threads()
        print()

        print("OUTPUT PROJECTION")
        attn_output = attn_output.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        output_proj = TransformerOpSimulator(self.model.self_attn.o_proj)
        duration, attn_out_proj = output_proj(attn_output)
        self.add_event("O Projection", duration, {
            "op_type": "MatMul",
            "input_size": list(attn_output.shape),
            "output_size": list(attn_out_proj.shape),
        })
        self.synchronize_threads()
        print()

        print("RESIDUAL CONNECTION")
        residual = TransformerOpSimulator(MatAdd())
        duration, attn_output = residual(attn_out_proj, self.inputs['input_embeds'])
        self.add_event("Residual Connection", duration, {
            "op_type": "MatAdd",
            "input_size": list(attn_out_proj.shape),
            "output_size": list(attn_output.shape),
        })
        print()

        print("RMSNorm (post)")
        post_layernorm = TransformerOpSimulator(self.model.post_attention_layernorm)
        duration, hidden_states = post_layernorm(attn_output)
        self.add_event("RMSNorm (post)", duration, {
            "op_type": "Norm",
            "input_size": list(attn_output.shape),
            "output_size": list(hidden_states.shape),
        })
        self.synchronize_threads()
        print()

        print("MLP PROJECTIONS")
        print("  1. Gate Proj")
        gate_proj = TransformerOpSimulator(self.model.mlp.gate_proj)
        duration, gate_out = gate_proj(hidden_states)
        self.add_event("Gate Projection", duration, {
            "op_type": "MatMul",
            "input_size": list(hidden_states.shape),
            "output_size": list(gate_out.shape),
        })
       
        print("  2. Up Proj")
        up_proj = TransformerOpSimulator(self.model.mlp.up_proj)
        duration, up_out = up_proj(hidden_states)
        self.add_event("Up Projection", duration, {
            "op_type": "MatMul",
            "input_size": list(hidden_states.shape),
            "output_size": list(up_out.shape),
        })
        
        # Wait for MLP projections to complete
        self.synchronize_threads()

        print("  3. SiLU Activation & Element-wise Multiply")
        silu = TransformerOpSimulator(self.model.mlp.act_fn)
        duration, hidden_states = silu(up_out)
        hidden_states *= gate_out
        self.add_event("SiLU & Mult", duration, {
            "op_type": "Activation",
            "input_size": list(up_out.shape),
            "output_size": list(hidden_states.shape),
        })
        self.synchronize_threads()
        
        print("  4. Down Proj")
        down_proj = TransformerOpSimulator(self.model.mlp.down_proj)
        duration, down_out = down_proj(hidden_states)
        self.add_event("Down Projection", duration, {
            "op_type": "MatMul",
            "input_size": list(hidden_states.shape),
            "output_size": list(down_out.shape),
        })
        self.synchronize_threads()
        print()

        print("RESIDUAL CONNECTION")
        residual = TransformerOpSimulator(MatAdd())
        duration, hidden_states = residual(down_out, attn_output)
        self.add_event("Residual Connection", duration, {
            "op_type": "MatAdd",
            "input_size": list(down_out.shape),
            "output_size": list(hidden_states.shape),
        })
        print()


    def save(self, trace_path):
        with open(trace_path, 'w') as f:
            json.dump(self.trace, f, indent=2)
            
