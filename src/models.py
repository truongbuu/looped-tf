import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

def build_general_model(conf, y_dim=1):
    if conf.family == "gpt2":
        model = GeneralTransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            linear_embedding=conf.linear_embedding,
        )
    else:
        raise NotImplementedError

    return model

class GeneralTransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, linear_embedding = False):
        super(GeneralTransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"
        self.n_positions = n_positions
        self.n_dims = n_dims
        if linear_embedding:
            self._read_in = nn.Linear(n_dims, n_embd)
        else:
            self._read_in = nn.Embedding(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)


    def forward_no_position(self, zs, attention_mask = None):
        embeds = self._read_in(zs)
        output = self._backbone.forward_no_position(inputs_embeds=embeds, attention_mask = attention_mask).last_hidden_state
        prediction = self._read_out(output)
        return prediction
    
    def forward_single(self, embeds, attention_mask = None):
        output = self._backbone.forward_no_position(inputs_embeds=embeds, attention_mask = attention_mask).last_hidden_state
        return output
    
    def looped_forward(self, zs, horizon, attention_mask = None):
        # input injection
        zs = self._read_in(zs)
        output = torch.zeros_like(zs).to(zs.device)
        output_list = []
        for i in range(horizon):
            output = self.forward_single(output+zs, attention_mask)
            output_list.append(self._read_out(output).clone())
        return output_list

    def looped_forward_without(self, zs, horizon, attention_mask = None):
        # no injection
        output = self._read_in(zs)
        output_list = []
        for i in range(horizon):
            output = self.forward_single(output, attention_mask)
            output_list.append(self._read_out(output).clone())
        return output_list
    