# %% 
import transformer_lens as tl
from transformer_lens import HookedTransformer
import torch

# Clear GPU memory and cache
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
if torch.cuda.is_available():
    torch.cuda.synchronize()

# %%
model = HookedTransformer.from_pretrained(
    "gpt2-small", 
    torch_dtype=torch.float16,
    device="cuda"
)

print(model.cfg.d_model)

# %%

prompt = """
<START>If I have a rubix cube, and I rotate it 4 times along one axis, 2 times along the other axis, and 0 times along the third axis, where is the initial face of the cube relative to the final face?
"""

logits, cache = model.run_with_cache(prompt)

# %%

residual_stream, labels = cache.decompose_resid(return_labels=True, mode="attn")
print(labels[0:3])
# %%

answer = " road" 
logit_attrs = cache.logit_attrs(residual_stream, answer)

print(model.W_K.shape)
print(model.cfg.n_layers)




# %%
