from copy import copy
from dataclasses import dataclass

import torch

from bon.core import SamplingFailure, generate as original_generate


@dataclass
class Prefix:
    pad_masks: torch.Tensor
    cache: object


def prepare_prefix(policy, batch, diagnostics=None):
    from lerobot.policies.pi05 import modeling_pi05 as M
    model = policy.model
    if model._rtc_enabled():
        raise ValueError('Shared-prefix extension is for synchronous PI05 BoN only')
    policy.eval()
    images, img_masks = policy._preprocess_images(batch)
    tokens, masks = batch[M.OBS_LANGUAGE_TOKENS], batch[M.OBS_LANGUAGE_ATTENTION_MASK]
    if tokens.shape[0] != 1:
        raise ValueError('Expected exactly one observation; candidate batch is created after prefix prefill')
    embeddings, pad, att = model.embed_prefix(images, img_masks, tokens, masks)
    attention = M.prepare_attention_masks_4d(M.make_att_2d_masks(pad, att))
    positions = torch.cumsum(pad, dim=1)-1
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = 'eager'
    _, cache = model.paligemma_with_expert.forward(
        attention_mask=attention, position_ids=positions, past_key_values=None,
        inputs_embeds=[embeddings, None], use_cache=True)
    if diagnostics is not None:
        diagnostics.update(prefix_observation_batch=1, prefix_calls=1,
                           prefix_tokens=int(pad.shape[1]), prefix_shared=True)
    return Prefix(pad, cache)


def expand_prefix(prefix, count):
    if count == 1:
        return prefix
    cache = copy(prefix.cache)
    cache.layers = []
    for source in prefix.cache.layers:
        keys, values = source.keys, source.values
        if keys.shape[0] != 1 or values.shape[0] != 1:
            raise ValueError('Prefix KV cache must contain a single observation')
        layer = copy(source)
        layer.keys = keys.expand(count, *keys.shape[1:])
        layer.values = values.expand(count, *values.shape[1:])
        cache.layers.append(layer)
    return Prefix(prefix.pad_masks.expand(count, -1), cache)


def denoise(policy, prefix, noise, cancelled=lambda:False):
    from lerobot.policies.common.flow_matching import euler_integrate
    model = policy.model

    def velocity(x, timestep):
        if cancelled():
            raise InterruptedError('Shared-prefix candidate generation cancelled')
        return model.denoise_step(prefix_pad_masks=prefix.pad_masks,
                                  past_key_values=prefix.cache, x_t=x, timestep=timestep)

    return euler_integrate(velocity, noise, model.config.num_inference_steps,
                           rtc_processor=model.rtc_processor, rtc_enabled=False)


@torch.no_grad()
def generate_shared(policy, batch, postprocessor, n, mode='batch', noise=None,
                    cancelled=lambda:False, diagnostics=None):
    if n == 1:
        if diagnostics is not None:
            diagnostics.update(n1_original_path=True, prefix_shared=False,
                               prefix_observation_batch=1, prefix_calls=1)
        return original_generate(policy,batch,postprocessor,1,'serial',noise,cancelled)
    if n < 1 or mode not in ('batch','serial'):
        raise ValueError('Expected n>=1 and batch/serial mode')
    if cancelled():
        return None
    if noise is None:
        device=next(v.device for v in batch.values() if torch.is_tensor(v))
        noise=policy.model.sample_noise((n,policy.config.chunk_size,policy.config.max_action_dim),device)
    if tuple(noise.shape)!=(n,policy.config.chunk_size,policy.config.max_action_dim):
        raise ValueError('Unexpected explicit noise shape')
    first=None
    try:
        prefix=prepare_prefix(policy,batch,diagnostics)
        dim=policy.config.output_features['action'].shape[0]
        if mode=='batch':
            expanded=expand_prefix(prefix,n)
            raw=denoise(policy,expanded,noise,cancelled)[:,:,:dim]
            absolute=postprocessor(raw).detach().cpu()
        else:
            chunks=[]
            for i in range(n):
                if cancelled():
                    return None
                raw=denoise(policy,prefix,noise[i:i+1],cancelled)[:,:,:dim]
                chunk=postprocessor(raw).detach().cpu()
                if i==0:
                    first=chunk[0].numpy().copy()
                chunks.append(chunk)
            absolute=torch.cat(chunks,dim=0)
        result=absolute.numpy().copy()
        if result.shape!=(n,policy.config.chunk_size,7) or not torch.isfinite(absolute).all():
            raise ValueError('Invalid absolute candidate actions')
        return None if cancelled() else result
    except InterruptedError:
        if cancelled():
            return None
        raise
    except Exception as exc:
        raise SamplingFailure(f'{type(exc).__name__}: {exc}',first,noise[:1]) from exc

