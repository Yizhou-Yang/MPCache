import torch, json
from pathlib import Path
ROOT = Path('/root/viewdit'); MODEL = ROOT/'weights/CogVideoX-2b'
CFG, NSTEP, LT = 6.0, 30, 5
import sys; sys.path.insert(0, str(ROOT/'experiments/video'))
from step_oracle import PROMPTS
from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
from transformers import T5EncoderModel, T5Tokenizer
tok = T5Tokenizer.from_pretrained(MODEL/'tokenizer')
enc = T5EncoderModel.from_pretrained(MODEL/'text_encoder', torch_dtype=torch.float16, attn_implementation='eager').cuda().eval()
ti = tok(PROMPTS[0], padding='max_length', max_length=226, truncation=True, return_tensors='pt')
pe = enc(ti.input_ids.cuda())[0].half()
ne = enc(tok('', padding='max_length', max_length=226, truncation=True, return_tensors='pt').input_ids.cuda())[0].half()
del enc; torch.cuda.empty_cache()
dit = CogVideoXTransformer3DModel.from_pretrained(MODEL/'transformer', torch_dtype=torch.float16).cuda().eval()
sch = CogVideoXDDIMScheduler.from_pretrained(MODEL/'scheduler'); sch.set_timesteps(NSTEP, device='cuda')
g = torch.Generator(device='cuda').manual_seed(3000)
lat = torch.randn((1, LT, 16, 60, 90), generator=g, device='cuda', dtype=torch.float16) * sch.init_noise_sigma
prev_u = prev_c = None
rows=[]
with torch.no_grad():
    for i, t in enumerate(sch.timesteps):
        out = dit(hidden_states=torch.cat([lat]*2), encoder_hidden_states=torch.cat([ne, pe]), timestep=t.expand(2), return_dict=False)[0].float()
        u, c = out.chunk(2)
        if prev_u is not None:
            rows.append(dict(i=i,
                u_chg=float((u-prev_u).norm()/(u.norm()+1e-9)),
                c_chg=float((c-prev_c).norm()/(c.norm()+1e-9)),
                v_chg=float(((u+CFG*(c-u))-(prev_u+CFG*(prev_c-prev_u))).norm()/((u+CFG*(c-u)).norm()+1e-9))))
        prev_u, prev_c = u, c
        lat = sch.step((u+CFG*(c-u)).half(), t, lat, eta=0., return_dict=False)[0].half()
import numpy as np
print("rel change per step (median over steps)")
print("  unconditional u: %.4f" % np.median([r['u_chg'] for r in rows]))
print("  conditional  c: %.4f" % np.median([r['c_chg'] for r in rows]))
print("  guided       v: %.4f" % np.median([r['v_chg'] for r in rows]))
