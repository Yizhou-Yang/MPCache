import os, sys, torch, numpy as np, imageio.v2 as iio
from diffusers import CogVideoXPipeline
M = "/root/viewdit/weights/CogVideoX-2b"; NSTEP = 30
USE = int(os.environ.get("USE_CORE", "0"))

def patchify(t):
    B, T, C, H, W = t.shape
    x = t.reshape(B, T, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(B, T * (H // 2) * (W // 2), C * 4)

class Diag:
    def __init__(s, dim=64): s.num=torch.zeros(dim); s.den=torch.zeros(dim); s.n=0; s.beta=1.0
    def observe(s, d, y):
        s.num += (y*d).sum(dim=(0,1)).cpu(); s.den += (d*d).sum(dim=(0,1)).cpu(); s.n+=1
        if s.n >= 2:
            A=(s.num/(s.den+1e-8)).cuda()
            em=(y-A*d).pow(2).mean().sqrt().item(); ee=y.pow(2).mean().sqrt().item()+1e-8
            s.beta = 0.7*s.beta + 0.3*(ee/(em+ee))
    def A(s): return (s.num/(s.den+1e-8)).cuda()

pipe = CogVideoXPipeline.from_pretrained(M, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
tf = pipe.transformer; orig = tf.forward
prompts = open("/root/viewdit/resources/prompts/prompt_video.txt").read().splitlines()

# front-loaded: first 10 steps dense, then 9 evals spread over the remaining 20
FRONT = list(range(10)) + [12, 15, 18, 21, 24, 26, 28, 29]

def run(pi, outdir):
    st={"i":-1,"h":[],"sm":Diag(),"nfe":0}
    fs=set(FRONT)
    def fwd(*args, **kw):
        st["i"]+=1; i=st["i"]; hs=kw.get("hidden_states", args[0] if args else None)
        xt=patchify(hs.detach().float()); h=st["h"]
        if i in fs or not h:
            out=orig(*args,**kw); o=out[0] if isinstance(out,(tuple,list)) else out
            vt=patchify(o.detach().float())
            if len(h)>=2:
                (i0,x0,v0),(i1,x1,v1)=h[-2],h[-1]
                r=(i-i1)/max(1,(i1-i0))
                st["sm"].observe((xt-x1)-r*(x1-x0), vt-(v1+r*(v1-v0)))
            h.append((i,xt,vt))
            if len(h)>2: h.pop(0)
            st["nfe"]+=1
            return out
        (i0,x0,v0),(i1,x1,v1)=h[-2],h[-1]
        r=(i-i1)/max(1,(i1-i0))
        if USE and st["sm"].n>=2:
            d=(xt-x1)-r*(x1-x0)
            vt=v1 + st["sm"].beta*(st["sm"].A()*d)
        else:
            vt=v1
        B,T,C,H,W=hs.shape
        x=vt.reshape(B,T,H//2,W//2,C,2,2).permute(0,1,4,2,5,3,6)
        return (x.reshape(B,T,C,H,W).to(hs.dtype),)
    tf.forward=fwd
    g=torch.Generator("cuda").manual_seed(pi)
    with torch.no_grad():
        out=pipe(prompt=prompts[pi], negative_prompt="animation", num_frames=17,
                 height=480, width=720, num_inference_steps=NSTEP, guidance_scale=6.0, generator=g)
    tf.forward=orig
    frames=np.stack([np.array(f).astype(np.uint8) for f in out.frames[0]])
    os.makedirs(outdir, exist_ok=True)
    iio.mimsave(os.path.join(outdir,"video_%06d.mp4"%pi), frames, fps=8, codec="libx264", macro_block_size=8)
    print("[done] p=%d nfe=%d core=%d beta=%.3f"%(pi,st["nfe"],USE,st["sm"].beta), flush=True)

for pi in range(20):
    run(pi, "results/cogcore/core%d" % USE)
print("CORE_DONE", flush=True)
