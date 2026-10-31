import os, sys, torch, numpy as np, imageio.v2 as iio
from diffusers import CogVideoXPipeline
M = "/root/viewdit/weights/CogVideoX-2b"; NSTEP = 30
SIG = os.environ.get("SIG", "model")   # model | input

def patchify(t):
    B, T, C, H, W = t.shape
    x = t.reshape(B, T, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(B, T * (H // 2) * (W // 2), C * 4)

class Diag:
    def __init__(s, dim=64): s.num=torch.zeros(dim); s.den=torch.zeros(dim); s.n=0
    def observe(s, d, y):
        s.num += (y*d).sum(dim=(0,1)).cpu(); s.den += (d*d).sum(dim=(0,1)).cpu(); s.n+=1
    def A(s): return (s.num/(s.den+1e-8)).cuda()

pipe = CogVideoXPipeline.from_pretrained(M, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
tf = pipe.transformer; orig = tf.forward
prompts = open("/root/viewdit/resources/prompts/prompt_video.txt").read().splitlines()

def run(pi, target, tau, outdir):
    st={"i":-1,"h":[],"sm":Diag(),"nfe":0}
    def fwd(*args, **kw):
        st["i"]+=1; i=st["i"]; hs=kw.get("hidden_states", args[0] if args else None)
        xt=patchify(hs.detach().float()); h=st["h"]
        due=(i+1)*target/NSTEP - st["nfe"]
        do=(i<4) or (i==NSTEP-1) or (not h) or (due>=1.0 and st["nfe"]<target)
        if not do and len(h)>=2:
            (i0,x0,v0),(i1,x1,v1)=h[-2],h[-1]
            r=(i-i1)/max(1,(i1-i0))
            d=(xt-x1)-r*(x1-x0)
            if SIG=="model" and st["sm"].n>=2:
                sig=((st["sm"].A()*d).pow(2).mean().sqrt()/(v1.pow(2).mean().sqrt()+1e-8)).item()
            else:
                sig=(d.abs().mean().item()/(x1.abs().mean().item()+1e-8))
            do = sig > tau
        if do:
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
        B,T,C,H,W=hs.shape
        vt=h[-1][2]
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
    print("[done] p=%d nfe=%d sig=%s"%(pi,st["nfe"],SIG), flush=True)

tau = float(os.environ.get("TAU", "0.10"))
for pi in range(20):
    run(pi, 19, tau, "results/cogsig/%s_%s" % (SIG, os.environ.get("TAU","0.10").replace(".","")))
print("SIG_DONE", flush=True)
