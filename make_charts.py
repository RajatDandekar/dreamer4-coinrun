import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

CREAM="#F0EEE6"; INK="#1A1A1A"; CLAY="#C15F3C"; TEAL="#3C7A7B"; LINE="#DCD8CC"; MUTED="#6B6B63"
plt.rcParams.update({"figure.facecolor":CREAM,"axes.facecolor":CREAM,"savefig.facecolor":CREAM,
    "text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
    "axes.edgecolor":LINE,"font.size":11,"font.family":"sans-serif"})

# ---- REAL observed points from our dynamics run (logged during training) ----
steps=[1277,4187,44337,52995,56137,57692,62025,62371,63600,63939,67160,70134,72048,73348,74682,75330]
flow =[0.5898,0.0679,0.0146,0.0162,0.0187,0.0177,0.0135,0.0135,0.0194,0.0194,0.0155,0.0114,0.0151,0.0118,0.0229,0.0143]
boot =[0,0,0.0073,0.0093,0.0082,0.0064,0.0068,0.0039,0.0023,0.0041,0.0076,0.0061,0.0070,0.0053,0.0067,0.0042]

fig,ax=plt.subplots(figsize=(9,4.6))
ax.axvspan(0,40000,color="#E8E5DA",alpha=.55,lw=0)
ax.axvspan(40000,80000,color="#DCD8CC",alpha=.55,lw=0)
ax.plot(steps,flow,color=CLAY,lw=2.2,marker="o",ms=4,label="flow_mse  (flow matching)")
ax.plot(steps,boot,color=TEAL,lw=2.2,marker="o",ms=4,label="boot_mse  (shortcut bootstrap)")
ax.axvline(40000,color=INK,ls="--",lw=1.2,alpha=.65)
ax.text(40600,0.42,"bootstrap_start\n(step 40,000)",fontsize=9.5,color=INK)
ax.text(8000,0.50,"Phase 1\nflow matching only",fontsize=10,color=MUTED,ha="center")
ax.text(60000,0.50,"Phase 2\n+ shortcut bootstrap",fontsize=10,color=MUTED,ha="center")
ax.set_yscale("log"); ax.set_xlabel("training step"); ax.set_ylabel("loss (log scale)")
ax.set_title("Dynamics training — measured losses",fontsize=13,pad=12,loc="left")
ax.legend(frameon=False,fontsize=10); ax.set_xlim(0,80000)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",color=LINE,lw=.7,alpha=.7)
plt.tight_layout(); plt.savefig("figures/chart_training.png",dpi=200); plt.close()

# ---- FVD decomposition (exact measured) ----
fig,ax=plt.subplots(figsize=(9,3.5))
labels=["Tokenizer ceiling\noriginal vs reconstruction","Dynamics only\nreconstruction vs dream","End-to-end\noriginal vs dream"]
vals=[16.59,23.29,32.19]; cols=[TEAL,CLAY,INK]
b=ax.barh(labels,vals,color=cols,height=.52)
for r,v in zip(b,vals): ax.text(v+.7,r.get_y()+r.get_height()/2,f"{v:.2f}",va="center",fontsize=12,color=INK,fontweight="bold")
ax.set_xlabel("FVD  (lower is better)"); ax.set_xlim(0,38); ax.invert_yaxis()
ax.set_title("Where the visual error comes from",fontsize=13,pad=12,loc="left")
for s in ("top","right","left"): ax.spines[s].set_visible(False)
ax.grid(axis="x",color=LINE,lw=.7,alpha=.7); ax.tick_params(length=0)
plt.tight_layout(); plt.savefig("figures/chart_fvd.png",dpi=200); plt.close()

# ---- Tokenizer PSNR progression (measured) ----
ts=[200,497,520,10000]; ps=[15.90,20.55,21.50,40.41]
fig,ax=plt.subplots(figsize=(9,4.2))
ax.plot(ts,ps,color=CLAY,lw=2.4,marker="o",ms=6)
for x,y in zip(ts,ps): ax.annotate(f"{y:.2f}",(x,y),textcoords="offset points",xytext=(0,10),fontsize=10,ha="center",color=INK)
for y,lab,c in [(35.7,"Genie paper  35.7",MUTED),(38.25,"GenieRedux  38.25",TEAL)]:
    ax.axhline(y,color=c,ls=":",lw=1.5); ax.text(180,y+0.6,lab,fontsize=9.5,color=c,va="bottom")
ax.set_xscale("log"); ax.set_xlabel("training step (log)"); ax.set_ylabel("reconstruction PSNR (dB)")
ax.set_title("Tokenizer quality — ours vs published baselines",fontsize=13,pad=12,loc="left")
ax.set_xlim(150,14000); ax.set_ylim(12,45)
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",color=LINE,lw=.7,alpha=.7)
plt.tight_layout(); plt.savefig("figures/chart_psnr.png",dpi=200); plt.close()
print("charts written")
