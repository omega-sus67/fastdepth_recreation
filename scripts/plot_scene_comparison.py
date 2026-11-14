#!/usr/bin/env python3
"""Plot the fixed scene-pruning tuning comparison without claiming significance."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--summary',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output exists')
 s=json.loads(a.summary.read_text());models=s['models'];methods=['original','weight','average_ablation','scene_aware','random']
 labels=['Original','Weight','Average\nablation','Scene-aware','Random']
 fig,ax=plt.subplots(figsize=(9,5.5),layout='constrained')
 offsets=[-0.18,0,0.18];colors=['#245f9c','#de7a33','#28836a']
 for i,method in enumerate(methods):
  values=[models[f'{method}_{seed}']['mean_image_rmse'] for seed in [42,43,44]]
  for k,(seed,value) in enumerate(zip([42,43,44],values)):
   ax.scatter(i+offsets[k],value,color=colors[k],s=55,label=f'Seed {seed}' if i==0 else None,zorder=3)
  ax.hlines(np.mean(values),i-0.27,i+0.27,color='#333333',linewidth=1.5,label='Three-seed mean' if i==0 else None)
 ax.set_xticks(range(len(methods)),labels);ax.set_ylabel('Mean per-image RMSE (native units; lower is better)')
 ax.set_title('Exploratory recovery comparison on 445 tuning images (zoomed scale)');ax.grid(axis='y',alpha=0.25);ax.legend(ncol=4,fontsize=9)
 fig.text(0.5,-0.06,'Previously inspected tuning set; unknown checkpoint pretraining overlap; not an Orin or final-test result.',ha='center',fontsize=8)
 a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170,bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
