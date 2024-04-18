#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import scanpy as sc
import os
import numpy as np
import pandas as pd
import leidenalg
import anndata as ad
import scrublet as scr
from matplotlib.pyplot import rc_context
import milopy
import milopy.core as milo
import scvelo as scv
import warnings
warnings.filterwarnings("ignore")




# In[ ]:


in_dat = sc.read_csv('/Users/tomwolstenholme-hogg/anaconda3/envs/dissertation/GSE138852_counts.csv').transpose()
adata = sc.AnnData(in_dat)
adata.var_names_make_unique()
adata.obs_names_make_unique()


# In[ ]:


print(os.getcwd())


# In[ ]:


labels = pd.read_csv('/Users/tomwolstenholme-hogg/anaconda3/envs/dissertation/scRNA_metadata.tsv', sep='\t')
selected_column = 'cellType'
adata.obs['cell_type'] = labels[selected_column].values
adata.obs


# In[ ]:


# Creating a column of just MT genes.
adata.var['mt'] = adata.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)


# In[ ]:


def scrubbing(datas):
    
    scrub = scr.Scrublet(datas.X)

    # Perform doublet detection
    scrub_results = scrub.scrub_doublets(min_counts=2, min_cells=3, n_prin_comps=30)

    datas.obs['doublet_scores'] = scrub_results[0]
    datas.obs['predicted_doublets'] = scrub_results[1]

    # Filter out predicted doublets
    datas = datas[datas.obs['predicted_doublets'] == False]

    # Ensure the AnnData object is properly updated
    datas.raw = datas.copy()
    scrub.plot_histogram()
    scrub.set_embedding('UMAP', scr.get_umap(scrub.manifold_obs_, 10, min_dist=0.3))
    scrub.plot_embedding('UMAP', order_points=True);
    datas.obs['predicted_doublets'].value_counts()
    return datas
adata = scrubbing(adata)


# In[ ]:


def plotting_violins(data):
    data.var['mt'] = data.var.index.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(data, keys=['total_counts', 'n_genes_by_counts', 'total_counts_mt'], jitter=0.4, multi_panel=True)
    return data


# In[ ]:


plotting_violins(adata)


# In[ ]:


def normalisation(data):
    sc.pp.normalize_total(data, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
    data.X.sum(axis=1)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes= 2000)
    data.raw = data
    return data

# Makes the numbers more comparable
adata = normalisation(adata)


# In[ ]:


# PCA function
def PCA(data, pcs):
    data = data[:, data.var.highly_variable]
    sc.pp.regress_out(data, ['total_counts', 'pct_counts_mt', 'n_genes_by_counts'])
    sc.pp.scale(data, max_value=10) 
    sc.tl.pca(data, svd_solver='arpack')
    sc.pl.pca_variance_ratio(data, log=True, n_pcs= pcs)
    data.raw = data
    return data

adata = PCA(adata, 50)


# In[ ]:


# Able to see eigen value for each PC, allows us to workout the the percentage
# variance that each PC shows.
d = adata.uns['pca']['variance'] 

def percentage_variance_calc(data):
    count = 0
    n = 0
    for i in data:
        n = n + (100*i/sum(data))
        print(n)
        count += 1
        print(count)
# for loop to workout the percentage variance so I know where to chop it off
percentage_variance_calc(d)


# In[ ]:


sc.pp.neighbors(adata, n_pcs= 30) # Forms neighbourhood matrices.
sc.tl.leiden(adata) # Needed to add this in for some reason when i re ran it, but now it works, looks different to before.
sc.tl.paga(adata)
sc.pl.paga(adata, plot=False) # Can remove plot=False if we want to remove the coarse-grained graph
sc.tl.umap(adata, init_pos='paga')


# In[ ]:


adata.obs['sample_ID'] = list(map(lambda x: "_".join(x.split("_")[1:]), adata.obs.index))
adata.obs['sequence'] = list(map(lambda x: "_".join(x.split("_")[0:1]), adata.obs.index))


# In[ ]:


adata.obs['Disease_status'] = list(map(lambda x: x[0:2], adata.obs['sample_ID']))
pd.DataFrame(adata.obs['Disease_status']).value_counts()


# In[ ]:


sc.pl.umap(adata, color=['Disease_status', 'cell_type'])


# In[ ]:


sc.pp.neighbors(adata, n_neighbors= 10)
milo.make_nhoods(adata)

# Counting the number of each cell type
milo.count_nhoods(adata, sample_col='sample_ID')

# Comparing between the case and control,
# tried using the sequence so that it would be unique but it still doesn't work
milo.DA_nhoods(adata, design= "~ Disease_status")

# Checking the results
milo_results = adata.uns['nhood_adata'].obs
print(milo_results)


# In[ ]:


milopy.utils.build_nhood_graph(adata)
milopy.plot.plot_nhood_graph(adata, alpha=0.2, min_size=5)
sc.pl.umap(adata, color=['Disease_status', 'cell_type'])


# In[ ]:


scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')


# In[ ]:


scv.tl.velocity(adata, mode='deterministic')


# In[ ]:


# Visualising the data, add same params to these
scv.pl.velocity_embedding(adata, basis='umap')
scv.pl.velocity_embedding_grid(adata, basis='umap')
scv.pl.velocity_embedding_stream(adata, basis= 'umap')


# In[ ]:


scv.pl.velocity_embedding(adata, arrow_length=3, arrow_size=2, dpi= 120)


# In[ ]:


# Doing the plotting counterpart, allows us to examine our results in detail
scv.pl.velocity(adata, var_names='cell_type')
scv.pl.velocity_graph(adata)


# In[ ]:


scv.pl.proportions(adata)


# In[ ]:


# Just another way to visualise the data, plot all of them to see which George prefers
scv.pl.scatter(adata, 'Cpe', color=['leiden', 'velocity'],
               add_outline='cell_type')


# In[ ]:


# Identifying the important genes

scv.tl.rank_velocity_genes(adata, groupby='leiden', min_corr= .3)
df = scv.DataFrame(adata.uns['rank_velocity_genes']['cell_types'])
df.head()


# In[ ]:


scv.tl.score_genes_cell_cycle(adata)
scv.pl.scatter(adata, color_gradients=['S_score', 'G2M_score'], smooth=True, perc=[5, 95])


# In[ ]:


# Doing the speed and coherence
# Tells us where cells differentiate at a slower/faster pace, and where direction is un-/determined
scv.tl.velocity_confidence(adata)
keys = 'velocity_length', 'velocity_confidence'
scv.pl.scatter(adata, c=keys, cmap='coolwarm', perc=[5, 95])


# In[ ]:


# Able to draw the descendents coming from a specified cell, so we can track an early cell
# To its potential fate

x, y = scv.utils.get_cell_transitions(adata, basis='umap', starting_cell=70)
ax = scv.pl.velocity_graph(adata, c='lightgrey', edge_width=.05, show= False)
ax = scv.pl.scatter(adata, x=x, y=y, s=120, c='ascending', cmap='gnuplot', ax=ax)


# In[ ]:


scv.tl.velocity_pseudotime(adata)
scv.pl.scatter(adata, color='velocity_pseudotime', cmap='gnuplot')

