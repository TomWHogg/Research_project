#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import os
import numpy as np
import pandas as pd
import leidenalg
import anndata as ad
import scrublet as scr
from matplotlib.pyplot import rc_context
import matplotlib.pyplot as plt
import milopy
import milopy.core as milo
import scvelo as scv
import warnings
warnings.filterwarnings("ignore")
import umap.plot
import milopy.plot as milopl
import milopy.utils
import plotly.express as px
import seaborn as sns


# In[2]:


in_dat = sc.read_csv('/Users/tomwolstenholme-hogg/anaconda3/envs/dissertation/GSE138852_counts.csv').transpose()
adata = sc.AnnData(in_dat)
adata.var_names_make_unique()
adata.obs_names_make_unique()


# In[3]:


print(os.getcwd())


# In[4]:


labels = pd.read_csv('/Users/tomwolstenholme-hogg/anaconda3/envs/dissertation/scRNA_metadata.tsv', sep='\t')
selected_column = 'cellType'
adata.obs['cell_type'] = labels[selected_column].values
adata.obs


# In[5]:


def plotting_violins(data, output_path='_plots.pdf'):
    data.var['mt'] = data.var.index.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(data, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save=output_path)
    return data

plotting_violins(adata)


# In[6]:


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


# In[7]:


plotting_violins(adata)


# In[8]:


def normalisation(data):
    sc.pp.normalize_total(data, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
    data.X.sum(axis=1)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes= 2000)
    data.raw = data
    return data

# Makes the numbers more comparable
adata = normalisation(adata)


# In[9]:


# PCA function
def PCA(data, pcs, output_path='pca_plot.pdf' ):
    data = data[:, data.var.highly_variable]
    sc.pp.regress_out(data, ['total_counts', 'pct_counts_mt', 'n_genes_by_counts'])
    sc.pp.scale(data, max_value=10) 
    sc.tl.pca(data, svd_solver='arpack')
    sc.pl.pca_variance_ratio(data, log=True, n_pcs= pcs, save=output_path)
    data.raw = data
    return data

adata = PCA(adata, 50)


# In[10]:


# Commented out for now as it's been calculated and we're using 85%

# # Able to see eigen value for each PC, allows us to workout the the percentage
# # variance that each PC shows.
# d = adata.uns['pca']['variance'] 

# def percentage_variance_calc(data):
#     count = 0
#     n = 0
#     for i in data:
#         n = n + (100*i/sum(data))
#         print(n)
#         count += 1
#         print(count)
# # for loop to workout the percentage variance so I know where to chop it off
# percentage_variance_calc(d)


# In[11]:


sc.pp.neighbors(adata, n_pcs= 32) # Forms neighbourhood matrices.
sc.tl.leiden(adata) # Needed to add this in for some reason when i re ran it, but now it works, looks different to before.
sc.tl.paga(adata)
sc.pl.paga(adata, plot=False) # Can remove plot=False if we want to remove the coarse-grained graph
sc.tl.umap(adata, init_pos='paga')


# In[12]:


adata.obs['sample_ID'] = list(map(lambda x: "_".join(x.split("_")[1:]), adata.obs.index))
adata.obs['sequence'] = list(map(lambda x: "_".join(x.split("_")[0:1]), adata.obs.index))


# In[13]:


adata.obs['Disease_status'] = list(map(lambda x: x[0:2], adata.obs['sample_ID']))
pd.DataFrame(adata.obs['Disease_status']).value_counts()


# In[15]:


custom_palette = {'AD': '#6FA0D6', 'Ct': '#FF8C99'}
fig, ax1 = plt.subplots(1, 2, figsize=(10, 4))  # Create subplots with two panels

# UMAP for Disease_status
sc.pl.umap(adata, color='Disease_status', palette=custom_palette, show=False, ax=ax1[0])

# UMAP for cell_type
sc.pl.umap(adata, color='cell_type', show=False, ax=ax1[1])

# Adjust layout
plt.tight_layout()

# Save or show the plot
plt.savefig('umap_combined.pdf')
plt.show()


# In[16]:


sc.pp.neighbors(adata, n_neighbors= 10)
milo.make_nhoods(adata)

adata[adata.obs['nhood_ixs_refined'] != 0].obs[['nhood_ixs_refined', 'nhood_kth_distance']]


# Counting the number of each cell type
milo.count_nhoods(adata, sample_col='sample_ID')


# Comparing between the case and control,
milo.DA_nhoods(adata, design= "~ Disease_status")

# Checking the results
adata.uns['nhood_adata'].obs


# In[50]:


milopy.utils.build_nhood_graph(adata)



fig2, ax2 = plt.subplots(1, 3, figsize=(24, 6))  # Create subplots with three panels
sc.pl.umap(adata, color='cell_type', show=False, ax= ax2[0])
sc.pl.umap(adata, color='Disease_status', palette=custom_palette, show=False, ax= ax2[1])

milopy.plot.plot_nhood_graph(adata, alpha=0.2, min_size=5, save=(True, '.pdf'), ax= ax2[2])



plt.tight_layout()

plt.savefig('Differential_abundance_with_umaps.pdf')
plt.show()


# In[27]:


milopy.utils.annotate_nhoods(adata, anno_col= 'cell_type')
adata.uns['nhood_adata'].obs


# In[61]:


fig1 = px.violin(x= adata.uns['nhood_adata'].obs['logFC'], color= adata.uns['nhood_adata'].obs['nhood_annotation'])
fig1.update_layout(xaxis_title='Log-Fold change',paper_bgcolor='white')
fig1.show()


# In[75]:


sc.pl.violin(adata.uns['nhood_adata'], 'logFC', groupby='nhood_annotation', show=False)
plt.yticks(np.arange(-15, 15, 2.5))
plt.axhline(y=0, color='black', linestyle='--');

plt.xlabel('Cell Type')
plt.ylabel('Log-Fold change')
plt.show()


# In[51]:


old_figsize = plt.rcParams["figure.figsize"]
plt.rcParams["figure.figsize"] = [10,5]
plt.subplot(1,2,1)
plt.hist(adata.uns["nhood_adata"].obs.PValue, bins=50);
plt.xlabel("P-Vals");
plt.subplot(1,2,2)
plt.plot(adata.uns["nhood_adata"].obs.logFC, -np.log10(adata.uns["nhood_adata"].obs.SpatialFDR), '.');
plt.xlabel("log-Fold Change");
plt.ylabel("- log10(Spatial FDR)");
plt.tight_layout()
plt.rcParams["figure.figsize"] = old_figsize


# In[ ]:


scv.settings.verbosity = 3
scv.settings.presenter_view = True
scv.set_figure_params('scvelo')


# In[ ]:


scv.pp.moments(adata)


# In[ ]:


#scv.tl.velocity(adata, mode='deterministic', use_raw = True)


# In[ ]:


adata.layers


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

