#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scanpy as sc
import scanpy as sc
import numpy as np
import pandas as pd
import leidenalg
from matplotlib.pyplot import rc_context
import anndata as ad


# In[2]:


# Reading the data files in scanpy
adata_A1 = sc.read_10x_mtx('/Users/tomwolstenholme-hogg/Documents/Research_project/GSE175814_RAW/GSM5348374_A1', var_names='gene_symbols')
adata_A2 = sc.read_10x_mtx('/Users/tomwolstenholme-hogg/Documents/Research_project/GSE175814_RAW/GSM5348374_A2', var_names='gene_symbols')
adata_A3 = sc.read_10x_mtx('/Users/tomwolstenholme-hogg/Documents/Research_project/GSE175814_RAW/GSM5348374_A3', var_names='gene_symbols')
adata_A4 = sc.read_10x_mtx('/Users/tomwolstenholme-hogg/Documents/Research_project/GSE175814_RAW/GSM5348374_A4', var_names='gene_symbols')
# Sample 1 = AD
# Sample 3 = AD?


# In[3]:


# Adding metadata for each sample
adata_A1.obs["sample_name"] = "A1"
adata_A2.obs['sample_name'] = 'A2'
adata_A3.obs['sample_name'] = 'A3'
adata_A4.obs['sample_name'] = 'A4'


# In[4]:


AD_adata = [adata_A1, adata_A3]
C_adata = [adata_A2, adata_A4]
AD_adata = ad.concat(AD_adata, merge='same')
C_adata = ad.concat(C_adata, merge='same')


# In[5]:


adatas = [adata_A1, adata_A2, adata_A3, adata_A4]
adatas = ad.concat(adatas, merge='same')
adatas.var_names_make_unique()
adatas.obs_names_make_unique()
adatas.obs
adatas.var


# In[6]:


sc.pl.highest_expr_genes(adatas, n_top=20, )


# In[7]:


# Creating a column of just MT genes.
adatas.var['mt'] = adatas.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adatas, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], groupby='sample_name', jitter=0.4, multi_panel=True)




# In[8]:


# Creating a column of just MT genes.
C_adata.var['mt'] = C_adata.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(C_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(C_adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], groupby='sample_name', jitter=0.4, multi_panel=True)



# In[9]:


# Creating a column of just MT genes for AD sample.
AD_adata.var['mt'] = AD_adata.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(AD_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(AD_adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], groupby='sample_name', jitter=0.4, multi_panel=True)


# In[10]:


# Finding the 98th percentile of counts
upper_lim = np.quantile(adatas.obs.n_genes_by_counts.values, .98)


# In[11]:


# Filtering them out,
adatas = adatas[adatas.obs.n_genes_by_counts < upper_lim]
adatas = adatas[adatas.obs.pct_counts_mt < 2.5]


# In[12]:


# Normalisation

# Makes the numbers more comparable
sc.pp.normalize_total(adatas, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
adatas.X.sum(axis= 1)
sc.pp.log1p(adatas)
sc.pp.highly_variable_genes(adatas, n_top_genes= 2000)
sc.pl.highly_variable_genes(adatas)
adatas.raw = adatas


# In[13]:


# Doing the same for alzheimer's samples
sc.pp.normalize_total(AD_adata, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
adatas.X.sum(axis= 1)
sc.pp.log1p(AD_adata)
sc.pp.highly_variable_genes(AD_adata, n_top_genes= 2000)
sc.pl.highly_variable_genes(AD_adata)
AD_adata.raw = AD_adata


# In[14]:


# Doing the same for control samples
sc.pp.normalize_total(C_adata, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
adatas.X.sum(axis= 1)
sc.pp.log1p(C_adata)
sc.pp.highly_variable_genes(C_adata, n_top_genes= 2000)
sc.pl.highly_variable_genes(C_adata)
AD_adata.raw = AD_adata


# In[15]:


# Using PCA to show variance of each components
adatas = adatas[:, adatas.var.highly_variable]
sc.pp.regress_out(adatas, ['total_counts', 'pct_counts_mt', 'n_genes_by_counts'])
sc.pp.scale(adatas, max_value=10) 
sc.tl.pca(adatas, svd_solver='arpack')
sc.pl.pca_variance_ratio(adatas, log=True, n_pcs= 100)
sc.pl.pca(adatas, color='CST3')
# I don't think CST3 is the best one to measure.
# The standard is 50 plots, want it logarythmic to see difference more easily
# Could change it down to around 12 as not much variance after that but will ask George.


# In[16]:


# Doing PCA on AD
adatas = adatas[:, adatas.var.highly_variable]
sc.pp.regress_out(adatas, ['total_counts', 'pct_counts_mt', 'n_genes_by_counts'])
sc.pp.scale(adatas, max_value=10) 
sc.tl.pca(adatas, svd_solver='arpack')
sc.pl.pca_variance_ratio(adatas, log=True, n_pcs= 500)


# In[17]:


# adatas.write(Research_Project), not sure how to do this, could pickle it?


# In[18]:


sc.pp.neighbors(adatas, n_pcs= 30) # Forms neighbourhood matrices.
sc.tl.leiden(adatas) # Needed to add this in for some reason when i re ran it, but now it works, looks different to before.
sc.tl.paga(adatas)
sc.pl.paga(adatas, plot=False) # Can remove plot=False if we want to remove the coarse-grained graph
sc.tl.umap(adatas, init_pos='paga')
sc.pl.umap(adatas, color=['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B'])


# In[19]:


# Doing the same for AD samples
sc.pp.neighbors(AD_adata, n_pcs= 30) # Forms neighbourhood matrices.
sc.tl.leiden(AD_adata) # Needed to add this in for some reason when i re ran it, but now it works, looks different to before.
sc.tl.paga(AD_adata)
sc.pl.paga(AD_adata, plot=False) # Can remove plot=False if we want to remove the coarse-grained graph
sc.tl.umap(AD_adata, init_pos='paga')
sc.pl.umap(AD_adata, color=['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B'])


# In[20]:


# Doing the same for control samples
sc.pp.neighbors(C_adata, n_pcs= 30) # Forms neighbourhood matrices.
sc.tl.leiden(C_adata) # Needed to add this in for some reason when i re ran it, but now it works, looks different to before.
sc.tl.paga(C_adata)
sc.pl.paga(C_adata, plot=False) # Can remove plot=False if we want to remove the coarse-grained graph
sc.tl.umap(C_adata, init_pos='paga')
sc.pl.umap(C_adata, color=['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B'])


# In[21]:


# Using leiden graph clustering method, directly clusters the neighbourhood matrices onn the previous graph
sc.tl.leiden(adatas, resolution= 0.5)
sc.pl.umap(adatas, color=['leiden','APBB1IP', 'ST18', 'SV2B','CD14',
                'LGALS3', 'S100A8', 'FCGR3A', 'CST3', 'PPBP'])


# In[22]:


# Using leiden clustering on just AD samples, able to see variation in APBB1IP, ST18
sc.tl.leiden(AD_adata, resolution= 0.5)
sc.pl.umap(AD_adata, color=['leiden', 'CTNNA3', 'RNF220', 'MBP', 'ST18', 'SLC44A1', 'SLC24A2', 'IL1RAPL1',
                           'PLP1', 'FRMD4B', 'FRMD5', 'PLEKHH1', 'PIP4K2A', 'PRUNE2', 'ZNF536',
                           'DOCK10', 'ANK3', 'SIK3', 'PHLPP1', 'PPP2R2B', 'UNC5C', 'TTLL7', 'ENPP2', 'TMTC2', 'PDE4B', 'CLMN'])


# In[23]:


# UMAP on control, able to see difference in ppbp, s100A8, sv2b
sc.tl.leiden(C_adata, resolution= 0.5)
sc.pl.umap(C_adata, color=['leiden', 'CTNNA3', 'RNF220', 'MBP', 'ST18', 'SLC44A1', 'SLC24A2', 'IL1RAPL1',
                           'PLP1', 'FRMD4B', 'FRMD5', 'PLEKHH1', 'PIP4K2A', 'PRUNE2', 'ZNF536',
                           'DOCK10', 'ANK3', 'SIK3', 'PHLPP1', 'PPP2R2B', 'UNC5C', 'TTLL7', 'ENPP2', 'TMTC2', 'PDE4B', 'CLMN'])


# In[24]:


# Now finding marker genes # Genes that show if a nucleic acid sequence has been inserted into something's DNA or not.
sc.tl.rank_genes_groups(adatas, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adatas, n_genes=25, sharey=False)
sc.settings.verbosity = 2 # Reduce the verbosity(number of samples?)


# In[25]:


# Ranking the genes using logistic regression
sc.tl.rank_genes_groups(adatas, 'leiden', method='logreg')
sc.pl.rank_genes_groups(adatas, n_genes=25, sharey=False)


# In[26]:


# Making a list of marker genes, not sure if we want to change these?
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
# Shows the top 10 ranked genes per cluster in a dataframe?


# In[27]:


pd.DataFrame(adatas.uns['rank_genes_groups']['names']).head(5)


# In[28]:


# Not sure about this
result = adatas.uns['rank_genes_groups']
"""groups = result['names'].dtype.names
pd.DataFrame({group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)"""


# In[29]:


# Compare it to a single cluster
sc.tl.rank_genes_groups(adatas, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adatas, groups=['0'], n_genes=20)


# In[30]:


# Now plotting it in a violin so that we can visualise it better
sc.pl.rank_genes_groups_violin(adatas, groups='0', n_genes= 7)
# Went with 7 genes as that is when the variation seems to stop.


# In[31]:


# Showing it with computed differential expression (comparing it with the rest of the groups)
#sc.pl.rank_genes_groups_violin(adatas, groups='0', n_genes=7)
# Shows the exact same graph for some reason


# In[32]:


# Comparing certain gene across groups?
sc.pl.violin(adatas, ['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B'], groupby='leiden')


# In[33]:


# Marking the clusters for umap
new_cluster_names = [
    'Inhibitory-iB', 'Inhibitory-iA', 'Microgia', 'Macroglial-Oligodendrocyte'
, 'Macroglial-Polydendrocyte', 'Astrocyte', 'Inhibitory_CXCL14', 
'Inhibitory GRIK-ERBB4', 'Excitatory', 'Endothelial-Pericyte','11','12',
    '13','14','15', '16','17','18','19','20','21']
adatas.rename_categories('leiden', new_cluster_names)
# I don't have all of the category names yet for some reason they're not in the paper
# Need to maybe make 2 umaps below 1 of the controls combined and one of the alzheimers so we can compare.
sc.pl.umap(adatas, color='leiden', legend_loc='on data', title='', frameon=False, save='.pdf')


# In[34]:


sc.pl.dotplot(adatas, marker_genes, groupby='leiden')


# In[35]:


sc.pl.stacked_violin(adatas, marker_genes, groupby='leiden', rotation=90)


# In[36]:


with rc_context({'figure.figsize': (3, 3)}):
    sc.pl.umap(adatas, color=['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B','KCNH1', 'KCNQ5', 'DLGAP2'
                              , 'FRMPD4', 'KHDRBS2', 'IQCJ-SCHIP1','n_genes_by_counts','sample_name'], s=50, frameon=False, ncols=4, vmax='p99')


# In[37]:


with rc_context({'figure.figsize': (3, 3)}):
    sc.pl.umap(AD_adata, color=['CTNNA3', 'RNF220', 'MBP', 'ST18', 'SLC44A1', 'SLC24A2', 'IL1RAPL1',
                           'PLP1', 'FRMD4B', 'FRMD5', 'PLEKHH1', 'PIP4K2A', 'PRUNE2', 'ZNF536',
                           'DOCK10', 'ANK3', 'SIK3', 'PHLPP1', 'PPP2R2B', 'UNC5C', 'TTLL7', 'ENPP2', 'TMTC2', 'PDE4B', 'CLMN','n_genes_by_counts','sample_name'], s=50, frameon=False, ncols=4, vmax='p99')


# In[38]:


with rc_context({'figure.figsize': (3, 3)}):
    sc.pl.umap(C_adata, color=['CTNNA3', 'RNF220', 'MBP', 'ST18', 'SLC44A1', 'SLC24A2', 'IL1RAPL1',
                           'PLP1', 'FRMD4B', 'FRMD5', 'PLEKHH1', 'PIP4K2A', 'PRUNE2', 'ZNF536',
                           'DOCK10', 'ANK3', 'SIK3', 'PHLPP1', 'PPP2R2B', 'UNC5C', 'TTLL7', 'ENPP2', 'TMTC2', 'PDE4B', 'CLMN','n_genes_by_counts','sample_name'], s=50, frameon=False, ncols=4, vmax='p99')


# In[39]:


sc.tl.leiden(adatas, key_added='clusters', resolution=0.5)
with rc_context ({'figure.figsize': (5, 5)}):
    sc.pl.umap(adatas, color='clusters', add_outline=True,
                legend_loc='on data', legend_fontsize=12, legend_fontoutline=2, frameon=False,
                title='Clustering of cells', palette='Set1')


# In[40]:


sc.tl.leiden(AD_adata, key_added='clusters', resolution=0.5)
with rc_context ({'figure.figsize': (5, 5)}):
    sc.pl.umap(AD_adata, color='clusters', add_outline=True,
                legend_loc='on data', legend_fontsize=12, legend_fontoutline=2, frameon=False,
                title='Clustering of cells', palette='Set1')


# In[41]:


sc.tl.leiden(C_adata, key_added='clusters', resolution=0.5)
with rc_context ({'figure.figsize': (5, 5)}):
    sc.pl.umap(C_adata, color='clusters', add_outline=True,
                legend_loc='on data', legend_fontsize=12, legend_fontoutline=2, frameon=False,
                title='Clustering of cells', palette='Set1')


# In[ ]:




