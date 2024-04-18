#!/usr/bin/env python
# coding: utf-8

import scanpy as sc
import scanpy as sc
import numpy as np
import pandas as pd
import leidenalg
from matplotlib.pyplot import rc_context
import anndata as ad
import celltypist
from celltypist import models


# Reading the data files in scanpy
adata_A1 = sc.read_10x_mtx('/Users/tomwolstenholme-hogg/Documents/Research_project/GSE175814_RAW/GSM5348374_A1', var_names='gene_symbols')
adata_A2 = sc.read_10x_mtx('/Users/tomwolstenholme-hogg/Documents/Research_project/GSE175814_RAW/GSM5348374_A2', var_names='gene_symbols')
adata_A3 = sc.read_10x_mtx('/Users/tomwolstenholme-hogg/Documents/Research_project/GSE175814_RAW/GSM5348374_A3', var_names='gene_symbols')
adata_A4 = sc.read_10x_mtx('/Users/tomwolstenholme-hogg/Documents/Research_project/GSE175814_RAW/GSM5348374_A4', var_names='gene_symbols')
# Sample 1 = AD
# Sample 3 = AD?


# Adding metadata for each sample
adata_A1.obs["sample_name"] = "A1"
adata_A2.obs['sample_name'] = 'A2'
adata_A3.obs['sample_name'] = 'A3'
adata_A4.obs['sample_name'] = 'A4'


adatas = [adata_A1, adata_A2, adata_A3, adata_A4]
adatas = ad.concat(adatas, merge='same')
adatas.var_names_make_unique()
adatas.obs_names_make_unique()





adata_A1.var['mt'] = adata_A1.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata_A1, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# Sorting data
adata_A2.var['mt'] = adata_A2.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata_A2, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata_A3.var['mt'] = adata_A3.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata_A3, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata_A4.var['mt'] = adata_A4.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata_A4, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)


# Creating a column of just MT genes.
adatas.var['mt'] = adatas.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adatas, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adatas, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], groupby='sample_name', jitter=0.4, multi_panel=True)




# Filtering the data for A1
adata_A1 = adata_A1[adata_A1.obs['n_genes_by_counts'] < 8000]
adata_A1 = adata_A1[adata_A1.obs['n_genes_by_counts'] > 1000]
           
adata_A1 = adata_A1[3000 < adata_A1.obs['total_counts']]
adata_A1 = adata_A1[50000 > adata_A1.obs['total_counts']]
adata_A1 = adata_A1[adata_A1.obs['pct_counts_mt'] < 10]

# Filtering for A2
adata_A2 = adata_A2[500 < adata_A2.obs['n_genes_by_counts']]
adata_A2 = adata_A2[6000 > adata_A2.obs['n_genes_by_counts']]

adata_A2 = adata_A2[2000 < adata_A2.obs['total_counts']]
adata_A2 = adata_A2[50000 > adata_A2.obs['total_counts']]
adata_A2 = adata_A2[adata_A2.obs['pct_counts_mt'] < 25]

# Filtering for A3
adata_A3 = adata_A3[1000 < adata_A3.obs['n_genes_by_counts']]
adata_A3 = adata_A3[7000 > adata_A3.obs['n_genes_by_counts']]


adata_A3 = adata_A3[5000 < adata_A3.obs['total_counts']]
adata_A3 = adata_A3[50000 > adata_A3.obs['total_counts']]
adata_A3 = adata_A3[adata_A3.obs.pct_counts_mt < 15]

# Filtering for A4
adata_A4 = adata_A4[500 < adata_A4.obs['n_genes_by_counts']]
adata_A4 = adata_A4[8000 > adata_A4.obs['n_genes_by_counts']]

adata_A4 = adata_A4[2000 < adata_A4.obs['total_counts']]
adata_A4 = adata_A4[60000 > adata_A4.obs['total_counts']]
                    
adata_A4 = adata_A4[adata_A4.obs.pct_counts_mt < 25]


# Forming the separate data, AD
AD_adata = [adata_A1, adata_A3]
AD_adata


AD_adata = ad.concat(AD_adata, merge='same')
AD_adata.var_names_make_unique()
AD_adata.obs_names_make_unique()

# Forming control

C_adata = [adata_A2, adata_A4]
C_adata = ad.concat(C_adata, merge='same')
C_adata.var_names_make_unique()
C_adata.obs_names_make_unique()


AD_adata.var['mt'] = AD_adata.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(AD_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(AD_adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], groupby='sample_name', jitter=0.4, multi_panel=True)


C_adata.var['mt'] = C_adata.var.index.str.startswith('MT-')
sc.pp.calculate_qc_metrics(C_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(C_adata, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], groupby='sample_name', jitter=0.4, multi_panel=True)


# Normalisation

# Makes the numbers more comparable
sc.pp.normalize_total(adatas, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
adatas.X.sum(axis= 1)
sc.pp.log1p(adatas)
sc.pp.highly_variable_genes(adatas, n_top_genes= 2000)
adatas.raw = adatas


# Doing the same for alzheimer's samples
sc.pp.normalize_total(AD_adata, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
adatas.X.sum(axis= 1)
sc.pp.log1p(AD_adata)
sc.pp.highly_variable_genes(AD_adata, n_top_genes=2000)
AD_adata.raw = AD_adata


# Doing the same for control samples
sc.pp.normalize_total(C_adata, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
adatas.X.sum(axis= 1)
sc.pp.log1p(C_adata)
sc.pp.highly_variable_genes(C_adata, n_top_genes= 2000)
C_adata.raw = C_adata


# # Using PCA to show variance of each components
# adatas = adatas[:, adatas.var.highly_variable]
# sc.pp.regress_out(adatas, ['total_counts', 'pct_counts_mt', 'n_genes_by_counts'])
# sc.pp.scale(adatas, max_value=10) 
# sc.tl.pca(adatas, svd_solver='arpack')
# sc.pl.pca_variance_ratio(adatas, log=True, n_pcs= 50)
# # The standard is 50 plots, want it logarythmic to see difference more easily
# # Could change it down to around 12 as not much variance after that but will ask George.


# Doing PCA on AD
AD_adata = AD_adata[:, AD_adata.var.highly_variable]
sc.pp.regress_out(AD_adata, ['total_counts', 'pct_counts_mt', 'n_genes_by_counts'])
sc.pp.scale(AD_adata, max_value=10) 
sc.tl.pca(AD_adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(AD_adata, log=True, n_pcs= 50)


# Reducing it to 22 PCs as that is when the variance becomes irrelevant
sc.pl.pca_variance_ratio(AD_adata, log=True, n_pcs= 15)
ad = AD_adata.uns['pca']['variance'] 
# Able to see eigen value for each PC, allows us to workout the the percentage
# variance that each PC shows.
def percentage_variance_calc(data):
    count = 0
    n = 0
    for i in data:
        n = n + (100*i/sum(data))
        print(n)
        count += 1
        print(count)
# for loop to workout the percentage variance so I know where to chop it off
percentage_variance_calc(ad)
    



# Doing PCA on control
C_adata = C_adata[:, C_adata.var.highly_variable]
sc.pp.regress_out(C_adata, ['total_counts', 'pct_counts_mt', 'n_genes_by_counts'])
sc.pp.scale(C_adata, max_value=10) 
sc.tl.pca(C_adata, svd_solver='arpack')
sc.pl.pca_variance_ratio(C_adata, log=True, n_pcs= 50)


sc.pl.pca_variance_ratio(C_adata, log=True, n_pcs= 17)
cd = C_adata.uns['pca']['variance'] 


percentage_variance_calc(cd)


# Doing the same for AD samples
sc.pp.neighbors(AD_adata, n_pcs= 15) # Forms neighbourhood matrices.
sc.tl.leiden(AD_adata) # Needed to add this in for some reason when i re ran it, but now it works, looks different to before.
sc.tl.paga(AD_adata)
sc.pl.paga(AD_adata, plot=False) # Can remove plot=False if we want to remove the coarse-grained graph
sc.tl.umap(AD_adata, init_pos='paga')
sc.pl.umap(AD_adata, color=['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B'])


# Doing the same for control samples
sc.pp.neighbors(C_adata, n_pcs= 18) # Forms neighbourhood matrices.
sc.tl.leiden(C_adata) # Needed to add this in for some reason when i re ran it, but now it works, looks different to before.
sc.tl.paga(C_adata)
sc.pl.paga(C_adata, plot=False) # Can remove plot=False if we want to remove the coarse-grained graph
sc.tl.umap(C_adata, init_pos='paga')
sc.pl.umap(C_adata, color=['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B'])


# # Using leiden graph clustering method, directly clusters the neighbourhood matrices onn the previous graph
# sc.tl.leiden(adatas, resolution= 0.5)
# sc.pl.umap(adatas, color=['leiden','APBB1IP', 'ST18', 'SV2B','CD14',
#                 'LGALS3', 'S100A8', 'FCGR3A', 'CST3', 'PPBP'])


# Using leiden clustering on just AD samples, able to see variation in APBB1IP, ST18
sc.tl.leiden(AD_adata, resolution= 0.5)
sc.pl.umap(AD_adata, color=['leiden', 'CTNNA3', 'RNF220', 'MBP', 'ST18', 'SLC44A1', 'SLC24A2', 'IL1RAPL1',
                           'PLP1', 'FRMD4B', 'FRMD5', 'PLEKHH1', 'PIP4K2A', 'PRUNE2', 'ZNF536',
                           'DOCK10', 'ANK3', 'SIK3', 'PHLPP1', 'PPP2R2B', 'UNC5C', 'TTLL7', 'ENPP2', 'TMTC2', 'PDE4B', 'CLMN'])


# UMAP on control, able to see difference in ppbp, s100A8, sv2b
sc.tl.leiden(C_adata, resolution= 0.5)
sc.pl.umap(C_adata, color=['leiden', 'CTNNA3', 'RNF220', 'MBP', 'ST18', 'SLC44A1', 'SLC24A2', 'IL1RAPL1',
                           'PLP1', 'FRMD4B', 'FRMD5', 'PLEKHH1', 'PIP4K2A', 'PRUNE2', 'ZNF536',
                           'DOCK10', 'ANK3', 'SIK3', 'PHLPP1', 'PPP2R2B', 'UNC5C', 'TTLL7', 'ENPP2', 'TMTC2', 'PDE4B', 'CLMN'])


sc.tl.leiden(C_adata, resolution= 0.5)
sc.pl.umap(C_adata, color='leiden', save=({'.png'}, True))


# # Now finding marker genes # Genes that show if a nucleic acid sequence has been inserted into something's DNA or not.
# sc.tl.rank_genes_groups(adatas, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adatas, n_genes=25, sharey=False)
# sc.settings.verbosity = 2 # Reduce the verbosity(number of samples?)


# Ranking the genes using logistic regression
# sc.tl.rank_genes_groups(adatas, 'leiden', method='logreg')
# sc.pl.rank_genes_groups(adatas, n_genes=25, sharey=False)


# # Making a list of marker genes, not sure if we want to change these?
# marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
#                 'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
#                 'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']
# # Shows the top 10 ranked genes per cluster in a dataframe?


# pd.DataFrame(adatas.uns['rank_genes_groups']['names']).head(5)


# # Not sure about this
# result = adatas.uns['rank_genes_groups']
# """groups = result['names'].dtype.names
# pd.DataFrame({group + '_' + key[:1]: result[key][group]
#     for group in groups for key in ['names', 'pvals']}).head(5)"""


# Compare it to a single cluster
# sc.tl.rank_genes_groups(adatas, 'leiden', groups=['0'], reference='1', method='wilcoxon')
# sc.pl.rank_genes_groups(adatas, groups=['0'], n_genes=20)


# Now plotting it in a violin so that we can visualise it better
# sc.pl.rank_genes_groups_violin(adatas, groups='0', n_genes= 7)
# Went with 7 genes as that is when the variation seems to stop.


# Showing it with computed differential expression (comparing it with the rest of the groups)
#sc.pl.rank_genes_groups_violin(adatas, groups='0', n_genes=7)
# Shows the exact same graph for some reason


# Comparing certain gene across groups?
# sc.pl.violin(adatas, ['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B'], groupby='leiden')


# Marking the clusters for umap
# new_cluster_names = [
#     'Inhibitory-iB', 'Inhibitory-iA', 'Microgia', 'Macroglial-Oligodendrocyte'
# , 'Macroglial-Polydendrocyte', 'Astrocyte', 'Inhibitory_CXCL14', 
# 'Inhibitory GRIK-ERBB4', 'Excitatory', 'Endothelial-Pericyte','11','12',
#     '13','14','15', '16','17','18','19','20','21']
# adatas.rename_categories('leiden', new_cluster_names)
# I don't have all of the category names yet for some reason they're not in the paper
# Need to maybe make 2 umaps below 1 of the controls combined and one of the alzheimers so we can compare.
# sc.pl.umap(adatas, color='leiden', legend_loc='on data', title='', frameon=False, save='.pdf')


# sc.pl.dotplot(adatas, marker_genes, groupby='leiden')


# sc.pl.stacked_violin(adatas, marker_genes, groupby='leiden', rotation=90)


# with rc_context({'figure.figsize': (3, 3)}):
#     sc.pl.umap(adatas, color=['APBB1IP', 'ST18', 'PDGFRA', 'GRIK3', 'SV2B','KCNH1', 'KCNQ5', 'DLGAP2'
#                               , 'FRMPD4', 'KHDRBS2', 'IQCJ-SCHIP1','n_genes_by_counts','sample_name'], s=50, frameon=False, ncols=4, vmax='p99')


with rc_context({'figure.figsize': (3, 3)}):
    sc.pl.umap(AD_adata, color=['CTNNA3', 'RNF220', 'MBP', 'ST18', 'SLC44A1', 'SLC24A2', 'IL1RAPL1',
                           'PLP1', 'FRMD4B', 'FRMD5', 'PLEKHH1', 'PIP4K2A', 'PRUNE2', 'ZNF536',
                           'DOCK10', 'ANK3', 'SIK3', 'PHLPP1', 'PPP2R2B', 'UNC5C', 'TTLL7', 'ENPP2', 'TMTC2', 'PDE4B', 'CLMN','n_genes_by_counts','sample_name'], s=50, frameon=False, ncols=4, vmax='p99')
    sc.pl.umap(AD_adata, color='sample_name', save=True)


with rc_context({'figure.figsize': (3, 3)}):
    sc.pl.umap(C_adata, color=['CTNNA3', 'RNF220', 'MBP', 'ST18', 'SLC44A1', 'SLC24A2', 'IL1RAPL1',
                           'PLP1', 'FRMD4B', 'FRMD5', 'PLEKHH1', 'PIP4K2A', 'PRUNE2', 'ZNF536',
                           'DOCK10', 'ANK3', 'SIK3', 'PHLPP1', 'PPP2R2B', 'UNC5C', 'TTLL7', 'ENPP2', 'TMTC2', 'PDE4B', 'CLMN','n_genes_by_counts','sample_name'], s=50, frameon=False, ncols=4, vmax='p99')
    sc.pl.umap(C_adata, color='sample_name', save=True)


# sc.tl.leiden(adatas, key_added='clusters', resolution=0.5)
# with rc_context ({'figure.figsize': (5, 5)}):
#     sc.pl.umap(adatas, color='clusters', add_outline=True,
#                 legend_loc='on data', legend_fontsize=12, legend_fontoutline=2, frameon=False,
#                 title='Clustering of cells', palette='Set1')


sc.tl.leiden(C_adata, key_added='clusters', resolution=0.5)
with rc_context ({'figure.figsize': (5, 5)}):
    sc.pl.umap(C_adata, color='clusters', add_outline=True,
                legend_loc='on data', legend_fontsize=12, legend_fontoutline=2, frameon=False,
                title='Clustering of cells', palette='Set1')


AD_adata.shape
#AD_adata.X.expm1().sum(axis = 1)


AD_adata.obs


models.download_models(force_update = True)


models.models_path


model = models.Model.load(model = 'Human_AdultAged_Hippocampus.pkl')
mouse_model = models.Model.load(model = 'Developing_Mouse_Brain.pkl')


predictions = celltypist.annotate(AD_adata, model= 'Human_AdultAged_Hippocampus.pkl',
                                 majority_voting = True)
Developing_predictions = celltypist.annotate(AD_adata, model='Developing_Mouse_Brain.pkl',
                                        majority_voting = True)
#predictions.predicted_labels.predicted_labels
AD_adata.obs["celltypist_preds"] = predictions.predicted_labels.predicted_labels
#predictions.predicted_labels.predicted_labels.index.value_counts()
#AD_adata
AD_adata.obs["developing_preds"] = Developing_predictions.predicted_labels.predicted_labels



sc.tl.rank_genes_groups(AD_adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(AD_adata, n_genes=25, sharey=False)


cluster_names = ['Oligodendrocyte', 'Microglia', 'Astrocyte','Astrocyte of the cerebral cortex',
                'L2/3-6 intratelencephalic projecting glutamatergic cortical neuron', 
                'Oligodendrocyte Precursor cell', 'caudal ganglionic eminence derived GABAergic cortical interneuron',
                'corticothalamic-projecting glutamatergic cortical neuron', 'Macroglial cell',
                'chandelier pvalb GABAergic cortical interneuron', 'Cerebral cortex endothelial cell', 'Pyramidal neuron',
                'st GABAergic cortical interneuron', 'lamp5 GABAergic cortical interneuron','Mature microglial cell',
                'Mature microglial cell2', ' Differentiation committed oligodendrocyte precursor', 'Astrocyte of the cerebral cortex2',
                'Fibroblast','']
AD_adata.rename_categories('leiden', cluster_names)



with rc_context ({'figure.figsize': (5, 5)}):
    sc.pl.umap(AD_adata, color='leiden', add_outline=True,
                legend_loc='on data', legend_fontsize=12, legend_fontoutline=2, frameon=False,
                title='Clustering of AD cells', palette='Set1')


sc.tl.umap(AD_adata)
#sc.pl.umap(AD_adata, color='leiden', legend_loc= 'on data')
sc.pl.umap(AD_adata, color='celltypist_preds', legend_loc= 'on data', save=True)




