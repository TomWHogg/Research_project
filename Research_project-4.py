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
from pyslingshot import Slingshot
import gseapy as gp
from gseapy import barplot, dotplot
from gseapy import Msigdb
from gseapy import enrichment_map
import networkx as nx
from scipy.stats import chi2_contingency
from matplotlib.patches import FancyArrowPatch



# In[2]:


in_dat = sc.read_csv('/Users/tomwolstenholme-hogg/anaconda3/envs/dissertation/GSE138852_counts.csv').transpose()
adata = sc.AnnData(in_dat)
adata.var_names_make_unique()
adata.obs_names_make_unique()


# In[3]:


labels = pd.read_csv('/Users/tomwolstenholme-hogg/anaconda3/envs/dissertation/scRNA_metadata.tsv', sep='\t')
selected_column = 'cellType'
adata.obs['cell_type'] = labels[selected_column].values

# Editing the names of the cell types so they are their actual names
replacement_dict = {'oligo': 'Oligodendrocyte', 'astro': 'Astrocyte', 'mg': 'Microglia',
                    'endo': 'Endothelial cell', 'neuron': 'Neuron', 'OPC': 'OPC', 'doublet':'Doublet',
                    'unID':'unID'}

adata.obs['cell_type'] = adata.obs['cell_type'].replace(replacement_dict)

adata.obs


# In[4]:


mask = ~adata.obs['cell_type'].isin(['Doublet', 'unID'])

# Subset the AnnData object to keep only the rows where mask is True
adata = adata[mask].copy()
adata.obs


# In[5]:


def plotting_violins(data, output_path='_plots.pdf'):
    data.var['mt'] = data.var.index.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(data, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(data, keys=['total_counts', 'n_genes_by_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save=output_path, stripplot=False)
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
    return datas
adata = scrubbing(adata)


# In[7]:


def normalisation(data):
    sc.pp.normalize_total(data, target_sum=1e4) # Normalises every cell to 10,000 UMI (10,000 unique molecular identifiers)
    data.X.sum(axis=1)
    sc.pp.log1p(data)
    sc.pp.highly_variable_genes(data, n_top_genes= 2000)
    data.raw = data
    return data

# Makes the numbers more comparable
adata = normalisation(adata)


# In[8]:


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


# In[9]:


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


# In[10]:


sc.pp.neighbors(adata, n_pcs= 32) # Forms neighbourhood matrices.
sc.tl.leiden(adata) # Needed to add this in for some reason when i re ran it, but now it works, looks different to before.
sc.tl.paga(adata)
sc.pl.paga(adata, plot=False) # Can remove plot=False if we want to remove the coarse-grained graph
sc.tl.umap(adata, init_pos='paga')


# In[11]:


adata.obs['sample_ID'] = list(map(lambda x: "_".join(x.split("_")[1:]), adata.obs.index))
adata.obs['Disease_status'] = list(map(lambda x: x[0:2], adata.obs['sample_ID']))


# In[12]:


adata.obs


# In[13]:


cell_type_disease_counts = adata.obs.groupby(['Disease_status', 'cell_type']).size().unstack(fill_value=0)
cell_type_disease_counts


# In[14]:


# Keeping this for George but don't actually need it for stats test calculation

sums_by_disease_status = cell_type_disease_counts.sum(axis=1)

# Divide each cell count by the sum to get the proportion
proportions = cell_type_disease_counts.div(sums_by_disease_status, axis=0)

proportions


# In[15]:


chi2, p, dof, expected = chi2_contingency(cell_type_disease_counts)
print('Chi-Square Statistic:', chi2, '\nP-value:', p)


# In[16]:


custom_palette = {'AD': '#6FA0D6', 'Ct': '#FF8C99'}
cell_type_palette = {'Oligodendrocyte': '#d6cfd3', 'Astrocyte': '#7a378b',
                     'Microglia': '#9fb6cd', 'OPC': '#000080',
                     'Endothelial cell': '#cf6ba9', 'Neuron': '#ab82ff'}
fig, ax1 = plt.subplots(1, 2, figsize=(10, 4))  # Create subplots with two panels

# UMAP for Disease_status
sc.pl.umap(adata, color='Disease_status', palette=custom_palette, show=False, ax=ax1[0])

# UMAP for cell_type
sc.pl.umap(adata, color='cell_type', palette= cell_type_palette, show=False, ax=ax1[1])

# Adjust layout
plt.tight_layout()

# Save or show the plot
plt.savefig('umap_combined.pdf')
plt.show()


# In[17]:


sc.pp.neighbors(adata, n_neighbors= 10)
milo.make_nhoods(adata)

adata[adata.obs['nhood_ixs_refined'] != 0].obs[['nhood_ixs_refined', 'nhood_kth_distance']]


# Counting the number of each cell type
milo.count_nhoods(adata, sample_col='sample_ID')


# Comparing between the case and control,
milo.DA_nhoods(adata, design= "~ Disease_status")


# In[18]:


milopy.utils.build_nhood_graph(adata)

# Plotting milo results next to disease status and cell type so I can compare.
fig2, ax2 = plt.subplots(1, 3, figsize=(26, 6)) 
sc.pl.umap(adata, color='cell_type', show=False, ax= ax2[0])
sc.pl.umap(adata, color='Disease_status', palette=custom_palette, show=False, ax= ax2[1])

milopy.plot.plot_nhood_graph(adata, alpha=0.2, min_size=5, save=(True, '.pdf'), ax= ax2[2])



plt.tight_layout()

plt.savefig('Differential_abundance_with_umaps.pdf')
plt.show()


# In[19]:


# Annotating cell_types onto milo data.
milopy.utils.annotate_nhoods(adata, anno_col= 'cell_type')


# In[20]:


# Extracting logFC values and cell types from nhood adata
logFC = np.array(adata.uns['nhood_adata'].obs['logFC'])
nhood_annotation = np.array(adata.uns['nhood_adata'].obs['nhood_annotation'])

# Find the cell types in the nhood_adata from milo.
groups = np.unique(nhood_annotation)
group_positions = range(1, len(groups) + 1)

# Creating dictionary where keys are groups and values are logFC for that group
grouped_data = {group: logFC[nhood_annotation == group] for group in groups}

# Data needs to be in list of arrays format for violinplot
data = [grouped_data[group] for group in groups]

# Creating figure
fig, ax = plt.subplots(figsize=(10, 6))

violin_parts = ax.violinplot(data, positions=group_positions,vert=False, showmeans=False, showmedians=False, showextrema=False)



# Setting colours to be the same as our umap
colors = ['#7a378b', '#cf6ba9', '#9fb6cd', '#ab82ff', '#000080', '#d6cfd3'] 


for partname in ('cbars','cmins','cmaxes','cmeans','cmedians','cquantiles'):
    vp = violin_parts.get(partname)
    if vp is not None:
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
        
# If you want to set individual colors for each violin, you can do it like this:
for vp, color in zip(violin_parts['bodies'], colors):
    vp.set_facecolor(color)
    vp.set_edgecolor('black')
    vp.set_alpha(0.75)
    
for i, group in enumerate(groups):
    scatter_data = grouped_data[group]
    jitter = np.random.normal(0, 0.04, size=len(scatter_data))
    ax.scatter(scatter_data, np.repeat(group_positions[i], len(scatter_data)) + jitter, alpha=0.45, color='black', s=0.25)
        

# cell type as ticks for y axis
ax.set_yticks(range(1, len(groups) + 1))
ax.set_yticklabels(groups)

# Set the labels for the axes
ax.set_ylabel('Cell Type')
ax.set_xlabel('Log-Fold Change')

# setting limits for LogFC
ax.set_xlim(-12.5, 12.5)

# vertical dashed line to show LogFC 0 
ax.axvline(x=0, color='black', linestyle='--')

# Remove top and right spines so that its open
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.invert_yaxis()


# Save the figure
plt.savefig('Violin_of_celltype_abundance_matplotlib.pdf')

# Show the plot
plt.show()       


# In[21]:


# Differentiaal expression
adata_oligo = adata[adata.obs['cell_type'] == 'Oligodendrocyte']
sc.tl.rank_genes_groups(adata_oligo, 'Disease_status')
result_oligo = sc.get.rank_genes_groups_df(adata_oligo, group='AD')

adata_astro = adata[adata.obs['cell_type'] == 'Astrocyte']
sc.tl.rank_genes_groups(adata_astro, 'Disease_status')
result_astro = sc.get.rank_genes_groups_df(adata_astro, group='AD')

adata_oligo_ct = adata[adata.obs['cell_type'] == 'Oligodendrocyte']
sc.tl.rank_genes_groups(adata_oligo_ct, 'Disease_status')
result_oligo_ct = sc.get.rank_genes_groups_df(adata_oligo_ct, group='Ct')

adata_astro_ct = adata[adata.obs['cell_type'] == 'Astrocyte']
sc.tl.rank_genes_groups(adata_astro_ct, 'Disease_status')
result_astro_ct = sc.get.rank_genes_groups_df(adata_astro_ct, group='Ct')




# In[22]:


# Making volcano plots that have the gene name labelled if there is difference > |7|

LOG_FOLD_CHANGE = 1.5
FDR = 0.1

def volcano_plot(cell_type, title, filename, ax=None, Ylim=50):
    result = cell_type.copy()
    result['genes'] = adata.var_names
    
    result["-logQ"] = -np.log10(result["pvals"].astype("float"))  # Note: Use log10 for Q-values
    lowqval_de = result.loc[abs(result["logfoldchanges"]) > LOG_FOLD_CHANGE]
    other_de = result.loc[abs(result["logfoldchanges"]) <= LOG_FOLD_CHANGE]

    if ax is None:
        fig, ax = plt.subplots()

    # Color code based on 'Disease_status'
    scatter_ct = sns.scatterplot(
        x=other_de["logfoldchanges"],
        y=other_de["-logQ"],
        s=6, color='#FF8C99',  # Use red for 'Ct'
        ax=ax,
    )
    scatter_ad = sns.scatterplot(
        x=lowqval_de["logfoldchanges"],
        y=lowqval_de["-logQ"],
        s=6, color='#6FA0D6',  # Use blue for 'AD'
        ax=ax,
    )
    ax.set_xlabel("log2 FC")
    ax.set_ylabel("-log Q-value")
    
    if title is None:
        title = "custom DE"
    plt.ylim(0, Ylim)
    plt.title(title)
    
    # Label genes with logFC less than -7 or greater than 7
    for i, row in result.iterrows():
        if abs(row['logfoldchanges']) > 7:
            ax.annotate(row['genes'],  # Using gene column i added to result df
                        xy=(row['logfoldchanges'], row['-logQ']),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom',
                        fontsize=8)
    
    plt.savefig(filename)
    plt.show()


# In[23]:


volcano_plot(result_oligo, 'Oligodendrocytes DE', 'Oligodendrocyte_DE.pdf', Ylim=60,)
volcano_plot(result_astro, 'Astrocytes DE','Astrocyte_DE.pdf', Ylim=60)


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 16))
custom_xlim = (-12, 12)
custom_ylim = (-12, 12)
# plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)

slingshot = Slingshot(adata, celltype_key="cell_type", start_node= 4, obsm_key="X_umap", debug_level='verbose')
slingshot.fit(num_epochs=1, debug_axes=axes)



# In[ ]:


#Getting the psuedotime
pseudotime = slingshot.unified_pseudotime

# Adding the pseudotime that I just accessed to the adata dataframe, so that I can compare between CT and AD
adata.obs['pseudotime'] = pseudotime


# In[ ]:


print(dir(slingshot.curves[0]))


# In[ ]:


def trajectories(axis):
    for pc in slingshot.curves:
        curve_points = pc.points
        axis.plot(pc.points[:, 0], pc.points[:, 1], color='black', lw=0.75)  # Plot on the first subplot

# See if George can get the arrow to work, not sure where I'm going wrong, maybe need to do a forloop?
    if len(curve_points) > 1:
        start_point = curve_points[-2]
        end_point = curve_points[-1]
        
        arrow = FancyArrowPatch(posA= start_point, posB= end_point,
                               arrowstyle= '->', mutation_scale=50,
                               color='black', lw=1)
        axis.add_patch(arrow)


# In[ ]:


fig3, ax3 = plt.subplots(1, 2, figsize=(20, 6))

# Plot the UMAP
sc.pl.umap(adata, color='cell_type', show=False, ax=ax3[0])
trajectories(ax3[0])

ax3[1].set_title('Pseudotime')
sc.pl.umap(adata, color='pseudotime', show=False, ax=ax3[1])
trajectories(ax3[1])
    
plt.savefig('Slingshot_with_trajectories.pdf')


plt.show()


# In[ ]:


# Putting the pseudotime in order for each Disease_status
pseudotime_ct = np.sort(adata.obs.loc[adata.obs['Disease_status'] == 'Ct', 'pseudotime'])
pseudotime_ad = np.sort(adata.obs.loc[adata.obs['Disease_status'] == 'AD', 'pseudotime'])

# Calculating the percentile ranks for each Disease_status directly
percentile_ranks_ct = np.linspace(0, 100, len(pseudotime_ct))
percentile_ranks_ad = np.linspace(0, 100, len(pseudotime_ad))

# Plotting our percentile ranks against pseudotime for both AD and CT
plt.figure(figsize=(12, 8))
plt.plot(pseudotime_ct, percentile_ranks_ct, label='Ct', color='#FF8C99')
plt.plot(pseudotime_ad, percentile_ranks_ad, label='AD', color='#6FA0D6')
plt.xlabel('Pseudotime')
plt.ylabel('Cumulative frequency')
plt.title('Pseudotime Distribution Across Percentiles by Disease Status')
plt.legend(loc='upper left', fontsize='large')
plt.grid(True)
plt.savefig('percentage_through_differentiation.pdf')
plt.show()


# In[ ]:


gene_list = adata.var_names
glist = gene_list.str.strip().to_list()


# In[ ]:


res = gp.gsea(data=adata.to_df().T, # row -> genes, column-> samples
        gene_sets='MSigDB_Hallmark_2020',
        cls=adata.obs['Disease_status'],
        outdir=None,
        method='s2n', # signal_to_noise
        threads= 16)


# In[ ]:


res.res2d.head(10)


# In[ ]:


enr_bg = gp.enrichr(gene_list= glist,
                gene_sets= 'MSigDB_Hallmark_2020',
                organism='human',
                outdir=None)


# In[ ]:


enr_bg.results.head(15)


# In[ ]:


p_values = enr_bg.results['Adjusted P-value'].head(20)
terms = enr_bg.results['Term'].head(20)

# Create a horizontal bar plot
plt.figure(figsize=(10, 8))
plt.barh(terms, p_values, color='skyblue')
plt.xlabel('Adjusted P-Value')
plt.ylabel('Terms')
plt.title('All of the terms that have an adjusted P value < 0.05')
plt.gca().invert_yaxis()  # Invert y-axis to display terms from top to bottom
plt.savefig('terms_with_p_less_than0.05.pdf')
plt.show()


# In[ ]:


enr = gp.enrich(gene_list=glist,
                 gene_sets='MSigDB_Hallmark_2020',
                 background=None,
                 outdir=None,
                 verbose=True)


# In[ ]:


p_values = enr.results['P-value'].head(23)
terms = enr.results['Term'].head(23)

# Create a horizontal bar plot
plt.figure(figsize=(10, 8))
plt.barh(terms, p_values, color='skyblue')
plt.xlabel('P-Value')
plt.ylabel('Terms')
plt.title('All of the terms that have a P value < 0.05')
plt.gca().invert_yaxis()  # Invert y-axis to display terms from top to bottom
plt.savefig('terms_with_p_less_than0.05.pdf')
plt.show()


# In[ ]:


ax = dotplot(enr.results,
              column="Adjusted P-value",
              x='Gene_set', # set x axis, so you could do a multi-sample/library comparsion
              size=10,
              top_term=5,
              figsize=(3,5),
              title = "MSigDB_Hallmark_2020",
              xticklabels_rot=45, # rotate xtick labels
              show_ring=True, # set to False to revmove outer ring
              marker='o',
             )


# In[ ]:


ax = barplot(enr.results,
              column="Adjusted P-value",
              group='Gene_set', # set group, so you could do a multi-sample/library comparsion
              size=10,
              top_term=5,
              figsize=(3,5),
              #color=['darkred', 'darkblue'] # set colors for group
              color = {'MSigDB_Hallmark_2020':'skyblue'}
             )


# In[ ]:


ax = dotplot(enr.res2d, title='Dot plot of results',cmap='viridis_r', size=10, figsize=(3,5))
ax = barplot(enr.res2d,title='Not sure about this', figsize=(4, 5), color='darkred')


# In[ ]:


pre_res = gp.prerank(rnk=result_oligo, 
                     gene_sets= 'MSigDB_Hallmark_2020',
                     outdir='gsea_result',
                     min_size=5, max_size=500)
pre_res.res2d.head(5)


# In[ ]:


ax = dotplot(pre_res.res2d,
             column="FDR q-val",
             title='MSigDB_Hallmark_2020',
             cmap='viridis_r',
             size=6, # adjust dot size
             figsize=(4,5), cutoff=0.25, show_ring=False)


# In[ ]:





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

