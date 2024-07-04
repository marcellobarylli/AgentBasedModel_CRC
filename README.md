# Colorectal cancer as an evolving ecosystem

This repo contains the code for modelling colorectal cancer as an evolving ecosystem in terms of an agent-based model.


A summary of the files and their purposes is given belwo:
- display_pathway_results.py: displays results of pathway_ranking and pathway_analysis in a nicely structured table
- analyse_pathway_results.py: summarise the results of the pathway analysis into lists over CMS or cell subtype overlap
- create_heatmap.py: creates heatmaps of the genes in the bulk dataset by the CRC consortium
- eda_gene_expression.py: data exploration of the CRC bulk dataset focussing on grouping and 
- explore_single_cell_data.py: create cut off graph from the differentially expressed genes, create sampled and average gene expression from the raw expression sets, save the the sampled gene expression data into a format to be used by scanpy, save differentially expressed genes to files to be used for pahtway detection, create normal vs. tumour heatmaps of selected genes
- extract_initial_pathway_activation.py: rewrites the gene expression values from the bulk dataset to values that can be used to start the agent-based model simulation with. Genes to regulate are gathered from the single cell analysis data.
- find_pathways_in_list.py: go from differentially expressed genes to the lists of pathways they match up with.
- light_gbm_sensitivity_analysis.py: map gene ids, do synthetic experiments to see how the samples with adjusted gene expressions are classified.
