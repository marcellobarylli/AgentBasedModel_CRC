import urllib.request
from prismx import prismx as px
import scanpy as sc
import pickle
import h5py
import os

# urllib.request.urlretrieve("https://mssm-seq-matrix.s3.amazonaws.com/mouse_matrix.h5", "mouse_matrix.h5")

def load_adata_file(cell_type_to_look_at, tissue_class, scale):
    adata = sc.read('../data/for_pyscan_{}_{}_{}.h5ad'.format(tissue_class, cell_type_to_look_at, scale))
    adata.obs.cell_type = adata.obs.subtype.astype('category')
    adata.var_names = adata.var.values.flatten()

    return adata

if __name__ == '__main__':

    correlationFolder = "correlatio1n_folder"
    clusterNumber = 200

    print(help(px))
    cell_type_to_look_at = 'T cells'
    regulated_pathway = 'TNF-alpha Signaling via NF-kB'
    pathway_cms = 2
    pathway_dir = 'neg'
    tissue_class = 'Tumor'
    scale = 'raw'
    #
    # print('starting to load mouse data')
    # mouse_data = sc.read('mouse_matrix.h5')
    # print('this is a description of the mouse_data: \n', mouse_data)

    print('load anndata file')
    loaded_file = load_adata_file(cell_type_to_look_at, tissue_class, scale)
    # print(loaded_file)
    # # loaded_file.rename_categories('X', 'data/expression')
    # # loaded_file.write('expression.hdf5')
    # #
    # print('create dataset')
    # f = h5py.File('mydataset6.hdf5', 'a')
    # #
    # # # print(map(int, loaded_file.var.values[:, 0]))
    #
    # gene_list = [string.encode('utf8') for string in loaded_file.var.values[:, 0]]
    # # gene_list = [i for i in range(len(loaded_file.var.values[:, 0]))]
    # # print(gene_list)
    # #
    # # print(len(loaded_file.var.values[:, 0]))
    # # print(loaded_file.X)
    # #
    # print('add data to dataset')
    # f.create_dataset('data/expression', data=loaded_file.X)
    # f.create_dataset('meta/genes', data=gene_list)
    # f.create_dataset('meta/Sample_geo_accession', data=['testtesttest'.encode('utf8') for i in range(len(
    #     loaded_file.var.values[0, :]))])
    #
    # # print('çlose file')
    # # f.close()
    #
    #
    # print('going to start now to process everything: ')
    # os.mkdir(correlationFolder)
    # matrix = px.createCorrelationMatrices('mydataset6.hdf5',
    #                          correlationFolder, clusterCount=clusterNumber, sampleCount=5000, verbose=True)


    predictionFolder = "prediction_folder"
    libs = px.listLibraries()
    i = 1

    # select GO: Biological Processes
    gmtFile = px.loadLibrary(libs[110])

    # px.correlationScores(gmtFile, correlationFolder, predictionFolder, verbose=True)

    # model = px.trainModel(predictionFolder, correlationFolder, gmtFile, trainingSize=300000, testTrainSplit=0.1,
    #                       samplePositive=40000, sampleNegative=200000, randomState=42, verbose=True)
    # pickle.dump(model, open("gobp_model.pkl", 'wb'))

    outfolder = "prismxresult"

    outname = libs[i]
    px.predictGMT("gobp_model.pkl", gmtFile, correlationFolder, predictionFolder, outfolder, outname, stepSize=200,
                  intersect=False, verbose=True)
