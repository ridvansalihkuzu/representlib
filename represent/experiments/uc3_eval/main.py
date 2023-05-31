import os
import pandas as pd
from geojson import Point, Feature, FeatureCollection, dump
import torch

if __name__ == '__main__':

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    input_file = './data/database_test.csv'
    # output directory
    output_dir = './output/'
    os.makedirs(output_dir, exist_ok=True)

    # output csv file
    csv_file = os.path.join(output_dir, 'out_%s.csv' % (input_file.split('/')[-1].split('.')[0]))
    # output geojson file
    geojson_file = os.path.join(output_dir, 'out_%s.geojson' % (input_file.split('/')[-1].split('.')[0]))

    from represent.models.uc3_gnn import GNN

    # GNN model
    model = GNN(pretrained=True).to(device)
    model.eval()

    from represent.datamodules.uc3_gnn_datamodule import BuildDataLoader

    # data loader
    loader = BuildDataLoader(input_file).get_data_loader()

    # dataframe
    df = []
    # geojson features list
    fea = []

    for x, fp_id in loader:
        # score value per building
        s = model.score(x.to(device)).cpu().numpy()
        # building centroids
        xyz = x.centroid.cpu().numpy().T

        df += [pd.DataFrame(data={'score': s, 'easting': xyz[0], 'northing': xyz[1], 'height': xyz[2]},
                            index=fp_id)]

        fea += [
            Feature(geometry=Point((float('{:.3f}'.format(xi[0])), float('{:.3f}'.format(xi[1])))),
                    properties={
                        'height': float('{:.2f}'.format(xi[2])),
                        'score': float('{:.1f}'.format(si)),
                        'footprint_id': int('{:}'.format(i))}
                    )
            for xi, si, i in zip(xyz.T, s, fp_id)]

    # write csv file
    df = pd.concat(df, axis=0)
    df.index.name = 'footprint_id'
    formats = {'easting': '{:.3f}', 'northing': '{:.3f}', 'score': '{:.1f}', 'height': '{:.2f}'}
    for col, f in formats.items():
        df[col] = df[col].map(lambda x: f.format(x))
    df.to_csv(csv_file)

    # crs
    crs = {
        'type': 'name',
        'properties': {
            'name': 'EPSG:3035'
        }
    }
    # write geojson file
    feature_collection = FeatureCollection(fea, crs=crs)
    with open(geojson_file, 'w') as f:
        dump(feature_collection, f)
