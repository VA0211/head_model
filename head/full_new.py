import sklearn.manifold
import sklearn.impute
import sklearn.neighbors
import sklearn.ensemble
import sklearn.linear_model
import sklearn.svm
import xgboost
import os
import argparse
import project
import logger
import features.knn
import models.pytorch_wrapper as ptw  # <--- Our new PyTorch adapter

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='../micro_rna/20160809_contestant/')
parser.add_argument('--working_dir', default=project.DEFAULT_WORKING_DIRECTORY_PATH)
parser.add_argument('--seeds', default=str(project.DEFAULT_SEED))
parser.add_argument('--n_folds', type=int, default=project.DEFAULT_N_FOLDS)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

def main():
    logger.setup_logger()
    seeds = map(int, args.seeds.split(','))
    
    # Force t-SNE to be faster using all cores
    # (Note: standard sklearn TSNE is slow, but n_jobs helps if you have modern sklearn)
    tsne_estimator = sklearn.manifold.TSNE(n_components=3, n_jobs=-1)

    for seed in seeds:
        pl = project.pipeline(
            args.working_dir, seed, args.n_folds,
            os.path.join(args.input_dir, 'train', 'feature_vectors.csv'),
            os.path.join(args.input_dir, 'test', 'feature_vectors.csv'),
            os.path.join(args.input_dir, 'train', 'labels.txt'),
            os.path.join(args.input_dir, 'label_names.txt')
        )

        # --- LEVEL 0 (Preprocessing) ---
        pl.transform('imputed', ['raw'], True, sklearn.impute.SimpleImputer(strategy='median'))
        pl.transform('normalized', ['imputed'], True, sklearn.preprocessing.Normalizer())
        
        # We enable t-SNE now!
        print("Running t-SNE (this might take 5-10 mins)...")
        pl.transform('tsne', ['imputed'], True, tsne_estimator)

        # --- LEVEL 1 (Base Models) ---
        # CPU Models (Accelerated)
        pl.predict_proba('lev1_random-forest', ['imputed'], 2, sklearn.ensemble.RandomForestClassifier(n_jobs=-1))
        pl.predict_proba('lev1_logistic-regression', ['imputed'], 2, sklearn.linear_model.LogisticRegression(n_jobs=-1))
        pl.predict_proba('lev1_extra-tree', ['imputed'], 2, sklearn.ensemble.ExtraTreesClassifier(n_jobs=-1))
        pl.decision_function('lev1_linear-svc', ['imputed'], 2, sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=50), version=1)

        # KNN (Accelerated)
        KS = [2, 4, 8, 16, 32, 64, 128, 256]
        for k in KS:
            pl.predict_proba(f'lev1_knn_k={k}', ['imputed'], 2, sklearn.neighbors.KNeighborsClassifier(n_neighbors=k, n_jobs=-1))
        
        pl.transform('lev1_knn_distances', ['imputed'], 2, features.knn.KNNDistanceFeature(ks=[1, 2, 4]))
        pl.transform('lev1_knn_distances_tsne', ['tsne'], 2, features.knn.KNNDistanceFeature(ks=[1]))

        # XGBoost (GPU Accelerated)
        pl.predict_proba('lev1_xgboost', ['raw', 'tsne'], 2, 
            xgboost.XGBClassifier(objective='multi:softmax', learning_rate=0.05, max_depth=5, 
                                n_estimators=1000, nthread=10, subsample=0.5, colsample_bytree=1.0, 
                                device='cuda'))

        # Neural Networks (PyTorch Replacement - GPU Accelerated)
        # We replace models.chainer.MLP3 with ptw.MLP3
        pl.predict_proba('lev1_mlp3', ['imputed', 'tsne'], 2,
            ptw.PyTorchClassifier(ptw.MLP3, gpu=0, n_epoch=100, n_out=len(pl.label_names)))
            
        pl.predict_proba('lev1_mlp4', ['imputed', 'tsne'], 2,
            ptw.PyTorchClassifier(ptw.MLP4, gpu=0, n_epoch=200, n_out=len(pl.label_names)))

        # --- LEVEL 2 (Stacking) ---
        LEVEL1_PREDICTIONS = [
            'lev1_random-forest', 'lev1_logistic-regression', 'lev1_extra-tree',
            'lev1_linear-svc', 'lev1_xgboost', 'lev1_mlp3', 'lev1_mlp4'
        ] + [f'lev1_knn_k={k}' for k in KS]
        
        LEVEL1_FEATURES = ['tsne', 'lev1_knn_distances', 'lev1_knn_distances_tsne']

        pl.predict_proba('lev2_logistic-regression', LEVEL1_PREDICTIONS, 1, 
            sklearn.linear_model.LogisticRegression(n_jobs=-1), version=1)

        pl.predict_proba('lev2_xgboost2', LEVEL1_PREDICTIONS + LEVEL1_FEATURES, 1,
            xgboost.XGBClassifier(objective='multi:softmax', learning_rate=0.1, max_depth=5, 
                                n_estimators=1000, nthread=10, subsample=0.9, colsample_bytree=0.7, 
                                device='cuda'), version=1)
        
        pl.predict_proba('lev2_mlp4', ['imputed'] + LEVEL1_PREDICTIONS + LEVEL1_FEATURES, 1,
            ptw.PyTorchClassifier(ptw.MLP4, gpu=0, n_epoch=200, n_out=len(pl.label_names)), version=1)

        pl.predict('lev2_linear-svc', ['imputed'], 1, 
            sklearn.svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=50))

if __name__ == '__main__':
    main()