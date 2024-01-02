import argparse
import os
import pandas as pd

from check_status import load_results
from plotter import heatmap_plot
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from loaders import train_test_val_loader
from sklearn.metrics import confusion_matrix


def main(args):
    in_dir = args.input

    results = []
    results_fn = in_dir + '/results' #os.path.join(in_dir, 'results')
    results = load_results(results_fn, args.measures)
    print('Read {} which contained {} results.'.format(
        results_fn, len(results)))

    df = pd.DataFrame.from_records(results)

    if args.output:
        df.to_csv(args.output)
        print('Wrote results to', args.output)

    best_per_testset = {}
    result_names = df.result_name.unique()
    for testset in sorted(result_names):
        idx = df['result_name'] == testset
        dft = df[idx]
        print()
        print('# {}, {} results'.format(testset, len(dft)))

        print('Best {} results so far according to {} {}.'.format(
            args.N, args.opt, args.meas))
        if args.opt == 'max':
            dfs = dft.nlargest(args.N, args.meas)
        else:
            dfs = dft.nsmallest(args.N, args.meas)
        best = dfs.iloc[0][args.meas]
        best_per_testset[testset] = best

        print(dfs.drop(columns='result_name'))
    if args.heatmap:
        m = args.heatmap.split(',')
        heatmap_data = df.pivot_table(index=m[0], columns=m[1], values=m[2], aggfunc='mean')

        # Create a heatmap using Seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.5f', linewidths=0.5)

        # Customize the labels and title
        plt.xlabel('Layer Size')
        plt.ylabel('Batch Size')
        plt.title('Hyperparameter Tuning Heatmap')

        # Show the heatmap
        plt.show()
    if args.confusion:
        if args.confusion == 0:
            # this should select the best model
            pass
        else:
            model_num = args.confusion
            model_path = in_dir + f'/{model_num}'
            model = tf.saved_model.load(model_path)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            test_filepath = dir_path + "/NM_simulations/masked_data3/test"
            test_data = train_test_val_loader(test_filepath)
            y_pred = model.predict(test_data)
            y_pred_labels = np.argmax(y_pred, axis=1)
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str,
                        help='directory with the results file')
    parser.add_argument('--output', '-O', type=str)
    parser.add_argument('--measure', type=str, default='P@5', required=False,
                        help='measure to optimize', dest='meas')
    parser.add_argument('-N', type=int, default=5,
                        help='number of top results to show')
    parser.add_argument('--opt', type=str,
                        choices=['max', 'min'], default='max', required=False,
                        help='whether to minimize or maximize the measure')
    parser.add_argument('--measures', type=str,
                        help='names of measures if missing from results, '
                        'e.g., --measures=P@1,P@3,P@5')
    parser.add_argument('--heatmap', type=str, 
                        help='make a heatmap from param1,param2,result')
    parser.add_argument('--confusion', type=int,
                        help='make a confusion matrix for the selected model number')
    args = parser.parse_args()

    main(args)
