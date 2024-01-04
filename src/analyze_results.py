import argparse
import os
import pandas as pd
import numpy as np
from check_status import load_results
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from loaders import train_test_val_loader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kurtosis, skew, norm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfReader, PdfWriter
import io

# Set the global theme
sns.set_theme(style="whitegrid")


def fig_to_pdf_page(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='pdf')
    buf.seek(0)
    return PdfReader(buf).pages[0]


def main(args):
    in_dir = args.input
    if not args.err_analysis:
        results = []
        results_fn = in_dir + '/results' #os.path.join(in_dir, 'results')
        results = load_results(results_fn, args.measures)
        if not args.param_id:
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
            if not args.param_id:
                print('# {}, {} results'.format(testset, len(dft)))

                print('Best {} results so far according to {} {}.'.format(
                    args.N, args.opt, args.meas))
            if args.opt == 'max':
                dfs = dft.nlargest(args.N, args.meas)
            else:
                dfs = dft.nsmallest(args.N, args.meas)
            best = dfs.iloc[0][args.meas]
            best_per_testset[testset] = best
            if not args.param_id:
                print(dfs.drop(columns='result_name'))
            if args.param_id:
                print(" ".join(map(str, dfs.get('param_id').tolist())))
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
    elif args.err_analysis:
        if args.err_analysis == 0:
            # this should select the best model
            pass
        else:
            if args.pdf_filename:
                pdf_writer = PdfWriter()
                pdf_filename = in_dir + f'/{args.pdf_filename}'
                # Check if the PDF file already exists
                if os.path.exists(pdf_filename):
                    # Open the existing PDF and add its pages to the writer
                    with open(pdf_filename, "rb") as existing_pdf:
                        pdf_reader = PdfReader(existing_pdf)
                        for page in range(len(pdf_reader.pages)):
                            pdf_writer.add_page(pdf_reader.pages[page])
                model_num = args.err_analysis
                model_path = in_dir + f'/{model_num}'
                model = tf.keras.models.load_model(model_path)

                dir_path = os.path.dirname(os.path.realpath(__file__))
                test_filepath = dir_path + "/NM_simulations/masked_data3/test"
                test_data = train_test_val_loader(test_filepath)
                x_test = test_data[:-1, :].T
                y_test = test_data[-1, :]
                y_test = y_test.flatten()
                y_pred = model.predict(x_test)
                y_pred = y_pred.flatten()
                residuals = y_test - y_pred
                df = pd.DataFrame(data={'y_pred': y_pred.tolist(), 'y_test': y_test.tolist()})

                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r_squared = r2_score(y_test, y_pred)

                # Scatter plot with regression line
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.regplot(x='y_pred', y='y_test', data=df, ci=95,
                            scatter_kws={"s": 80, "alpha": 0.7, "edgecolor": 'k', "linewidths": 0.5},
                            line_kws={"color": "red"})
                plt.xlabel('True Values', fontsize=12)
                plt.ylabel('Predicted Values', fontsize=12)
                plt.title('True vs. Predicted Values', fontsize=14)

                # Inset for metrics
                axins = inset_axes(plt.gca(), width="40%", height="30%", loc='lower right')
                axins.text(0.05, 0.7, f'MAE = {mae:.3f}', fontsize=10)
                axins.text(0.05, 0.5, f'RMSE = {rmse:.3f}', fontsize=10)
                axins.text(0.05, 0.3, f'$R^2 = {r_squared:.3f}$', fontsize=10)
                axins.axis('off')
                plt.tight_layout()
                pdf_writer.add_page(fig_to_pdf_page(fig))
                plt.close(fig)

                # Residual plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=y_pred, y=residuals, alpha=0.7,
                                edgecolor='k', linewidth=0.5, s=80)
                plt.xlabel('Predicted Values', fontsize=12)
                plt.ylabel('Residuals', fontsize=12)
                plt.title('Residual Plot', fontsize=14)
                pdf_writer.add_page(fig_to_pdf_page(fig))
                plt.close(fig)

                # Histogram of residuals with fitted normal distribution
                residual_kurtosis, residual_skewness = kurtosis(residuals), skew(residuals)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(residuals, bins=30, kde=True, color='skyblue',
                             edgecolor='k', linewidth=0.5)
                plt.xlabel('Residuals', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title('Histogram of Residuals', fontsize=14)

                # Fit a normal distribution to residuals
                mu, std = norm.fit(residuals)
                x = np.linspace(min(residuals), max(residuals), 100)
                pdf = norm.pdf(x, mu, std)

                # Scale the PDF to match the scale of the histogram
                bin_width = (max(residuals) - min(residuals)) / 30
                scaled_pdf = pdf * len(residuals) * bin_width
                plt.plot(x, scaled_pdf, 'r-', lw=2)
                plt.legend(['Fitted Normal Distribution'])

                # Calculate FWHM
                fwhm = 2 * np.sqrt(2 * np.log(2)) * std

                # Inset for kurtosis, skewness, center, and FWHM, with grid and axes removed
                axins_hist = inset_axes(plt.gca(), width="40%", height="40%", loc='upper right')
                axins_hist.text(0.3, 0.7, f'Kurtosis = {residual_kurtosis:.2f}', fontsize=10)
                axins_hist.text(0.3, 0.5, f'Skewness = {residual_skewness:.2f}', fontsize=10)
                axins_hist.text(0.3, 0.3, f'Center = {mu:.2f}', fontsize=10)
                axins_hist.text(0.3, 0.1, f'FWHM = {fwhm:.2f}', fontsize=10)
                axins_hist.axis('off')  # This turns off the axis labels and grid
                pdf_writer.add_page(fig_to_pdf_page(fig))
                plt.close(fig)

                with open(pdf_filename, "wb") as out:
                    pdf_writer.write(out)

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
    parser.add_argument('--err_analysis', type=int,
                        help='make a plots based on the selected model')
    parser.add_argument('--pdf_filename', type=str,
                        help='indicate the filename you would like to give the pdf')
    parser.add_argument('--param_id', action='store_true',
                        help='print the top -N param_ids if this flag is set')
    args = parser.parse_args()

    main(args)
