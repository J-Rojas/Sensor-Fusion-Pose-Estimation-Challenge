import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_filtered_pr_curve(outdir, data, maxgap=0.03):
    df = pd.DataFrame(data=data)
    df['recall_grp'] = (df['recall']/maxgap).astype(int)*maxgap
    aggregation = {'recall': {
                             'mean_recall': 'mean'
                             },
                   'precision': {'max_precision': lambda x: max(x),
                                 'min_precision': lambda x: min(x),
                                 'mean_precision': 'mean'
                                 }}
    grouped_recall_with_precision = df.groupby('recall_grp').agg(aggregation)
    plt.clf()
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'max_precision')], lw=2, color='teal', label='Max Precision')
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'mean_precision')], lw=2, color='cornflowerblue', label='Mean Precision')
    plt.plot(grouped_recall_with_precision[('recall', 'mean_recall')],
             grouped_recall_with_precision[('precision', 'min_precision')], lw=2, color='turquoise', label='Min Precision')

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Filtered Precision-Recall Curve, filterGap={0}'.format(maxgap))
    plt.legend(loc="upper right")
    fname = '{0}/precision_recall_curve_filtered.png'.format(outdir)
    plt.savefig(fname)
    print 'Filtered Precision-Recall Curve saved at ' + fname


def plot_pr_curve(losses, precision_arr, recall_arr, outdir):
    data = np.array(zip(precision_arr, recall_arr,losses), dtype=[('precision', float), ('recall', float), ('loss', float)])
    data_filename = 'precision_recall_value.csv'
    np.savetxt('{0}/{1}'.format(outdir,data_filename), data, delimiter=",")
    print 'Plotting Precision-Recall Curve...'
    plt.clf()
    plt.plot(data['recall'], data['precision'], lw=2, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")

    fname = '{0}/precision_recall_curve.png'.format(outdir)
    plt.savefig(fname)
    print 'Precision-Recall Curve saved at '+ fname
    #plt.show()

    plot_filtered_pr_curve(outdir, data)