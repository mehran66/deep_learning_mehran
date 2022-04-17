import tqdm as tq
from sklearn.metrics import fbeta_score

#https://www.kaggle.com/code/nhaphan0411/0-90-mobilenetv2-optimized-with-tfrecord
def perf_grid(y_hat_val, y_val, label_names, n_thresh=100):
    # Find label frequencies in the validation set
    label_freq = y_val.sum(axis=0)

    # Define thresholds
    thresholds = np.linspace(0, 1, n_thresh + 1).astype(np.float32)

    # Compute all metrics for all labels
    ids, labels, freqs, tps, fps, fns, precisions, recalls, f1s, f2s = [], [], [], [], [], [], [], [], [], []

    for i in tq.tqdm(range(len(label_names))):
        for thresh in thresholds:
            ids.append(i)
            labels.append(label_names[i])
            freqs.append(round(label_freq[i] / len(y_val), 2))

            y = y_val[:, i]
            y_pred = y_hat_val[:, i] > thresh

            tp = np.count_nonzero(y_pred * y)
            fp = np.count_nonzero(y_pred * (1 - y))
            fn = np.count_nonzero((1 - y_pred) * y)
            precision = tp / (tp + fp + 1e-16)
            recall = tp / (tp + fn + 1e-16)
            f1 = 2 * tp / (2 * tp + fn + fp + 1e-16)
            f2 = fbeta_score(y, y_pred, average='weighted', beta=2)

            tps.append(tp)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            f2s.append(f2)

    # Create the performance dataframe
    grid = pd.DataFrame({'id': ids,
                         'label': labels,
                         'freq': freqs,
                         'threshold': list(thresholds) * len(label_names),
                         'tp': tps,
                         'fp': fps,
                         'fn': fns,
                         'precision': precisions,
                         'recall': recalls,
                         'f1': f1s,
                         'f2': f2s})

    return grid

grid = perf_grid(y_pred, np.array(y_test), np.array(list(label_to_class.keys())))
grid[grid['label'].str.contains('primary')].head(20)
# Choose the best threshold of
grid_max = grid.loc[grid.groupby(['id', 'label'])[['f2']].idxmax()['f2'].values]
print(grid_max)