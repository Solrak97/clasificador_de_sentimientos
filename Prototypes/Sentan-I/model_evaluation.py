import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn.metrics

# Graficación de matríz de confusión


def confussion_matrix(y_true, y_pred, classes):
    cf_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i[0] for i in classes],
                         columns=[i[0] for i in classes])

    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True)

    plt.show()
