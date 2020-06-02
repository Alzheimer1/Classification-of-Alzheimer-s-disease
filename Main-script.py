import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# load the dataset
feature1 = pd.read_csv('................')
feature2 = pd.read_csv('................')
feature3 = pd.read_csv('................')
feature4 = pd.read_csv('................')
feature5 = pd.read_csv('................')
feature6 = pd.read_csv('................')

X1 = feature1.iloc[:, 1:].values
y1 = feature1.iloc[:, 0].values
X2 = feature2.iloc[:, 1:].values
y2 = feature2.iloc[:, 0].values
X3 = feature3.iloc[:, 1:].values
y3 = feature3.iloc[:, 0].values
X4 = feature4.iloc[:, 1:].values
y4 = feature4.iloc[:, 0].values
X5 = feature5.iloc[:, 1:].values
y5 = feature5.iloc[:, 0].values
X6 = feature6.iloc[:, 1:].values
y6 = feature6.iloc[:, 0].values

# Preprocess data
from MKLpy.preprocessing import normalization, rescale_01

X1 = rescale_01(X1)  # feature scaling in [0,1]
X1 = normalization(X1)  # ||X_i||_2^2 = 1
X2 = rescale_01(X2)
X2 = normalization(X2)
X3 = rescale_01(X3)
X3 = normalization(X3)
X4 = rescale_01(X4)
X4 = normalization(X4)
X5 = rescale_01(X5)
X5 = normalization(X5)
X6 = rescale_01(X6)
X6 = normalization(X6)

# # train/test
X_tr_A, X_te_A, y_tr_A, y_te_A = train_test_split(X1, y1, test_size=0.3, random_state=42)
X_tr_F, X_te_F, y_tr_F, y_te_F = train_test_split(X2, y2, test_size=0.3, random_state=42)
X_tr_AV, X_te_AV, y_tr_AV, y_te_AV = train_test_split(X3, y3, test_size=0.3, random_state=42)
X_tr_FM, X_te_FM, y_tr_FM, y_te_FM = train_test_split(X4, y4, test_size=0.3, random_state=42)
X_tr_S, X_te_S, y_tr_S, y_te_S = train_test_split(X5, y5, test_size=0.3, random_state=42)
X_tr_D, X_te_D, y_tr_D, y_te_D = train_test_split(X6, y6, test_size=0.3, random_state=42)

# Applying Polynomial kernel
from MKLpy.metrics import pairwise

k1 = [pairwise.homogeneous_polynomial_kernel(X_tr_A, degree=d) for d in range(5)]
k11 = [pairwise.homogeneous_polynomial_kernel(X_te_A, X_tr_A, degree=d) for d in range(5)]
k2 = [pairwise.homogeneous_polynomial_kernel(X_tr_F, degree=d) for d in range(5)]
k22 = [pairwise.homogeneous_polynomial_kernel(X_te_F, X_tr_F, degree=d) for d in range(5)]
k3 = [pairwise.homogeneous_polynomial_kernel(X_tr_AV, degree=d) for d in range(5)]
k33 = [pairwise.homogeneous_polynomial_kernel(X_te_AV, X_tr_AV, degree=d) for d in range(5)]
k4 = [pairwise.homogeneous_polynomial_kernel(X_tr_FM, degree=d) for d in range(5)]
k44 = [pairwise.homogeneous_polynomial_kernel(X_te_FM, X_tr_FM, degree=d) for d in range(5)]
k5 = [pairwise.homogeneous_polynomial_kernel(X_tr_S, degree=d) for d in range(5)]
k55 = [pairwise.homogeneous_polynomial_kernel(X_te_S, X_tr_S, degree=d) for d in range(5)]
k6 = [pairwise.homogeneous_polynomial_kernel(X_tr_D, degree=d) for d in range(5)]
k66 = [pairwise.homogeneous_polynomial_kernel(X_te_D, X_tr_D, degree=d) for d in range(5)]
##################################################################################################
K7 = (k1 + k2 + k3 + k4 + k5 + k6)
k8 = (k11 + k22 + k33 + k44 + k55 + k66)

from MKLpy.algorithms import EasyMKL
from MKLpy.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import svm
from itertools import product

# base_learner1 = SVC(C=3, gamma=0.001)
param_grid = {
    'gamma': np.logspace(-3, 0, 7),
    'C': range(1, 10),
    'kernel': ['poly', 'rbf', 'linear', 'sigmoid']
}
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
cv = LeaveOneOut()

# print(base_learner)
###########################################################################################
best_results = {}

for lam in [0, 0.0001, 0.0009, 0.001, 0.009, 0.01, 0.09, 0.1, 0.2, 0.9, 1]:
    base_learner = GridSearchCV(svm.SVC(probability=True), param_grid=param_grid, cv=cv, refit='AUC',
                                error_score=0, pre_dispatch='1*n_jobs', n_jobs=1)
    scores = cross_val_score(k7, y_tr_A, EasyMKL(learner=base_learner, lam=lam), n_folds=5, scoring='accuracy')
    print(lam, scores)
    acc = np.mean(scores)
    if not best_results or best_results['score'] < acc:
        best_results = {'lam': lam, 'score': acc}

# EasyMKL-BASED
#############################################################################################
clf = EasyMKL(learner=base_learner, lam=best_results['lam']).fit(k7, y_tr_A)
print(clf)
#############################################################################################
# evaluate the solution
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred = clf.predict(k8)  # predictions
y_score = clf.decision_function(k8)  # rank
accuracy = accuracy_score(y_te_A, y_pred)
roc_auc = roc_auc_score(y_te_A, y_score)
print('Accuracy score: %.3f, roc_AUC score: %.3f' % (accuracy, roc_auc))
print('accuracy on the test set: %.3f, with lambda=%.2f' % (accuracy, best_results['lam']))
###############################################################################################
# # Calling confusion matrix and plotting classification report
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score, roc_curve, auc, roc_auc_score

print(classification_report(y_te_A, y_pred))
cm = cnf_matrix = confusion_matrix(y_te_A, y_pred)
print(cm)
total = sum(sum(cnf_matrix))
ACC = (cnf_matrix[0, 0] + cnf_matrix[1, 1]) / total
sensitivity = Recall = cnf_matrix[0, 0] / (cnf_matrix[0, 0] + cnf_matrix[1, 0])
specificity = cnf_matrix[1, 1] / (cnf_matrix[1, 1] + cnf_matrix[0, 1])
precision = cnf_matrix[0, 0] / (cnf_matrix[0, 0] + cnf_matrix[0, 1])
f1score = 2 * ((precision * Recall) / (precision + Recall))
##################################################################################################
fpr1, tpr1, thresholds1 = roc_curve(y_te_A, y_score)
print(fpr1)
print(tpr1)
print(thresholds1)
roc_auc1 = auc(fpr1, tpr1)
print('Roc_auc1: %.4f' % roc_auc1)
#################################################################################################
# # calculate AUC
AUC = roc_auc_score(y_te_A, y_score)
# calculate AUC
print('AUC1: %.4f' % AUC)
print('Accuracy1: %.4f' % ACC)
print('Sensitivity1: %.4f' % sensitivity)
print('Specificity1: %.4f' % specificity)
print('Precision1: %.4f' % precision)
print('F1score1: %.4f' % f1score)
# # print(Recall)
Cohen = cohen_kappa_score(y_te_A, y_pred)
print('Cohen1: %.4f' % Cohen)
###################################################################################################
####################################################################################
lw = 2
plt.figure()
plt.plot(fpr1, tpr1, 'o-', ms=2, label='Combined-ROI(AUC = %0.4f)' % roc_auc1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.yticks(np.arange(0, 1.05, step=0.1))
plt.xlabel('1-Specificity (False Positive Rate)', fontsize='large', fontweight='bold')
plt.ylabel('Sensitivity (True Positive Rate)', fontsize='large', fontweight='bold')
plt.title('ROC curve for AD vs. HC', fontsize='large', fontweight='bold')
plt.legend(loc="lower right")
plt.show()
# plt.savefig('roc_auc.png')
plt.close()
#####################################################################################
