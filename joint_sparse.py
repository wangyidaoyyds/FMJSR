import numpy as np
import time
import pickle
from tqdm import tqdm
import itertools
import copy
import scipy.io as scio


def get_matrix_vectorb(path_1, path_2):
    data_1 = scio.loadmat(path_1)
    data_2 = scio.loadmat(path_2)
    GalleryFeature_1 = np.array(data_1['train_feature'], dtype=np.float64)
    GalleryFeature_2 = np.array(data_2['train_feature'], dtype=np.float64)
    GalleryLabel_temp = data_1['all_dict'][0:553]
    TestLabel_1_temp = data_1['all_dict'][553:]
    TestLabel_2_temp = data_2['all_dict'][553:]
    TestLabel_1 = []
    TestLabel_2 = []
    GalleryLabel = []
    for i, value in enumerate(TestLabel_1_temp):
        TestLabel_1.append(value.split('/')[0])
    for i, value in enumerate(TestLabel_2_temp):
        TestLabel_2.append(value.split('/')[0])
    for i, value in enumerate(GalleryLabel_temp):
        GalleryLabel.append(value.split('/')[0])
    GalleryLabel = np.array(GalleryLabel)
    GalleryLabelSet = []
    for i in GalleryLabel:
        if i not in GalleryLabelSet:
            GalleryLabelSet.append(i)
    GalleryLabelSet = list(map(int, GalleryLabelSet))
    TEMP_TestFeature_1 = np.array(data_1['test_feature'], dtype=np.float64)
    TEMP_TestFeature_2 = np.array(data_2['test_feature'], dtype=np.float64)
    GalleryFeature_1 = GalleryFeature_1.transpose()
    GalleryFeature_2 = GalleryFeature_2.transpose()
    matrixA = {}
    matrixA[0] = GalleryFeature_1
    matrixA[1] = GalleryFeature_2
    TestLabel_1, TestLabel_2, TestFeature_1, TestFeature_2 = TransFormFeature(TestLabel_1, TestLabel_2,
                                                                              TEMP_TestFeature_1, TEMP_TestFeature_2,
                                                                              79, 'AR_79', 'EAR3_79')
    TestFeature = {}
    TestFeature[0] = TestFeature_1
    TestFeature[1] = TestFeature_2
    TestLabel = np.array(TestLabel_1)
    return matrixA, TestFeature, GalleryLabelSet, TestLabel


def lasso_value(matrixA, vectorb, lamb, x):
    return np.linalg.norm(np.dot(matrixA, x) - vectorb, ord=2)


def proximalGradiant(x_k, matrixA, vectorb, lamb, maxiter, epsilon):
    value_list = []
    eigenvalue, featurevector = np.linalg.eig(np.dot(matrixA.T, matrixA))
    l = np.max(eigenvalue)
    for i in range(maxiter):
        value_old = lasso_value(matrixA, vectorb, lamb, x_k)
        nabla_h = np.dot(matrixA.T, (np.dot(matrixA, x_k) - vectorb))
        temp_x = x_k - nabla_h / l
        temp = np.zeros(temp_x.shape)
        temp = np.concatenate((np.abs(temp_x) - lamb / l, temp), axis=1)
        x_k = np.sign(temp_x) * np.max(temp, axis=1).reshape(temp_x.shape)
        value_new = lasso_value(matrixA, vectorb, lamb, x_k)
        if i % 1 == 0:
            print("the %dth iter 's value= %f" % (i * 1, value_new[0][0]))
        value_list.append(value_new)
        if (abs(value_old[0][0] - value_new[0][0])) <= epsilon:
            return x_k, value_new, value_list
    return x_k, value_new, value_list


def Atom_APG_Joint(x_k, matrixA, vectorb, lamb, maxiter, epsilon):
    x_kminus1 = x_k
    l = 1
    value_list = []
    for i in range(1, maxiter + 1):
        value_old_1 = lasso_value(matrixA[0], vectorb[0], lamb, x_k[:, [0]])
        value_old_2 = lasso_value(matrixA[1], vectorb[1], lamb, x_k[:, [1]])
        value_old = (value_old_1 + value_old_2) / 2
        for m in range(2):
            x_k_hat = x_k[:, [m]] + (i - 2) / (i + 1) * (x_k[:, [m]] - x_kminus1[:, [m]])
            x_kminus1[:, [m]] = x_k[:, [m]]
            nabla_h = np.dot(matrixA[m].T, (np.dot(matrixA[m], x_k_hat) - vectorb[m]))
            temp_x = x_k_hat - nabla_h / l
            temp = np.zeros(temp_x.shape)
            temp = np.concatenate((np.abs(temp_x) - lamb / l, temp), axis=1)
            x_k[:, [m]] = np.sign(temp_x) * np.max(temp, axis=1).reshape(temp_x.shape)
        value_new_1 = lasso_value(matrixA[0], vectorb[0], lamb, x_k[:, [0]])
        value_new_2 = lasso_value(matrixA[1], vectorb[1], lamb, x_k[:, [1]])
        value_new = (value_new_1 + value_new_2) / 2
        if i % 20 == 0:
            print("the %dth iter 's value= %f" % (i * 1, value_new))
        value_list.append(value_new)
        if (abs(value_old - value_new)) <= epsilon:
            return x_k
    return x_k


def APG(x_k, matrixA, vectorb, lamb, maxiter, epsilon):
    x_kminus1 = x_k
    eigenvalue, featurevector = np.linalg.eig(np.dot(matrixA.T, matrixA))
    l = np.max(eigenvalue)
    value_list = []
    for i in range(1, maxiter + 1):
        value_old = lasso_value(matrixA, vectorb, lamb, x_k)
        x_k_hat = x_k + (i - 2) / (i + 1) * (x_k - x_kminus1)
        x_kminus1 = x_k
        nabla_h = np.dot(matrixA.T, (np.dot(matrixA, x_k_hat) - vectorb))
        temp_x = x_k_hat - nabla_h / l
        temp = np.zeros(temp_x.shape)
        temp = np.concatenate((np.abs(temp_x) - lamb / l, temp), axis=1)
        x_k = np.sign(temp_x) * np.max(temp, axis=1).reshape(temp_x.shape)
        value_new = lasso_value(matrixA, vectorb, lamb, x_k)
        if i % 1 == 0:
            print("the %dth iter 's value= %f" % (i * 1, value_new[0][0]))
        value_list.append(value_new)
        if (abs(value_old[0][0] - value_new[0][0])) <= epsilon:
            return x_k, value_new, value_list
    return x_k, value_new, value_list


def Atom_Level_ADMM_joint_sparse(A, x_k, matrixA, vectorb, lamb_1, lamb_2, lamb_3, pho, maxiter, epsilon):
    value_list_1 = []
    l = pho
    z_k_1 = copy.deepcopy(x_k)
    np.random.seed(0)
    w_k_1 = np.random.rand(matrixA[0].shape[1], 2)

    for i in range(1, maxiter + 1):
        value_old_1 = lasso_value(matrixA[0], vectorb[0], lamb_1, x_k[:, [0]])
        value_old_2 = lasso_value(matrixA[1], vectorb[1], lamb_1, x_k[:, [1]])
        value_old = (value_old_1 + value_old_2) / 2
        for m in range(2):
            pho_i = np.identity(matrixA[m].shape[1]) * l
            x_k[:, [m]] = np.dot(np.linalg.inv(np.dot(matrixA[m].T, matrixA[m]) + pho_i),
                                 (np.dot(matrixA[m].T, vectorb[m]) + l * z_k_1[:, [m]] - w_k_1[:, [m]]))

            temp_1 = np.zeros(z_k_1[:, [m]].shape)
            temp_1 = np.concatenate((np.abs(x_k[:, [m]] + w_k_1[:, [m]]) - lamb_1 / l, temp_1), axis=1)
            z_k_1[:, [m]] = np.sign(x_k[:, [m]] + w_k_1[:, [m]]) * np.max(temp_1, axis=1).reshape(z_k_1[:, [m]].shape)
            w_k_1[:, [m]] = w_k_1[:, [m]] + x_k[:, [m]] - z_k_1[:, [m]]

        value_new_1 = lasso_value(matrixA[0], vectorb[0], lamb_1, x_k[:, [0]])
        value_new_2 = lasso_value(matrixA[1], vectorb[1], lamb_1, x_k[:, [1]])
        value_new = (value_new_1 + value_new_2) / 2
        value_list_1.append(value_new_1)
        if (abs(value_old - value_new)) <= epsilon:
            return x_k, value_new, value_list_1
    return x_k, value_new_1, value_list_1


def ADMM_joint_sparse(A, x_k, matrixA, vectorb, lamb_1, lamb_2, lamb_3, pho, maxiter, epsilon):
    value_list_1 = []
    l = pho
    z_k_1 = copy.deepcopy(x_k)
    np.random.seed(0)
    w_k_1 = np.random.rand(matrixA[0].shape[1], 2)

    z_k_3 = copy.deepcopy(np.dot(A, x_k))
    np.random.seed(0)
    w_k_3 = np.random.rand(A.shape[0], 2)

    for i in range(1, maxiter + 1):
        value_old_1 = lasso_value(matrixA[0], vectorb[0], lamb_1, x_k[:, [0]])
        value_old_2 = lasso_value(matrixA[1], vectorb[1], lamb_1, x_k[:, [1]])
        value_old = (value_old_1 + value_old_2) / 2
        for m in range(2):
            pho_i = np.identity(matrixA[m].shape[1]) * l
            x_k[:, [m]] = np.dot(np.linalg.inv(np.dot(matrixA[m].T, matrixA[m]) + pho_i + l * np.dot(A.T, A)), (
                        np.dot(matrixA[m].T, vectorb[m]) + l * z_k_1[:, [m]] - w_k_1[:, [m]] + l * np.dot(A.T, z_k_3[:,
                                                                                                               [
                                                                                                                   m]] - w_k_3[
                                                                                                                         :,
                                                                                                                         [
                                                                                                                             m]])))

            temp_1 = np.zeros(z_k_1[:, [m]].shape)
            temp_1 = np.concatenate((np.abs(x_k[:, [m]] + w_k_1[:, [m]]) - lamb_1 / l, temp_1), axis=1)
            z_k_1[:, [m]] = np.sign(x_k[:, [m]] + w_k_1[:, [m]]) * np.max(temp_1, axis=1).reshape(z_k_1[:, [m]].shape)

            w_k_1[:, [m]] = w_k_1[:, [m]] + x_k[:, [m]] - z_k_1[:, [m]]

        for j in range(z_k_3.shape[0]):
            eta = lamb_3 / l
            zi = z_k_3[j] + w_k_3[j]
            z_k_3[j] = max((1 - eta / np.linalg.norm(zi, ord=2)), 0) * zi
        for m in range(2):
            w_k_3[:, [m]] = w_k_3[:, [m]] + np.dot(A, x_k[:, [m]]) - z_k_3[:, [m]]

        value_new_1 = lasso_value(matrixA[0], vectorb[0], lamb_1, x_k[:, [0]])
        value_new_2 = lasso_value(matrixA[1], vectorb[1], lamb_1, x_k[:, [1]])
        value_new = (value_new_1 + value_new_2) / 2
        value_list_1.append(value_new_1)
        if (abs(value_old - value_new)) <= epsilon:
            return x_k, value_new, value_list_1
    return x_k, value_new_1, value_list_1


def SR_dict(admm_x_sr, FeaturesOftest, DictOfLabel, DictOfTrains):
    y_1 = FeaturesOftest[0]
    y_2 = FeaturesOftest[1]

    sce_list = []
    for i in range(len(DictOfLabel)):
        A_temp_1 = DictOfTrains[0][:, i * 7: i * 7 + 7]
        A_temp_2 = DictOfTrains[1][:, i * 7: i * 7 + 7]
        x_temp_1 = admm_x_sr[i * 7: i * 7 + 7, [0]]
        x_temp_2 = admm_x_sr[i * 7: i * 7 + 7, [1]]
        Ax_1 = np.dot(A_temp_1, x_temp_1)
        Ax_2 = np.dot(A_temp_2, x_temp_2)
        isre_1 = y_1 - Ax_1
        isre_2 = y_2 - Ax_2
        sce_1 = np.linalg.norm(isre_1, ord=2)
        sce_2 = np.linalg.norm(isre_2, ord=2)
        scc_1 = np.linalg.norm(x_temp_1, ord=1)
        scc_2 = np.linalg.norm(x_temp_2, ord=1)
        scc = scc_1 + scc_2
        sce = sce_1 + sce_2
        sce = 1 - np.exp(-np.multiply(3, (scc / sce)))
        sce_list.append(sce)
    return sce_list


def TransFormFeature(TestLabel_1, TestLabel_2, TEMP_TestFeature_1, TEMP_TestFeature_2, NumofSubject, NameofDataset_1,
                     NameofDataset_2):

    return TestLabel_1, TestLabel_2, TEMP_TestFeature_1, TEMP_TestFeature_2
