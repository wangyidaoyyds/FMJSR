import os
from config_unimodal import get_config_multimodal
import numpy as np
import os
import copy
import sys
import itertools
from joint_sparse import ADMM_joint_sparse, Atom_Level_ADMM_joint_sparse

from tqdm import tqdm

sys.path.append("/usr/local/MATLAB/R2017b/extern/engines/python/build/lib.linux-x86_64-2.7/")
sys.__egginsert = len(sys.path)
import matlab.engine

eng = matlab.engine.start_matlab()
import scipy.io as scio


def GetRealScore(dataFile, IoCoB, scope, dctordeep):
    real_score = []
    if dataFile == 'MDII':
        real_score_txt = open(
            'Result/multimodal/AR_79_EAR3_79/eval_multimodal_20225280101/real_{}_{}_pipeilist_joint_sparse.txt'.format(
                scope, IoCoB))
    else:
        real_score_txt = open(
            'Result/multimodal/AR_100_PolyU1/eval_multimodal_20225280101/real_{}_{}_pipeilist_joint_sparse.txt'.format(
                scope, IoCoB))
    real_file_lines = real_score_txt.readlines()
    real_score_list = []
    for string in real_file_lines:
        string = string.split(', ')
        real_score_list.append(string)
    real_score_list = np.asarray(real_score_list)
    for i in range(len(real_score_list[0])):
        real_score.append(float(real_score_list[0][i]))
    return real_score


def update_sce_list(sce_list_1, sce_list_2, DictRealLabel, label):
    return sce_list_1[DictRealLabel.index(int(label))] + sce_list_2[DictRealLabel.index(int(label))]


def TransFormFeature(data_1, data_2, feature_1, feature_2, scope, gap_1, gap_2):
    temp_1 = 0
    temp_2 = 0
    MultiModal_1 = []
    MultiModal_2 = []
    FeatureModal_1 = []
    FeatureModal_2 = []
    for i in range(min(len(data_1) // gap_1, len(data_2) // gap_2)):
        data_temp_1 = data_1[temp_1: temp_1 + gap_1]
        data_temp_2 = data_2[temp_2: temp_2 + gap_2]
        feature_temp_1 = feature_1[temp_1: temp_1 + gap_1, :]
        feature_temp_2 = feature_2[temp_2: temp_2 + gap_2, :]
        for tuples in itertools.product(data_temp_1, data_temp_2):
            MultiModal_1.append(np.asarray(list(tuples[0])))
            MultiModal_2.append(np.asarray(list(tuples[1])))
        for tuples in itertools.product(feature_temp_1, feature_temp_2):
            FeatureModal_1.append(np.asarray(list(tuples[0])))
            FeatureModal_2.append(np.asarray(list(tuples[1])))
        temp_1 += gap_1
        temp_2 += gap_2
    FeatureModal_1 = np.asarray(FeatureModal_1)
    FeatureModal_2 = np.asarray(FeatureModal_2)
    MultiModal_1 = np.asarray(MultiModal_1)
    MultiModal_2 = np.asarray(MultiModal_2)
    return MultiModal_1, MultiModal_2, FeatureModal_1, FeatureModal_2


class Eval(object):
    def __init__(self, dataFile, IoCoB, scope, order, rank_reshold, have_dict, cuda_num, dctOrDeep, lambda_1, lambda_2,
                 lambda_3, worst, epsilon):
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
        self.rank_reshold = rank_reshold
        self.IoCoB = IoCoB
        self.scope = scope
        self.dctOrDeep = dctOrDeep
        self.order = order
        self.worst = worst
        self.dataFile = dataFile
        print("-------------------------------------------")
        print(dataFile)
        print("self.scope: ", self.scope)
        print("self.IoCoB: ", self.IoCoB)
        print("self.rank: ", self.rank_reshold)
        print("self.dctOrdeep: ", self.dctOrDeep)
        print("self.worst: ", self.worst)
        print("lambda_1: ", lambda_1)
        print("lambda_3: ", lambda_3)
        print("epsilon: ", epsilon)
        print("-------------------------------------------")
        if self.worst == 'worst':
            if dataFile == 'MDII':
                self.dataFile_1 = '79AR_GT'
                self.dataFile_2 = 'EAR3_79'
                self.dataFile_3 = 'AR_79'
                self.dataFile_4 = 'EAR3_77EAR2'
                self.gap_1 = 8
                self.gap_2 = 11
                self.gap_3 = 7
                self.gap_4 = 4
            else:
                self.dataFile_1 = '100AR_GT'
                self.dataFile_2 = 'PolyU1'
                self.dataFile_3 = 'AR_100'
                self.dataFile_4 = 'PolyU1_PolyU2'
                self.gap_1 = 8
                self.gap_2 = 10
                self.gap_3 = 7
                self.gap_4 = 10
        else:
            if dataFile == 'MDII':
                self.dataFile_1 = '79AR_GT'
                self.dataFile_2 = '79EAR3_OCC20EAR_50'
                self.dataFile_3 = '79AR_OCC79AR'
                self.dataFile_4 = 'EAR3_77EAR2'
                self.gap_1 = 8
                self.gap_2 = 11
                self.gap_3 = 12
                self.gap_4 = 4
            else:
                self.dataFile_1 = '100AR_GT'
                self.dataFile_2 = 'PolyU1_20PolyU1'
                self.dataFile_3 = '100AR_OCC100AR'
                self.dataFile_4 = 'PolyU1_PolyU2'
                self.gap_1 = 8
                self.gap_2 = 10
                self.gap_3 = 12
                self.gap_4 = 10
        conf_1 = get_config_multimodal(self.dataFile_1, self.dataFile_2, dctOrDeep)
        conf_2 = get_config_multimodal(self.dataFile_3, self.dataFile_4, dctOrDeep)

        self.All_Rank = []
        self.All_score_no_1 = []
        self.All_score = []
        self.real_score = []
        self.spoof_score = []

        self.All_Rank_left = []
        self.All_score_left = []
        self.real_score_left = []
        self.spoof_score_left = []

        self.All_Rank_right = []
        self.All_score_right = []
        self.spoof_score_right = []
        self.spoof_score_right = []
        save_label_for_mat_1 = []
        save_label_for_mat_2 = []
        save_x_sr_for_mat_1 = []
        save_x_sr_for_mat_2 = []

        if have_dict == 'have_dict':
            if self.dataFile == 'MDII':
                self.class_dict = list(np.load(
                    'Result/Dict/eval_multimodal_xiaozidian_AR_79_EAR3_79_{}_{}.npy'.format(self.scope, self.dctOrDeep),
                    allow_pickle=True))
            else:
                self.class_dict = list(np.load(
                    'Result/Dict/eval_multimodal_xiaozidian_AR_100_PolyU1_{}_{}.npy'.format(self.scope, self.dctOrDeep),
                    allow_pickle=True))
        else:
            self.class_dict = [i + 1 for i in range(self.scope)]

        for which_spoof in range(2):
            if which_spoof == 0:
                GAP_1 = self.gap_1
                GAP_2 = self.gap_2
                data_1 = scio.loadmat(conf_1.MatName_1)
                data_2 = scio.loadmat(conf_1.MatName_2)
                GalleryLabel_1 = data_1['all_dict'][0:conf_1.GalleryLen]
                GalleryLabel_2 = data_2['all_dict'][0:conf_1.GalleryLen]
                TestLabel_1 = data_1['all_dict'][conf_1.GalleryLen:]
                TestLabel_2 = data_2['all_dict'][conf_1.GalleryLen:]
                GalleryLen = conf_1.GalleryLen
            else:
                GAP_1 = self.gap_3
                GAP_2 = self.gap_4
                data_1 = scio.loadmat(conf_2.MatName_1)
                data_2 = scio.loadmat(conf_2.MatName_2)
                GalleryLabel_1 = data_1['all_dict'][0:conf_2.GalleryLen]
                GalleryLabel_2 = data_2['all_dict'][0:conf_2.GalleryLen]
                TestLabel_1 = data_1['all_dict'][conf_2.GalleryLen:]
                TestLabel_2 = data_2['all_dict'][conf_2.GalleryLen:]
                GalleryLen = conf_2.GalleryLen

            self.GalleryFeature_1 = np.asarray(data_1['train_feature'], dtype=np.float64)
            self.GalleryFeature_2 = np.asarray(data_2['train_feature'], dtype=np.float64)
            self.TestFeature_1 = np.asarray(data_1['test_feature'], dtype=np.float64)
            self.TestFeature_2 = np.asarray(data_2['test_feature'], dtype=np.float64)
            self.TestLabel_1 = []
            self.TestLabel_2 = []
            TestLabel_1_num = []
            TestLabel_2_num = []
            self.TestLabel = []

            for i, value in enumerate(TestLabel_1):
                self.TestLabel_1.append(value.split('/')[0])
                TestLabel_1_num.append(int(value.split('/')[0]))
            for i, value in enumerate(TestLabel_2):
                self.TestLabel_2.append(value.split('/')[0])
                TestLabel_2_num.append(int(value.split('/')[0]))

            match_sort = sorted(list(zip(TestLabel_1_num, self.TestLabel_1, self.TestFeature_1)), key=lambda x: x[0])
            _, self.TestLabel_1, self.TestFeature_1 = zip(*match_sort)
            self.TestLabel_1 = list(self.TestLabel_1)
            self.TestFeature_1 = np.asarray(self.TestFeature_1)
            match_sort = sorted(list(zip(TestLabel_2_num, self.TestLabel_2, self.TestFeature_2)), key=lambda x: x[0])
            _, self.TestLabel_2, self.TestFeature_2 = zip(*match_sort)
            self.TestLabel_2 = list(self.TestLabel_2)
            self.TestFeature_2 = np.asarray(self.TestFeature_2)

            self.TestLabel_1, self.TestLabel_2, self.TestFeature_1, self.TestFeature_2 = TransFormFeature(
                self.TestLabel_1, self.TestLabel_2, self.TestFeature_1, self.TestFeature_2, GalleryLen // 7, GAP_1,
                GAP_2)

            self.GalleryLabel_1 = []
            self.GalleryLabel_2 = []
            GalleryLabel_1_num = []
            GalleryLabel_2_num = []
            for i, value in enumerate(GalleryLabel_1):
                self.GalleryLabel_1.append(value.split('/')[0])
                GalleryLabel_1_num.append(int(value.split('/')[0]))
            for i, value in enumerate(GalleryLabel_2):
                self.GalleryLabel_2.append(value.split('/')[0])
                GalleryLabel_2_num.append(int(value.split('/')[0]))

            match_sort = sorted(list(zip(GalleryLabel_1_num, self.GalleryLabel_1, self.GalleryFeature_1)),
                                key=lambda x: x[0])
            _, self.GalleryLabel_1, self.GalleryFeature_1 = zip(*match_sort)
            self.GalleryLabel_1 = list(self.GalleryLabel_1)
            self.GalleryFeature_1 = np.asarray(self.GalleryFeature_1)

            match_sort = sorted(list(zip(GalleryLabel_2_num, self.GalleryLabel_2, self.GalleryFeature_2)),
                                key=lambda x: x[0])
            _, self.GalleryLabel_2, self.GalleryFeature_2 = zip(*match_sort)
            self.GalleryLabel_2 = list(self.GalleryLabel_2)
            self.GalleryFeature_2 = np.asarray(self.GalleryFeature_2)

            self.GalleryFeature_1 = self.GalleryFeature_1.transpose()
            self.GalleryFeature_2 = self.GalleryFeature_2.transpose()
            self.GalleryLabel = np.asarray(self.GalleryLabel_1)
            self.TestLabel = np.asarray(self.TestLabel_1)
            if self.IoCoB == 'baseline':
                self.GalleryLabel = [int(i) for i in self.GalleryLabel]

            self.GalleryLabelSet = []
            for i in self.GalleryLabel:
                if i not in self.GalleryLabelSet:
                    self.GalleryLabelSet.append(i)
            self.GalleryLabelSet = list(map(int, self.GalleryLabelSet))
            for i, label in tqdm(enumerate(self.TestLabel)):
                label = "".join(self.TestLabel[i])
                if which_spoof == 0:
                    save_label_for_mat_1.extend(label)
                else:
                    save_label_for_mat_2.extend(label)
                FeaturesOftests_1 = self.TestFeature_1[i]
                FeaturesOftests_2 = self.TestFeature_2[i]
                if int(label) not in self.class_dict:
                    class_dict = copy.deepcopy(self.class_dict)
                    class_dict.append(int(label))
                else:
                    class_dict = copy.deepcopy(self.class_dict)
                DictOfTrains_1 = np.zeros([200, len(class_dict) * 7])
                DictOfTrains_2 = np.zeros([200, len(class_dict) * 7])
                DictRealLabel = []
                gap = 0
                for LabelInGallery in self.GalleryLabelSet:
                    if LabelInGallery in class_dict:
                        start = self.GalleryLabelSet.index(LabelInGallery) * 7
                        end = start + 7
                        DictOfTrains_1[:, gap: gap + 7] = self.GalleryFeature_1[:, start: end]
                        DictOfTrains_2[:, gap: gap + 7] = self.GalleryFeature_2[:, start: end]
                        gap += 7
                        DictRealLabel.append(LabelInGallery)
                AFORX = np.zeros((len(DictRealLabel), len(DictRealLabel) * 7))
                for aforx in range(0, len(DictRealLabel)):
                    AFORX[aforx, aforx * 7: aforx * 7 + 7] = 1
                matrixA = {}
                matrixA[0] = DictOfTrains_1
                matrixA[1] = DictOfTrains_2
                vectorb = {}
                vectorb[0] = np.asarray([FeaturesOftests_1]).T
                vectorb[1] = np.asarray([FeaturesOftests_2]).T
                np.random.seed(0)
                admm_x_k = np.random.randn(len(DictRealLabel) * 7, 2)
                self.admm_x_sr, _, _ = Atom_Level_ADMM_joint_sparse(AFORX, admm_x_k, matrixA, vectorb, lambda_1,
                                                                    lambda_2, lambda_3, 1, 50000, epsilon)
                if which_spoof == 0:
                    save_x_sr_for_mat_1.extend(self.admm_x_sr)
                else:
                    save_x_sr_for_mat_2.extend(self.admm_x_sr)
                sce_list, sce_list_left, sce_list_right = self.SR_dict(FeaturesOftests_1, FeaturesOftests_2,
                                                                       DictRealLabel, matrixA,
                                                                       DictRealLabel.index(int(label)))

                rank_list = [0 for x in range(len(sce_list))]
                sorted_sce_list = copy.deepcopy(sce_list)
                sorted_sce_list = sorted(list(set(sorted_sce_list)), reverse=True)
                sce_list_no_1 = [sorted_sce_list[0] for x in sce_list]
                for j in range(len(sce_list)):
                    rank_list[j] = sorted_sce_list.index(sce_list[j]) + 1
                self.All_Rank.append(rank_list[DictRealLabel.index(int(label))])
                self.All_score_no_1.append(sce_list_no_1[DictRealLabel.index(int(label))])
                self.spoof_score.append(sce_list[DictRealLabel.index(int(label))])

                rank_list_left = [0 for x in range(len(sce_list_left))]
                sorted_sce_list_left = copy.deepcopy(sce_list_left)
                sorted_sce_list_left = sorted(list(set(sorted_sce_list_left)), reverse=True)
                for j in range(len(sce_list_left)):
                    rank_list_left[j] = sorted_sce_list_left.index(sce_list_left[j]) + 1
                self.All_Rank_left.append(rank_list_left[DictRealLabel.index(int(label))])

                rank_list_right = [0 for x in range(len(sce_list_right))]
                sorted_sce_list_right = copy.deepcopy(sce_list_right)
                sorted_sce_list_right = sorted(list(set(sorted_sce_list_right)), reverse=True)
                for j in range(len(sce_list_right)):
                    rank_list_right[j] = sorted_sce_list_right.index(sce_list_right[j]) + 1
                self.All_Rank_right.append(rank_list_right[DictRealLabel.index(int(label))])

            self.All_Label = [int(0)] * len(self.spoof_score)
            if not os.path.exists('Result/multimodal/{}/eval_multimodal_{}'.format(dataFile, self.order)):
                os.makedirs('Result/multimodal/{}/eval_multimodal_{}'.format(dataFile, self.order))
            spoof_score = open(
                'Result/multimodal/{}/eval_multimodal_{}/spoof_{}_{}_pipeilist_joint_sparse_{}.txt'.format(dataFile,
                                                                                                           self.order,
                                                                                                           self.scope,
                                                                                                           self.IoCoB,
                                                                                                           self.worst),
                'w+')
            spoof_score.write(str(self.spoof_score))
            spoof_score.close()
            scio.savemat('{}_x_sr_{}_spoof_{}.mat'.format(self.dataFile, dctOrDeep, self.worst),
                         {'label_test_1': save_label_for_mat_1, 'label_test_2': save_label_for_mat_2,
                          'x_sr_1': save_x_sr_for_mat_1, 'x_sr_2': save_x_sr_for_mat_2})
            if self.IoCoB == 'baseline':
                pass
            self.All_score = self.spoof_score

            real_score = GetRealScore(self.dataFile, self.IoCoB, self.scope, self.dctOrDeep)

            TestLabel = [int(1)] * len(real_score)
            rank_list = [int(1)] * len(real_score)

            self.All_Rank.extend(rank_list)
            self.All_Rank_left.extend(rank_list)
            self.All_Rank_right.extend(rank_list)

            self.All_score.extend(real_score)
            self.All_score_no_1.extend(real_score)
            self.All_Label.extend(TestLabel)

            self.cal_accuracy(rank_reshold)

    def cal_accuracy(self, rank_threshold):
        score = np.asarray(list(self.All_score))
        score_left = np.asarray(list(self.All_score_left))
        score_right = np.asarray(list(self.All_score_right))
        true_label = np.asarray(list(self.All_Label))

        all_rank = np.asarray(self.All_Rank)
        all_rank_left = np.asarray(self.All_Rank_left)
        all_rank_right = np.asarray(self.All_Rank_right)

        frrlist = []
        farlist = []
        min_eer_index = 3

        EER_result = open('Result/multimodal_spoof_EER_result.txt', 'a+')
        choose_score = list(set(list(score)))
        for i in tqdm(range(len(choose_score))):
            th = choose_score[i]
            classOfpredict = (score >= th)
            classOfpredict_score = abs(self.All_score_no_1 - score) < 0.4
            classOfpredict = np.logical_and(classOfpredict_score, classOfpredict)
            accOfpredict = (classOfpredict & true_label)
            TP = np.sum(accOfpredict)
            FN = np.sum(classOfpredict < true_label)
            TP = float(TP)
            FN = float(FN)
            TPR = (TP / (TP + FN))

            FP = np.sum(classOfpredict > true_label)
            TN = len(score) - TP - FN - FP
            FP = float(FP)
            FN = float(FN)
            FPR = (FP / (FP + TN))

            FAR = FPR
            FRR = 1 - TPR

            frrlist.append(FRR)
            farlist.append(FAR)
            if abs(FAR - FRR) <= min_eer_index:
                min_eer_index = abs(FAR - FRR)
                eer_min = (FAR + FRR) / 2
                print("---------------")
                print("th: ", th)
                print("FRR: ", FRR)
                print("FAR: ", FAR)
                print("eer: ", eer_min)
                print("TP: ", TP)
                print("FN: ", FN)
                print("FP: ", FP)
                print("TN: ", TN)

        np.savetxt(
            'Result/multimodal/{}/eval_multimodal_{}/{}_{}_rank_reshold_0_far_joint_sparse_{}.txt'.format(self.dataFile,
                                                                                                          self.order,
                                                                                                          self.scope,
                                                                                                          self.IoCoB,
                                                                                                          self.worst),
            farlist, fmt='%f', delimiter=',')
        np.savetxt(
            'Result/multimodal/{}/eval_multimodal_{}/{}_{}_rank_reshold_0_frr_joint_sparse_{}.txt'.format(self.dataFile,
                                                                                                          self.order,
                                                                                                          self.scope,
                                                                                                          self.IoCoB,
                                                                                                          self.worst),
            frrlist, fmt='%f', delimiter=',')
        v = list(map(lambda x: abs(x[0] - x[1]), zip(farlist, frrlist)))
        minindex = v.index(min(v))
        eer = float(frrlist[minindex]) + float(farlist[minindex])
        eer = eer / 2
        print("eer:{}%".format(eer))
        EER_result_txt = "Result_multimodal_{}_{}_{}_eval_multimodal_{}_{}_{}_RANK_{}_joint_sparse_spoof_{}: {}\n".format(
            self.dataFile_1, self.dataFile_2, self.worst, self.order, self.scope, self.IoCoB, '0', self.dctOrDeep,
            str(eer))
        EER_result.write(EER_result_txt)
        if self.rank_reshold != 0:
            for rank_th in [5]:
                frrlist = []
                farlist = []
                classOfpredict_rank = all_rank <= rank_th
                classOfpredict_rank_left = all_rank_left <= rank_th
                classOfpredict_rank_right = all_rank_right <= rank_th
                if self.dctOrDeep == "deep":
                    classOfpredict_score = abs(self.All_score_no_1 - score) < 0.4
                else:
                    classOfpredict_score = abs(self.All_score_no_1 - score) < 0.8
                classOfpredict_r_temp = np.logical_and(classOfpredict_rank, classOfpredict_rank_left)
                classOfpredict_r = np.logical_and(classOfpredict_r_temp, classOfpredict_rank_right)
                classOfpredict_s_r = np.logical_and(classOfpredict_r, classOfpredict_score)
                for i in tqdm(range(len(choose_score))):
                    th = choose_score[i]
                    classOfpredict = (score >= th)
                    classOfpredict = np.logical_and(classOfpredict_s_r, classOfpredict)
                    accOfpredict = (
                            classOfpredict & true_label)
                    TP = np.sum(accOfpredict)
                    FN = np.sum(classOfpredict < true_label)
                    TP = float(TP)
                    FN = float(FN)
                    TPR = (TP / (TP + FN))

                    FP = np.sum(classOfpredict > true_label)
                    TN = len(score) - TP - FN - FP
                    FP = float(FP)
                    FN = float(FN)
                    FPR = (FP / (FP + TN))

                    FAR = FPR
                    FRR = 1 - TPR

                    frrlist.append(FRR)
                    farlist.append(FAR)

                    if abs(FAR - FRR) < min_eer_index:
                        min_eer_index = abs(FAR - FRR)
                        eer_min = (FAR + FRR) / 2
                        print("---------------")
                        print("th: ", th)
                        print("FRR: ", FRR)
                        print("FAR: ", FAR)
                        print("eer: ", eer_min)
                        print("TP: ", TP)
                        print("FN: ", FN)
                        print("FP: ", FP)
                        print("TN: ", TN)
                np.savetxt(
                    'Result/multimodal/{}/eval_multimodal_{}/{}_{}_rank_reshold_{}_far_joint_sparse_{}_sorted_by_cer.txt'.format(
                        self.dataFile, self.order, self.scope, self.IoCoB, self.rank_reshold, self.worst), farlist,
                    fmt='%f', delimiter=',')
                np.savetxt(
                    'Result/multimodal/{}/eval_multimodal_{}/{}_{}_rank_reshold_{}_frr_joint_sparse_{}_sorted_by_cer.txt'.format(
                        self.dataFile, self.order, self.scope, self.IoCoB, self.rank_reshold, self.worst), frrlist,
                    fmt='%f', delimiter=',')
                v = list(map(lambda x: abs(x[0] - x[1]), zip(farlist, frrlist)))
                minindex = v.index(min(v))
                eer = float(frrlist[minindex]) + float(farlist[minindex])
                eer = eer / 2
                print("eer:{}%".format(eer))
                EER_result_txt = "Result_multimodal_{}_{}_{}_eval_multimodal_{}_{}_{}_RANK_{}_joint_sparse_spoof_{}_sorted_by_cer: {}\n".format(
                    self.dataFile_1, self.dataFile_2, self.worst, self.order, self.scope, self.IoCoB, rank_th,
                    self.dctOrDeep, str(eer))
                EER_result.write(EER_result_txt)
        EER_result.close()

    def update_dict(self, temp_class_dict, label):
        TEMP_INDEX = temp_class_dict[label]
        temp_INDEX = self.class_dict[label]
        val = list(set(TEMP_INDEX).intersection(set(temp_INDEX)))
        if len(val) >= self.scope:
            k = 0
            real_k = 0
            for keys in self.class_dict[label]:
                if keys not in val:
                    k += 1
                    continue
                self.class_dict[label][real_k] = self.class_dict[label][k]
                k += 1
                real_k += 1
            del self.class_dict[label][real_k:]

    def SR_dict(self, FeaturesOftest_1, FeaturesOftest_2, DictOfLabel, DictOfTrains, index_label):
        y_1 = np.transpose([FeaturesOftest_1])
        y_2 = np.transpose([FeaturesOftest_2])

        sce_list = []
        sce_list_left = []
        sce_list_right = []
        sce_list_left_cer = []
        sce_list_right_cer = []
        a = []
        b = []
        for i in range(len(DictOfLabel)):
            A_temp_1 = DictOfTrains[0][:, i * 7: i * 7 + 7]
            A_temp_2 = DictOfTrains[1][:, i * 7: i * 7 + 7]
            x_temp_1 = self.admm_x_sr[i * 7: i * 7 + 7, [0]]
            x_temp_2 = self.admm_x_sr[i * 7: i * 7 + 7, [1]]
            Ax_1 = np.dot(A_temp_1, x_temp_1)
            Ax_2 = np.dot(A_temp_2, x_temp_2)
            isre_1 = y_1 - Ax_1
            isre_2 = y_2 - Ax_2
            sce_1 = np.linalg.norm(isre_1, ord=2)
            sce_2 = np.linalg.norm(isre_2, ord=2)
            scc_1 = np.linalg.norm(x_temp_1, ord=1)
            scc_2 = np.linalg.norm(x_temp_2, ord=1)
            cer_1 = 1 - np.exp(-np.multiply(3, (scc_1 / sce_1)))
            cer_2 = 1 - np.exp(-np.multiply(3, (scc_2 / sce_2)))
            sce = cer_1 + cer_2

            sce_list_left.append(sce_1)
            sce_list_right.append(sce_2)
            sce_list_left_cer.append(cer_1)
            sce_list_right_cer.append(cer_2)
            sce_list.append(sce)
        return sce_list, sce_list_left_cer, sce_list_right_cer

    def l1_ls(self, DictOfTrain, FeaturesOftest):
        y = np.transpose([FeaturesOftest])
        A = DictOfTrain.tolist()
        y = y.tolist()
        A = matlab.double(A)
        y = matlab.double(y)
        lambdaa = 0.05
        rel_tol = 0.01
        x = eng.l1_ls(A, y, lambdaa, rel_tol)
        x_sr = np.array(x)
        return x_sr

    def base(self, FeaturesOftests):
        base_list = []
        t = 0
        for i in range(0, len(self.DictOfTrain)):
            dist = np.linalg.norm(self.DictOfTrain[i] - FeaturesOftests)
            base_list.append(dist)
        return base_list


if __name__ == "__main__":
    import time

    print("sleeping")
    print("begin")

    eval = Eval('MDII', 'CER', 79,  5, '1', 'no_need', 'deep', 0.1, 0.006, 0.1, 'worst', 1e-6)
    eval = Eval('MDII', 'CER', 79,  5, '1', 'no_need', 'deep', 0.1, 0.006, 0.1, 'no_worst', 1e-6)
    eval = Eval('MDIII', 'CER', 100, 5, '1', 'no_need', 'deep', 0.1, 0.006, 0.1, 'worst', 1e-6)
    eval = Eval('MDIII', 'CER', 100,  5, '1', 'no_need', 'deep', 0.1, 0.006, 0.1, 'no_worst', 1e-6)
