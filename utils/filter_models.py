import sys
sys.path.append('../')
from utils.evaluation import *
from utils.net_classification import *
from GA.geneticAlgorithm import TwoLevelGA
from GA.random_search import RandomSearcher

import numpy as np


# Load data
def get_data(dataset, parent_dir, loader_object):
    runs = os.listdir(parent_dir)
    sorted_h, sorted_ph = get_sorted_individuals_from_evolutions(dataset, parent_dir, runs, loader_object)
    table_h, table_ph = get_sorted_tables(sorted_h=sorted_h, sorted_ph=sorted_ph)
    return table_h, table_ph

def get_proportion(bads, goods, fitness_lim):
    total = len(bads)
    VP = np.sum(bads > fitness_lim)
    FP = np.sum(bads <= fitness_lim)
    VN = np.sum(goods <= fitness_lim)
    FN = np.sum(goods > fitness_lim)
    FPR = FP / (FP + VN)
    TPR = VP / (VP + FN)
    return FPR, TPR, total / (FP + 1)

def search_thr(table_, column, fitness_lim=0.25):
    values = [v for v in table_[column] if v is not None]
    values = list(np.sort(np.unique(values)))
    proportions = []
    FPR = []
    TPR = []
    for v in values:
        lower_group = table_[table_[column] < v]['error']
        upper_group = table_[table_[column] >= v]['error']
        
        good_group = lower_group if np.min(lower_group) < np.min(upper_group) else upper_group
        bad_group = upper_group if np.min(lower_group) < np.min(upper_group) else lower_group
        fpr, tpr, dist = get_proportion(bad_group, good_group, fitness_lim)
        proportions.append(dist)
        FPR.append(fpr)
        TPR.append(tpr)
    sorted_ids = np.argsort(proportions)[::-1]
    id1 = sorted_ids[0]
    if len(proportions) > 1:
        id2 = sorted_ids[1]
        for id2_aux in sorted_ids:
            if np.abs(id1  - id2_aux) > fitness_lim * len(proportions):
                id2 = id2_aux
                break
    else:
        id2 = sorted_ids[0]
    return [values[id1], values[id2]], [proportions[id1], proportions[id2]]


def compare_features(table_, delta=0.05):
    def convert(x):
        return 1. / (x - delta)
    def invert(x):
        return 1. / x + delta
   
    feats = {}
    thr_d = {}
    for c in table_.columns:
        if c in ['code', 'id', 'error']:
            continue
        ths, scores = search_thr(table_, c)
        thr_d[c] = ths
        total_filtered = 0
        segments = []
        for k in range(len(ths)):
            if scores[k] < 2:
                continue
            lower_group = table_[table_[c] < thr_d[c][k]]['error']
            upper_group = table_[table_[c] >= thr_d[c][k]]['error']

            good_group = lower_group if np.min(lower_group) < np.min(upper_group) else upper_group
            bad_group = upper_group if np.min(lower_group) < np.min(upper_group) else lower_group

            total = len(good_group) + len(bad_group)
            total_filtered += 100*len(bad_group)/total 
            segments.append('lower' if np.min(lower_group) < np.min(upper_group) else 'upper')
            
        if scores[0] > 2 and total_filtered > 1:
            for k in range(len(ths)):
                if scores[k] < 2:
                    continue
                lower_group = table_[table_[c] < thr_d[c][k]]['error']
                upper_group = table_[table_[c] >= thr_d[c][k]]['error']

                good_group = lower_group if segments[k]=='lower' else upper_group
                bad_group = upper_group if segments[k]=='lower' else lower_group

                total = len(good_group) + len(bad_group)
                if k == 0:
                    feats[c] = [scores[0] , [segments[0]]]
                else: 
                    feats[c] = [scores[0], [segments[0], segments[1]]]
    feats = {k: v for k, v in sorted(feats.items(), key=lambda item: item[1][0], reverse=True)}
    for k,v in feats.items():
        feats[k] = (thr_d[k][0:len(v[1])], v[1])
    return feats, thr_d

def filter_table(table_orig, order_feats, delta=0.05):
    table_ = table_orig.copy()
    def convert(x):
            return 1. / (x - delta)
    def invert(x):
        return 1. / x + delta
    order_feats_ = {}
    final_thr = {}
    for c in order_feats.keys():
        ths, segs = order_feats[c]
        bads = []
        orig_size = len(table_)
        for k in range(len(ths)):
            th = ths[k]
            seg = segs[k]
            lower_group = table_[table_[c] < th]
            upper_group = table_[table_[c] >= th]

            good_group = lower_group if seg == 'lower' else upper_group
            bad_group = upper_group if seg == 'lower' else lower_group

            table_ = good_group
            bads.append(bad_group)
            
        bad_group = pd.concat(bads)
        good_group = good_group['error']
        bad_group = bad_group['error']
        if len(bad_group) > 0:
            order_feats_[c] = order_feats[c]
            
        total = len(good_group) + len(bad_group)
        assert total == orig_size        
    return table_, order_feats_

# Compare results
def get_all_results(m, PARENT_DIR, DATASET='MRDBI'):
    dataset_names_ = ['test_validation3', 'validation', 'population_opt']
    dataset_names_18 = ['test_validation3', 'validation', '1level/18_epochs', 
                     '1level_20gen/18_epochs', 'population_opt', 'random_search/18_epochs']
    for dname in dataset_names_18:
        print(dname)
        table_h, table_ph = get_data(DATASET, PARENT_DIR % dname, TwoLevelGA)
        pos = []
        neg = []
        for row in table_h.iterrows():
            k, c = row
            code = c['code']
            x = m.is_model_ok(code)
            if x:
                pos.append(c['error'])
            else:
                neg.append(c['error'])
        total = np.array(pos + neg)

        pos_np = np.array(pos)
        neg_np = np.array(neg)
        old_total_bads = np.sum(total >= 0.25) #len(table_h_[table_h_['error'] >= 0.25])
        new_total_bads = np.sum(pos_np >= 0.25) #len(new_table_[new_table_['error'] >= 0.25])
        old_total_goods = np.sum(total < 0.25) #len(table_h_[table_h_['error'] < 0.25])
        new_total_goods = np.sum(pos_np < 0.25) #len(new_table_[new_table_['error'] < 0.25])
        print("Original --> Bads: %d. Goods: %d" % (old_total_bads, old_total_goods))
        print("New      --> Bads: %d. Goods: %d" % (new_total_bads, new_total_goods))
        plt.hist([1. / (pos_np - m.DELTA), 1. / (total - m.DELTA)], label=['positive', 'total'] , bins=20)
        plt.xlabel('1 / (Error - 0.05)')
        plt.ylabel("n°")
        plt.legend()
        plt.title("Histogram of evaluated architectures")
        plt.show()

class ModelFilterV2:
    DATASET = 'MRDBI'
    PARENT_DIR = '../../../../evolved_data/%s'
    DELTA = 0.05
    MODEL_PATH = 'filter_tmp'
    
    def __init__(self, load_file=None):
        if load_file is None:
            self.filters, self.table_columns = ModelFilterV2.build_system()
            self.save(ModelFilterV2.MODEL_PATH)
        else:
            self = self.load(load_file)
            
    @staticmethod
    def build_system():
        table_h, table_ph = get_data(ModelFilterV2.DATASET, ModelFilterV2.PARENT_DIR % 'population_opt', TwoLevelGA)
        feats, thrs = compare_features(table_h, ModelFilterV2.DELTA)
        l1 = ['dropout_std', 'filter_mul_std', 'filter_mul_mean', 'filter_mul_max', 'dropout_max', 'convs_less_identity',
              'final_node_is_conv', 'final_node_conv', 'kernel_5', 'filter_mul_min']
        l2 = ['final_node_is_conv', 'dropout_std', 'filter_mul_std', 'convs_less_identity', 'filter_mul_mean', 
                  'filter_mul_max', 'filter_mul_min', 'kernel_5']
        for key in l1:
            try:
                del feats[key]
            except KeyError:
                pass
        new_table, order_feats_final = filter_table(table_h, feats, ModelFilterV2.DELTA)
        for k,v in order_feats_final.items():
            print(k,'\t\t', v)
        return order_feats_final, table_h.columns
    
    def is_model_ok(self, model):
        table_ = self.get_features_from_model(model)
        for c in self.filters.keys():
            ths, segs = self.filters[c]
            for k in range(len(ths)):
                th = ths[k]
                seg = segs[k]
                feat = table_.get(c).values[0]
                if (feat < th and seg == 'lower') or (feat >= th and seg == 'upper'):
                    continue
                else:
                    return False
        return True
        
    def get_features_from_model(self, model):
        b = -1
        features_dict = {b: {}}
        for value, function in get_util_functions().items():
            features_dict[b][value] = function(str(model))
        c = pd.DataFrame(features_dict, index=self.table_columns).T
        #x = get_Xr(c)
        return c
    
    def save(self, filename):
        outfile = open(filename, 'wb')
        pickle.dump(self, outfile)
        outfile.close()

    @staticmethod
    def load(file):
        infile = open(file, 'rb')
        obj = pickle.load(infile)
        infile.close()
        return obj