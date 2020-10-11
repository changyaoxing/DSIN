# coding: utf-8
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import FRAC

if __name__ == "__main__":

    # 用户基本信息表user_profile
    # 本数据集涵盖了raw_sample中全部用户的基本信息。字段说明如下：
    # (1)
    # userid：脱敏过的用户ID；
    # (2)
    # cms_segid：微群ID；
    # (3)
    # cms_group_id：cms_group_id；
    # (4)
    # final_gender_code：性别
    # 1: 男, 2: 女；
    # (5)
    # age_level：年龄层次；
    # (6)
    # pvalue_level：消费档次，1: 低档，2: 中档，3: 高档；
    # (7)
    # shopping_level：购物深度，1: 浅层用户, 2: 中度用户, 3: 深度用户
    # (8)
    # occupation：是否大学生 ，1: 是, 0: 否
    # (9)
    # new_user_class_level：城市层级
    user = pd.read_csv('../raw_data/user_profile.csv')

    # 原始样本骨架raw_sample
    # 我们从淘宝网站中随机抽样了114万用户8天内的广告展示 / 点击日志（2600
    # 万条记录），构成原始的样本骨架。
    # 字段说明如下：
    # (1)
    # user_id：脱敏过的用户ID；
    # (2)
    # adgroup_id：脱敏过的广告单元ID；
    # (3)
    # time_stamp：时间戳；
    # (4)
    # pid：资源位；
    # (5)
    # noclk：为1代表没有点击；为0代表点击；
    # (6)
    # clk：为0代表没有点击；为1代表点击；
    # 我们用前面7天的做训练样本（20170506 - 20170512），用第8天的做测试样本（20170513）。
    sample = pd.read_csv('../raw_data/raw_sample.csv')

    if not os.path.exists('../sampled_data/'):
        os.mkdir('../sampled_data/')

    if os.path.exists('../sampled_data/user_profile_' + str(FRAC) + '_.pkl') and os.path.exists(
            '../sampled_data/raw_sample_' + str(FRAC) + '_.pkl'):
        user_sub = pd.read_pickle(
            '../sampled_data/user_profile_' + str(FRAC) + '_.pkl')
        sample_sub = pd.read_pickle(
            '../sampled_data/raw_sample_' + str(FRAC) + '_.pkl')
    else:

        if FRAC < 1.0:
            user_sub = user.sample(frac=FRAC, random_state=1024)
        else:
            user_sub = user
        sample_sub = sample.loc[sample.user.isin(user_sub.userid.unique())]
        pd.to_pickle(user_sub, '../sampled_data/user_profile_' +
                     str(FRAC) + '.pkl')
        pd.to_pickle(sample_sub, '../sampled_data/raw_sample_' +
                     str(FRAC) + '.pkl')

    # 用户的行为日志behavior_log
    # 本数据集涵盖了raw_sample中全部用户22天内的购物行为(共七亿条记录)。字段说明如下：
    # (1)
    # user：脱敏过的用户ID；
    # (2)
    # time_stamp：时间戳；
    # (3)
    # btag：行为类型, 包括以下四种：
    # ipv
    # 浏览
    # cart
    # 加入购物车
    # fav
    # 喜欢
    # buy
    # 购买
    # (4)
    # cate：脱敏过的商品类目；
    # (5)
    # brand: 脱敏过的品牌词；
    # 这里以user + time_stamp为key，会有很多重复的记录；这是因为我们的不同的类型的行为数据是不同部门记录的，
    # 在打包到一起的时候，实际上会有小的偏差（即两个一样的time_stamp实际上是差异比较小的两个时间）。
    if os.path.exists('../raw_data/behavior_log_pv.pkl'):
        log = pd.read_pickle('../raw_data/behavior_log_pv.pkl')
    else:


        log = pd.read_csv('../raw_data/behavior_log.csv', sep=',', engine='python', iterator=True)
        loop = True
        chunkSize = 1000
        chunks = []
        index = 0
        while loop:
            try:
                print(index)
                chunk = log.get_chunk(chunkSize)
                chunks.append(chunk)
                index += 1
                if index >= 10000:
                    break

            except StopIteration:
                loop = False
                print("Iteration is stopped."+index)
        print('开始合并')
        log = pd.concat(chunks, ignore_index=True)

        log = log.loc[log['btag'] == 'pv']
        pd.to_pickle(log, '../raw_data/behavior_log_pv.pkl')

    userset = user_sub.userid.unique()
    log = log.loc[log.user.isin(userset)]
    # pd.to_pickle(log, '../sampled_data/behavior_log_pv_user_filter_' + str(FRAC) + '_.pkl')

    ad = pd.read_csv('../raw_data/ad_feature.csv')
    ad['brand'] = ad['brand'].fillna(-1)

    lbe = LabelEncoder()
    # unique_cate_id = ad['cate_id'].unique()
    # log = log.loc[log.cate.isin(unique_cate_id)]

    unique_cate_id = np.concatenate((ad['cate_id'].unique(), log['cate'].unique()))

    lbe.fit(unique_cate_id)
    ad['cate_id'] = lbe.transform(ad['cate_id']) + 1
    log['cate'] = lbe.transform(log['cate']) + 1

    lbe = LabelEncoder()
    # unique_brand = np.ad['brand'].unique()
    # log = log.loc[log.brand.isin(unique_brand)]

    unique_brand = np.concatenate(
        (ad['brand'].unique(), log['brand'].unique()))

    lbe.fit(unique_brand)
    ad['brand'] = lbe.transform(ad['brand']) + 1
    log['brand'] = lbe.transform(log['brand']) + 1

    log = log.loc[log.user.isin(sample_sub.user.unique())]
    log.drop(columns=['btag'], inplace=True)
    log = log.loc[log['time_stamp'] > 0]

    pd.to_pickle(ad, '../sampled_data/ad_feature_enc_' + str(FRAC) + '.pkl')
    pd.to_pickle(
        log, '../sampled_data/behavior_log_pv_user_filter_enc_' + str(FRAC) + '.pkl')

    print("0_gen_sampled_data done")
