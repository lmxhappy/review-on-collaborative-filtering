# coding: utf-8

import os
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from recsys.datasets import ml100k
from recsys.preprocessing import ids_encoder
from recsys.preprocessing import train_test_split, get_examples


def evaluate(x_test, y_test):
    print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
    preds = list(predict(u, i) for (u, i) in x_test)
    mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
    print('\nMAE :', mae)
    return mae


def my_test():
    from recsys.memories.UserToUser import UserToUser

    # load ml100k ratings
    ratings, movies = ml100k.load()

    # prepare data
    ratings, uencoder, iencoder = ids_encoder(ratings)

    # get examples as tuples of userids and itemids and labels from normalize ratings
    raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

    # train test split
    (x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

    # In[22]:

    # create the user-based CF
    usertouser = UserToUser(ratings, movies, metric='cosine')

    # In[23]:

    # evaluate the user-based CF on the ml100k test data
    usertouser.evaluate(x_test, y_test)

    # #### Evaluation on the ML-1M dataset (this may take some time)

    # In[ ]:

    from recsys.datasets import ml1m
    from recsys.preprocessing import ids_encoder, get_examples, train_test_split
    from recsys.memories.UserToUser import UserToUser

    # load ml100k ratings
    ratings, movies = ml1m.load()

    # prepare data
    ratings, uencoder, iencoder = ids_encoder(ratings)

    # get examples as tuples of userids and itemids and labels from normalize ratings
    raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

    # train test split
    (x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

    # create the user-based CF
    usertouser = UserToUser(ratings, movies, k=20, metric='cosine')

    # evaluate the user-based CF on the ml1m test data
    print("==========================")
    usertouser.evaluate(x_test, y_test)


def evaluate(x_test, y_test):
    print('Evaluate the model on {} test data ...'.format(x_test.shape[0]))
    preds = list(predict(u, i) for (u, i) in x_test)
    mae = np.sum(np.absolute(y_test - np.array(preds))) / x_test.shape[0]
    print('\nMAE :', mae)
    return mae


class UserCF:

    def __init__(self, saved_predictions='predictions.csv', metric='cosine'):
        ratings, self.movies = ml100k.load()

        # create the encoder
        self.ratings, self.uencoder, self.iencoder = ids_encoder(ratings)

        # csr_matrix. [user_num, item_num]
        self.R = self.ratings_matrix()

        # Step 1. Identify $G_u$, the set of $k$ users similar to an active user $u$
        self.model = self.create_model(rating_matrix=self.R,
                                       metric=metric)  # we can also use the 'euclidian' distance

        self.similarities, self.neighbors = self.nearest_neighbors()

        # mean ratings for each user
        mean = self.ratings.groupby(by='userid', as_index=False)['rating'].mean()
        self.mean = mean.to_numpy()[:, 1]

        mean_ratings = pd.merge(ratings, mean, suffixes=('', '_mean'), on='userid')

        # normalized ratings for each items
        # mean_ratings:Index(['userid', 'itemid', 'rating', 'rating_mean', 'norm_rating'], dtype='object')
        mean_ratings['norm_rating'] = mean_ratings['rating'] - mean_ratings['rating_mean']

        # nump shape [100000,5]
        self.np_ratings = mean_ratings.to_numpy()

        if os.path.exists(saved_predictions):
            os.remove(saved_predictions)

        self.saved_predictions = saved_predictions

        self.user2userCF()

    def create_model(self, rating_matrix, metric='cosine'):
        """
        - create the nearest neighbors model——NearestNeighbors with the corresponding similarity metric
        - fit the model

        @param rating_matrix: csr_matrix. [user_num, item_num]
        """
        model = NearestNeighbors(metric=metric, n_neighbors=21, algorithm='brute')
        model.fit(rating_matrix)

        return model

    def nearest_neighbors(self):
        """
        为每个user计算近邻。

        :param rating_matrix : rating matrix of shape (nb_users, nb_items)
        :param model : nearest neighbors model
        :return
            近的邻居，以及相应的相似度。
            - similarities : distances of the neighbors from the referenced user. numpy. shape[user_num, 20]
            - neighbors : neighbors of the referenced user in decreasing order of similarities. numpy. shape[user_num, 20]
        """
        similarities, neighbors = self.model.kneighbors(self.R)

        similarities = similarities[:, 1:]
        neighbors = neighbors[:, 1:]

        return similarities, neighbors

    def user2userCF(self):
        """
        Make predictions for each user in the database.
        """
        # get list of users in the database
        users = self.ratings.userid.unique()

        for count, userid in enumerate(users):
            # make rating predictions for the current user
            self.user2userPredictions(userid)

            sys.stdout.write('\rRating predictions. Progress status : %.1f%%' % (float(count / len(users)) * 100.0))
            sys.stdout.flush()

    def user2userPredictions(self, userid):
        """
        Make rating prediction for the active user on each candidate item and save in file prediction.csv

        :param
            - userid : id of the active user
            - pred_path : where to save predictions
        """
        # find candidate items for the active user
        # numpy
        candidates = self.find_candidate_items(userid)

        # loop over candidates items to make predictions
        for itemid in candidates:
            # prediction for userid on itemid
            # scalar value.
            r_hat = self.predict(userid, itemid)

            # save predictions
            with open(self.saved_predictions, 'a+') as file:
                line = '{},{},{}\n'.format(userid, itemid, r_hat)
                file.write(line)

    def find_candidate_items(self, userid):
        """
        Find candidate items for an active user

        邻居买过的item，排除了已经买过的，就是candidate items。

        :param userid : active user
        """
        user_neighbors = self.neighbors[userid]

        # dataframe. Index(['userid', 'itemid', 'rating'])
        activities = self.ratings.loc[self.ratings.userid.isin(user_neighbors)]

        # sort items in decreasing order of frequency
        # 每个itemid的rating次数
        # series。index是itemid，value是frequency。
        frequency = activities.groupby('itemid')
        frequency = frequency['rating']
        frequency = frequency.count()

        # Gu_items_0 = frequency.index.to_numpy()

        # dataframe。Index(['itemid', 'count']')
        frequency = frequency.reset_index(name='count')
        frequency = frequency.sort_values(['count'], ascending=False)
        Gu_items = frequency.itemid
        top_1 = Gu_items.iloc[0]
        # items already purchased by the active user
        active_items = self.ratings.loc[self.ratings.userid == userid].itemid.to_list()

        candidates = np.setdiff1d(Gu_items, active_items, assume_unique=True)[:30]
        ret = np.isin(top_1, candidates)
        return candidates

    def find_user_item_neighbors(self, userid, itemid):
        """
        返回userid相似的、对itemid有过评分的邻居。
        :return
            - 邻居
            - 距离
        """
        user_similarities = self.similarities[userid]
        user_neighbors = self.neighbors[userid]

        # find users who rated item 'itemid'
        iratings = self.np_ratings[self.np_ratings[:, 1].astype('int') == itemid]

        # find similar users to 'userid' who rated item 'itemid'
        suri = iratings[np.isin(iratings[:, 0], user_neighbors)]

        suri_userids = suri[:, 0].astype('int')
        indexes = [np.where(user_neighbors == uid)[0][0] for uid in suri_userids]

        sims = user_similarities[indexes]

        return suri, sims

    def predict(self, userid, itemid):
        """
        predict what score userid would have given to itemid.

        :param
            - userid : user id for which we want to make prediction
            - itemid : item id on which we want to make prediction

        :return
            - r_hat : predicted rating of user userid on item itemid
        """

        # get mean rating of user userid
        user_mean = self.mean[userid]

        suri, sims = self.find_user_item_neighbors(userid, itemid)

        # 如果没有发现对itemid有互动的近邻
        if suri.size == 0:
            return user_mean

        # similar users who rated current item (surci)
        normalized_ratings = suri[:, 4]

        num = np.dot(normalized_ratings, sims)
        den = np.sum(np.abs(sims))

        if num == 0 or den == 0:
            return user_mean

        r_hat = user_mean + num / den

        return r_hat

    def user2userRecommendation(self, userid):
        """
        """
        # encode the userid
        uid = self.uencoder.transform([userid])[0]
        saved_predictions = 'predictions.csv'

        predictions = pd.read_csv(saved_predictions, sep=',', names=['userid', 'itemid', 'predicted_rating'])
        predictions = predictions[predictions.userid == uid]
        List = predictions.sort_values(by=['predicted_rating'], ascending=False)

        List.userid = self.uencoder.inverse_transform(List.userid.tolist())
        List.itemid = self.iencoder.inverse_transform(List.itemid.tolist())

        List = pd.merge(List, self.movies, on='itemid', how='inner')

        return List

    # ### Transform rating dataframe to matrix
    def ratings_matrix(self):
        """
        制作user-item矩阵。

        :return    csr_matrix. [user_num, item_num]
        """
        # dataframe。[user_num, item_num]
        val = pd.crosstab(self.ratings.userid, self.ratings.itemid, self.ratings.rating, aggfunc=sum)

        val = val.fillna(0).values

        return csr_matrix(val)


def main():
    # ### Load MovieLen ratings
    user_cf = UserCF()

    user_cf.user2userRecommendation(212)

    # get examples as tuples of userids and itemids and labels from normalize ratings
    raw_examples, raw_labels = get_examples(ratings, labels_column='rating')

    # train test split
    (x_train, x_test), (y_train, y_test) = train_test_split(examples=raw_examples, labels=raw_labels)

    evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
