import cv2
import os
import glob
import numpy as np
import pickle
from sklearn.cluster import KMeans
import time


def Features_extraction(img, num_features=1000):
    # Finding the key points
    sift = cv2.SIFT_create(nfeatures=num_features)
    kps, des = sift.detectAndCompute(img, None)
    return kps, des


def get_server(num_features=1000, num_obj=50):
    try:
        with open('Data2/server/sift_features.pkl', 'rb') as file:
            server = pickle.load(file)
            print('Read server data')
        return server

    except FileNotFoundError:
        descriptors = []
        for i in np.arange(num_obj) + 1:
            img_paths = glob.glob('Data2/server/' + 'obj' + str(i) + '_*.JPG')
            descriptor = []
            for path in img_paths:
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                _, des = Features_extraction(img, num_features=num_features)
                descriptor.append(des[:num_features])
            descriptor = np.array(descriptor)
            descriptor = descriptor.reshape(-1, descriptor.shape[2])
            descriptors.append(descriptor)
        file = open('Data2/server/sift_features.pkl', 'wb')
        pickle.dump(np.array(descriptors), file)
        file.close()

        with open('Data2/server/sift_features.pkl', 'rb') as file:
            server = pickle.load(file)
            return server


def get_client(num_features=1000, num_obj=50):
    try:
        with open('Data2/client/sift_features.pkl', 'rb') as file:
            client = pickle.load(file)
            print('Read client data')
        return client

    except FileNotFoundError:
        descriptors = []
        for i in np.arange(num_obj) + 1:
            img_path = glob.glob('Data2/client/' + 'obj' + str(i) + '_*.JPG')[0]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            _, des = Features_extraction(img,num_features=num_features)
            des = des[:num_features].reshape((des[:num_features].shape[0], des[:num_features].shape[1]))
            descriptors.append(des)
        file = open('Data2/client/sift_features.pkl', 'wb')
        pickle.dump(np.array(descriptors), file)
        file.close()

        with open('Data2/client/sift_features.pkl', 'rb') as file:
            client = pickle.load(file)
            return client


def hi_kmeans(data, b, depth):
    server_list = np.concatenate(data, axis=0)
    num_obj = data.shape[0]

    idx = np.arange(server_list.shape[0])
    tree = recursive(server_list, idx, b, depth, 0, num_obj)
    tree = sorted(tree, key=lambda k: k['i'])

    for i, obj in enumerate(data):
        for feature in obj:
            leaf = recursiveNext(tree, feature.reshape(1, -1), 0)
            tree[leaf]['objects'][i] += 1

    leaves = list((filter(lambda x: len(x['children']) == 0, tree)))
    return tree, leaves


def recursiveNext(tree, feature, node):
    if tree[node]['model'] != None:
        cluster = tree[node]['model'].predict(feature)
        node = recursiveNext(tree, feature, tree[node]['children'][cluster[0]])
    return node


def recursive(data, idx, b, depth, n, num_obj):
    features = data[idx]
    tree = []
    children = []
    kmeans = None
    obj_array = None
    if features.shape[0] >= b and depth > 1:
        kmeans = KMeans(n_clusters=b, random_state=0).fit(features)
        for i in range(b):
            new_idx = [idx[p] for p, q in enumerate(kmeans.labels_) if q == i]
            new_depth = depth-1
            new_n = n+len(tree)+1
            new_tree = recursive(data, new_idx, b, new_depth, new_n, num_obj)
            tree += new_tree
            children.append(new_tree[-1]['i'])
    else:
        obj_array = np.zeros(num_obj)
    tree.append({'i': n, 'model': kmeans, 'children': children, 'objects': obj_array})
    return tree


def tf_idf(leaves, data):
    num_obj = data.shape[0]
    f = np.array(list(map(lambda x: x['objects'], leaves)))
    F = np.array(list(map(lambda x: x.shape[0], data)))
    K = np.array(list(map(lambda x: np.sum(x != 0), f)))
    W = f / F * np.log2(K / num_obj).reshape(-1, 1)
    return W, f


def query(tree, client, leaves):
    result = np.zeros((client.shape[0], len(leaves)))
    idx = np.array(list(map(lambda x: x['i'], leaves)))
    for i, obj in enumerate(client):
        for feature in obj:
            leaf = recursiveNext(tree, feature.reshape(1, -1), 0)
            leaf = np.where(idx == leaf)[0][0]
            result[i][leaf] += 1
    return result


def score(W, f, n):
    q = n @ W
    d = (f.T @ W).T
    s = []
    for i, j in enumerate(q):
        s.append(np.linalg.norm(j / np.linalg.norm(j) - d / np.linalg.norm(d), axis=1))
    s = np.array(s)
    sorted_s = np.argsort(s, axis=1)
    return sorted_s


def main():
    # Image Feature Extraction
    server = get_server(1000, 50)
    client = get_client(1000, 50)

    # Vocabulary Tree Construction
    b = 5
    depth = 7
    rate = 1
    num_features = int(1000 * rate)

    tree, leaves = hi_kmeans(server, b, depth)
    W, f = tf_idf(leaves, server)

    # Querying
    new_client = []
    for i in np.arange(client.shape[0]):
        temp = client[i][:num_features, :]
        new_client.append(temp)
    new_client = np.array(new_client)

    result = query(tree, new_client, leaves)
    s = score(W, f, result)
    top1 = np.sum([np.any(i == j) for i, j in enumerate(s[:, :1])])/50
    top5 = np.sum([np.any(i == j) for i, j in enumerate(s[:, :5])])/50

    print('features:', num_features, '\tb:', b, '\tdepth:', depth, '\trate:', rate, '\ttop1:', top1, '\ttop5:', top5)


if __name__ == '__main__':
    main()
