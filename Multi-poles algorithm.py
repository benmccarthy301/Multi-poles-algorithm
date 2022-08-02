#imports
import numpy as np
from os.path import exists
import pandas as pd

#cartesian coordinates
def dist1(star_i, star_j):
    return np.arccos(np.dot(star_i, star_j))

# Create Feature Set
def create_feature_set(catalog, FOV):
    feature_set = [] # [(id(0), id(1)), theta < FOV]
    for i in range(len(catalog)):
        for j in range(i+1, len(catalog)):
            theta = dist1(catalog['position'][i], catalog['position'][j])
            if theta <= FOV:
                feature_set.append((catalog['id'][i], catalog['id'][j], theta))
    return feature_set

def initialise(stars, ids, num_poles):
    distance = []
    init = []
    for i in range(len(stars)):
        init.append((stars[i], ids[i]))
        distance.append((stars[i], dist1(stars[i], [0, 0, 1]), ids[i]))
    distance = np.array(list(distance), dtype=[('position', object), ('distance', object), ('id', object)])
    init = np.array(list(init), dtype=[('position', object), ('id', object)])
    distance = np.sort(distance, order='distance')
    #return distance[:num_poles]
    return init[:num_poles]

#Create neighbour set of first pole using angular distances
def create_distance_set(stars, pole):
    distance = [(star, pole, dist1(star, pole)) for star in stars if not np.array_equal(star, pole)]
    return np.array(distance, dtype=[('star', object), ('pole', object), ('distance', object)])

def create_pair_list_simple(feature_set, distance, eps_pi):
    pairs = []
    distances = []
    for i in range(len(feature_set)):
        if np.abs(feature_set[i][2] - distance) <= eps_pi:
            pairs.append((feature_set[i][0], feature_set[i][1]))
        distances.append(((feature_set[i][0], feature_set[i][1]), feature_set[i][2], distance, np.abs(feature_set[i][2] - distance)))
    distances = np.array(distances, dtype = [("feature ids", object), ("ground truth", object), ("calculated", object), ("distance", object)])
    distances = np.sort(distances, order='distance')
    return pairs

def create_pair_list_long(feature_set, distances, eps_pi, N):
    pairs = []
    for j, distance in enumerate(distances):
        for i in range(len(feature_set)):
            if np.abs(feature_set[i][2] - distance['distance']) <= eps_pi:
                if j >= N:
                    pairs.append((j+1, (feature_set[i][0], feature_set[i][1])))
                else:
                    pairs.append((j, (feature_set[i][0], feature_set[i][1])))
    return np.array(pairs, dtype=[('index', object), ('feature ids', tuple)])

#pole is one that occurs most in list

def find_pole_in_pairs(pair_list):
    counts = {}
    positions = []
    for i in range(len(pair_list)):
        pair = pair_list[i]
        for star_id in pair['feature ids']:
            if star_id in counts:
                counts[star_id] += 1
            else:
                counts[star_id] = 1
                positions.append(pair['index'])
    max_count = max(list(counts.values()))
    position = list(counts.values()).index(max_count)
    return list(counts.keys())[position]

# neighbours are all stars paired with pole

def create_neighbour_set(pairs, pole):
    pole_neighbours = []
    for pair in pairs:
        if pair['feature ids'][0] == pole:
            pole_neighbours.append((pair['index'], pair['feature ids'][1]))
        elif pair['feature ids'][1] == pole:
            pole_neighbours.append((pair['index'], pair['feature ids'][0]))
    pole_neighbours = np.array(pole_neighbours, dtype=[('index', object), ('id', object)])
    return pole_neighbours

def create_neighbour_id_set(pairs, pole):
    return [ID for pair in pairs for ID in pair if pole in pair and ID != pole]

def intersection(set_1, set_2):
    intersection = [(star['index'], star['id']) for star in set_1 if star in set_2]
    return intersection

#returns intersection list if u and w true, else returns 0
def verification_phase(pole_i, neighbours_i, pole_j, neighbours_j):
    th = 5
    u = (pole_i['id'] in neighbours_j['id']) and (pole_j['id'] in neighbours_i['id'])
    if u:
        V = intersection(neighbours_i, neighbours_j)
        w = len(V) >= th
        if w:
            position = np.where(neighbours_i['id'] == pole_j['id'])
            V.append(neighbours_i[position][0])
            V = np.array(V, dtype = [('index', object), ('id', object)])
            return V
    return False

def confirmation_phase(verified, confirmed, features, stars):
    if len(verified) == 0:
        return confirmed
    v = verified[0]
    verified = verified[1:]
    v_pos = stars[v['index']]
    c_pos = confirmed[-1]['position']
    distance = dist1(v_pos.tolist(), c_pos.tolist())
    confirmed_features = [feature for feature in features if confirmed[-1]['id'] in feature]
    pairs = create_pair_list_simple(confirmed_features, distance, eps_pi)
    pairs = [pair for pair in pairs if v['id'] in pair]

    if len(pairs) == 0:
        return confirmation_phase(verified, confirmed, features, stars)
    else:
        dtype = confirmed.dtype
        confirmed = list(confirmed)
        confirmed.append((v['id'], v_pos))
        confirmed = np.array(confirmed, dtype=dtype)
        return confirmation_phase(verified, confirmed, features, stars)
    return

#returns pole and neighbour ids
def acceptance_phase(stars, features, candidate_pole, eps_pi, N):
    distances = create_distance_set(stars, candidate_pole)
    pairs = create_pair_list_long(features, distances, eps_pi, N)
    pole = find_pole_in_pairs(pairs)
    neighbours = create_neighbour_set(pairs, pole)
    pole = np.array((pole, candidate_pole), dtype=[('id', object), ('position', object)])
    return pole, neighbours, pairs


def recursive_MPA(stars, pole_i, neighbours_i, candidate_poles, N, features):
    u, w = False, False
    pole_j, neighbours_j, pairs = acceptance_phase(stars, features, candidate_poles['position'][N], eps_pi, N)
    print("GROUND TRUTH POLE FOUND: ", pole_j['id'], candidate_poles['id'][N] == pole_j['id'])
    verified = verification_phase(pole_i, neighbours_i, pole_j, neighbours_j)
    print("VERIFIED RESULT: ", verified)
    if not isinstance(verified, bool):
        confirmed = np.array([(pole_i['id'], pole_i['position'])], dtype=[('id', int), ('position', list)])
        return confirmation_phase(verified, confirmed, features, stars)
    if N == 0:
        return -1
    return recursive_MPA(stars, pole_i, neighbours_i, candidate_poles, N-1, features)


def get_data_from_file_arr(file):
    catalog = []
    for i, line in enumerate(file):
        line = list(map(float, line.split()))
        line = line/np.linalg.norm(line)
        if np.linalg.norm(line) - 1.54 <= 5.5:
            catalog.append((i, line))
    return np.array(catalog, dtype=[('id', object), ('position', object)])

def get_data_from_file(file):
    catalog = []
    for i, line in enumerate(file):
        line = list(map(float, line.split()))
        line = line/np.linalg.norm(line)
        catalog.append(list(line))
    return np.array(catalog)

def get_results(i, ids, chain_set):
    if not exists('output/results.csv'):
        f = open('output/results.csv', 'w')
        f.write('TP,FP,NR\n0,0,0\n')
        f.close()
    df = pd.read_csv('output/results.csv')
    #chain set is -1 if loops out of verification so can't get id_rate or error rate
    no_result = isinstance(chain_set, int)
    if no_result:
        df["NR"] += no_result
        return
    id_rate = len([link for link in chain_set['id'] if link in ids])/len(ids) >= 0.7
    error_rate = (len([link for link in chain_set['id'] if not link in ids])+len(ids)-ids.count(-1)-len(chain_set['id']))/len(ids) >= 0.7
    if id_rate + error_rate + no_result != 1:
        print("error: results don't sum to 1")
    df["TP"] += id_rate
    df["FP"] += error_rate
    df["NR"] += no_result
    df.to_csv('output/results.csv', index=False)

FOV = np.deg2rad(14)
eps_pi = 0.00028

file = open("HIP_catalog_vectors_with_mag_less_than_6", "r")
catalog = get_data_from_file_arr(file)
features = create_feature_set(catalog, FOV)

for i in range(1):
    print(i, end='\r')
    file = open(f"5/stars_{0}")
    stars = get_data_from_file(file)
    file = open(f"5/id_{0}")

    ids = []
    for line in file:
        ids.append(int(line))
    candidate_poles = initialise(stars, ids, int(len(ids)/2))
    N = len(candidate_poles)-1

    pole_i, neighbours_i, pairs = acceptance_phase(stars,
                                        features,
                                        candidate_poles['position'][N],
                                        eps_pi,
                                        N)

    print("GROUND TRUTH POLE FOUND: ", pole_i['id'], candidate_poles['id'][N] == pole_i['id'])
    chain_set = recursive_MPA(stars,
                        pole_i, #first pole
                        neighbours_i, #first pole's neighbours
                        candidate_poles, #rest of poles
                        N-1, #j-th index
                        features) #[id0, id1, theta]
    if not isinstance(chain_set, int):
        print("calculated ids", chain_set['id'])
        print("ground truth ids", ids)
    else:
        print("chain set not found")
    get_results(i, ids, chain_set)

