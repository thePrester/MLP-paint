import numpy as np

CLASS_MAP = {
    "alpha":   "1,0,0,0,0",
    "beta":    "0,1,0,0,0",
    "gamma":   "0,0,1,0,0",
    "delta":   "0,0,0,1,0",
    "epsilon": "0,0,0,0,1"
}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_features(points, M):
    if not points or not M:
        raise ValueError
    translation_invariant = np.array(points) - np.mean(points, axis=0)
    m = np.max(np.max(np.abs(translation_invariant), axis=0))
    scale_invariant = translation_invariant / m
    pairs = [*zip(scale_invariant, scale_invariant[1:])]
    interpoint_distances = [np.linalg.norm(x1 - x2) for x1, x2 in pairs]
    D = np.sum(interpoint_distances)
    distances_x = _calculate_distances(interpoint_distances)
    represent = list()
    for k in range(M):
        new_distance = k * D / (M - 1)
        rep_x, rep_y = _interpolate(scale_invariant, distances_x, new_distance)
        represent.append(rep_x)
        represent.append(rep_y)
    return represent

def _calculate_distances(interpoint_distances):
    distances_x = [0] + [np.sum(interpoint_distances[:index]) \
        for index in range(1, len(interpoint_distances) + 1)]
    return distances_x

def _interpolate(points, values, new_value):
    if new_value == values[-1]:
        new_point = points[-1]
        return new_point
    for index in range(len(points) - 1):
        if new_value > values[index] and new_value < values[index + 1]:
            l_value, r_value = values[index], values[index + 1]
            l_weight = (r_value - new_value) / (r_value - l_value)
            r_weight = (new_value - l_value) / (r_value - l_value)

            l_point, r_point = points[index], points[index + 1]
            new_point = (l_weight * l_point[0] + r_weight * r_point[0], l_weight * l_point[1] + r_weight * r_point[1])
            break
        elif new_value == values[index]:
            new_point = points[index]
            break
        else:
            new_point = None
    if new_point is None:
        raise ValueError("Can't find interpolated point")
    return new_point
