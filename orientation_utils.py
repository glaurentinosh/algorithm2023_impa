def compare_ccw(point1, point2, origin):
    '''arr = np.array([
        [1,1,1],
        [point1[0], point2[0], origin[0]],
        [point1[1], point2[1], origin[1]]
    ])'''

    #det = np.linalg.det(arr)
    det = (point2[0]-point1[0])*(origin[1]-point1[1])-(origin[0]-point1[0])*(point2[1]-point1[1])
    if det == 0:
        dist1, dist2 = distance(point1, origin), distance(point2, origin)
        if dist1 == dist2:
            return 0
        return -1 if dist1 < dist2 else 1
    return 1 if det < 0 else -1

def distance(p1, p2):
    return ((p1[0] - p2[0]) * (p1[0] - p2[0]) +
            (p1[1] - p2[1]) * (p1[1] - p2[1]))