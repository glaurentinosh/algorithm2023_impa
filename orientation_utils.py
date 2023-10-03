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

def determinant(point1, point2, origin):
    return (point2[0]-point1[0])*(origin[1]-point1[1])-(origin[0]-point1[0])*(point2[1]-point1[1])

def baricentricCoord(point, triangle):
    t0, t1, t2 = triangle[0], triangle[1], triangle[2]
    det0 = determinant(t0,t1,t2)
    det1 = determinant(t0,t1,point)
    det2 = determinant(t1,t2,point)
    det3 = determinant(t2,t0,point)
    
    if det0 > 0:
        return det1>=0 and det2>=0 and det3>=0
    if det0 < 0:
        return det1<=0 and det2<=0 and det3<=0
    if det0 == 0:
        d0 = max([distance(t0,t1), distance(t1,t2), distance(t2,t0)])
        d1 = max([distance(t0,point), distance(t1,point), distance(t2,point)])
        return d1<=d0
    
def baricentricCoordNew(point, triangle):
    t0, t1, t2 = triangle[0], triangle[1], triangle[2]
    det0 = determinant(t0,t1,t2)
    det1 = determinant(t0,t1,point)
    det2 = determinant(t1,t2,point)
    det3 = determinant(t2,t0,point)
    
    has_neg = (det1 < 0) or (det2 < 0) or (det3 < 0)
    has_pos = (det1 > 0) or (det2 > 0) or (det3 > 0)

    if det0 != 0:
        return not (has_neg and has_pos)
    if det0 == 0:
        d0 = max([distance(t0,t1), distance(t1,t2), distance(t2,t0)])
        d1 = max([distance(t0,point), distance(t1,point), distance(t2,point)])
        return d1<=d0    