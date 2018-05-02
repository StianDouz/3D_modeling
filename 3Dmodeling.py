import math
import matplotlib.pyplot as plt
 
def new_matrix(rows, cols, fill_value):
        return [[ fill_value for j in range(cols)] for i in range(rows)]
 
def draw_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    plt.plot( [x1,x2], [y1,y2], "r-")
def draw_blue(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    plt.plot( [x1,x2], [y1,y2], "k-")
 
 
def mat_eye(A):
    rows, cols = len(A), len(A[0])
 
    print "Shape: ", rows,"x", cols
 
    for r in range(rows):
        for c in range(cols):
            print A[r][c],
        print
    return A
 
 
def mat_scale(sx, sy, sz):
 
    B = [[sx, 0, 0, 0], [0, sy, 0, 0], [0, 0, sz, 0], [0, 0, 0, 1]]
    return B
 
def mat_translate(tx, ty, tz):
 
    C = [[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]]
    return C
 
def mat_rotate_y (theta):
 
    D = [[int(math.cos(theta)), 0, int(math.sin(theta)), 0], [0, 1, 0, 0], [int(-math.sin(theta)), 0, int(math.cos(theta)), 0], [0, 0, 0, 1]]
    return D
 
def mat_projection(fovy, aspect, near, far):
 
    f = 1 / (math.tan(fovy/2))
    nf = 1.0 / (near - far)
 
    D = [[f / aspect , 0, 0, 0], [0, f, 0, 0], [0, 0, (near+far)*nf, -1], [0, 0, 2*far*near*nf, 0]]
    return D
 
def mat_vec_mul(M, v):
    Mv = [0.0, 0.0, 0.0, 0.0]
    for i in range(0,4):
        for j in range(0,4):
            Mv[i] += M[i][j] * v[j]
    return Mv
 
def mat_mat_mul(M1, M2):
    M = new_matrix(4,4,0)
 
    for k in range(0,4):
        for i in range(0,4):
            for j in range(0,4):
                M[i][j] += M1[i][k] * M2[k][j]
    return M
 
def vec_project(M,p):
    a = mat_vec_mul(M,p)
    X = a[0]
    Y = a[1]
    W = a[3]
    X /= W
    Y /= W
    return (X,Y)

P3d = [[0,0,0],[10,0,0],[10,0,10],[0,0,10],[0,10,0],[10,10,0],[10,10,10],[0,10,10],[1,12,1],[7,12,1],[7,12,9],[1,12,9],[2,15,2],[4,15,2],[4,15,8],[2,15,8]]
L = [(0,1),(0,3),(1,2),(2,3),(4,5),(4,7),(5,6),(6,7),(0,4),(1,5),(2,6),(3,7)]
L2 = [(8,9),(8,11),(9,10),(10,11),(4,8),(5,9),(6,10),(7,11)]
L3 = [(12,13),(12,15),(13,14),(14,15),(8,12),(9,13),(10,14),(11,15)]
 
P2d = []
Mv1 = mat_translate(-5, -8, -20)
Mv2 = mat_rotate_y(math.pi * (3/4))
Mv3 = mat_scale(1.5, 0.6, 1.1)
Mv4 = mat_mat_mul(Mv1, Mv2)
Mv = mat_mat_mul(Mv3, Mv4)
Mp = mat_projection(math.pi / 3.0, 1.0, 0.1, 10.0)
M = mat_mat_mul(Mp, Mv)
 
 
 
 
for point in P3d:
    point.append(1.0)
    P2d.append(vec_project(M,point))
 
for i,j in L:
    p1 = P2d[i]
    p2 = P2d[j]
    draw_line(p1,p2)
for i,j in L2:
    p1 = P2d[i]
    p2 = P2d[j]
    draw_blue(p1,p2)
for i,j in L3:
    p1 = P2d[i]
    p2 = P2d[j]
    draw_line(p1,p2)
 
plt.xlim(-15,15)
plt.ylim(-15,15)
plt.show()