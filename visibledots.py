import math
from tkinter import *
import numpy as np
from sympy import *
import random as random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def fx_dal(x0,x_sat):
    '''
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    r= np.zeros((len(list(x_sat))))
    for i in range(len(list(x_sat))):
      r[i]=np.sqrt((x0[0]-x_sat[i][0])**2+(x0[1]-x_sat[i][1])**2)
    return r
def Hx_dal(x_prev,x_sat):
    '''
    Градиентная матрица
    :param x_prev:
    :param x_sat:
    :return:
    '''
    #ДАЛЬНОМЕРНЫЙ
    H = np.zeros((len(list(x_sat)), 2))
    for i in range(len(list(x_sat))):
        H[i][0] = -(x_sat[i][0] - x_prev[0]) / np.sqrt((x_prev[0]-x_sat[i][0])**2+(x_prev[1]-x_sat[i][1])**2)
        H[i][1] = -(x_sat[i][1] - x_prev[1]) / np.sqrt((x_prev[0]-x_sat[i][0])**2+(x_prev[1]-x_sat[i][1])**2)

    return H

def descent_process(R,x_sat,x_prev, Hx, fxx):
    '''
    Итерация градиентного спуска (процесс)
    :param R: наблюдения
    :param x_sat: координаты спутников (якорей)
    :param x_prev: оценка координат объекта на k-1 итерации
    :return: оценка координат объекта на k итерации
    '''


    a = np.linalg.inv(np.dot(Hx(x_prev,x_sat).T, Hx(x_prev,x_sat)))
    b = Hx(x_prev,x_sat).T
    c  = R - fxx(x_prev,x_sat)
    x_new = x_prev + np.dot(a,np.dot(b,c))
    return x_new

def Gradient_descent(R,x_sat, Hx, fx, x_prev=np.array([0.1,0.1]),epsilon=0.1):
    '''
    Алгоритм градиентного спуска
    :param R: наблюдения
    :param x_sat: координаты спутников (якорей)
    :param x_prev: начальные условия по координатам
    :param epsilon: критерий остановы
    :param show_plots: показать графики
    :return: оценка координат
    '''
    history = []
    i = 1
    while True:
        x_new = descent_process(R,x_sat,x_prev, Hx, fx)
        #print(f'x_est объекта на {i} итерации равен {x_new}')
        Euc =  np.sqrt((x_new[0]-x_prev[0])**2+(x_new[1]-x_prev[1])**2)
        history.append(Euc)
        if Euc<=epsilon:
            #print('\nОстановка расчета!')
            winner= x_new
            break
        i+=1
        x_prev = x_new.copy()

    return winner
class Edges:
    edges_count = 0

    def __init__(self, sx, sy, ex, ey):
        self.sx = sx
        self.sy = sy
        self.ex = ex
        self.ey = ey
        Edges.edges_count += 1


class VisibleLines:
    lines_count = 0

    def __init__(self, line_angle, x, y):
        self.line_angle = line_angle
        self.x = x
        self.y = y
        VisibleLines.lines_count += 1


def mmotion(event):
    x, y = (event.x//n)*n, (event.y//m)*m
    x1, y1 = None, None
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1, width=5, fill='black')
    canvas.old_coords = x, y
    if (x1):
        edge_lines.append(Edges(x, y, x1, y1))
        canvas.old_coords = None


def edges_print(event):
    for i in edge_lines:
        print(i.sx, i.sy, i.ex, i.ey)


def drag(event):
    points = []
    visibility_bool = 0
    mouse_x = ((canvas.winfo_pointerx() - canvas.winfo_rootx())//n)*n
    mouse_y = ((canvas.winfo_pointery() - canvas.winfo_rooty())//m)*m
    marker_coordinates = [mouse_x, mouse_y]
    event.widget.place(x=mouse_x, y=mouse_y, anchor=CENTER)
    calculating_visibility(marker_coordinates[0], marker_coordinates[1], 300, edge_lines, points)
    points = sorted(points, key=lambda x: x.line_angle)
    if (triangles):
        for i in range(len(triangles)):
            canvas.delete(triangles[i])
    triangles.clear()
    for i in range(len(points) - 1):
        print(points[i].line_angle, points[i].x, points[i].y, points[i + 1].line_angle, points[i + 1].x,
              points[i + 1].y)
        triangle = canvas.create_polygon(marker_coordinates[0], marker_coordinates[1], points[i].x, points[i].y,
                                         points[i + 1].x, points[i + 1].y, width=1,
                                         fill='green')
        vis_condition1 = (points[i].x - target_coordinates[0]) * (points[i + 1].y - points[i].y) - \
                         (points[i + 1].x - points[i].x) * (points[i].y - target_coordinates[1])
        vis_condition2 = (points[i + 1].x - target_coordinates[0]) * (marker_coordinates[1] - points[i + 1].y) - \
                         (marker_coordinates[0] - points[i + 1].x) * (points[i + 1].y - target_coordinates[1])
        vis_condition3 = (marker_coordinates[0] - target_coordinates[0]) * (points[i].y - marker_coordinates[1]) - \
                         (points[i].x - marker_coordinates[0]) * (marker_coordinates[1] - target_coordinates[1])
        if (vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0):
            visibility_bool = visibility_bool + 1
        triangles.append(triangle)
    triangle = canvas.create_polygon(marker_coordinates[0], marker_coordinates[1], points[len(points) - 1].x,
                                     points[len(points) - 1].y, points[0].x, points[0].y, width=1,
                                     fill='green')
    vis_condition1 = (points[len(points) - 1].x - target_coordinates[0]) * (points[0].y - points[len(points) - 1].y) - \
                     (points[0].x - points[len(points) - 1].x) * (points[len(points) - 1].y - target_coordinates[1])
    vis_condition2 = (points[0].x - target_coordinates[0]) * (marker_coordinates[1] - points[0].y) - \
                     (marker_coordinates[0] - points[0].x) * (points[0].y - target_coordinates[1])
    vis_condition3 = (marker_coordinates[0] - target_coordinates[0]) * (
                      points[len(points) - 1].y - marker_coordinates[1]) - \
                     (points[len(points) - 1].x - marker_coordinates[0]) * (
                                 marker_coordinates[1] - target_coordinates[1])
    if (vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0):
        visibility_bool = visibility_bool + 1
    triangles.append(triangle)
    if visibility_bool > 0:
        canvas_target.create_rectangle(0, 0, 10, 10, fill='green', width=0, outline='green')
    else:
        canvas_target.create_rectangle(0, 0, 10, 10, fill='red', width=0, outline='red')
    visibility_bool = 0
    # for i in range(0, 720, n):
    #     canvas.create_line(0, i, 1280, i)
    # for i in range(0, 1280, m):
    #     canvas.create_line(i, 0, i, 720)


def drag2(event):
    mouse_x = ((canvas.winfo_pointerx() - canvas.winfo_rootx()) // n) * n
    mouse_y = ((canvas.winfo_pointery() - canvas.winfo_rooty()) // m) * m
    global target_coordinates
    target_coordinates = [mouse_x, mouse_y]
    event.widget.place(x=mouse_x, y=mouse_y, anchor=CENTER)


def calculating_visibility(ox, oy, radius, edge_lines, points):
    points.clear()
    for i in edge_lines:
        for j in range(2):
            if j == 0:
                rdx = float(i.sx - ox)
                rdy = float(i.sy - oy)
            else:
                rdx = float(i.ex - ox)
                rdy = float(i.ey - oy)
            base_ang = float((math.atan2(rdy, rdx)))
            ang = 0
            for k in range(3):
                # setup angles
                if k == 0:
                    ang = base_ang - 0.01
                if k == 1:
                    ang = base_ang
                if k == 2:
                    ang = base_ang + 0.01
                # creating ray
                rdx = float(radius * math.cos(ang))
                rdy = float(radius * math.sin(ang))
                min_t1 = float(1000000000000000000000000)
                min_px = float(0)
                min_py = float(0)
                min_ang = float(0)
                # checking ray intersection
                for e in edge_lines:
                    sdx = float(e.ex - e.sx)
                    sdy = float(e.ey - e.sy)
                    if (math.fabs((sdx - rdx)) > 0) & (math.fabs((sdy-rdy)) > 0) & ((sdx * rdy - sdy * rdx) != 0):
                        t2 = float((rdx+0.0001) * (e.sy - oy) + (rdy * (ox - e.sx))) / (sdx * rdy - sdy * (rdx+0.0001))
                        t1 = float((e.sx + sdx * t2 - ox) / (rdx+0.0001))
                        if (t1 > 0) & (t2 >= 0) & (t2 <= 1):
                            if t1 < min_t1:
                                min_t1 = float(t1)
                                min_px = float(ox + (rdx+0.0001) * t1)
                                min_py = float(oy + rdy * t1)
                                min_ang = math.degrees(float(math.atan2(min_py-oy, min_px-ox)))
                line = VisibleLines(min_ang, min_px, min_py)
                points.append(line)
    points = sorted(points, key=lambda x: x.line_angle)


# Tkinter setup
tk = Tk()
tk.title('Mark Casting')
tk.geometry('1280x720')
tk.resizable(False, False)

n = 20
m = 20

# 1 in code == 1 sm
ln = [0, 0, 1280, 0]     # NORTH LINE
lw = [0, 0, 0, 720]  # WEST LINE
ls = [0, 720, 1280, 720]   # SOUTH LINE
le = [1280, 720, 1280, 0]  # EAST LINE

# Generating simple room
canvas = Canvas(tk, bg='white', width=1280, height=720)

canvas.old_coords = None

canvas.create_line((ln[0]), (ln[1]), (ln[2]), (ln[3]), width=3, fill='black')
canvas.create_line((lw[0]), (lw[1]), (lw[2]), (lw[3]), width=3, fill='black')
canvas.create_line((ls[0]), (ls[1]), (ls[2]), (ls[3]), width=3, fill='black')
canvas.create_line(le[0], le[1], (le[2]), (le[3]), width=3, fill='black')

# Marker setups
coord_mayak = [[10,10],[1270,10],[10,700],[1270,700]]



marker_coordinates = [200-5, 220-5]

# canvas_marker.place(x=coord_mayak[0][0], y=coord_mayak[0][1])
# canvas_marker1.place(x=coord_mayak[1][0], y=coord_mayak[1][1])
# canvas_marker2.place(x=coord_mayak[2][0], y=coord_mayak[2][1])
# canvas_marker3.place(x=coord_mayak[3][0], y=coord_mayak[3][1])


# Target Setup
# canvas_target.place(x=970-5, y=310-5)
target_coordinates = [970-5, 310-5]

# canvas_marker.bind('<B1-Motion>', drag)
# canvas_marker1.bind('<B1-Motion>', drag)
# canvas_marker2.bind('<B1-Motion>', drag)
# canvas_marker3.bind('<B1-Motion>', drag)
# canvas_target.bind('<B1-Motion>', drag2)
# canvas.bind('<Button-2>', mmotion)
# canvas.bind('<Button-3>', edges_print)

#координаты коробки в центре и её отрисовка
nw = Edges(480, 500, 800, 500)
ww = Edges(480, 220, 480, 500)
sw = Edges(480, 220, 800, 220)
ew = Edges(800, 220, 800, 500)
canvas.create_line((nw.sx), (nw.sy), (nw.ex), (nw.ey), width=3, fill='black')
canvas.create_line((ww.sx), (ww.sy), (ww.ex), (ww.ey), width=3, fill='black')
canvas.create_line((sw.sx), (sw.sy), (sw.ex), (sw.ey), width=3, fill='black')
canvas.create_line((ew.sx), (ew.sy), (ew.ex), (ew.ey), width=3, fill='black')

ne = Edges(-1, -1, 1281, -1)
we = Edges(-1, -1, -1, 721)
se = Edges(-1, 721, 1281, 721)
ee = Edges(1281, 721, 1281, -1)

edge_lines = [ne, we, se, ee, nw, ww, sw, ew]
points = []

calculating_visibility(marker_coordinates[0], marker_coordinates[1], 300, edge_lines, points)

points = sorted(points, key=lambda x: x.line_angle)

# target_angle = math.atan2(target_coordinates[1]-marker_coordinates[1], target_coordinates[0]-marker_coordinates[0])
# print(target_angle)



triangles = []
visibility_bool = 0
# for i in range(len(points) - 1):
#     # print(points[i].line_angle, points[i].x, points[i].y, points[i+1].line_angle, points[i+1].x, points[i+1].y)
#     triangle = canvas.create_polygon(marker_coordinates[0], marker_coordinates[1], points[i].x, points[i].y,
#                                      points[i + 1].x, points[i + 1].y, width=1,
#                                      fill='green')
#     vis_condition1 = (points[i].x - target_coordinates[0]) * (points[i+1].y - points[i].y) - \
#                      (points[i+1].x - points[i].x) * (points[i].y - target_coordinates[1])
#     vis_condition2 = (points[i+1].x - target_coordinates[0]) * (marker_coordinates[1] - points[i+1].y) - \
#                      (marker_coordinates[0] - points[i+1].x) * (points[i+1].y - target_coordinates[1])
#     vis_condition3 = (marker_coordinates[0] - target_coordinates[0]) * (points[i].y - marker_coordinates[1]) - \
#                      (points[i].x - marker_coordinates[0]) * (marker_coordinates[1] - target_coordinates[1])
#     if (vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0):
#         visibility_bool = visibility_bool + 1
#     triangles.append(triangle)
# triangle = canvas.create_polygon(marker_coordinates[0], marker_coordinates[1], points[len(points) - 1].x,
#                                  points[len(points) - 1].y, points[0].x, points[0].y, width=1,
#                                  fill='green')
# vis_condition1 = (points[len(points) - 1].x - target_coordinates[0]) * (points[0].y - points[len(points) - 1].y) - \
#              (points[0].x - points[len(points) - 1].x) * (points[len(points) - 1].y - target_coordinates[1])
# vis_condition2 = (points[0].x - target_coordinates[0]) * (marker_coordinates[1] - points[0].y) - \
#                  (marker_coordinates[0] - points[0].x) * (points[0].y - target_coordinates[1])
# vis_condition3 = (marker_coordinates[0] - target_coordinates[0]) * (points[len(points) - 1].y - marker_coordinates[1]) - \
#                  (points[len(points) - 1].x - marker_coordinates[0]) * (marker_coordinates[1] - target_coordinates[1])
# if (vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0):
#     visibility_bool = visibility_bool + 1
# triangles.append(triangle)
#
# if visibility_bool > 0:
#     canvas_target.create_rectangle(0, 0, 10, 10, fill='green', width=0, outline='green')
# else:
#     canvas_target.create_rectangle(0, 0, 10, 10, fill='red', width=0, outline='red')
#
# visibility_bool = 0
#     # triangle = canvas.create_line(marker_coordinates[0], marker_coordinates[1], points[i].x, points[i].y,
#     #                               width=1,
#     #                               fill='green')
#     # triangle = canvas.create_line(marker_coordinates[0], marker_coordinates[1], points[i + 1].x, points[i + 1].y,
#     #                               width=1,
#     #                               fill='green')



for i in range(0, 720, n):
    canvas.create_line(0, i, 1280, i)
for i in range(0, 1280, m):
    canvas.create_line(i, 0, i, 720)

pointvisx =[]
pointvisy =[]
vidimost = []
R_dal = []
for i in range(0,int(720/n)):
    for j in range(0,int(1280/m)):
        if not((j*n<=nw.ex) and (j*n>=nw.sx) and (i*m<=ew.ey) and (i*m>=ew.sy)):
            pointvisx.append(j*n)
            pointvisy.append(i*m)
            R_dal.append(0)
            vidimost.append(0)
            canvas_marker = Canvas(tk, width=5, height=5)
            # canvas_marker.create_rectangle(0, 0, 5, 5, fill='pink', width=0, outline='black')
            # canvas_marker.place(x=j*n, y=i*m,anchor=CENTER)
bool_number_mayak = np.zeros((len(pointvisx),4))
for i in range(4):
    points = []
    calculating_visibility(coord_mayak[i][0], coord_mayak[i][1], 300, edge_lines, points)
    points = sorted(points, key=lambda x: x.line_angle)
    for j in range(len(pointvisx)):
        visibility_bool = False
        for g in range(len(points) - 1):
            # triangle = canvas.create_line(coord_mayak[i][0], coord_mayak[i][1], points[g].x, points[g].y,
            #                                   width=1,
            #                                   fill='green')

            vis_condition1 = (points[g].x - pointvisx[j]) * (points[g + 1].y - points[g].y) - \
                             (points[g + 1].x - points[g].x) * (points[g].y - pointvisy[j])
            vis_condition2 = (points[g + 1].x - pointvisx[j]) * (coord_mayak[i][1] - points[g + 1].y) - \
                             (coord_mayak[i][0] - points[g + 1].x) * (points[g + 1].y - pointvisy[j])
            vis_condition3 = (coord_mayak[i][0] - pointvisx[j]) * (points[g].y - coord_mayak[i][1]) - \
                             (points[g].x - coord_mayak[i][0]) * (coord_mayak[i][1] - pointvisy[j])
            if ((vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0)) or ((vis_condition1 < 0) & (vis_condition2 < 0) & (vis_condition3 < 0)):
                visibility_bool = True
        vis_condition1 = (points[len(points) - 1].x - pointvisx[j]) * (
                                points[0].y - points[len(points) - 1].y) - \
                                     (points[0].x - points[len(points) - 1].x) * (points[len(points) - 1].y - pointvisy[j])
        vis_condition2 = (points[0].x - pointvisx[j]) * (coord_mayak[i][1] - points[0].y) - \
                         (coord_mayak[i][0] - points[0].x) * (points[0].y - pointvisy[j])
        vis_condition3 = (coord_mayak[i][0] - pointvisx[j]) * (
                points[len(points) - 1].y - coord_mayak[i][1]) - \
                         (points[len(points) - 1].x - coord_mayak[i][0]) * (
                                 coord_mayak[i][1] - pointvisy[j])
        if ((vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0)) or ((vis_condition1 < 0) & (vis_condition2 < 0) & (vis_condition3 < 0)):
            visibility_bool = True



        if visibility_bool == True:
            vidimost[j] += 1
            bool_number_mayak[j][i] = 1

number_mayak = []
for j in range(len(pointvisx)):
    number_mayak.append([])
    add_mayak = []
    for i in range(4):
        if bool_number_mayak[j][i] == 1:
            add_mayak.append(i)
    number_mayak[j] = add_mayak

# теперь number_mayak содержит в j элементе какие маяки видят j-тую точку комнаты


 # vis_condition1 = (points[i].x - target_coordinates[0]) * (points[i+1].y - points[i].y) - \
#                      (points[i+1].x - points[i].x) * (points[i].y - target_coordinates[1])
#     vis_condition2 = (points[i+1].x - target_coordinates[0]) * (marker_coordinates[1] - points[i+1].y) - \
#                      (marker_coordinates[0] - points[i+1].x) * (points[i+1].y - target_coordinates[1])
#     vis_condition3 = (marker_coordinates[0] - target_coordinates[0]) * (points[i].y - marker_coordinates[1]) - \
#                      (points[i].x - marker_coordinates[0]) * (marker_coordinates[1] - target_coordinates[1])

# for i in range(1):
#     points = []
#     calculating_visibility(coord_mayak[i][0], coord_mayak[i][1], 300, edge_lines, points)
#     points = sorted(points, key=lambda x: x.line_angle)
#     for j in range(len(pointvisx)):
#         visibility_bool = 0
#         vis_condition1 = (points[len(points) - 1].x - pointvisx[j]) * (
#                     points[0].y - points[len(points) - 1].y) - \
#                          (points[0].x - points[len(points) - 1].x) * (points[len(points) - 1].y - pointvisy[j])
#         vis_condition2 = (points[0].x - pointvisx[j]) * (coord_mayak[i][1] - points[0].y) - \
#                          (coord_mayak[i][0] - points[0].x) * (points[0].y - pointvisy[j])
#         vis_condition3 = (coord_mayak[i][0] - pointvisx[j]) * (
#                 points[len(points) - 1].y - coord_mayak[i][1]) - \
#                          (points[len(points) - 1].x - coord_mayak[i][0]) * (
#                                  coord_mayak[i][1] - pointvisy[j])
#         if ((vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0)) or ((vis_condition1 < 0) & (vis_condition2 < 0) & (vis_condition3 < 0)):
#             visibility_bool = visibility_bool + 1
#         if visibility_bool > 0:
#             vidimost[j] += 1
q = 3 #необходимое кол-во видимых маяков
pointnecx = [] #координаты точек, видимых q кол-вом маяков
pointnecy = []

coord_mayak_2 = []
winner_1 = []
Hx_dots = []
geom_f = []
for i in range(len(pointvisx)):
    coord_mayak_2.append([])
    for j in number_mayak[i]:
        coord_mayak_2[i].append(coord_mayak[j])

    R_dal[i] = fx_dal([pointvisx[i],pointvisy[i]], coord_mayak_2[i])
    Hx_dots = Hx_dal([pointvisx[i],pointvisy[i]], coord_mayak_2[i])
    multi = np.dot(Hx_dots.T,Hx_dots)
    inv_Hx = np.linalg.inv(multi)
    geom_f.append(sqrt(np.trace(inv_Hx)))

for j in range(len(pointvisx)):
    print(str(j), ' - ', vidimost[j], 'X: ',pointvisx[j],'Y: ', pointvisy[j], 'Number_Mayak: ',number_mayak[j],'Geom_Factor: ', geom_f[j] )
    if vidimost[j] == 1:
        canvas_vidim = Canvas(tk, width=10, height=10)
        canvas_vidim.create_rectangle(0, 0, 10, 10, fill='red')
        canvas_vidim.place(x=pointvisx[j], y=pointvisy[j],anchor=CENTER)
    if vidimost[j] == 2:
        canvas_vidim = Canvas(tk, width=10, height=10)
        canvas_vidim.create_rectangle(0, 0, 10, 10, fill='yellow')
        canvas_vidim.place(x=pointvisx[j], y=pointvisy[j],anchor=CENTER)
    if vidimost[j] == 3:
        canvas_vidim = Canvas(tk, width=10, height=10)
        canvas_vidim.create_rectangle(0, 0, 10, 10, fill='green')
        canvas_vidim.place(x=pointvisx[j], y=pointvisy[j],anchor=CENTER)
    if vidimost[j] == 4:
        canvas_vidim = Canvas(tk, width=10, height=10)
        canvas_vidim.create_rectangle(0, 0, 10, 10, fill='purple')
        canvas_vidim.place(x=pointvisx[j], y=pointvisy[j],anchor=CENTER)
    # if vidimost[j] == q:
    #     canvas_vidim = Canvas(tk, width=10, height=10)
    #     canvas_vidim.create_rectangle(0, 0, 10, 10, fill='green')
    #     canvas_vidim.place(x=pointvisx[j], y=pointvisy[j],anchor=CENTER)
    #     # pointnecx.append(pointvisx)
    #     # pointnecy.append(pointvisy)
# print(R_dal[40])
# print(R_dal[160])

canvas_marker = Canvas(tk, width=15, height=15)
canvas_marker.create_rectangle(1, 1,15,15, fill='black', width=1, outline='red')
canvas_marker1 = Canvas(tk, width=15, height=15)
canvas_marker1.create_rectangle(1, 1, 15, 15, fill='black', width=1, outline='red')
canvas_marker2 = Canvas(tk, width=15, height=15)
canvas_marker2.create_rectangle(1, 1, 15, 15, fill='black', width=1, outline='red')
canvas_marker3 = Canvas(tk, width=15, height=15)
canvas_marker3.create_rectangle(1, 1, 15, 15, fill='black', width=1, outline='red')
canvas_target = Canvas(tk, width=15, height=15)
canvas_target.create_rectangle(0, 0, 10, 10, fill='gray', width=0, outline='gray')

canvas_marker.place(x=coord_mayak[0][0], y=coord_mayak[0][1],anchor= CENTER)
canvas_marker1.place(x=coord_mayak[1][0], y=coord_mayak[1][1],anchor= CENTER)
canvas_marker2.place(x=coord_mayak[2][0], y=coord_mayak[2][1],anchor= CENTER)
canvas_marker3.place(x=coord_mayak[3][0], y=coord_mayak[3][1],anchor= CENTER)

canvas.pack()

fig = plt.figure(figsize = (12,8))
ax = plt.axes(projection = "3d")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Г')

ax.scatter(pointvisx, pointvisy, geom_f, s=1)

plt.show()

tk.mainloop()

