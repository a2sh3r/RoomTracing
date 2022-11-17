import math
from tkinter import *
import numpy as np
from sympy import *
import random as random
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from MNK_trace import *

word = 'У'
# coord_mayak = [[10,10],[1270,10],[10,700],[1270,700]] # - для Д 4 маяка по углам
# coord_mayak = [[10,360],[1270,360],[640,10],[640,720]] # - для Д 4 маяка на серединах стен
# coord_mayak = [[10,10],[1270,10],[200,700]] #- для Д 3 маяка
# coord_mayak = [[10,10],[1270,700]] # - для ДУ 2 маяка по углам
coord_mayak = [[20,100],[100,20],[60,660],[1260,620],[1180,700],[1220,60]] # - для ДУ 6 маяка
# coord_mayak = [[10,360],[1270,360]] # - для ДУ 2 маяка по бокам
# coord_mayak = [[100,100],[1200,100],[100,600],[1200,600]] # - тест с Угломерным



def motion0(event):
    points = []
    visibility_bool = 0
    mouse_x = int(((canvas.winfo_pointerx() - canvas.winfo_rootx())//n)*n)
    mouse_y = int(((canvas.winfo_pointery() - canvas.winfo_rooty())//m)*m)
    event.widget.place(x=mouse_x, y=mouse_y, anchor=CENTER)
    coord_mayak[0] = [mouse_x + random.gauss(0,0.0001), mouse_y+ random.gauss(0,0.0001)]

def motion1(event):
    points = []
    visibility_bool = 0
    mouse_x = int(((canvas.winfo_pointerx() - canvas.winfo_rootx())//n)*n)
    mouse_y = int(((canvas.winfo_pointery() - canvas.winfo_rooty())//m)*m)
    event.widget.place(x=mouse_x, y=mouse_y, anchor=CENTER)
    coord_mayak[1] =[mouse_x + random.gauss(0,0.0001), mouse_y+ random.gauss(0,0.0001)]

def motion2(event):
    points = []
    visibility_bool = 0
    mouse_x = int(((canvas.winfo_pointerx() - canvas.winfo_rootx())//n)*n)
    mouse_y = int(((canvas.winfo_pointery() - canvas.winfo_rooty())//m)*m)
    event.widget.place(x=mouse_x, y=mouse_y, anchor=CENTER)
    coord_mayak[2] = [mouse_x + random.gauss(0,0.0001), mouse_y+ random.gauss(0,0.0001)]

def motion3(event):
    points = []
    visibility_bool = 0
    mouse_x = int(((canvas.winfo_pointerx() - canvas.winfo_rootx())//n)*n)
    mouse_y = int(((canvas.winfo_pointery() - canvas.winfo_rooty())//m)*m)
    event.widget.place(x=mouse_x, y=mouse_y, anchor=CENTER)
    coord_mayak[3] = [mouse_x + random.gauss(0,0.0001), mouse_y+ random.gauss(0,0.0001)]

def motion4(event):
    points = []
    visibility_bool = 0
    mouse_x = int(((canvas.winfo_pointerx() - canvas.winfo_rootx())//n)*n)
    mouse_y = int(((canvas.winfo_pointery() - canvas.winfo_rooty())//m)*m)
    event.widget.place(x=mouse_x, y=mouse_y, anchor=CENTER)
    coord_mayak[4] = [mouse_x + random.gauss(0,0.0001), mouse_y+ random.gauss(0,0.0001)]

def motion5(event):
    points = []
    visibility_bool = 0
    mouse_x = int(((canvas.winfo_pointerx() - canvas.winfo_rootx())//n)*n)
    mouse_y = int(((canvas.winfo_pointery() - canvas.winfo_rooty())//m)*m)
    event.widget.place(x=mouse_x, y=mouse_y, anchor=CENTER)
    coord_mayak[5] = [mouse_x + random.gauss(0,0.0001), mouse_y+ random.gauss(0,0.0001)]


motion = [motion0, motion1, motion2, motion3, motion4, motion5]


def change(event):
    print('Смена метода, выберите новый метод: ')
    print(("Д - дальномерный, УД - дальномерный-уголомерный, "))
    word = input("У - угломерный, РД - разностно-дальномерный, УРД - угломерно-разностнодальномерный: ")

def calculation(event):
    pointvisx = []
    pointvisy = []
    vidimost = []
    R_dal = []
    # заполнение векторов координат сетки, которые находятся в комнате
    for i in range(0, int(720 / n)):
        for j in range(0, int(1280 / m)):
            pointvisx.append(j * n)
            pointvisy.append(i * m)
            R_dal.append(0)
            vidimost.append(0)
            canvas_marker = Canvas(tk, width=5, height=5)
            # canvas_marker.create_rectangle(0, 0, 5, 5, fill='pink', width=0, outline='black')
            # canvas_marker.place(x=j*n, y=i*m,anchor=CENTER)

    bool_number_mayak = np.zeros((len(pointvisx), len(coord_mayak)))

    # цикл проверки каждой точкой видимости каждого маяка
    for i in range(len(coord_mayak)):
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
                if ((vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0)) or (
                        (vis_condition1 < 0) & (vis_condition2 < 0) & (vis_condition3 < 0)):
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
            if ((vis_condition1 >= 0) & (vis_condition2 >= 0) & (vis_condition3 >= 0)) or (
                    (vis_condition1 < 0) & (vis_condition2 < 0) & (vis_condition3 < 0)):
                visibility_bool = True

            if visibility_bool == True:
                vidimost[j] += 1
                bool_number_mayak[j][i] = 1
            if ((pointvisx[j] <= nw.ex) and (pointvisx[j] >= nw.sx) and (pointvisy[j] <= ew.ey) and (
                    pointvisy[j] >= ew.sy)):
                vidimost[j] = -1

    number_mayak = []
    for j in range(len(pointvisx)):
        number_mayak.append([])
        add_mayak = []
        for i in range(len(coord_mayak)):
            if bool_number_mayak[j][i] == 1:
                add_mayak.append(i)
        number_mayak[j] = add_mayak



    coord_mayak_2 = []

    geom_f = []


    print(min_number_mayak)
    for i in range(len(pointvisx)):
        coord_mayak_2.append([])
        for j in number_mayak[i]:
            coord_mayak_2[i].append(coord_mayak[j])

        if word == 'Д':
            if vidimost[i] >= min_number_mayak:
                # R_dal[i] = fx_dal_ideal([pointnecx[i],pointnecy[i]], coord_mayak_2[i])
                Hx_dots = Hx_dal([pointvisx[i], pointvisy[i]], coord_mayak_2[i])
                multi = np.dot(Hx_dots.T, Hx_dots)
                print(i, ' матрица, определитель: ', np.linalg.det(multi))
                inv_Hx = np.linalg.inv(multi)
                geom_f.append(sqrt(np.trace(inv_Hx)))
            else:
                geom_f.append(0)

        if word == 'ПД':
            if vidimost[i] >= min_number_mayak:
                # R_dal[i] = fx_dal_ideal([pointnecx[i],pointnecy[i]], coord_mayak_2[i])
                Hx_dots = Hx_pseudo_dal([pointvisx[i], pointvisy[i]], coord_mayak_2[i])
                multi = np.dot(Hx_dots.T, Hx_dots)
                print(i, ' матрица, определитель: ', np.linalg.det(multi))
                inv_Hx = np.linalg.inv(multi)
                geom_f.append(sqrt(np.trace(inv_Hx)))
            else:
                geom_f.append(0)

        if word == 'РД':
            if vidimost[i] >= min_number_mayak:
                # R_dal[i] = fx_dal_ideal([pointnecx[i],pointnecy[i]], coord_mayak_2[i])
                Hx_dots = Hx_dal_difference([pointvisx[i], pointvisy[i]], coord_mayak_2[i])
                multi = np.dot(Hx_dots.T, Hx_dots)
                print(i, ' матрица, определитель: ', np.linalg.det(multi))
                inv_Hx = np.linalg.inv(multi)
                geom_f.append(sqrt(np.trace(inv_Hx)))
            else:
                geom_f.append(0)

        if word == 'УРД':
            if vidimost[i] >= min_number_mayak:
                # R_dal[i] = fx_dal_ideal([pointnecx[i],pointnecy[i]], coord_mayak_2[i])
                Hx_dots = Hx_angle_dal_difference([pointvisx[i], pointvisy[i]], coord_mayak_2[i])
                multi = np.dot(Hx_dots.T, Hx_dots)
                print(i, ' матрица, определитель: ', np.linalg.det(multi))
                inv_Hx = np.linalg.inv(multi)
                geom_f.append(sqrt(np.trace(inv_Hx)))
            else:
                geom_f.append(0)

        if word == 'У':
            if vidimost[i] >= min_number_mayak:
                # R_dal[i] = fx_angle([pointnecx[i],pointnecy[i]], coord_mayak_2[i])
                Hx_dots = Hx_angle([pointvisx[i], pointvisy[i]], coord_mayak_2[i])
                multi = np.dot(Hx_dots.T, Hx_dots)
                print(i, ' матрица, определитель: ', np.linalg.det(multi))
                inv_Hx = np.linalg.inv(multi)
                geom_f.append(sqrt(np.trace(inv_Hx)))
            else:
                geom_f.append(0)

        if word == 'УД':
            if vidimost[i] >= min_number_mayak:
                # TODO Разобраться в сингулярной матрице, определители порядка 10^-16

                # R_dal[i] = fx_angle_dal_ideal([pointnecx[i],pointnecy[i]], coord_mayak_2[i])
                Hx_dots = Hx_angle_dal([pointvisx[i], pointvisy[i]], coord_mayak_2[i])
                multi = np.dot(Hx_dots.T, Hx_dots)
                print(i, ' матрица, определитель: ', np.linalg.det(multi))
                inv_Hx = np.linalg.inv(multi)
                geom_f.append(sqrt(np.trace(inv_Hx)))
            else:
                geom_f.append(0)

    for j in range(len(pointvisx)):
        print(str(j), ' - ', vidimost[j], 'X: ', pointvisx[j], 'Y: ', pointvisy[j], 'Number_Mayak: ', number_mayak[j],
              'Geom_Factor: ', geom_f[j])
        if vidimost[j] == 0:
            canvas_vidim = Canvas(tk, width=10, height=10)
            canvas_vidim.create_rectangle(0, 0, 10, 10, fill='white')
            canvas_vidim.place(x=pointvisx[j], y=pointvisy[j], anchor=CENTER)
        if vidimost[j] == 1:
            canvas_vidim = Canvas(tk, width=10, height=10)
            canvas_vidim.create_rectangle(0, 0, 10, 10, fill='#8B0000')
            canvas_vidim.place(x=pointvisx[j], y=pointvisy[j], anchor=CENTER)
        if vidimost[j] == 2:
            canvas_vidim = Canvas(tk, width=10, height=10)
            canvas_vidim.create_rectangle(0, 0, 10, 10, fill='#FFFF00')
            canvas_vidim.place(x=pointvisx[j], y=pointvisy[j], anchor=CENTER)
        if vidimost[j] == 3:
            canvas_vidim = Canvas(tk, width=10, height=10)
            canvas_vidim.create_rectangle(0, 0, 10, 10, fill='#006400')
            canvas_vidim.place(x=pointvisx[j], y=pointvisy[j], anchor=CENTER)
        if vidimost[j] == 4:
            canvas_vidim = Canvas(tk, width=10, height=10)
            canvas_vidim.create_rectangle(0, 0, 10, 10, fill='#4B0082')
            canvas_vidim.place(x=pointvisx[j], y=pointvisy[j], anchor=CENTER)
        if vidimost[j] == 5:
            canvas_vidim = Canvas(tk, width=10, height=10)
            canvas_vidim.create_rectangle(0, 0, 10, 10, fill='#FF1493')
            canvas_vidim.place(x=pointvisx[j], y=pointvisy[j], anchor=CENTER)
        if vidimost[j] == 6:
            canvas_vidim = Canvas(tk, width=10, height=10)
            canvas_vidim.create_rectangle(0, 0, 10, 10, fill='#2F4F4F')
            canvas_vidim.place(x=pointvisx[j], y=pointvisy[j], anchor=CENTER)
        # if vidimost[j] == q:
        #     canvas_vidim = Canvas(tk, width=10, height=10)
        #     canvas_vidim.create_rectangle(0, 0, 10, 10, fill='green')
        #     canvas_vidim.place(x=pointvisx[j], y=pointvisy[j],anchor=CENTER)
        #     # pointnecx.append(pointvisx)
        #     # pointnecy.append(pointvisy)
    # print(R_dal[40])
    # print(R_dal[160])
    sum_Z = 0
    k = 0
    for i in range(len(geom_f)):

        if geom_f[i] != 0:
            sum_Z += geom_f[i]
            k += 1
        if geom_f[i] > 30:
            geom_f[i] = 30
    canvas_marker = []
    for i in range(len(coord_mayak)):

        canvas_marker.append(0)

        canvas_marker[i] = Canvas(tk, width=15, height=15)
        canvas_marker[i].create_rectangle(1, 1, 15, 15, fill='black', width=1, outline='red')
        canvas_marker[i].place(x=coord_mayak[i][0], y=coord_mayak[i][1], anchor=CENTER)
        canvas_marker[i].bind('<B1-Motion>', motion[i])


    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection="3d")
    my_cmap = plt.get_cmap('magma')

    pointvisx = np.array(pointvisx)
    pointvisy = np.array(pointvisy)
    Z = np.array(geom_f).astype('float')

    mean_Z = sum_Z / k
    print('Был использован', word, 'метод')
    print('Средний геометрический фактор по комнате: ', mean_Z)
    print('Для перемещения маяков используйте ЛКМ')
    print('Для расчета нажмите СКМ')
    print('Для выбора нового метода позиционирования нажмите ПКМ')
    ax.plot_trisurf(pointvisx, pointvisy, Z, cmap=my_cmap, edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('DOP')
    plt.show()


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


for i in range(len(coord_mayak)):
    for j in range(len(coord_mayak[0][:])):
        coord_mayak[i][j] += 0.002

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


#отрисовка сетки
for i in range(0, 720, n):
    canvas.create_line(0, i, 1280, i)
for i in range(0, 1280, m):
    canvas.create_line(i, 0, i, 720)

print(("Выберите метод расчета: Д - дальномерный, УД - дальномерный-уголомерный, "))
print("У - угломерный, РД - разностно-дальномерный, УРД - угломерно-разностнодальномерный: ")
word = input()


if word == 'РД' or word == 'ПД':
    min_number_mayak = 3
if word == 'Д' or word == 'У'or word == 'УРД':
    min_number_mayak = 2 #минимальное кол-во маяков, необходимое для определения координаты
if word == 'УД':
    min_number_mayak = 1 #минимальное кол-во маяков, необходимое для определения координаты



canvas_marker = []
i = 0
while i < len(coord_mayak):
    canvas_marker.append(0)
    canvas_marker[i] = Canvas(tk, width=15, height=15)
    canvas_marker[i].create_rectangle(1, 1, 15, 15, fill='black', width=1, outline='red')

    canvas_marker[i].place(x=coord_mayak[i][0], y=coord_mayak[i][1], anchor=CENTER)
    canvas_marker[i].bind('<B1-Motion>', motion[i])

    i += 1







print('Для перемещения маяков используйте ЛКМ')
print('Для расчета нажмите СКМ')
print('Для выбора нового метода позиционирования нажмите ПКМ')

canvas.bind('<Button-2>', calculation)
canvas.bind('<Button-3>', change)

canvas.pack()
tk.mainloop()

