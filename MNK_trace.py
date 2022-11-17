import math

import numpy as np
import matplotlib.pyplot as plt
from math import *
from sympy import *
import random as random
from sklearn.preprocessing import StandardScaler

import matplotlib.path as pltPath
import plotly.graph_objects as go
import traceback


def get_intersections(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    if d > r0 + r1:
        return None
    # One circle within other
    if d < abs(r0 - r1):
        return None
    # coincident circles
    if d == 0 and r0 == r1:
        return None
    else:
        a = (r0 ** 2 - r1 ** 2 + d ** 2) / (2 * d)
        h = sqrt(r0 ** 2 - a ** 2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = float(x2 + h * (y1 - y0) / d)
        y3 = float(y2 - h * (x1 - x0) / d)

        x4 = float(x2 - h * (y1 - y0) / d)
        y4 = float(y2 + h * (x1 - x0) / d)

        return [x3, y3, x4, y4]


def trace_hyperbole(m1, m2, target, Targetsxu, Targetsyu, Targetsxd, Targetsyd):
    d = float((sqrt(((target[0] - m1[0])**2) + (target[1]-m1[1])**2) - sqrt(((target[0] - m2[0])**2) + (target[1]-m2[1])**2)) + 0.3)
    a = d/2
    l = float(sqrt(((m2[0] - m1[0])**2) + (m2[1]-m1[1])**2))
    c = l/2
    for i in range((int(c-a)+1)*100, int(l)*200):
        R2 = abs(i/100)
        R1 = abs(d + R2)
        target = get_intersections(m1[0], m1[1], R1, m2[0], m2[1], R2)
        Targetsxd.append(target[0])
        Targetsyd.append(target[1])
        Targetsxu.append(target[2])
        Targetsyu.append(target[3])


def fx_dal(x0,x_sat):
    '''
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    r= np.zeros((len(list(x_sat))))
    for i in range(len(list(x_sat))):
      r[i]=np.sqrt((x0[0]-x_sat[i][0])**2+(x0[1]-x_sat[i][1])**2)+random.normalvariate(0,0.05)
    return r


def fx_dal_ideal(x0, x_sat):
    '''
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    rid = np.zeros((len(list(x_sat))))
    for i in range(len(list(x_sat))):
      rid[i] = np.sqrt((x0[0]-x_sat[i][0])**2+(x0[1]-x_sat[i][1])**2)
    return rid


def fx_pseudo_dal(x0,x_sat):
    '''
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    # var= np.zeros((len(list(x_sat))-1))
    # for i in range(len(list(x_sat))-1):
    #     if i==0:continue
    #     var[i-1]= np.linalg.norm(x_sat[i]-x0)-np.linalg.norm(x_sat[0]-x0)
    # c=3e8
    #one_matrix=np.zeros((len(list(x_sat)),1))
    r= np.zeros((len(list(x_sat))))
    #one_matrix = np.array(list(map(lambda x: +1, one_matrix)))
    for i in range(len(list(x_sat))):
        r[i]=np.sqrt((x0[0]-x_sat[i][0])**2+(x0[1]-x_sat[i][1])**2)

    #r=np.array(list(map(lambda x: np.linalg.norm(x-x0),x_sat)))
    #delta=c*x0[2]*one_matrix
    #R=delta+r
    return r


def fx_angle(x0,x_sat):
    '''
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    var= np.zeros((len(list(x_sat))))
    for i in range(len(x_sat)):
        var[i]=atan2((x_sat[i][1]-x0[1]),(x_sat[i][0]-x0[0])) + random.gauss(0, 0.02)

    return var


def fx_angle_pseudo_dal(x0,x_sat):
    '''
    УГЛОМЕРНЫЙ-ПСЕВДОДАЛЬНОМЕРНЫЙ
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    i=0
    j=0
    c=3e8
    var= np.zeros((len(list(x_sat)))*2)
    # for i in range(len(x_sat)):
    #     var[i]=atan2((x_sat[i][1]-x0[1]),(x_sat[i][0]-x0[0]))+np.sqrt((x0[0]-x_sat[i][0])**2+(x0[1]-x_sat[i][1])**2)+c*x0[2]

    while i<(len(list(x_sat)))*2:
        var[i] = atan2((x_sat[j][1] - x0[1]), (x_sat[j][0] - x0[0]))
        i+=1
        var[i]=np.sqrt((x0[0]-x_sat[j][0])**2+(x0[1]-x_sat[j][1])**2)+c*x0[2]
        i+=1
        j+=1

    return var


def fx_dal_difference(x0,x_sat):
    '''
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    k=0
    l=0
    for n in range(len(list(x_sat))):
        for m in range(len(list(x_sat))):
            if m>n: k+=1
    var= np.zeros(k)
    for i in range(len(list(x_sat))):
        for j in range(len(list(x_sat))):
            if j>i:
                var[l]= np.linalg.norm(x_sat[i]-x0)-np.linalg.norm(x_sat[j]-x0)+random.normalvariate(0,0.05)*2
                l+=1
    R=np.array(list(map(lambda x: np.linalg.norm(x-x0),x_sat)))
    return var



#через мастера
# def Hx(x_prev,x_sat):def Hx(x_prev,x_sat):
# #     '''
# #     Градиентная матрица
# #     :param x_prev:
# #     :param x_sat:
# #     :return:
# #     '''
# #
# #     return np.array(list(map(lambda x: -(x-x_prev)/np.linalg.norm(x-x_prev) if np.linalg.norm(x-
#     '''
#     Градиентная матрица
#     :param x_prev:
#     :param x_sat:
#     :return:
#     '''
#     gradMatrix = np.zeros((len(list(x_sat))-1, 2))
#     delta = np.zeros((len(list(x_sat))))
#     master = x_sat[0]
#     for i in range(len(list(x_sat))):
#        if master[0]==x_sat[i][0] and  master[1]==x_sat[i][1]:
#            #gradMatrix[i]=np.zeros(x_prev.shape)
#            continue
#        delta_master = np.sqrt((x_prev[0] - master[0]) ** 2 + (x_prev[1] - master[1]) ** 2)
#        delta_anchor = np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2)
#        #delta[i-1] = np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2)
#        try:
#            gradMatrix[i-1] = (x_prev - x_sat[i])/delta_anchor - (x_prev-master)/delta_master
#        except:
#            print("error")
#            gradMatrix[i-1] = np.zeros(x_prev.shape)
#
#     return gradMatrix


def fx_angle_dal(x0,x_sat):
    '''
    УГЛОМЕРНЫЙ-ДАЛЬНОМЕРНЫЙ
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    i=0
    j=0
    # var= np.zeros((len(list(x_sat))))
    # for i in range(len(x_sat)):
    #     var[i]=atan2((x_sat[i][1]-x0[1]),(x_sat[i][0]-x0[0]))+np.sqrt((x0[0]-x_sat[i][0])**2+(x0[1]-x_sat[i][1])**2)

    while i<(len(list(x_sat)))*2:
        var[i] = atan2((x_sat[j][1] - x0[1]), (x_sat[j][0] - x0[0]))
        i+=1
        var[i]=np.sqrt((x0[0]-x_sat[j][0])**2+(x0[1]-x_sat[j][1])**2)
        i+=1
        j+=1

    return var

# Псевдодальномерный
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

def Hx_pseudo_dal(x_prev,x_sat):
    '''
    Градиентная матрица
    :param x_prev:
    :param x_sat:
    :return:
    '''
    #ДАЛЬНОМЕРНЫЙ
    H = np.zeros((len(list(x_sat)), 3))
    for i in range(len(list(x_sat))):
        H[i][0] = -(x_sat[i][0] - x_prev[0]) / np.sqrt((x_prev[0]-x_sat[i][0])**2+(x_prev[1]-x_sat[i][1])**2)
        H[i][1] = -(x_sat[i][1] - x_prev[1]) / np.sqrt((x_prev[0]-x_sat[i][0])**2+(x_prev[1]-x_sat[i][1])**2)
        H[i][2] = 3e8


    return H

# РАЗНОСТНО-ДАЛЬНОМЕРНЫЙ
def Hx_dal_difference(x_prev,x_sat):
    '''
    Градиентная матрица
    :param x_prev:
    :param x_sat:
    :return:
    '''
    k=0
    l=0
    for n in range(len(list(x_sat))):
        for m in range(len(list(x_sat))):
            if m > n: k += 1
    gradMatrix = np.zeros((k, 2))
    delta = np.zeros((len(list(x_sat))))
    for i in range(len(list(x_sat))):
        for j in range(len(list(x_sat))):
            if j>i:
               delta_j = np.sqrt((x_prev[0] - x_sat[j][0]) ** 2 + (x_prev[1] - x_sat[j][1]) ** 2)
               delta_i = np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2)
               #delta[i-1] = np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2)
               try:
                   gradMatrix[l] = (x_prev - x_sat[i])/delta_i - (x_prev-x_sat[j])/delta_j
                   l+=1
                   if l>6: print("error")
               except:
                   print("error")


    return gradMatrix


#УГЛОМЕРНЫЙ
def Hx_angle(x_prev,x_sat):
    '''
    Градиентная матрица
    :param x_prev:
    :param x_sat:
    :return:
    '''

    gradMatrix = np.zeros((len(list(x_sat)), 2))
    for i in range(len(list(x_sat))):
        #corner=atan2((x_sat[i][1]-x_prev[1]),(x_sat[i][0]-x_prev[0]))
        gradMatrix[i][0]= (x_sat[i][1]-x_prev[1])/(x_sat[i][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[i][1]**2-2*x_sat[i][1]*x_prev[1]+x_prev[1]**2)*180/math.pi
        gradMatrix[i][1]= (x_prev[0]-x_sat[i][0])/((x_sat[i][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[i][1]**2-2*x_sat[i][1]*x_prev[1]+x_prev[1]**2))*180/math.pi

    return gradMatrix


def fx_angle_dal_diference(x0,x_sat):
    '''
    УГЛОМЕРНО -РАЗНОСТОДАЛЬНОМЕРНЫЙ
    Вектор функциональной связи между наблюдениями и координатами объекта
    :param x0:
    :param x_sat:
    :return:
    '''
    k=0
    l=0
    for n in range(len(list(x_sat))):
        for m in range(len(list(x_sat))):
            if m>n: k+=1
    var= np.zeros(k)
    for i in range(len(list(x_sat))):
        for j in range(len(list(x_sat))):
            if j>i:
                var[l]= np.linalg.norm(x_sat[i]-x0)-np.linalg.norm(x_sat[j]-x0)+atan2((x_sat[i][1]-x0[1]),(x_sat[i][0]-x0[0]))+atan2((x_sat[j][1]-x0[1]),(x_sat[j][0]-x0[0]))
                l+=1
    R=np.array(list(map(lambda x: np.linalg.norm(x-x0),x_sat)))
    return var




def Hx_angle_dal(x_prev,x_sat):
    '''
    УГЛОМЕРНЫЙ-ДАЛЬНОМЕРНЫЙ
    Градиентная матрица
    :param x_prev:
    :param x_sat:
    :return:
    '''
    i = 0
    j = 1
    gradMatrix = np.zeros((len(list(x_sat))*2, 2))
    for i in range(len(list(x_sat))):
        #corner=atan2((x_sat[i][1]-x_prev[1]),(x_sat[i][0]-x_prev[0]))
        gradMatrix[j][0] = (x_prev[0] - x_sat[i][0]) / (np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2))
        gradMatrix[i][0]=((x_sat[i][1]-x_prev[1])/(x_sat[i][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[i][1]**2-2*x_sat[i][1]*x_prev[1]+x_prev[1]**2 + 0.001))*180/math.pi
        gradMatrix[j][1]= (x_prev[1] - x_sat[i][1]) / (np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2))
        gradMatrix[i][1]=((x_prev[0]-x_sat[i][0])/((x_sat[i][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[i][1]**2-2*x_sat[i][1]*x_prev[1]+x_prev[1]**2+ 0.001)))*180/math.pi
        i +=2
        j+=2

    return gradMatrix



def Hx_angle_dal_difference(x_prev,x_sat):
    '''
    УГЛОМЕРНЫЙ -РАЗНОСТНОДАЛЬНОМЕРНЫЙ
    Градиентная матрица
    :param x_prev:
    :param x_sat:
    :return:
    '''
    k=0
    l=0
    for n in range(len(list(x_sat))):
        for m in range(len(list(x_sat))):
            if m > n: k += 1
    gradMatrix = np.zeros((k+len(list(x_sat)), 2))
    delta = np.zeros((len(list(x_sat))))
    for i in range(len(list(x_sat))):
        for j in range(len(list(x_sat))):
            if j>i:
               delta_j = np.sqrt((x_prev[0] - x_sat[j][0]) ** 2 + (x_prev[1] - x_sat[j][1]) ** 2)
               delta_i = np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2)
               #delta[i-1] = np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2)
               try:
                   gradMatrix[l][0] = (x_prev[0] - x_sat[i][0])/delta_i - (x_prev[0]-x_sat[j][0])/delta_j+((x_sat[i][1]-x_prev[1])/(x_sat[i][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[i][1]**2-2*x_sat[i][1]*x_prev[1]+x_prev[1]**2)) +(((x_sat[j][1]-x_prev[1])/(x_sat[j][0]**2-2*x_sat[j][0]*x_prev[0]+x_prev[0]**2+x_sat[j][1]**2-2*x_sat[j][1]*x_prev[1]+x_prev[1]**2)))
                   gradMatrix[l][1] = (x_prev[1] - x_sat[i][1]) / delta_i - (x_prev[1] - x_sat[j][1]) / delta_j +((x_prev[0]-x_sat[i][0])/((x_sat[i][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[i][1]**2-2*x_sat[i][1]*x_prev[1]+x_prev[1]**2))) +(((x_prev[0]-x_sat[j][0])/((x_sat[j][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[j][1]**2-2*x_sat[j][1]*x_prev[1]+x_prev[1]**2))))
                   l+=1
                   if l>6: print("error")
               except:
                   print("error")
                   gradMatrix[l] = np.zeros(x_prev.shape)
                   l += 1
                   if l > 6: print("error")

    for i in range(len(list(x_sat))):
        gradMatrix[l][0] = (x_sat[i][1] - x_prev[1]) / (
                    x_sat[i][0] ** 2 - 2 * x_sat[i][0] * x_prev[0] + x_prev[0] ** 2 + x_sat[i][1] ** 2 - 2 * x_sat[i][
                1] * x_prev[1] + x_prev[1] ** 2) *180/math.pi
        gradMatrix[l][1] = (x_prev[0] - x_sat[i][0]) / ((
                    x_sat[i][0] ** 2 - 2 * x_sat[i][0] * x_prev[0] + x_prev[0] ** 2 + x_sat[i][1] ** 2 - 2 * x_sat[i][
                1] * x_prev[1] + x_prev[1] ** 2)) *180/math.pi
        l+=1

    return gradMatrix


def Hx_angle_pseudo_dal(x_prev,x_sat):
    '''
    УГЛОМЕРНЫЙ-ПСЕВДОДАЛЬНОМЕРНЫЙ
    Градиентная матрица
    :param x_prev:
    :param x_sat:
    :return:
    '''
    gradMatrix = np.zeros((len(list(x_sat))*2, 3))
    # for i in range(len(list(x_sat))):
    #     #corner=atan2((x_sat[i][1]-x_prev[1]),(x_sat[i][0]-x_prev[0]))
    #     gradMatrix[i][0]=((x_sat[i][1]-x_prev[1])/(x_sat[i][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[i][1]**2-2*x_sat[i][1]*x_prev[1]+x_prev[1]**2))+(x_prev[0]-x_sat[i][0])/ (np.sqrt((x_prev[0]-x_sat[i][0])**2+(x_prev[1]-x_sat[i][1])**2))
    #     gradMatrix[i][1]=((x_prev[0]-x_sat[i][0])/((x_sat[i][0]**2-2*x_sat[i][0]*x_prev[0]+x_prev[0]**2+x_sat[i][1]**2-2*x_sat[i][1]*x_prev[1]+x_prev[1]**2)))+(x_prev[1] - x_sat[i][1]) / (np.sqrt((x_prev[0] - x_sat[i][0]) ** 2 + (x_prev[1] - x_sat[i][1]) ** 2))
    #     gradMatrix[i][2]=3e8
    i=0
    j=1
    k=0
    while i<(len(list(x_sat)))*2:
        gradMatrix[i][0] = ((x_sat[k][1] - x_prev[1]) / (x_sat[k][0] ** 2 - 2 * x_sat[k][0] * x_prev[0] + x_prev[0] ** 2 + x_sat[k][1] ** 2 - 2 * x_sat[k][1] * x_prev[1] + x_prev[1] ** 2))
        gradMatrix[i][1] = ((x_prev[0]-x_sat[k][0])/((x_sat[k][0]**2-2*x_sat[k][0]*x_prev[0]+x_prev[0]**2+x_sat[k][1]**2-2*x_sat[k][1]*x_prev[1]+x_prev[1]**2)))
        gradMatrix[i][2] = 0
        gradMatrix[j][0] = (x_prev[0]-x_sat[k][0])/ (np.sqrt((x_prev[0]-x_sat[k][0])**2+(x_prev[1]-x_sat[k][1])**2))
        gradMatrix[j][1] = (x_prev[1] - x_sat[k][1]) / (np.sqrt((x_prev[0] - x_sat[k][0]) ** 2 + (x_prev[1] - x_sat[k][1]) ** 2))
        gradMatrix[j][2] = 3e8
        i+=2
        j+=2
        k+=1
    return gradMatrix


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


def gradient_descent(R,x_sat, Hx, fx, x_prev=np.array([0.1,0.1]),epsilon=0.1,show_plots=''):
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
        print(f'x_est объекта на {i} итерации равен {x_new}')
        Euc =  np.sqrt((x_new[0]-x_prev[0])**2+(x_new[1]-x_prev[1])**2)
        history.append(Euc)
        if Euc<=epsilon:
            print('\nОстановка расчета!')
            winner= x_new
            break
        i+=1
        x_prev = x_new.copy()

    if show_plots:
        plt.figure(1)
        plt.plot(list(range(1,len(history)+1)),history, 'o--', linewidth=2, markersize=12, color = 'magenta')
        plt.grid()
        plt.xlabel('номер итерации,k')
        plt.ylabel('norm(x_new-x_prev)')
        plt.show()
        return x_new
    else:
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


if __name__=='__main__':
    word = input()
    if word == 'Дальномерный':
        history_1 = []
        Rs = []
        for i in range(100):
            x = np.array([3,3])
            x_sat = np.array([[0, 0],
                             [5, 0],
                             [0, 5]])
            R = fx_dal(x, x_sat)
            R_dal_ideal = fx_dal_ideal(x, x_sat)
            Rs.append(R_dal_ideal)
            # x_new = gradient_descent(R, x_sat,show_plots='True')
            winner_1 = Gradient_descent(R, x_sat, Hx_dal, fx_dal, x_prev=np.array([0.1, 0.1]), epsilon=0.1)
            history_1.append(winner_1)
            # print("разносно-дальномерный")
        history_1 = np.array(history_1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_1[:, 0], y=history_1[:, 1],
                                     mode='markers',
                                     name='coord mayak',marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[0,1,2,3,4,5], y=[0,1,2,3,4,5],
                                     mode='markers',
                                     name='path'))

        sigma = 0.05
        # for ii in range(len(Rs[:, 1])):
        ii = 0
        for jj in range(3):
            Rs[ii][jj] = Rs[ii][jj] + 3 * sigma
            fig.add_shape(type="circle",
                          xref="x", yref="y",
                          x0=x_sat[jj][0] - Rs[ii][jj], y0=x_sat[jj][1] - Rs[ii][jj],
                          x1=x_sat[jj][0] + Rs[ii][jj], y1=x_sat[jj][1] + Rs[ii][jj],
                          line_color="LightSeaGreen",
                          )

        fig.update_xaxes(range=[0, 10], zeroline=False)
        fig.update_yaxes(range=[0, 10])
        fig.update_layout(width=1200, height=1200)
        fig.show()

    if word == 'Угломерный':
        history_1 = []
        for i in range(100):
            x = np.array([3,3])
            x_sat = np.array([[0, 0],
                             [5, 0],
                             [0, 5]])
            R = fx_angle(x, x_sat)

            # x_new = gradient_descent(R, x_sat,show_plots='True')У
            winner_1 = Gradient_descent(R, x_sat, Hx_angle, fx_angle, x_prev=np.array([0.1, 0.1]), epsilon=0.1)
            history_1.append(winner_1)
            # print("разносно-дальномерный")
        history_1 = np.array(history_1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_1[:, 0], y=history_1[:, 1],
                                     mode='markers',
                                     name='coord mayak',marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[0,1,2,3,4,5], y=[0,1,2,3,4,5],
                                     mode='markers',
                                     name='path'))
        sigma_angle = 0.02
        radius = 25
        for ii in range(len(x_sat)):
            angle_base = float(atan2(-(x_sat[ii][1] - x[1]), -(x_sat[ii][0] - x[0]))) + 3 * sigma_angle
            print(angle_base)
            # if angle_base > 0:
            #     angle_base = abs(angle_base + 3 * sigma_angle)
            # elif angle_base < 0:
            #     angle_base = -1*abs(angle_base - 3 * sigma_angle)
            print(angle_base)
            x2 = float(radius * cos(angle_base) + x_sat[ii][0])
            y2 = float(radius * sin(angle_base) + x_sat[ii][1])
            # print(angle_base, x_sat[ii][0], x2, x_sat[ii][1], y2)
            fig.add_trace(go.Scatter(x=[x_sat[ii][0], x2], y=[x_sat[ii][1], y2],
                                     mode='lines',
                                     name='lines'))
        for ii in range(len(x_sat)):
            angle_base = float(atan2(-(x_sat[ii][1] - x[1]), -(x_sat[ii][0] - x[0]))) - 3 * sigma_angle
            print(angle_base)
            # if angle_base > 0:
            #     angle_base = abs(angle_base + 3 * sigma_angle)
            # elif angle_base < 0:
            #     angle_base = -1*abs(angle_base - 3 * sigma_angle)
            print(angle_base)
            x2 = float(radius * cos(angle_base) + x_sat[ii][0])
            y2 = float(radius * sin(angle_base) + x_sat[ii][1])
            # print(angle_base, x_sat[ii][0], x2, x_sat[ii][1], y2)
            fig.add_trace(go.Scatter(x=[x_sat[ii][0], x2], y=[x_sat[ii][1], y2],
                                     mode='lines',
                                     name='lines'))
        fig.update_xaxes(range=[0, 10], zeroline=False)
        fig.update_yaxes(range=[0, 10])
        fig.update_layout(width=1200, height=1200)
        fig.show()

    if word == 'РД':
        history_1 = []
        for i in range(400):
            x = np.array([2,4])
            x_sat = np.array([[0, 0],
                             [10, 0],
                             [0, 10]])
            R = fx_dal_difference(x, x_sat)

            # x_new = gradient_descent(R, x_sat,show_plots='True')У
            winner_1 = Gradient_descent(R, x_sat, Hx_dal_difference, fx_dal_difference, x_prev=np.array([0.1, 0.1]), epsilon=0.1)
            history_1.append(winner_1)
            # print("разносно-дальномерный")
        history_1 = np.array(history_1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history_1[:, 0], y=history_1[:, 1],
                                     mode='markers',
                                     name='coord mayak',marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[0,1,2,3,4,5], y=[0,1,2,3,4,5],
                                     mode='markers',
                                     name='path'))

        ReTargetsxu = []
        ReTargetsyu = []
        ReTargetsxd = []
        ReTargetsyd = []

        for i in range(len(x_sat)-1):
            m1 = x_sat[i]
            m2 = x_sat[i+1]
            target = x
            trace_hyperbole(m1, m2, target, ReTargetsxu, ReTargetsyu, ReTargetsxd, ReTargetsyd)
            ReTargetsxu = list(reversed(ReTargetsxu))
            ReTargetsyu = list(reversed(ReTargetsyu))
            fig.add_trace(go.Scatter(x=ReTargetsxu + ReTargetsxd, y=ReTargetsyu + ReTargetsyd,
                                     mode='lines',
                                     name='lines'))

            ReTargetsxu, ReTargetsyu, ReTargetsxd, ReTargetsyd = [], [], [], []
        m1 = x_sat[len(x_sat)-1]
        m2 = x_sat[0]
        target = x
        trace_hyperbole(m1, m2, target, ReTargetsxu, ReTargetsyu, ReTargetsxd, ReTargetsyd)
        ReTargetsxu = list(reversed(ReTargetsxu))
        ReTargetsyu = list(reversed(ReTargetsyu))
        fig.add_trace(go.Scatter(x=ReTargetsxu + ReTargetsxd, y=ReTargetsyu + ReTargetsyd,
                                 mode='lines',
                                 name='lines'))
        ReTargetsxu, ReTargetsyu, ReTargetsxd, ReTargetsyd = [], [], [], []

        fig.update_layout(width=1200, height=1200)
        fig.show()
