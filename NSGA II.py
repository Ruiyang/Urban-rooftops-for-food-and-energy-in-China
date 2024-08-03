import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.optimize import minimize
from matplotlib import pyplot as plt
from pymoo.decomposition.asf import ASF
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import MaxNLocator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

S = pd.read_excel("Data.xlsx", sheet_name="S", header=None).values[:, 0]
# S = S.reshape(S.shape[0], 1)
print(S.shape)

P = pd.read_excel("Data.xlsx", sheet_name="P", header=None).values[0][0]
print(P)

F = pd.read_excel("Data.xlsx", sheet_name="self", header=None).values[:, 0]
# F = F.reshape(F.shape[0], 1)
print(F.shape)

V = pd.read_excel("Data.xlsx", sheet_name="一年产量", header=None).values[:, 0]
# V = V.reshape(V.shape[0], 1)
print(V.shape)

E = pd.read_excel("Data.xlsx", sheet_name="需求", header=None).values[:, 0]
# E = E.reshape(E.shape[0], 1)
print(E.shape)

W = pd.read_excel("Data.xlsx", sheet_name="经济", header=None).values[:, 0]
# W = W.reshape(W.shape[0], 1)
print(W.shape)

G = pd.read_excel("Data.xlsx", sheet_name="最大安装容量", header=None).values[:, 0]
# G = G.reshape(G.shape[0], 1)
print(G.shape)

C = pd.read_excel("Data.xlsx", sheet_name="减排系数", header=None).values[:, 0]
# C = C.reshape(C.shape[0], 1)
print(C.shape)


def constraint_condition(s):
    condition_one = s[:124, ] + s[124:, ] - S[124:, ]
    # condition_two = s - S
    condition_three = np.sum(s[124:, ] / S[124:, ] * G[124:, ]) - P
    condition_four = s / S * V / E - F
    condition_five = s[124:, ] / S[124:, ] * V[124:, ] * C[124:, ] + s[:124, ] / S[:124, ] * V[:124, ] * C[:124, ]
    # condition_six = s[:124, ] - S[:124, ]
    # condition_seven = s[124:, ] - S[124:, ]
    return np.append(
        np.concatenate((condition_one,  condition_four, condition_five)),
        condition_three)


def target(s):
    t1 = -s[:124, ] / S[:124, ] * V[:124, ] / E[:124, ]
    t2 = -s[124:, ] / S[124:, ] * V[124:, ] / E[124:, ]
    t3 = s / S * W
    # min_val = min(t3)
    # max_val = max(t3)
    # scaled_t3 = [(x - min_val) / (max_val - min_val) for x in t3]
    target_one = np.mean(t1)
    target_two = np.mean(t2)
    target_three = -np.sum(t3)/1000000000000
    return target_one, target_two, target_three


class MyProblem(ElementwiseProblem):
    def __init__(self, **kwargs):
        # self.cost_matrix = cost_matrix
        super().__init__(n_var=248,  # 变量数
                         n_obj=3,  # 目标数
                         n_ieq_constr=497,  # 约束数993,745,497
                         xl=np.zeros((248,)),  # 变量下界
                         xu=S,  # 变量上界
                         **kwargs)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array(target(x))
        out["G"] = np.array(constraint_condition(x))


# 定义遗传算法
algorithm = NSGA2(
    pop_size=400,  # 种群数量
    n_offsprings=100,  # 每次迭代产生新的个体
    eliminate_duplicates=True
)

res = minimize(MyProblem(),
               algorithm,
               ('n_gen', 3500),  # 迭代次数
               seed=1,
               verbose=True,
               save_history=True,
               )

F = res.F  # Pareto前沿（Pareto Front）
X = res.X  # Pareto最优解（Pareto Optimal Solution）


pd.DataFrame(X, columns=None).to_excel("Pareto最优解.xlsx", index=False, header=None)
pd.DataFrame(F, columns=None).to_excel("Pareto前沿.xlsx", index=False, header=None)


# 熵权法分配目标权重
def Entropy(data):
    s_i = data / data.sum(axis=0)  # 计算每个属性在所有样本中的比例
    e_i = (-1 / np.log(data.shape[0])) * s_i * np.log(s_i + 1e-8)  # 计算每个属性的熵值权重
    e_i = np.where(np.isnan(e_i), 0.0, e_i)  # 处理可能存在的NaN值
    return (1 - e_i.sum(axis=0)) / (1 - e_i.sum(axis=0)).sum()  # 标准化


nF = MinMaxScaler().fit_transform(-F)  # 规范到0~1之间

weights = Entropy(nF)                 # 负熵 i=51
# weights = np.array([1/3, 1/3, 1/3])    i=34
print("weights", weights)
decomp = ASF()
# 将多个目标函数组合成一个单一的标量函数来实现问题的优化
i = decomp.do(nF, 1 / weights).argmin()
print("Best regarding ASF: Point \ni = %s\nF = %s\nX = %s" % (i, F[i], X[i]))
plt.figure(figsize=(70, 50))
plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
plt.title("Objective Space")
plt.show()

# # 直接传递权重
# i = decomp.do(nF, weights).argmin()
# print("Best regarding ASF: Point \ni = %s\nF = %s\nX = %s" % (i, F[i], X[i]))
# plt.figure(figsize=(70, 50))
# plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
# plt.title("Objective Space")
# plt.show()
# # 伪权重
# from pymoo.mcdm.pseudo_weights import PseudoWeights
# i = PseudoWeights(weights).do(nF)
# print("Best regarding Pseudo Weights: Point \ni = %s\nF = %s" % (i, F[i]))
# plt.figure(figsize=(70, 50))
# plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
# plt.scatter(F[i, 0], F[i, 1], marker="x", color="red", s=200)
# plt.title("Objective Space")
# plt.show()

# 散点图
from pymoo.visualization.scatter import Scatter
plot = Scatter(title=("Scatter 3D", {'pad': 30}),
               n_ticks=10,
               legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
               figsize=(70, 50),
               labels=["Ra", "Rpv", "Profit"])
plot.add(nF, color="grey", s=20)
plot.add(nF[i], color="red", s=70, label="Solution")
# plot.add(nF[51], color="purple", s=70, label="Solution 51")
# plot.add(nF[45], color="blue", s=70, label="Solution 45")
plot.show()
# 平行坐标图
from pymoo.visualization.pcp import PCP
plot = PCP(title=("Run", {'pad': 30}),
           n_ticks=10,
           legend=(True, {'loc': "upper left"}),
           figsize=(70, 50),
           labels=["Ra", "Rpv", "Profit"]
           )
plot.set_axis_style(color="grey", alpha=1)
plot.add(nF, color="grey", alpha=0.3)
plot.add(nF[i], linewidth=5, color="red", label="Solution")
# plot.add(nF[51], linewidth=5, color="purple", label="Solution 51")
# plot.add(nF[45], linewidth=5, color="blue", label="Solution 45")
plot.show()

# Radviz图
from pymoo.visualization.radviz import Radviz
plot = Radviz(title="Optimization",
              legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
              figsize=(70, 50),
              labels=["Ra", "Rpv", "Profit"],
              endpoint_style={"s": 70, "color": "green"})
plot.set_axis_style(color="black", alpha=1.0)
plot.add(nF, color="grey", s=20)
plot.add(nF[i], color="red", s=70, label="Solution")
# plot.add(nF[51], color="purple", s=70, label="Solution 51")
# plot.add(nF[45], color="blue", s=70, label="Solution 45")
plot.show()
# 星坐标图
from pymoo.visualization.star_coordinate import StarCoordinate
plot = StarCoordinate(title="Optimization",
                      legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
                      labels=["Ra", "Rpv", "Profit"],
                      axis_style={"color": "blue", 'alpha': 0.7},
                      figsize=(70, 50),
                      arrow_style={"head_length": 0.015, "head_width": 0.03})
plot.add(nF, color="grey", s=20)
plot.add(nF[i], color="red", s=70, label="Solution")
# plot.add(nF[51], color="purple", s=70, label="Solution 51")
# plot.add(nF[45], color="blue", s=70, label="Solution 45")
plot.show()


# 确定多少代满足约束
n_evals = []  # corresponding number of function evaluations\
hist_F = []  # the objective space values in each generation
hist_cv = []  # constraint violation in each generation
hist_cv_avg = []  # average constraint violation in the whole population

hist = res.history
for algo in hist:
    # store the number of function evaluations
    n_evals.append(algo.evaluator.n_eval)

    # retrieve the optimum from the algorithm
    opt = algo.opt

    # store the least contraint violation and the average in each population
    hist_cv.append(opt.get("CV").min())
    hist_cv_avg.append(algo.pop.get("CV").mean())

    # filter out only the feasible and append and objective space values
    feas = np.where(opt.get("feasible"))[0]
    hist_F.append(opt.get("F")[feas])

vals = hist_cv_avg

k = np.where(np.array(vals) <= 0.0)[0].min()
print(f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations.")

plt.figure(figsize=(70, 50))
plt.plot(n_evals, vals, color='black', lw=0.7, label="Avg. CV of Pop")  # 折线图
plt.scatter(n_evals, vals, facecolor="none", edgecolor='black', marker="p")  # 散点图
# 设置x轴刻度的最大数量为15
maxn = 15
plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=maxn, prune='both'))
plt.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
plt.title("Convergence")
plt.xlabel("Function Evaluations")
plt.ylabel("Constraint Violation")
plt.legend()
plt.show()

# from pymoo.indicators.igd import IGD
#
# metric = IGD(F, zero_to_one=True)
#
# igd = [metric.do(_F) for _F in hist_F]
#
# plt.figure(figsize=(70, 50))
# plt.plot(n_evals, igd,  color='black', lw=0.7, label="Avg. CV of Pop")
# plt.scatter(n_evals, igd,  facecolor="none", edgecolor='black', marker="p")
# plt.axhline(10**-2, color="red", label="10^-2", linestyle="--")
# plt.title("Convergence")
# plt.xlabel("Function Evaluations")
# plt.ylabel("IGD")
# plt.yscale("log")
# plt.legend()
# plt.show()
#
# from pymoo.indicators.igd_plus import IGDPlus
#
# metric = IGDPlus(F, zero_to_one=True)
#
# igd = [metric.do(_F) for _F in hist_F]
#
# plt.figure(figsize=(70, 50))
# plt.plot(n_evals, igd,  color='black', lw=0.7, label="Avg. CV of Pop")
# plt.scatter(n_evals, igd,  facecolor="none", edgecolor='black', marker="p")
# plt.axhline(10**-2, color="red", label="10^-2", linestyle="--")
# plt.title("Convergence")
# plt.xlabel("Function Evaluations")
# plt.ylabel("IGD+")
# plt.yscale("log")
# plt.legend()
# plt.show()

