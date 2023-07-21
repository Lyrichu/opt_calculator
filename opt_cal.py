# -*- coding:utf-8 -*-
# author:lyrichu@foxmail.com
# @Time: 2023/7/20 15:06

import random
import re
import sys
import time
import traceback
import numpy as np

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QLineEdit, QPlainTextEdit, QLabel, \
    QComboBox, QWidget, QGridLayout, QDoubleSpinBox, QSpinBox
from deap import base, creator, tools, algorithms
from pyswarm import pso
from scipy.optimize import minimize, NonlinearConstraint
from sympy import symbols, lambdify


class EvoWorker(QThread):
    result = Signal(float, list, str, str)  # Changed here

    def __init__(self, window, func, func_constraints, bounds, method):
        super().__init__()
        self.window = window
        self.func = func
        self.constraints_func = func_constraints
        self.bounds = bounds
        self.method = method

    def _get_ga_params_by_name(self, name):
        box = QSpinBox if "size" in name else QDoubleSpinBox
        return self.window.gaParamsGridWidget.findChild(box, name).value()

    def _get_pso_params_by_name(self, name):
        box = QSpinBox if "size" in name else QDoubleSpinBox
        return self.window.psoParamsGridWidget.findChild(box, name).value()

    def ga_optimize(self, func, lb, ub):

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # 目标是最小化问题，权重为负
        creator.create("Individual", list, fitness=creator.FitnessMin)  # 创建个体类

        toolbox = base.Toolbox()  # 工具箱
        hof = tools.HallOfFame(1)

        # 为每个基因单独设定范围
        for i in range(len(lb)):
            toolbox.register(f"attr_float{i}", random.uniform, lb[i], ub[i])

        # 个体由 len(lb) 个基因组成
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         [toolbox.__getattribute__(f"attr_float{i}") for i in range(len(lb))],
                         n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 种群由多个个体组成

        # 目标函数
        def objective(individual):
            return func(*individual),  # 注意，这里返回的是一个元组，用 * 将 individual 展开为多个参数

        # 可行性判断,用于处理约束项
        def feasible(individual):
            if len(self.constraints_func) == 0:
                return True
            # 逐个判断约束函数是否满足
            for _func, _, op, _ in self.constraints_func:
                if op == "=":  # 等式约束
                    if abs(_func(*individual)) > 1e-6:
                        return False
                else:  # 不等式约束(<=)
                    if _func(*individual) > 0:
                        return False
            return True

        toolbox.register("evaluate", objective)  # 注册目标函数
        toolbox.decorate("evaluate", tools.DeltaPenality(feasible, 1e6))  # 当个体不满足约束条件时，其适应度会被惩罚1e6

        toolbox.register("mate", tools.cxBlend, alpha=self._get_ga_params_by_name("cross_prob"))  # 交叉操作
        toolbox.register("mutate", tools.mutGaussian,
                         mu=self._get_ga_params_by_name("gaussian_mu"),
                         sigma=self._get_ga_params_by_name("gaussian_sigma"),
                         indpb=self._get_ga_params_by_name("mutate_prob"))  # 突变操作
        toolbox.register("select", tools.selTournament, tournsize=self._get_ga_params_by_name("select_size"))  # 选择操作

        # 初始化种群
        pop = toolbox.population(n=self._get_ga_params_by_name("population_size"))  # 种群大小为50

        # 执行遗传算法
        result = algorithms.eaSimple(pop, toolbox,
                                     cxpb=self._get_ga_params_by_name("cross_prob"),
                                     mutpb=self._get_ga_params_by_name("mutate_prob"),
                                     ngen=self._get_ga_params_by_name("generation_size"),
                                     halloffame=hof,
                                     verbose=False)  # 执行100代遗传算法
        best_individual = tools.selBest(pop, k=1)[0]  # 选择最优个体
        return best_individual, best_individual.fitness.values[0]  # 返回最优解和对应的函数值

    def run(self):
        try:
            start_time = time.time()
            lb = [b[0] for b in self.bounds]
            ub = [b[1] for b in self.bounds]
            if self.method == 'GA':
                solution, value = self.ga_optimize(self.func, lb, ub)
            elif self.method == 'PSO':
                solution, value = pso(lambda x: self.func(*x),
                                      lb,
                                      ub,
                                      swarmsize=self._get_pso_params_by_name("swarm_size"),
                                      omega=self._get_pso_params_by_name("omega"),
                                      phip=self._get_pso_params_by_name("phip"),
                                      phig=self._get_pso_params_by_name("phig"),
                                      maxiter=self._get_pso_params_by_name("maxiter_size"),
                                      minstep=self._get_pso_params_by_name("min_step"),
                                      minfunc=self._get_pso_params_by_name("min_func"))
            else:
                constraints = []
                for con in self.constraints_func:
                    con_func, _, op, v = con
                    if op == "=":
                        constraints.append(NonlinearConstraint(lambda x: con_func(*x), v, v))
                    else:
                        constraints.append(NonlinearConstraint(lambda x: con_func(*x), -np.inf, 0))
                res = minimize(lambda x: self.func(*x), [b[0] for b in self.bounds], bounds=self.bounds,
                               method=self.method, constraints=constraints)
                solution, value = res.x, res.fun
            # check constraints if all satisfied
            constraints_status_text = ""
            if self.constraints_func:
                constraints_status_text = "check constraints satisfied status:\n"
                for _func, func_str, op, v in self.constraints_func:
                    res = _func(*solution)
                    if op == "=":
                        status = "√" if abs(res) < 1e-6 else "×"
                        constraints_status_text += f"{func_str} {op} 0(left = {res}) {status}\n"
                    else:
                        status = "√" if res <= 0 else "×"
                        _op = op
                        if op == ">":
                            _op = "<"
                        elif op == ">=":
                            _op = "<="
                        constraints_status_text += f"{func_str} {_op} 0(left = {res}) {status}\n"

            elapsed_time = time.time() - start_time
            self.result.emit(value, solution, constraints_status_text, f'Elapsed time: {elapsed_time} seconds')
        except Exception as e:
            self.window.log_area.appendPlainText(traceback.format_exc())


class HelpWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Help")
        layout = QVBoxLayout()

        self.help_area = QPlainTextEdit()
        self.help_area.setReadOnly(True)
        layout.addWidget(self.help_area)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def display_help(self, text):
        self.help_area.setPlainText(text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.psoParamsGridWidget = None
        self.gaParamsGridWidget = None
        self.setWindowTitle("Function Optimizer")

        help_menu = self.menuBar().addMenu("Help")
        help_action = QAction("How to use", self)
        help_action.triggered.connect(self.show_help)
        help_menu.addAction(help_action)
        self.help_window = HelpWindow()

        self.layout = QVBoxLayout()

        # 在你的界面类的__init__方法中
        self.variables_input = QLineEdit()
        self.layout.addWidget(QLabel("Enter variable names separated by comma:"))
        self.layout.addWidget(self.variables_input)

        self.function_input = QLineEdit()
        self.layout.addWidget(QLabel("Enter the function:"))
        self.layout.addWidget(self.function_input)

        self.bounds_input = QLineEdit()
        self.layout.addWidget(QLabel("Enter the bounds (option,comma separated):"))
        self.layout.addWidget(self.bounds_input)

        self.constraints_input = QPlainTextEdit()
        self.constraints_input.setVisible(False)
        self.constraints_label = QLabel("Enter the constraints(option,newline seperated):")
        self.constraints_label.setVisible(False)
        self.layout.addWidget(self.constraints_label)
        self.layout.addWidget(self.constraints_input)

        self.target_select = QComboBox()
        self.target_select.addItems(["Minimize", "Maximize"])
        self.layout.addWidget(QLabel("Select the target:"))
        self.layout.addWidget(self.target_select)

        self.algorithm_select = QComboBox()
        self.algorithm_select.addItems(
            ['Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr',
             'GA', 'PSO'])
        self.algorithm_select.currentIndexChanged.connect(self.display_option_params)
        self.layout.addWidget(QLabel("Select the algorithm:"))
        self.layout.addWidget(self.algorithm_select)

        self.add_ga_params(False)
        self.add_pso_params(False)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_optimization)
        self.layout.addWidget(self.start_button)

        self.result_area = QPlainTextEdit()
        self.result_area.setReadOnly(True)
        self.layout.addWidget(QLabel("Result:"))
        self.layout.addWidget(self.result_area)

        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)
        self.layout.addWidget(QLabel("Logs:"))
        self.layout.addWidget(self.log_area)

        self.time_label = QLabel()
        self.layout.addWidget(QLabel("Elapsed time:"))
        self.layout.addWidget(self.time_label)

        self.container = QWidget()
        self.container.setLayout(self.layout)
        self.setCentralWidget(self.container)

    def add_ga_params(self, visible=False):
        grid = QGridLayout()
        spinBoxes = [QSpinBox(), QSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(),
                     QDoubleSpinBox(), QDoubleSpinBox(), QSpinBox()]
        tooltips = ['generation_size', 'population_size', 'cross_prob', 'mutate_prob',
                    'gaussian_mu', 'gaussian_sigma', 'select_size']
        labels = ['GA Evolution Generations',
                  'GA Population Size',
                  'GA Crossover Probability',
                  'GA Mutation Probability',
                  'GA Gaussian Mutation Mean',
                  'GA Gaussian Mutation Standard Deviation',
                  'GA Selection Size']
        default_values = [100, 500, 0.5, 0.1, 0, 1, 3]
        min_max_values = [
            (10, 50000),
            (50, 50000),
            (0.01, 0.99),
            (0.01, 0.99),
            (0, 1),
            (0.1, 3),
            (1, 100)
        ]
        assert len(tooltips) == len(labels) == len(default_values) == len(min_max_values)

        # 创建若干个QDoubleSpinBox并添加到网格布局中
        for i in range(len(labels)):
            label = QLabel(tooltips[i])
            label.setToolTip(labels[i])  # 设置鼠标悬停提示
            spinbox = spinBoxes[i]
            spinbox.setMinimum(min_max_values[i][0])
            spinbox.setMaximum(min_max_values[i][1])
            spinbox.setObjectName(tooltips[i])
            spinbox.setValue(default_values[i])
            grid.addWidget(label, i // 2, 2 * (i % 2))  # 添加标签到网格布局，3 * 2布局
            grid.addWidget(spinbox, i // 2, 2 * (i % 2) + 1)  # 添加spinbox到网格布局，3 * 2布局

        # 创建一个新的QWidget，将网格布局添加到新的QWidget中
        self.gaParamsGridWidget = QWidget()
        self.gaParamsGridWidget.setLayout(grid)
        self.gaParamsGridWidget.setVisible(visible)
        self.gaParamsGridWidget.setObjectName("ga_prams")
        # 将新的QWidget添加到垂直布局中
        self.layout.addWidget(self.gaParamsGridWidget)

    def add_pso_params(self, visible=False):
        grid = QGridLayout()
        spinBoxes = [QSpinBox(), QSpinBox(), QDoubleSpinBox(), QDoubleSpinBox(),
                     QDoubleSpinBox(), QDoubleSpinBox(), QDoubleSpinBox()]
        tooltips = ['swarm_size', 'maxiter_size', 'omega', 'phip',
                    'phig', 'min_step', 'min_func']
        labels = ['The number of particles in the swarm (Default: 100)',
                  'The maximum number of iterations for the swarm to search (Default: 100)',
                  'Particle velocity scaling factor (Default: 0.5)',
                  'Scaling factor to search away from the particle\'s best known position (Default: 0.5)',
                  'Scaling factor to search away from the swarm\'s best known position (Default: 0.5)',
                  'The minimum stepsize of swarm\'s best position before the search terminates (Default: 1e-8)',
                  'The minimum change of swarm\'s best objective value before the search terminates (Default: 1e-8)'
                  ]
        default_values = [100, 100, 0.5, 0.5, 0.5, 1e-8, 1e-8]
        min_max_values = [
            (10, 50000),
            (10, 50000),
            (0.01, 0.99),
            (0.01, 0.99),
            (0.01, 0.99),
            (1e-9, 0.99),
            (1e-9, 0.99)
        ]
        assert len(tooltips) == len(labels) == len(default_values) == len(min_max_values)

        # 创建若干个QDoubleSpinBox并添加到网格布局中
        for i in range(len(labels)):
            label = QLabel(tooltips[i])
            label.setToolTip(labels[i])  # 设置鼠标悬停提示
            spinbox = spinBoxes[i]
            spinbox.setMinimum(min_max_values[i][0])
            spinbox.setMaximum(min_max_values[i][1])
            spinbox.setObjectName(tooltips[i])
            spinbox.setValue(default_values[i])
            grid.addWidget(label, i // 2, 2 * (i % 2))  # 添加标签到网格布局，3 * 2布局
            grid.addWidget(spinbox, i // 2, 2 * (i % 2) + 1)  # 添加spinbox到网格布局，3 * 2布局

        # 创建一个新的QWidget，将网格布局添加到新的QWidget中
        self.psoParamsGridWidget = QWidget()
        self.psoParamsGridWidget.setLayout(grid)
        self.psoParamsGridWidget.setVisible(visible)
        self.psoParamsGridWidget.setObjectName("pso_prams")
        # 将新的QWidget添加到垂直布局中
        self.layout.addWidget(self.psoParamsGridWidget)

    def display_option_params(self):
        self._display_evo_params()
        self._display_constraints_params()

    def _display_evo_params(self):
        if self.algorithm_select.currentText() == "GA":
            self.gaParamsGridWidget.setVisible(True)
            self.psoParamsGridWidget.setVisible(False)
        elif self.algorithm_select.currentText() == "PSO":
            self.psoParamsGridWidget.setVisible(True)
            self.gaParamsGridWidget.setVisible(False)
        else:
            self.psoParamsGridWidget.setVisible(False)
            self.gaParamsGridWidget.setVisible(False)

    def _display_constraints_params(self):
        if self.algorithm_select.currentText() in ["trust-constr", "GA"]:
            self.constraints_label.setVisible(True)
            self.constraints_input.setVisible(True)
        else:
            self.constraints_label.setVisible(False)
            self.constraints_input.setVisible(False)

    def start_optimization(self):
        self.log_area.clear()
        self.result_area.clear()
        self.log_area.appendPlainText("Starting optimization...")
        try:
            # 假设变量名由用户输入，并用逗号分隔
            variables_input = self.variables_input.text()
            var_names = variables_input.split(',')

            # 使用sympy.symbols来定义这些变量
            vars_sympy = symbols(var_names)

            func_str = self.function_input.text()

            # 确保在eval的环境中定义了所有的变量
            eval_env = {var: vars_sympy[i] for i, var in enumerate(var_names)}

            func_sympy = eval(func_str, eval_env)

            if self.target_select.currentText() == "Maximize":
                func_sympy = -func_sympy

            func = lambdify(vars_sympy, func_sympy, 'numpy')

            bounds_str = self.bounds_input.text().split(',')
            if len(bounds_str) == 0:
                bounds = [(-1e6, 1e6) for _ in range(len(var_names))]
            else:
                bounds = [(float(b.split(':')[0]), float(b.split(':')[1])) for b in bounds_str]

            constraints_str = self.constraints_input.toPlainText()
            constraints = constraints_str.split("\n")
            func_constraints = []
            if len(constraints_str) > 0:
                for con in constraints:
                    a, b = re.split(">=|<=|>|<|=", con)
                    if ">=" in con:
                        func_str = f"{b} - ({a})"
                    else:
                        func_str = f"{a} - {b}"
                    func_sympy = eval(func_str, eval_env)
                    tmp_func = lambdify(vars_sympy, func_sympy, 'numpy')
                    if ">" not in con and "<" not in con and "=" in con:
                        func_constraints.append((tmp_func, func_str, "=", float(b)))
                    else:
                        op = None
                        for s in [">=", "<=", ">", "<"]:
                            if s in con:
                                op = s
                                break
                        func_constraints.append((tmp_func, func_str, op, float(b)))

            method = self.algorithm_select.currentText()

            self.worker = EvoWorker(self, func, func_constraints, bounds, method)
            self.worker.result.connect(self.on_result)
            self.worker.start()
        except Exception as e:
            self.log_area.appendPlainText(traceback.format_exc())

    def on_result(self, value, x, constraints_status_text, time_str):
        self.log_area.appendPlainText(f'Found solution: x = {x}, y = {value}\n' + constraints_status_text)
        self.result_area.setPlainText(f'x = {x}, y = {value}\n')
        self.time_label.setText(time_str)

    def show_help(self):
        help_text = """
        This is a function optimizer. To use it, follow these steps:
        1. Enter the function to optimize in the 'Enter the function:' field. Use 'x1' and 'x2' as variables. Example: '(x1-1)**2 + (x1-x2)**4 + (x2-3)**2'
        2. Enter the bounds for the variables in the 'Enter the bounds:' field. The bounds are comma separated, and each bound is in the form 'min:max'. Example: '0:10, 1:10'
        3. Select the optimization target in the 'Select the target:' drop-down menu. Choose 'Minimize' to find the minimum of the function, or 'Maximize' to find the maximum.
        4. Select the optimization algorithm in the 'Select the algorithm:' drop-down menu. You can choose from the available algorithms.
        5. Click the 'Start' button to start the optimization. The result will be shown in the 'Logs:' area, and the elapsed time will be shown below.
        Note: The program uses numerical optimization methods, and the result may not be accurate for certain complex functions or for functions with multiple local minima/maxima.
        """
        self.help_window.display_help(help_text)
        self.help_window.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    with open("style.qss", "r") as fin:
        app.setStyleSheet(fin.read())
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
