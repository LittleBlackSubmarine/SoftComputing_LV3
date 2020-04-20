# -*- coding: utf-8 -*-

import sys
import os
import random
import math
import operator
import csv
import statistics

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPainter, QBrush, QPen, QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtChart import QLineSeries, QChart, QValueAxis, QChartView
from PyQt5.QtWidgets import QFileDialog

from deap import base, creator, tools

#Change current working directory to this script path
import pathlib
pathlib.Path(__file__).parent.absolute()
os.chdir(pathlib.Path(__file__).parent.absolute())

####General parameters####
NO_DIMS = 5 #Size of the individual (number of dimensions)
DIM_SIZES = [5, 10]
NGEN = 5000 #number of generations
POP_SIZE = 100  #population size
A = 10
#########################
######GA parameters######
GA_MUTPB = 0.05 #probability for mutating an individual
MUTATION_SIZES = [0.05, 0.1, 0.2]
GA_NELT = 4    #number of elite individuals
ELITE_SIZES = [4, 8, 16]
GA_MAX_ABS = 0.1 #maximum absolute value a mutated gene can have
MAX_ABS_SIZES = [0.1, 0.4, 0.8]
#########################
#####PSO parameters######
PSO_INERTIA = 0.0   #inertia factor
INERTIA_SIZES = [0.0, 0.37, 0.74]
PSO_PERSONAL = 0.5  #personal factor
PERSONAL_F_SIZES = [0.5, 1.0, 1.5]
PSO_SOCIAL = 0.5    #social factor
SOCIAL_F_SIZES = [0.5, 1.0, 1.5]
#########################

####Other global variables####
F_MIN = -5.0    #Rastrigin function interval
F_MAX = 5.0     #Rastrigin function interval
stop_evolution = False
q_min_series = QLineSeries()
q_min_series.setName("MIN")
q_max_series = QLineSeries()
q_max_series.setName("MAX")
q_avg_series = QLineSeries()
q_avg_series.setName("AVG")
combination_series = []
best_fit_values = []
##############################

#Define evaluation (fitness) function for individual (cromosome or particle)
#Rastrigin function
#Fitness function -- added by Denis Lazor
def evaluateInd(individual):
    fit_val = 0
    r_sum = 0

    for x in individual:
        r_sum += pow(x, 2) - A * math.cos(2*math.pi*x)

    fit_val = A*len(individual) + r_sum

    return fit_val,

def generateParticle(cr, size, min_val, max_val):
    #generate particle with random position
    particle = cr.Particle(random.uniform(min_val, max_val) for _ in range(size))
    #generate random speed vector for particle
    particle.speed = list(random.uniform(min_val, max_val) for _ in range(size))
    particle.best = particle[:]
    return particle
#v(k+1) = inert_fact*v(k) + ind_fact*rand()*(p(k) – x(k)) + soc_fact*rand()*(g(k) – x(k))

def updateParticle(particle, g_best, inert_fact, ind_fact, soc_fact):
    first_part = list(inert_fact * particle.speed[i] for i in range(len(particle.speed)))
    second_part = list(ind_fact * random.random() * (particle.best[i] - particle[i]) for i in range(len(particle.best)))
    third_part = list(soc_fact * random.random() * (g_best[i] - particle[i]) for i in range(len(g_best)))
    #Update particle speed by adding all three parts of the formula
    particle.speed = list(map(operator.add, map(operator.add, first_part, second_part), third_part))
    #Update particle position by adding speed to previous particle position
    particle[:] = list(map(operator.add, particle, particle.speed))


# Print function added by -- Denis Lazor
def print_results_GA( p, m, e, ma, mean_gen_fit, best_gen_fit, gen,d):
    print("\nCombination:  Population = " + str(p) + ", Mutation = " + str(m*100) + "%, Elitism = " + str(e) + ", MaxAbsMutation = " + str(ma), ", Generations = " + str(gen) + ", Dimensions: " + str(d))
    print("Mean fitness: " + str(mean_gen_fit) + "\nBest fitness: " + str(best_gen_fit))


# Print function added by -- Denis Lazor
def print_results_PSO(p, in_f, pers_f, soc_f, mean_gen_fit, best_gen_fit, gen,d):
    print("\nCombination:  Population = " + str(p) + ", Inertia = " + str(in_f) + ", Personal_factor = " + str(pers_f) + ", Social_factor = " + str(soc_f) + ", Generations = " + str(gen) + ", Dimensions: " + str(d))
    print("Mean fitness: " + str(mean_gen_fit) + "\nBest fitness: " + str(best_gen_fit))


# Clearing all .csv files on program start -- Added by Denis Lazor
def clear_all_csv(csv_type):
    directory = 'graphs_csv/'

    if csv_type == "ga":
        extension = "ga.csv"
    else:
        extension = "pso.csv"

    for filename in os.listdir(directory):
        if filename.endswith(extension):
            f = open("graphs_csv/" + filename, 'w')
            f.truncate()
            continue
        else:
            continue


# Writing min_series for mean result to .csv files. Added by -- Denis Lazor
def write_to_file(series, parameter_name, algorithm):
    if algorithm == "ga":
        parameter_name = parameter_name + "_ga"
    else:
        parameter_name = parameter_name + "_pso"

    with open(("graphs_csv/" + parameter_name + ".csv"), "a") as file:
        writer = csv.writer(file, lineterminator='\n')
        iterations = []
        fitness_values = []
        for i in range(len(series)):
            iterations.append(i)
            fitness_values.append(series[i])
        writer.writerow(iterations)
        writer.writerow(fitness_values)

class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(850, 550)
        self.setWindowTitle("Rastrigin")
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        
        self.frameChart = QChartView(self.centralwidget)
        self.frameChart.setGeometry(QtCore.QRect(10, 10, 620, 500))
        self.frameChart.setFrameShape(QtWidgets.QFrame.Box)
        self.frameChart.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.frameChart.setRenderHint(QPainter.Antialiasing)
        self.frameChart.setObjectName("frameChart")
        
        self.genParams = QtWidgets.QGroupBox(self.centralwidget)
        self.genParams.setGeometry(QtCore.QRect(650, 10, 161, 110))
        self.genParams.setObjectName("genParams")
        self.genParams.setTitle("General parameters")
        
        self.label1 = QtWidgets.QLabel(self.genParams)
        self.label1.setGeometry(QtCore.QRect(10, 20, 61, 16))
        self.label1.setObjectName("label1")
        self.label1.setText("Population:")
        
        self.label2 = QtWidgets.QLabel(self.genParams)
        self.label2.setGeometry(QtCore.QRect(10, 50, 91, 16))
        self.label2.setObjectName("label2")
        self.label2.setText("No. generations:")
        
        self.label3 = QtWidgets.QLabel(self.genParams)
        self.label3.setGeometry(QtCore.QRect(10, 80, 81, 16))
        self.label3.setObjectName("label3")
        self.label3.setText("No. dimensions:")
        
        self.tbxPopulation = QtWidgets.QLineEdit(self.genParams)
        self.tbxPopulation.setGeometry(QtCore.QRect(100, 20, 51, 20))
        self.tbxPopulation.setObjectName("tbxPopulation")
        
        self.tbxGenerations = QtWidgets.QLineEdit(self.genParams)
        self.tbxGenerations.setGeometry(QtCore.QRect(100, 50, 51, 20))
        self.tbxGenerations.setObjectName("tbxGenerations")
        
        self.tbxDimensions = QtWidgets.QLineEdit(self.genParams)
        self.tbxDimensions.setGeometry(QtCore.QRect(100, 80, 51, 20))
        self.tbxDimensions.setObjectName("tbxDimensions")
        
        self.gaParams = QtWidgets.QGroupBox(self.centralwidget)
        self.gaParams.setGeometry(QtCore.QRect(650, 130, 191, 105))
        self.gaParams.setObjectName("gaParams")
        self.gaParams.setTitle("GA parameters")
        
        self.label4 = QtWidgets.QLabel(self.gaParams)
        self.label4.setGeometry(QtCore.QRect(10, 20, 61, 16))
        self.label4.setObjectName("label4")
        self.label4.setText("Mutation:")
        
        self.label5 = QtWidgets.QLabel(self.gaParams)
        self.label5.setGeometry(QtCore.QRect(10, 50, 91, 16))
        self.label5.setObjectName("label5")
        self.label5.setText("Elite members:")

        self.label9 = QtWidgets.QLabel(self.gaParams)
        self.label9.setGeometry(QtCore.QRect(10, 80, 61, 16))
        self.label9.setObjectName("label9")
        self.label9.setText("Max abs.:")
        
        self.tbxMutation = QtWidgets.QLineEdit(self.gaParams)
        self.tbxMutation.setGeometry(QtCore.QRect(100, 20, 51, 20))
        self.tbxMutation.setObjectName("tbxMutation")
        
        self.tbxElite = QtWidgets.QLineEdit(self.gaParams)
        self.tbxElite.setGeometry(QtCore.QRect(100, 50, 51, 20))
        self.tbxElite.setObjectName("tbxElite")

        self.tbxMaxAbs = QtWidgets.QLineEdit(self.gaParams)
        self.tbxMaxAbs.setGeometry(QtCore.QRect(100, 80, 51, 20))
        self.tbxMaxAbs.setObjectName("tbxMAxAbs")
        
        self.psoParams = QtWidgets.QGroupBox(self.centralwidget)
        self.psoParams.setGeometry(QtCore.QRect(650, 240, 161, 110))
        self.psoParams.setObjectName("psoParams")
        self.psoParams.setTitle("PSO parameters")
        
        self.label6 = QtWidgets.QLabel(self.psoParams)
        self.label6.setGeometry(QtCore.QRect(10, 20, 61, 16))
        self.label6.setObjectName("label6")
        self.label6.setText("Inertia factor:")
        
        self.label7 = QtWidgets.QLabel(self.psoParams)
        self.label7.setGeometry(QtCore.QRect(10, 50, 91, 16))
        self.label7.setObjectName("label7")
        self.label7.setText("Personal factor:")
        
        self.label8 = QtWidgets.QLabel(self.psoParams)
        self.label8.setGeometry(QtCore.QRect(10, 80, 81, 16))
        self.label8.setObjectName("label8")
        self.label8.setText("Social factor:")
        
        self.tbxInertia = QtWidgets.QLineEdit(self.psoParams)
        self.tbxInertia.setGeometry(QtCore.QRect(100, 20, 51, 20))
        self.tbxInertia.setObjectName("tbxInertia")
        
        self.tbxPersonal = QtWidgets.QLineEdit(self.psoParams)
        self.tbxPersonal.setGeometry(QtCore.QRect(100, 50, 51, 20))
        self.tbxPersonal.setObjectName("tbxPersonal")
        
        self.tbxSocial = QtWidgets.QLineEdit(self.psoParams)
        self.tbxSocial.setGeometry(QtCore.QRect(100, 80, 51, 20))
        self.tbxSocial.setObjectName("tbxSocial")
        
        self.cbxNoVis = QtWidgets.QCheckBox(self.centralwidget)
        self.cbxNoVis.setGeometry(QtCore.QRect(650, 350, 170, 17))
        self.cbxNoVis.setObjectName("cbxNoVis")
        self.cbxNoVis.setText("No visualization per generation")
        
        self.btnStartGA = QtWidgets.QPushButton(self.centralwidget)
        self.btnStartGA.setGeometry(QtCore.QRect(650, 370, 75, 23))
        self.btnStartGA.setObjectName("btnStartGA")
        self.btnStartGA.setText("Start GA")
        
        self.btnStartPSO = QtWidgets.QPushButton(self.centralwidget)
        self.btnStartPSO.setGeometry(QtCore.QRect(650, 400, 75, 23))
        self.btnStartPSO.setObjectName("btnStartPSO")
        self.btnStartPSO.setText("Start PSO")
        
        self.btnStop = QtWidgets.QPushButton(self.centralwidget)
        self.btnStop.setEnabled(False)
        self.btnStop.setGeometry(QtCore.QRect(740, 370, 75, 53))
        self.btnStop.setObjectName("btnStop")
        self.btnStop.setText("Stop")
        
        self.btnSaveChart = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveChart.setGeometry(QtCore.QRect(650, 450, 121, 41))
        self.btnSaveChart.setObjectName("btnSaveChart")
        self.btnSaveChart.setText("Save chart as image")
        
        self.btnSaveChartSeries = QtWidgets.QPushButton(self.centralwidget)
        self.btnSaveChartSeries.setGeometry(QtCore.QRect(650, 500, 121, 41))
        self.btnSaveChartSeries.setObjectName("btnSaveChartSeries")
        self.btnSaveChartSeries.setText("Save chart as series")
        
        self.setCentralWidget(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self)
        
        #Connect events
        self.btnStartGA.clicked.connect(self.btnStartGA_Click)
        self.btnStartPSO.clicked.connect(self.btnStartPSO_Click)
        self.btnStop.clicked.connect(self.btnStop_Click)
        self.btnSaveChart.clicked.connect(self.btnSaveChart_CLick)
        self.btnSaveChartSeries.clicked.connect(self.btnSaveChartSeries_Click)
        
        #Set default variables
        self.tbxGenerations.insert(str(NGEN))
        self.tbxPopulation.insert(str(POP_SIZE))
        self.tbxDimensions.insert(str(NO_DIMS))
        self.tbxMutation.insert(str(GA_MUTPB))
        self.tbxElite.insert(str(GA_NELT))
        self.tbxMaxAbs.insert(str(GA_MAX_ABS))
        self.tbxInertia.insert(str(PSO_INERTIA))
        self.tbxPersonal.insert(str(PSO_PERSONAL))
        self.tbxSocial.insert(str(PSO_SOCIAL))
        
    def btnStartGA_Click(self):

        global combination_series  # List of lists containing min_series of 5   results -- Added by Denis Lazor
        global parameter_name  # Name of parameter used for writing its data to .csv file -- Added by Denis Lazor
        global best_fit_values  # List containing best fitness values for every of 5 experiments per combination -- Added by Denis Lazor

        global DIM_SIZES
        global ELITE_SIZES
        global MAX_ABS_SIZES
        global MUTATION_SIZES

        # Checking if files are empty or not -- Added by Denis Lazor
        csv_contains_ga = os.stat("graphs_csv/original_ga.csv").st_size != 0

        # Clearing non empty files if we are trying to write to them -- Added by Denis Lazor
        if csv_contains_ga:
            clear_all_csv("ga")

        parameter_name = "original"
        n = 5000
        print("GA:\n")

        # Automation for all necessary combinations -- Added by Denis Lazor
        for d in DIM_SIZES:
            MUTATION_SIZES = [0.05, 0.1, 0.2]
            ELITE_SIZES = [4, 8, 16]
            MAX_ABS_SIZES = [0.4]

            for m in MUTATION_SIZES:
                for e in ELITE_SIZES:
                    for ma in MAX_ABS_SIZES:

                        for i in range(5):
                            # Set global variables
                            global stop_evolution
                            global q_min_series
                            global q_max_series
                            global q_avg_series
                            stop_evolution = False
                            q_min_series.clear()
                            q_max_series.clear()
                            q_avg_series.clear()

                            # Set global variables from information on UI
                            global NGEN
                            global POP_SIZE
                            global GA_MUTPB
                            global GA_NELT
                            global GA_MAX_ABS
                            NGEN = int(self.tbxGenerations.text())
                            POP_SIZE = int(self.tbxPopulation.text())
                            GA_MUTPB = m
                            GA_NELT = e
                            GA_MAX_ABS = ma

                            ####Initialize deap GA objects####

                            # Make creator that minimize. If it would be 1.0 instead od -1.0 than it would be maxmize
                            self.creator = creator
                            self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

                            # Create an individual (a blueprint for cromosomes) as a list with a specified fitness type
                            self.creator.create("Individual", list, fitness=self.creator.FitnessMin)

                            # Create base toolbox for finishing creation of a individual (cromosome)
                            self.toolbox = base.Toolbox()

                            # Define what type of data (number, gene) will it be in the cromosome
                            self.toolbox.register("attr_float", random.uniform, F_MIN, F_MAX)
                            # Initialization procedure (initRepeat) for the cromosome. For the individual to be completed we need to run initRepeat for the amaout of genes the cromosome includes
                            self.toolbox.register("individual", tools.initRepeat, self.creator.Individual,
                                                  self.toolbox.attr_float, n=NO_DIMS)

                            # Create a population of individuals (cromosomes). The population is then created by toolbox.population(n=300) where 'n' is the number of cromosomes in population
                            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

                            # Register evaluation function
                            self.toolbox.register("evaluate", evaluateInd)

                            # Register what genetic operators to use
                            # Standard coding
                            self.toolbox.register("mate", tools.cxTwoPoint)  # Use two point recombination
                            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=GA_MAX_ABS, indpb=0.5)

                            self.toolbox.register("select", tools.selTournament,
                                                  tournsize=3)  # Use tournament selection

                            ##################################

                            # Generate initial poplation. Will be a member variable so we can easely pass everything to new thread
                            self.pop = self.toolbox.population(n=POP_SIZE)

                            # Evaluate initial population, we map() the evaluation function to every individual and then assign their respective fitness, map runs evaluate function for each individual in pop
                            fitnesses = list(map(self.toolbox.evaluate, self.pop))
                            for ind, fit in zip(self.pop, fitnesses):
                                ind.fitness.values = fit  # Assign calcualted fitness value to individuals

                            # Extracting all the fitnesses of all individuals in a population so we can monitor and evovlve the algorithm until it reaches 0 or max number of generation is reached
                            self.fits = [ind.fitness.values[0] for ind in self.pop]

                            # Disable start and enable stop
                            self.btnStartGA.setEnabled(False)
                            self.btnStartPSO.setEnabled(False)
                            self.btnStop.setEnabled(False)
                            self.genParams.setEnabled(False)
                            self.gaParams.setEnabled(False)
                            self.psoParams.setEnabled(False)
                            self.cbxNoVis.setEnabled(False)

                            # Start evolution
                            self.evolveGA()


                        # Best fitness value -- Added by Denis Lazor

                        best_fit = np.array(min(best_fit_values))[0]
                        mean_fit = np.array(min(best_fit_values, key=lambda x: abs(x - statistics.mean(np.asarray(best_fit_values).flatten()))))[0]

                        # Index of best fitness value -- Added by Denis Lazor
                        mean_fit_idx = best_fit_values.index(mean_fit)

                        write_to_file(combination_series[mean_fit_idx], parameter_name, "ga")

                        # First name will be "original", second one "max_abs" -- Added by Denis Lazor
                        parameter_name = "max_abs"

                        print_results_GA(POP_SIZE, m, e, ma, mean_fit, best_fit, NGEN, d)

                        # Clearing past lists  -- Added by Denis Lazor
                        combination_series = []
                        best_fit_values = []

                    # Reducing number of combinations and changing .csv file for writing -- Added by Denis Lazor
                    MAX_ABS_SIZES = MAX_ABS_SIZES[0:1]
                    parameter_name = "elites"

                ELITE_SIZES = ELITE_SIZES[0:1]
                parameter_name = "mutation"

            MUTATION_SIZES = MUTATION_SIZES[0:1]
            parameter_name = "original"


        
    def btnStartPSO_Click(self):
        global combination_series  # List of lists containing min_series of 5   results -- Added by Denis Lazor
        global parameter_name  # Name of parameter used for writing its data to .csv file -- Added by Denis Lazor
        global best_fit_values  # List containing best fitness values for every of 5 experiments per combination -- Added by Denis Lazor

        global DIM_SIZES
        global INERTIA_SIZES
        global PERSONAL_F_SIZES
        global SOCIAL_F_SIZES

        # Checking if files are empty or not -- Added by Denis Lazor
        csv_contains_pso = os.stat("graphs_csv/original_pso.csv").st_size != 0

        # Clearing non empty files if we are trying to write to them -- Added by Denis Lazor
        if csv_contains_pso:
            clear_all_csv("pso")

        n = 5000
        parameter_name = "original"
        print("PSO:\n")

        # Automation for all necessary combinations -- Added by Denis Lazor
        for d in DIM_SIZES:
            INERTIA_SIZES = [0.0, 0.37, 0.74]
            PERSONAL_F_SIZES = [0.5, 1.0, 1.5]
            SOCIAL_F_SIZES = [0.5, 1.0, 1.5]

            for in_f in INERTIA_SIZES:
                for pers_f in PERSONAL_F_SIZES:
                    for soc_f in SOCIAL_F_SIZES:

                        for i in range(5):
                            # Set global variables
                            global stop_evolution
                            global q_min_series
                            global q_max_series

                            global q_avg_series
                            stop_evolution = False
                            q_min_series.clear()
                            q_max_series.clear()
                            q_avg_series.clear()

                            # Set global variables from information on UI
                            global NGEN
                            global POP_SIZE
                            global PSO_INERTIA
                            global PSO_PERSONAL
                            global PSO_SOCIAL
                            NGEN = int(self.tbxGenerations.text())
                            POP_SIZE = int(self.tbxPopulation.text())
                            PSO_INERTIA = in_f
                            PSO_PERSONAL = pers_f
                            PSO_SOCIAL = soc_f

                            ####Initialize deap PSO objects####

                            # Make creator that minimize. If it would be 1.0 instead od -1.0 than it would be maxmize
                            self.creator = creator
                            self.creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

                            # Create an individual (a blueprint for cromosomes) as a list with a specified fitness type
                            self.creator.create("Particle", list, fitness=self.creator.FitnessMin, speed=list,
                                                best=None)

                            # Create base toolbox for finishing creation of a individual (particle) and population
                            self.toolbox = base.Toolbox()

                            # Particle initialization
                            self.toolbox.register("particle", generateParticle, cr=self.creator, size=NO_DIMS,
                                                  min_val=F_MIN, max_val=F_MAX)

                            # Create a population of individuals (particles). The population is then created by e.g. toolbox.population(n=300) where 'n' is the number of particles in population
                            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)

                            # Update function for each particle
                            self.toolbox.register("update", updateParticle)

                            # Evaluation function for each particle
                            self.toolbox.register("evaluate", evaluateInd)

                            ##################################

                            # Create population
                            self.pop = self.toolbox.population(n=POP_SIZE)

                            # Evaluate initial population, we map() the evaluation function to every individual and then assign their respective fitness, map runs emaluet function for each individual in pop
                            fitnesses = list(map(self.toolbox.evaluate, self.pop))
                            for ind, fit in zip(self.pop, fitnesses):
                                ind.fitness.values = fit

                            # Extracting all the fitnesses of all individuals in a population so we can monitor and evovlve the algorithm until it reaches 0 or max number of generation is reached
                            self.fits = [ind.fitness.values[0] for ind in self.pop]

                            # Extraction current best position
                            self.global_best_position = tools.selBest(self.pop, 1)[0][:]

                            # Disable start and enable stop
                            self.btnStartGA.setEnabled(False)
                            self.btnStartPSO.setEnabled(False)
                            self.btnStop.setEnabled(False)
                            self.genParams.setEnabled(False)
                            self.gaParams.setEnabled(False)
                            self.psoParams.setEnabled(False)
                            self.cbxNoVis.setEnabled(False)

                            # Start evolution
                            self.evolvePSO()

                        # Best fitness value -- Added by Denis Lazor
                        best_fit = np.array(min(best_fit_values))[0]
                        mean_fit = np.array(min(best_fit_values, key=lambda x: abs(x - statistics.mean(np.asarray(best_fit_values).flatten()))))[0]

                        # Index of best fitness value -- Added by Denis Lazor
                        mean_fit_idx = best_fit_values.index(mean_fit)

                        write_to_file(combination_series[mean_fit_idx], parameter_name, "pso")

                        # First name will be "original", second one "social_factor" -- Added by Denis Lazor
                        parameter_name = "social_factor"

                        print_results_PSO(POP_SIZE, in_f, pers_f, soc_f, mean_fit, best_fit, NGEN, d)

                        # Clearing past lists  -- Added by Denis Lazor
                        print(best_fit_values)
                        combination_series = []
                        best_fit_values = []

                    # Reducing number of combinations and changing .csv file for writing -- Added by Denis Lazor
                    SOCIAL_F_SIZES = SOCIAL_F_SIZES[0:1]
                    parameter_name = "personal_factor"

                PERSONAL_F_SIZES = PERSONAL_F_SIZES[0:1]
                parameter_name = "inertia"

            INERTIA_SIZES = INERTIA_SIZES[0:1]
            parameter_name = "original"

    
    def btnStop_Click(self):
        global stop_evolution
        stop_evolution = True
        #Disable stop and enable start
        self.btnStartGA.setEnabled(True)
        self.btnStartPSO.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.genParams.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.psoParams.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
    
    #Function for GA evolution
    def evolveGA(self):
        global q_min_series
        global q_max_series
        global q_avg_series

        global combination_series
        global best_fit_values

        combination_current_series = []  # Clearing fitness values series -- Added by Denis Lazor

        # Variable for keeping track of the number of generations
        curr_g = 0
        
        # Begin the evolution till goal is reached or max number of generation is reached
        while min(self.fits) != 0 and curr_g < NGEN:
            #Check if evolution and thread need to stop
            if stop_evolution:
                break #Break the evolution loop
            
            # A new generation
            curr_g = curr_g + 1
            # print("-- Generation %i --" % curr_g)
            
            # Select the next generation individuals
            #Select POP_SIZE - NELT number of individuals. Since recombination is between neigbours, not two naighbours should be the clone of the same individual
            offspring = []
            offspring.append(self.toolbox.select(self.pop, 1)[0])    #add first selected individual
            for i in range(POP_SIZE - GA_NELT - 1):    # -1 because the first seleceted individual is already added
                while True:
                    new_o = self.toolbox.select(self.pop, 1)[0]
                    if new_o != offspring[len(offspring) - 1]:   #if it is different than the last inserted then add to offspring and break
                        offspring.append(new_o)
                        break
            
            # Clone the selected individuals because all of the changes are inplace
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Apply crossover on the selected offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)    #inplace recombination
                #Invalidate new children fitness values
                del child1.fitness.values
                del child2.fitness.values
    
            #Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < GA_MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            #Add elite individuals #Is clonning needed?
            offspring.extend(list(map(self.toolbox.clone, tools.selBest(self.pop, GA_NELT))))         
                    
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            # print("  Evaluated %i individuals" % len(invalid_ind))
            
            #Replace population with offspring
            self.pop[:] = offspring
            
            # Gather all the fitnesses in one list and print the stats
            self.fits = [ind.fitness.values[0] for ind in self.pop]
            
            length = len(self.pop)
            mean = sum(self.fits) / length
            sum2 = sum(x*x for x in self.fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            q_min_series.append(curr_g, min(self.fits))
            q_max_series.append(curr_g, max(self.fits))
            q_avg_series.append(curr_g, mean)

            combination_current_series.append(min(self.fits))  # Saving min_series fitness values of an experiment -- Added by Denis Lazor
                      
            # print("  Min %s" % q_min_series.at(q_min_series.count()-1).y())
            # print("  Max %s" % q_max_series.at(q_max_series.count()-1).y())
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)
            #
            if self.cbxNoVis.isChecked():
                app.processEvents()
            else:
                self.chart = QChart()
                self.chart.addSeries(q_min_series)
                self.chart.addSeries(q_max_series)
                self.chart.addSeries(q_avg_series)
                self.chart.setTitle("Fitness value over time")
                self.chart.setAnimationOptions(QChart.NoAnimation)
                self.chart.createDefaultAxes()
                self.frameChart.setChart(self.chart)
                self.frameChart.repaint()
                app.processEvents()
                      
        #Printing best individual
        best_ind = tools.selBest(self.pop, 1)[0]
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        
        #Visulaize final solution
        if self.cbxNoVis.isChecked():
            self.chart = QChart()
            self.chart.addSeries(q_min_series)
            self.chart.addSeries(q_max_series)
            self.chart.addSeries(q_avg_series)
            self.chart.setTitle("Fitness value over time")
            self.chart.setAnimationOptions(QChart.NoAnimation)
            self.chart.createDefaultAxes()
            self.frameChart.setChart(self.chart)
            self.frameChart.repaint()
        
        #Disable stop and enable start
        self.btnStartGA.setEnabled(True)
        self.btnStartPSO.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.genParams.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.psoParams.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
        app.processEvents()

        combination_series.append(combination_current_series)  # Saving 5 results -- Added by Denis Lazor
        best_fit_values.append(best_ind.fitness.values)  # Adding best fitness value of experiment -- Added by Denis Lazor

    
    #Function for GA evolution
    def evolvePSO(self):
        global q_min_series
        global q_max_series
        global q_avg_series

        global combination_series
        global best_fit_values

        combination_current_series = []  # Clearing fitness values series -- Added by Denis Lazor
        
        # Variable for keeping track of the number of generations
        curr_g = 0
        
        while min(self.fits) != 0.0 and curr_g < NGEN:
            #Check if evolution and thread need to stop
            if stop_evolution:
                break #Break the evolution loop
                
            # A new generation
            curr_g = curr_g + 1
            # print("-- Generation %i --" % curr_g)
            
            #Update particle position and evaluate particle
            for particle in self.pop:
                #Update
                self.toolbox.update(particle, self.global_best_position, PSO_INERTIA, PSO_PERSONAL, PSO_SOCIAL)
                #Evaluate
                fit = self.toolbox.evaluate(particle)
                #Update best position
                if fit[0] < particle.fitness.values[0]:
                    particle.best = particle[:]
                #Update fitness
                particle.fitness.values = fit            
            
            #Extracting all the fitnesses of all individuals in a population so we can monitor and evovlve the algorithm until it reaches 0 or max number of generation is reached
            self.fits = [ind.fitness.values[0] for ind in self.pop]
        
            #Extraction current best position
            self.global_best_position = tools.selBest(self.pop, 1)[0][:]
            
            #Stats
            length = len(self.pop)
            mean = sum(self.fits) / length
            sum2 = sum(x*x for x in self.fits)
            std = abs(sum2 / length - mean**2)**0.5
            
            q_min_series.append(curr_g, min(self.fits))
            q_max_series.append(curr_g, max(self.fits))
            q_avg_series.append(curr_g, mean)

            combination_current_series.append(min(self.fits))  # Saving min_series fitness values of an experiment -- Added by Denis Lazor
                      
            # print("  Min %s" % q_min_series.at(q_min_series.count()-1).y())
            # print("  Max %s" % q_max_series.at(q_max_series.count()-1).y())
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)
            #
            if self.cbxNoVis.isChecked():
                app.processEvents()
            else:
                self.chart = QChart()
                self.chart.addSeries(q_min_series)
                self.chart.addSeries(q_max_series)
                self.chart.addSeries(q_avg_series)
                self.chart.setTitle("Fitness value over time")
                self.chart.setAnimationOptions(QChart.NoAnimation)
                self.chart.createDefaultAxes()
                self.frameChart.setChart(self.chart)
                self.frameChart.repaint()
                app.processEvents()
        
        #Printing best individual
        best_ind = tools.selBest(self.pop, 1)[0]
        # print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
        
        #Visulaize final solution
        if self.cbxNoVis.isChecked():
            self.chart = QChart()
            self.chart.addSeries(q_min_series)
            self.chart.addSeries(q_max_series)
            self.chart.addSeries(q_avg_series)
            self.chart.setTitle("Fitness value over time")
            self.chart.setAnimationOptions(QChart.NoAnimation)
            self.chart.createDefaultAxes()
            self.frameChart.setChart(self.chart)
            self.frameChart.repaint()
        
        #Disable stop and enable start
        self.btnStartGA.setEnabled(True)
        self.btnStartPSO.setEnabled(True)
        self.btnStop.setEnabled(False)
        self.genParams.setEnabled(True)
        self.gaParams.setEnabled(True)
        self.psoParams.setEnabled(True)
        self.cbxNoVis.setEnabled(True)
        app.processEvents()

        combination_series.append(combination_current_series)  # Saving 5 results -- Added by Denis Lazor
        best_fit_values.append(best_ind.fitness.values)  # Adding best fitness value of experiment -- Added by Denis Lazor
    
    def btnSaveChart_CLick(self):
        p = self.frameChart.grab()
        filename, _ = QFileDialog.getSaveFileName(None,"Save series chart as a image","","Image Files (*.png)")
        p.save(filename, "PNG");
        print ("Chart series image saved to: ", filename)
    
    def btnSaveChartSeries_Click(self):
        global q_min_series
        global q_max_series
        global q_avg_series
        filename, _ = QFileDialog.getSaveFileName(None,"Save series to text file","","Text Files (*.txt, *.csv)")
        with open(filename, 'w') as dat:
            for i in range(q_min_series.count()):
                dat.write('%f,%f,%f\n' % (q_min_series.at(i).y(), q_avg_series.at(i).y(), q_max_series.at(i).y()))
        print ("Chart series saved to: ", filename)
    


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.setupUi()
    ui.show()
    sys.exit(app.exec_())
