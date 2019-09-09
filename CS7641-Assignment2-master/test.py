import time
from array import array
from itertools import product
from time import clock

import sys

sys.path.append("./ABAGAIL.jar")
import java.util.Random as Random

print "Hello world!\n"

import dist.DiscreteDependencyTree as DiscreteDependencyTree
import dist.DiscreteUniformDistribution as DiscreteUniformDistribution
import dist.DiscretePermutationDistribution as DiscretePermutationDistribution
import opt.GenericHillClimbingProblem as GenericHillClimbingProblem
import opt.RandomizedHillClimbing as RandomizedHillClimbing
import opt.SimulatedAnnealing as SimulatedAnnealing
import opt.ga.GenericGeneticAlgorithmProblem as GenericGeneticAlgorithmProblem
import opt.ga.StandardGeneticAlgorithm as StandardGeneticAlgorithm
import opt.prob.GenericProbabilisticOptimizationProblem as GenericProbabilisticOptimizationProblem
import opt.prob.MIMIC as MIMIC
import shared.FixedIterationTrainer as FixedIterationTrainer
import opt.example.TravelingSalesmanRouteEvaluationFunction as TravelingSalesmanRouteEvaluationFunction
import opt.SwapNeighbor as SwapNeighbor
import opt.ga.SwapMutation as SwapMutation
import opt.example.TravelingSalesmanCrossOver as TravelingSalesmanCrossOver
import opt.example.TravelingSalesmanSortEvaluationFunction as TravelingSalesmanSortEvaluationFunction
from base import *

print "Finish loading functions!\n"

# set N value.  This is the number of points
N = 50
random = Random()
maxIters = 5001
numTrials = 5

points = [[0 for x in xrange(2)] for x in xrange(N)]
for i in range(0, len(points)):
    points[i][0] = random.nextDouble()
    points[i][1] = random.nextDouble()
outfile = OUTPUT_DIRECTORY + '/TSP/TSP_{}_{}_LOG.csv'
ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscretePermutationDistribution(N)
nf = SwapNeighbor()
mf = SwapMutation()
cf = TravelingSalesmanCrossOver(ef)
hcp = GenericHillClimbingProblem(ef, odd, nf)
gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)

print "Finish Generating Points!\n"

# MIMIC
fill = [N] * N
ranges = array('i', fill)
ef = TravelingSalesmanRouteEvaluationFunction(points)
odd = DiscreteUniformDistribution(ranges)

for t in range(numTrials):
    for samples, keep, m in product([200,150], [100,75], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        fname = outfile.format('MIMIC{}_{}_{}'.format(samples, keep, m), str(t + 1))
        df = DiscreteDependencyTree(m, ranges)
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        ef = TravelingSalesmanRouteEvaluationFunction(points)
        pop = GenericProbabilisticOptimizationProblem(ef, odd, df)
        mimic = MIMIC(samples, keep, pop)
        fit = FixedIterationTrainer(mimic, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            fevals = ef.fevals
            score = ef.value(mimic.getOptimal())
            ef.fevals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print st
            with open(fname, 'a') as f:
                f.write(st)

print "Finish MIMIC!\n"

# RHC
for t in range(numTrials):
    fname = outfile.format('RHC', str(t + 1))
    with open(fname, 'w') as f:
        f.write('iterations,fitness,time,fevals\n')
    ef = TravelingSalesmanRouteEvaluationFunction(points)
    hcp = GenericHillClimbingProblem(ef, odd, nf)
    rhc = RandomizedHillClimbing(hcp)
    fit = FixedIterationTrainer(rhc, 10)
    times = [0]
    for i in range(0, maxIters, 10):
        start = clock()
        fit.train()
        elapsed = time.clock() - start
        times.append(times[-1] + elapsed)
        fevals = ef.fevals
        score = ef.value(rhc.getOptimal())
        ef.fevals -= 1
        st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
        print st
        with open(fname, 'a') as f:
            f.write(st)

print "Finish RHC!\n"

# SA
for t in range(numTrials):
    for CE in [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]:
        fname = outfile.format('SA{}'.format(CE), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        ef = TravelingSalesmanRouteEvaluationFunction(points)
        hcp = GenericHillClimbingProblem(ef, odd, nf)
        sa = SimulatedAnnealing(1E10, CE, hcp)
        fit = FixedIterationTrainer(sa, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            fevals = ef.fevals
            score = ef.value(sa.getOptimal())
            ef.fevals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print st
            with open(fname, 'a') as f:
                f.write(st)

print "Finish SA!\n"

# GA
for t in range(numTrials):
    for pop, mate, mutate in product([200, 100, 50], [100, 50, 25], [100, 50, 25]):
        fname = outfile.format('GA{}_{}_{}'.format(pop, mate, mutate), str(t + 1))
        with open(fname, 'w') as f:
            f.write('iterations,fitness,time,fevals\n')
        ef = TravelingSalesmanRouteEvaluationFunction(points)
        gap = GenericGeneticAlgorithmProblem(ef, odd, mf, cf)
        ga = StandardGeneticAlgorithm(pop, mate, mutate, gap)
        fit = FixedIterationTrainer(ga, 10)
        times = [0]
        for i in range(0, maxIters, 10):
            start = clock()
            fit.train()
            elapsed = time.clock() - start
            times.append(times[-1] + elapsed)
            fevals = ef.fevals
            score = ef.value(ga.getOptimal())
            ef.fevals -= 1
            st = '{},{},{},{}\n'.format(i, score, times[-1], fevals)
            print st
            with open(fname, 'a') as f:
                f.write(st)

print "Finish GA!\n"