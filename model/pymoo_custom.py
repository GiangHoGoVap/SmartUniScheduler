import numpy as np
import config

from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from model.scheduler import evaluate_fitness_generic
from model.decoder import decode_solution
from config import CLASS_BLOCK_SIZE, VALID_SESSIONS

class TimetableProblem(Problem):
    def __init__(self, class_ids, is_lab=False, constraints_manager=None, lecture_pop=None):
        self._class_ids = class_ids
        self._is_lab = is_lab
        self.constraints_manager = constraints_manager
        self._lecture_pop = lecture_pop
        n_var = len(class_ids) * CLASS_BLOCK_SIZE

        super().__init__(n_var=n_var, n_obj=1, xl=np.zeros(n_var, dtype=int), xu=np.full(n_var, 15, dtype=int))

    def _evaluate(self, X, out, *args, **kwargs):
        F = []
        for sol in X:
            inds = decode_solution(sol, self._class_ids, self._is_lab, config.lab_rooms if self._is_lab else None)
            f = evaluate_fitness_generic(inds, self.constraints_manager, self._is_lab, self._lecture_pop)
            F.append(f)                   
        out["F"] = np.array(F)

class MultiTimetableProblem(Problem):
    def __init__(self, class_ids, is_lab=False, constraints_manager=None, lecture_pop=None):
        self.class_ids = class_ids
        self.is_lab = is_lab
        self.constraints_manager = constraints_manager
        self.lecture_pop = lecture_pop
        n_var = len(class_ids) * CLASS_BLOCK_SIZE

        super().__init__(n_var=n_var, n_obj=2, n_constr=0, xl=np.zeros(n_var, int), xu=np.full(n_var, 15, int))

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.zeros((len(X), 2)) # (pop, n_obj)

        for i, sol in enumerate(X):
            inds = decode_solution(sol, self.class_ids, self.is_lab, config.lab_rooms if self.is_lab else None)

            hard_v, soft_v = self.constraints_manager.count_violations(inds, lecture_population=self.lecture_pop, verbose=False)

            hard_total = sum(hard_v.values())
            soft_score = np.mean(list(soft_v.values()) or [0])

            F[i, 0] = hard_total          
            F[i, 1] = soft_score           
        out["F"] = F

class BlockUniformCrossover(Crossover):
    def __init__(self, class_count: int):
        super().__init__(n_parents=2, n_offsprings=1)
        self.class_count = class_count

    def _do(self, problem, X, **kwargs):
        # X shape:  (n_parents, n_matings, n_var)
        n_matings, n_var = X.shape[1], X.shape[2]
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=int)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]
            child  = p1.copy()
            for i in range(self.class_count):
                s = i * CLASS_BLOCK_SIZE
                e = s + CLASS_BLOCK_SIZE
                if np.random.rand() < .5:
                    child[s:e] = p2[s:e]
            Y[0, k] = child
        return Y

class TimetableMutation(Mutation):
    def __init__(self, class_count: int, valid_days, is_lab=False):
        super().__init__()
        self.class_count = class_count
        self.valid_days = valid_days
        self.is_lab = is_lab

    def _do(self, problem, X, **kwargs):
        # X shape (n_individuals, n_var)
        for indiv in X:
            i = np.random.randint(self.class_count)
            block_start = i * CLASS_BLOCK_SIZE
            choice = np.random.choice(["day", "session", "week"])

            if choice == "day":
                indiv[block_start] = np.random.choice(self.valid_days)
            elif choice == "session":
                indiv[block_start + 1] = np.random.choice(VALID_SESSIONS)
            else:                              
                bit = np.random.randint(16)
                idx = block_start + 2 + bit
                indiv[idx] ^= 1

        return X

class FixedSampling(Sampling):
    def __init__(self, initial_pop):
        super().__init__()
        self.initial_pop = np.asarray(initial_pop, dtype=int)

    def _do(self, problem, n_samples, **kwargs):
        return self.initial_pop