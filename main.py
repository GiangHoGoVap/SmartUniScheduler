from model.scheduler import (create_initial_population, create_initial_population_lab)
from model.encoder import read_excel, preprocess, encode
from model.decoder import decode_solution, decode_solution_to_dataframe, save_decoded_schedule
from model.pymoo_custom import TimetableProblem, BlockUniformCrossover, TimetableMutation, FixedSampling, MultiTimetableProblem
from model.constraints import (
    ValidDayConstraint, SessionStartConstraint, LunchBreakConstraint,
    MidtermBreakConstraint, CourseDurationConstraint,
    CourseSameSemesterConstraint, CourseSameSemesterLabConstraint,
    LectureBeforeLabConstraint, LabSessionSpacingConstraint,
    ConstraintsManager, SoftSlotPreferenceConstraint
)
from config import (SHEET_FILE_NAME, SHEET_NAMES, VALID_DAYS, VALID_DAYS_LAB)

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

import os
import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt

def initialize_constraints(df1, df2, is_lab=False):
    cm = ConstraintsManager()
    cm.add_constraint(MidtermBreakConstraint())
    cm.add_constraint(SessionStartConstraint(df1, is_lab))
    cm.add_constraint(SessionStartConstraint(df2, is_lab))
    cm.add_constraint(CourseDurationConstraint(df1, is_lab))
    cm.add_constraint(CourseDurationConstraint(df2, is_lab))

    if is_lab:
        cm.add_constraint(ValidDayConstraint(VALID_DAYS_LAB, is_lab))
        cm.add_constraint(CourseSameSemesterLabConstraint(df1, config.df_course_lookup))
        cm.add_constraint(CourseSameSemesterLabConstraint(df2, config.df_course_lookup))
        cm.add_constraint(LectureBeforeLabConstraint())
        cm.add_constraint(LabSessionSpacingConstraint())
    else:
        cm.add_constraint(ValidDayConstraint(VALID_DAYS))
        cm.add_constraint(LunchBreakConstraint())
        cm.add_constraint(CourseSameSemesterConstraint(df1))
        cm.add_constraint(CourseSameSemesterConstraint(df2))
        cm.add_constraint(SoftSlotPreferenceConstraint(config.PREFS_slot, config.COURSE_TEACHER, config.duration_lookup))
    return cm

def main():
    file_path = os.path.join("data", SHEET_FILE_NAME)
    df, df_khmt, df_ktmt, df_pref = [read_excel(file_path, s) for s in SHEET_NAMES]

    df_main = preprocess(df, 0)
    df_khmt = preprocess(df_khmt, 1)
    df_ktmt = preprocess(df_ktmt, 2)

    config.df_course_lookup = pd.concat([df_khmt, df_ktmt], ignore_index=True).drop_duplicates(subset="course_id")

    config.duration_lookup = (
        pd.concat(
            [df_khmt[["course_id", "num_sessions"]],
             df_ktmt[["course_id", "num_sessions"]]],
            ignore_index=True)
        .drop_duplicates(subset="course_id")
        .set_index("course_id")["num_sessions"]
        .astype(int)
        .to_dict()
    )

    df_pref = preprocess(df_pref, 3)          

    chromosomes_df, chromosomes_lab_df = encode(df_main)

    constraints_manager_lec = initialize_constraints(df_khmt, df_ktmt, is_lab=False)
    constraints_manager_lab = initialize_constraints(df_khmt, df_ktmt, is_lab=True)

    prob_lec = TimetableProblem(chromosomes_df, is_lab=False, constraints_manager=constraints_manager_lec)
    cx_lec = BlockUniformCrossover(len(chromosomes_df))
    mut_lec = TimetableMutation(len(chromosomes_df), valid_days=VALID_DAYS)

    init_pop_lec = create_initial_population(30, chromosomes_df, df_khmt, df_ktmt)
    ga_lec = GA(pop_size=30, sampling=FixedSampling(init_pop_lec), crossover=cx_lec, mutation=mut_lec, eliminate_duplicates=True)
    res_lec = minimize(prob_lec, ga_lec, ("n_gen", 50), seed=1, verbose=True, save_history=True)

    # Extract history of best fitness values
    best_fitness_per_gen_lec = [algo.pop.get("F").min() for algo in res_lec.history]

    # Plot it
    # plt.plot(best_fitness_per_gen_lec, marker='o', linestyle='-')
    # plt.title("Best Fitness Value per Generation (Lecture)")
    # plt.xlabel("Generation")
    # plt.ylabel("Fitness")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    best_solution_lec = res_lec.X
    best_fitness_lec = res_lec.F[0]          
    config.lecture_population = decode_solution(best_solution_lec, chromosomes_df, is_lab=False)
    df_lecture = decode_solution_to_dataframe(best_solution_lec, chromosomes_df, is_lab=False)
    best_inds_lecture = decode_solution(best_solution_lec, chromosomes_df, is_lab=False)
    
    prob_lab = TimetableProblem(chromosomes_lab_df, is_lab=True, constraints_manager=constraints_manager_lab, lecture_pop=config.lecture_population)   
    cx_lab = BlockUniformCrossover(len(chromosomes_lab_df))
    mut_lab = TimetableMutation(len(chromosomes_lab_df), valid_days=VALID_DAYS_LAB, is_lab=True)

    init_pop_lab = create_initial_population_lab(30, chromosomes_lab_df, df_khmt, df_ktmt, config.lecture_population)
    ga_lab = GA(pop_size=30, sampling=FixedSampling(init_pop_lab), crossover=cx_lab, mutation=mut_lab, eliminate_duplicates=True)
    res_lab = minimize(prob_lab, ga_lab, ("n_gen", 200), seed=1, verbose=True, save_history=True)

    best_solution_lab = res_lab.X
    best_fitness_lab = res_lab.F[0]
    df_lab = decode_solution_to_dataframe(best_solution_lab, chromosomes_lab_df, is_lab=True, room_list=config.lab_rooms)
    best_inds_lab = decode_solution(best_solution_lab, chromosomes_lab_df, is_lab=True, room_list=config.lab_rooms)

    hard_lec, soft_lec = prob_lec.constraints_manager.count_violations(best_inds_lecture, verbose=True)
    hard_lab, soft_lab = prob_lab.constraints_manager.count_violations(best_inds_lab, lecture_population=best_inds_lecture, verbose=True)

    print("-" * 50)
    print("Best fitness (Lecture):", best_fitness_lec)
    print("Best fitness (Lab)    :", best_fitness_lab)

    save_decoded_schedule(df_lecture=df_lecture, df_lab=df_lab)

    # Extract history of best fitness values
    best_fitness_per_gen_lab = [algo.pop.get("F").min() for algo in res_lab.history]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_per_gen_lec, label="Lecture", marker='o')
    plt.plot(best_fitness_per_gen_lab, label="Lab", marker='x')

    plt.title("Best Fitness Over Generations For First Semester 2024-2025")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # prob_lec_multi = MultiTimetableProblem(chromosomes_df, is_lab=False, constraints_manager=constraints_manager_lec)
    # nsga = NSGA2(pop_size=30, sampling=FixedSampling(init_pop_lec), crossover=cx_lec, mutation=mut_lec, eliminate_duplicates=True)
    # res_lec_multi = minimize(prob_lec_multi, nsga, ("n_gen", 100), seed=1, verbose=True)

    # F = res_lec_multi.F        
    # X_pop = res_lec_multi.X        

    # mask_feasible = F[:,0] == 0
    # if not mask_feasible.any():
    #     mask_feasible = np.ones(len(F), dtype=bool)

    # F_feas = F[mask_feasible]
    # X_feas = X_pop[mask_feasible]

    # if len(F_feas) > 2:
    #     p_min = F_feas.min(axis=0)
    #     p_max = F_feas.max(axis=0)
    #     vec   = p_max - p_min
    #     vec_n = vec / np.linalg.norm(vec)

    #     def dist_to_line(pt):
    #         return np.linalg.norm( (pt - p_min) - np.dot(pt - p_min, vec_n) * vec_n )

    #     dists = np.apply_along_axis(dist_to_line, 1, F_feas)
    #     knee_idx  = dists.argmax()
    # else:
    #     knee_idx  = 0                # only one or two feasible → take first

    # X_knee = X_feas[knee_idx]
    # F_knee = F_feas[knee_idx]

    # # 3)  Decode & print
    # knee_inds  = decode_solution(X_knee, chromosomes_df, is_lab=False)
    # df_knee    = decode_solution_to_dataframe(X_knee, chromosomes_df, is_lab=False)

    # print("\n" + "="*70)
    # print("NSGA‑II  Pareto knee (lecture timetable)")
    # print(f"  Hard violations : {int(F_knee[0])}")
    # print(f"  Soft score      : {F_knee[1]:.3f}")
    # print("="*70 + "\n")

    # hard_lec_multi, soft_lec_multi = prob_lec_multi.constraints_manager.count_violations(knee_inds, verbose=True)

    # prob_lab_multi = MultiTimetableProblem(chromosomes_lab_df, is_lab=True, constraints_manager = constraints_manager_lab, lecture_pop = knee_inds)

    # init_pop_lab = create_initial_population_lab(30, chromosomes_lab_df, df_khmt, df_ktmt, knee_inds)

    # nsga_lab = NSGA2(pop_size=30, sampling=FixedSampling(init_pop_lab), crossover=cx_lab, mutation=mut_lab, eliminate_duplicates=True)

    # res_lab_multi = minimize(prob_lab_multi, nsga_lab, ("n_gen", 100), seed=1, verbose=True)

    # F_lab   = res_lab_multi.F
    # X_lab   = res_lab_multi.X

    # mask_feasible = F_lab[:, 0] == 0
    # if not mask_feasible.any():
    #     mask_feasible = np.ones(len(F_lab), dtype=bool)

    # F_lab_feas = F_lab[mask_feasible]
    # X_lab_feas = X_lab[mask_feasible]

    # if len(F_lab_feas) > 2:
    #     p_min = F_lab_feas.min(axis=0)
    #     p_max = F_lab_feas.max(axis=0)
    #     vec_n = (p_max - p_min) / np.linalg.norm(p_max - p_min)

    #     def _dist(pt):
    #         return np.linalg.norm((pt - p_min) -
    #                             np.dot(pt - p_min, vec_n) * vec_n)

    #     knee_lab_idx = np.argmax([_dist(pt) for pt in F_lab_feas])
    # else:
    #     knee_lab_idx = 0

    # X_knee_lab = X_lab_feas[knee_lab_idx]
    # F_knee_lab = F_lab_feas[knee_lab_idx]

    # knee_lab_inds = decode_solution(X_knee_lab, chromosomes_lab_df, is_lab=True, room_list=config.lab_rooms)

    # df_knee_lab = decode_solution_to_dataframe(X_knee_lab, chromosomes_lab_df, is_lab=True, room_list=config.lab_rooms)

    # print("\n" + "="*70)
    # print("NSGA‑II  Pareto knee (lab timetable)")
    # print(f"  Hard violations : {int(F_knee_lab[0])}")
    # print(f"  Soft score      : {F_knee_lab[1]:.3f}")
    # print("="*70 + "\n")

    # hard_lab_multi, soft_lab_multi = prob_lab_multi.constraints_manager.count_violations(knee_lab_inds, lecture_population=knee_lab_inds, verbose=True)    

if __name__ == "__main__":
    main()
