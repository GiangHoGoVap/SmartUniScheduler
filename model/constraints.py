from model.utils import UNIVERSE, slot_of, _pretty_table
from model.individual import Individual

import pandas as pd
import numpy as np
import skfuzzy as fuzz

class Constraint:
    def __init__(self, name, is_soft=False, weight=1):
        self.name = name
        self.violations = 0
        self.is_soft = is_soft
        self.weight = weight

    def evaluate(self, individual: Individual):
        return 0

class ValidDayConstraint(Constraint):
    def __init__(self, valid_days, is_lab=False):
        super().__init__('Valid day constraint' if not is_lab else 'Valid day lab constraint')
        self.valid_days = valid_days

    def evaluate(self, individual: Individual):
        self.violations = 0
        day = int(individual.bitstring[:4], 2)
        if day not in self.valid_days:
            self.violations = 1
            individual.add_violation(ValidDayConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations 

class SessionStartConstraint(Constraint):
    def __init__(self, course_info, is_lab=False):
        super().__init__('Session start constraint' if not is_lab else 'Session start lab constraint')
        self.course_info = course_info
        self.is_lab = is_lab

    def evaluate(self, individual: Individual):
        self.violations = 0
        session_start = int(individual.bitstring[4:8], 2)
        
        if self.is_lab:
            if session_start not in {2, 8}:
                self.violations = 1
                individual.add_violation(SessionStartConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        else:
            for index, row in self.course_info.iterrows():
                if row['course_id'] == individual.course_id:
                    if session_start + row['num_sessions'] - 1 not in range(3, 13):
                        self.violations = 1
                        individual.add_violation(SessionStartConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations

class LunchBreakConstraint(Constraint):
    def __init__(self):
        super().__init__('Lunch break constraint')

    def evaluate(self, individual: Individual):
        self.violations = 0
        session_start = int(individual.bitstring[4:8], 2)
        if session_start == 6:
            self.violations = 1
            individual.add_violation(LunchBreakConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations
    
class MidtermBreakConstraint(Constraint):
    def __init__(self):
        super().__init__('Midterm break constraint')

    def evaluate(self, individual: Individual):
        self.violations = 0
        weeks = list(individual.bitstring[8:])
        if weeks[7] == '1':
            self.violations = 1
            individual.add_violation(MidtermBreakConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations
    
class CourseDurationConstraint(Constraint):
    def __init__(self, course_info, is_lab=False):
        super().__init__('Course duration constraint' if not is_lab else 'Course duration lab constraint')
        self.course_info = course_info
        self.is_lab = is_lab

    def evaluate(self, individual: Individual):
        self.violations = 0
        weeks = list(individual.bitstring[8:])
        total_duration = sum(int(week) for week in weeks)

        for index, row in self.course_info.iterrows():
            if row['course_id'] == individual.course_id:
                if self.is_lab:
                    expected_duration = row['num_weeks_lab']
                else:
                    expected_duration = row['num_weeks']
                if total_duration != expected_duration:
                    self.violations = 1
                    individual.add_violation(CourseDurationConstraint.__name__, [f"{individual.course_id}-{individual.group_id}"])
        return self.violations

class CourseSameSemesterConstraint(Constraint):
    def __init__(self, course_info: pd.DataFrame):
        super().__init__("Course same semester constraint")
        self.semester_courses = { sem: set(course_info.loc[course_info.semester == sem, "course_id"]) for sem in range(1, 9) }
        self.duration = dict(course_info[["course_id", "num_sessions"]].itertuples(index=False))

    @staticmethod
    def _day(bitstring: str) -> int:
        return int(bitstring[:4], 2)

    @staticmethod
    def _start(bitstring: str) -> int:
        return int(bitstring[4:8], 2)

    @staticmethod
    def _weeks_mask(bitstring: str) -> int:
        return int(bitstring[8:], 2)

    def evaluate_population(self, population: list["Individual"]) -> int:
        self.violations = 0

        # Bucket by semester → group → day  (dict of dicts of lists)
        bucket: dict[int, dict[str, dict[int, list[tuple]]]] = {s: {} for s in range(1, 9)}

        for ind in population:
            # skip courses not in the lookup (safety)
            for sem, course_set in self.semester_courses.items():
                if ind.course_id in course_set:
                    gdict = bucket[sem].setdefault(ind.group_id, {})
                    dlist = gdict.setdefault(self._day(ind.bitstring), [])
                    dlist.append(ind)
                    break  # found its semester

        # Now sweep
        for sem_dict in bucket.values():
            for day_dict in sem_dict.values():
                for day_inds in day_dict.values():
                    # sort by session_start
                    day_inds.sort(key=lambda x: self._start(x.bitstring))

                    prev_end = -1
                    prev_mask = 0
                    prev_ind = None

                    for ind in day_inds:
                        start = self._start(ind.bitstring)
                        dur   = self.duration[ind.course_id]
                        end   = start + dur
                        mask  = self._weeks_mask(ind.bitstring)

                        if start < prev_end and (mask & prev_mask):
                            # overlap in time AND same weeks
                            ind.add_violation(self.name, [prev_ind.course_id])
                            prev_ind.add_violation(self.name, [ind.course_id])
                            self.violations += 1

                        # update sweep pointer
                        if end > prev_end:
                            prev_end, prev_mask, prev_ind = end, mask, ind

        return self.violations

class CourseSameSemesterLabConstraint(Constraint):
    def __init__(self, course_info: pd.DataFrame, course_lookup: pd.DataFrame):
        super().__init__("Course same semester lab constraint")

        self.semester_courses = { sem: set(course_info.loc[course_info.semester == sem, "course_id"]) for sem in range(1, 9) }
        self.lab_dur = dict(course_lookup[["course_id", "num_lab_sessions"]].dropna().itertuples(index=False))
        self.lec_dur = dict(course_lookup[["course_id", "num_sessions"]].dropna().itertuples(index=False))

    @staticmethod
    def _day(bitstring: str) -> int:
        return int(bitstring[:4], 2)

    @staticmethod
    def _start(bitstring: str) -> int:
        return int(bitstring[4:8], 2)

    @staticmethod
    def _weeks_mask(bitstring: str) -> int:
        return int(bitstring[8:], 2)         

    def evaluate_population(self, lab_pop: list["Individual"], lecture_pop: list["Individual"]) -> int:
        self.violations = 0

        # ---- Build buckets -------------------------------------------------
        by_sem_grp_day: dict[int, dict[str, dict[int, list["Individual"]]]] = { s: {} for s in range(1, 9) }

        # bucket labs
        for lab in lab_pop:
            for sem, course_set in self.semester_courses.items():
                if lab.course_id in course_set:
                    gdict = by_sem_grp_day[sem].setdefault(lab.group_id.split("-")[1], {})
                    gdict.setdefault(self._day(lab.bitstring), []).append(lab)
                    break  # found semester

        # bucket lectures (keyed by full group id)
        lect_by_sem_grp_day: dict[int, dict[str, dict[int, list["Individual"]]]] = { s: {} for s in range(1, 9) }
        for lec in lecture_pop:
            for sem, course_set in self.semester_courses.items():
                if lec.course_id in course_set:
                    gdict = lect_by_sem_grp_day[sem].setdefault(lec.group_id, {})
                    gdict.setdefault(self._day(lec.bitstring), []).append(lec)
                    break

        # ---- Check lab ↔ lab overlaps in same group & semester -------------
        for sem_dict in by_sem_grp_day.values():
            for grp_dict in sem_dict.values():
                for day, labs in grp_dict.items():
                    # sort by start
                    labs.sort(key=lambda x: self._start(x.bitstring))

                    prev_end = -1
                    prev_mask = 0
                    prev_lab = None

                    for lab in labs:
                        s = self._start(lab.bitstring)
                        d = self.lab_dur.get(lab.course_id, 2)
                        e = s + d
                        m = self._weeks_mask(lab.bitstring)

                        if s < prev_end and (m & prev_mask):
                            lab.add_violation(self.name, [prev_lab.course_id])
                            prev_lab.add_violation(self.name, [lab.course_id])
                            self.violations += 1

                        if e > prev_end:
                            prev_end, prev_mask, prev_lab = e, m, lab

        # ---- Check lab ↔ lecture overlaps within same group ----------------
        for sem in range(1, 9):
            lab_grp = by_sem_grp_day[sem]
            lec_grp = lect_by_sem_grp_day[sem]

            for grp_id, labs_by_day in lab_grp.items():
                if grp_id not in lec_grp:
                    continue
                for day, labs in labs_by_day.items():
                    lectures = lec_grp[grp_id].get(day, [])
                    if not lectures:
                        continue

                    # One pass over lectures per day (rarely many)
                    for lab in labs:
                        ls = self._start(lab.bitstring)
                        ld = self.lab_dur.get(lab.course_id, 2)
                        le = ls + ld
                        lm = self._weeks_mask(lab.bitstring)

                        for lec in lectures:
                            ss = self._start(lec.bitstring)
                            dd = self.lec_dur.get(lec.course_id, 2)
                            ee = ss + dd
                            mm = self._weeks_mask(lec.bitstring)

                            if ls < ee and ss < le and (lm & mm):
                                self.violations += 1

        # ---- Room conflicts across all groups ------------------------------
        room_day_map: dict[tuple[str, int], list[tuple[int, int, int]]] = {}
        # key → list of (start, end, week_mask)

        for lab in lab_pop:
            key = (lab.room, self._day(lab.bitstring))
            lst = room_day_map.setdefault(key, [])
            s   = self._start(lab.bitstring)
            d   = self.lab_dur.get(lab.course_id, 2)
            e   = s + d
            m   = self._weeks_mask(lab.bitstring)

            lst.append((s, e, m))

        # For each room‑day, sort and sweep
        for lst in room_day_map.values():
            lst.sort()
            prev_s, prev_e, prev_m = lst[0]
            for s, e, m in lst[1:]:
                if s < prev_e and (m & prev_m):
                    self.violations += 1
                if e > prev_e:                 # update sweep pointer
                    prev_s, prev_e, prev_m = s, e, m

        return self.violations

class LectureBeforeLabConstraint(Constraint):
    def __init__(self, min_weeks_after=3, penalty_weight=1):
        super().__init__('Lecture-before-laboratory dependency constraint')
        self.min_weeks_after = min_weeks_after  # Minimum number of weeks the lab must start after the lecture
        self.penalty_weight = penalty_weight  # Scaling factor for the penalty

    def evaluate_population(self, lab_population, lecture_population):
        penalties = [0] * len(lab_population)

        # Precompute lecture start weeks for faster lookup
        lecture_start_weeks = {}
        for lecture in lecture_population:
            course_id, group_id, bitstring = lecture.course_id, lecture.group_id, lecture.bitstring
            weeks = list(bitstring[8:])  
            start_week = weeks.index('1') if '1' in weeks else -1  
            lecture_start_weeks[f"{course_id}-{group_id}"] = start_week

        # Check labs and assign penalties
        for i, lab in enumerate(lab_population):
            lab_course_id, lab_group_id, lab_bitstring = lab.course_id, lab.group_id.split('-')[1], lab.bitstring
            lab_weeks = list(lab_bitstring[8:])  
            lab_start_week = lab_weeks.index('1') if '1' in lab_weeks else -1 

            lecture_key = f"{lab_course_id}-{lab_group_id}"
            lecture_start_week = lecture_start_weeks.get(lecture_key, -1)

            if lecture_start_week != -1 and lab_start_week != -1:
                actual_gap = lab_start_week - lecture_start_week
                violation_amount = max(0, self.min_weeks_after - actual_gap)
                penalties[i] = violation_amount * self.penalty_weight  # Dynamic penalty
            else:
                # Handle invalid cases (e.g., no '1' in weeks or no corresponding lecture)
                penalties[i] = self.penalty_weight  # Assign a fixed penalty for invalid cases

        return penalties

class LabSessionSpacingConstraint(Constraint):
    def __init__(self):
        super().__init__('Lab session spacing constraint')
        
    def evaluate(self, individual: Individual):
        self.violations = 0

        weeks = list(individual.bitstring[8:])
        scheduled_weeks = [idx for idx, week in enumerate(weeks) if week == '1']

        # Check for consecutive weeks
        for j in range(len(scheduled_weeks) - 1):
            if scheduled_weeks[j + 1] == scheduled_weeks[j] + 1:
                self.violations += 1

        return self.violations

class SoftSlotPreferenceConstraint(Constraint):
    def __init__(self, prefs_slot: dict, course_teacher: dict, duration_lookup: dict):    
        super().__init__("Soft slot preference", is_soft=True)
        self.prefs_slot = prefs_slot
        self.course_teacher = course_teacher
        self.duration_lookup = duration_lookup

    @staticmethod
    def rect_mf(x, start_idx, end_idx):
        return np.where((x >= start_idx) & (x <= end_idx), 1.0, 0.0)

    def _membership_array(self, spec):
        if spec["shape"] == "rect":
            return self.rect_mf(UNIVERSE, spec["start_idx"], spec["end_idx"])
        else:
            return fuzz.trapmf(UNIVERSE, [spec["a"], spec["b"], spec["c"], spec["d"]])

    def evaluate(self, ind: Individual) -> float | None:
        key = f"{ind.course_id}-{ind.group_id}"
        teacher = self.course_teacher.get(key)
        if teacher is None:
            return None                     

        spec = self.prefs_slot.get(teacher, {}).get(key)
        if spec is None:
            return None

        mu_all = self._membership_array(spec)   

        # scheduled rectangle indices
        day = int(ind.bitstring[:4], 2)
        session_start = int(ind.bitstring[4:8], 2)
        start_idx = slot_of.get((day, session_start))
        duration = self.duration_lookup.get(ind.course_id, 1)
        block = slice(start_idx, start_idx + duration)

        # area of overlap and total preference area
        overlap = mu_all[block].sum()
        pref_area = mu_all.sum()
        if pref_area == 0:
            return 1.0            

        return overlap / pref_area      

class ConstraintsManager:
    def __init__(self):
        self.constraints = []

    def add_constraint(self, constraint):
        self.constraints.append(constraint)

    def evaluate(self, individual: Individual):
        total_violations = 0
        for constraint in self.constraints:
            if constraint.is_soft is False:
                violations = constraint.evaluate(individual)
                if violations is not None:
                    total_violations += violations
        return 1 / (1 + total_violations)

    def evaluate_population(self, population, is_lab=False, lecture_population=None):
        total_penalties = [0] * len(population)  
        
        for constraint in self.constraints:
            if isinstance(constraint, LectureBeforeLabConstraint):
                if lecture_population is None:
                    raise ValueError("Lecture population is required for LectureBeforeLabConstraint.")
                penalties = constraint.evaluate_population(population, lecture_population)
                if penalties is not None:
                    for i, penalty in enumerate(penalties):
                        total_penalties[i] += penalty
            else:
                if is_lab and constraint.name == 'Course same semester lab constraint':
                    violations = constraint.evaluate_population(population, lecture_population)
                    if violations is not None:
                        for i in range(len(population)):
                            total_penalties[i] += violations  
                elif not is_lab and constraint.name == 'Course same semester constraint':
                    violations = constraint.evaluate_population(population)
                    if violations is not None:
                        for i in range(len(population)):
                            total_penalties[i] += violations  
                else:
                    continue  
        
        fitness_scores = [1 / (1 + penalty**2) for penalty in total_penalties]
        return fitness_scores

    def evaluate_soft_individual(self, individual: Individual):
        total_score = 0
        for constraint in self.constraints:
            if constraint.is_soft:
                score = constraint.evaluate(individual)
                if score is not None:
                    total_score += score
        return total_score
    
    def reset_violations(self):
        for constraint in self.constraints:
            constraint.violations = 0

    def count_violations(self, population: list["Individual"], lecture_population: list["Individual"] | None = None, verbose: bool = True):
        hard_dict: dict[str, int]    = {}
        soft_dict: dict[str, float]  = {}

        # ---------- initialise ----------------------------------------
        for idx, c in enumerate(self.constraints):
            key = f"{c.name}_{idx}"
            if getattr(c, "is_soft", False):
                soft_dict[key] = 0.0
            else:
                hard_dict[key] = 0

        # ---------- population‑level constraints ----------------------
        for idx, c in enumerate(self.constraints):
            key = f"{c.name}_{idx}"
            if isinstance(c, LectureBeforeLabConstraint):
                if lecture_population is None:
                    raise ValueError("Lecture population required for LectureBeforeLabConstraint")
                pen = c.evaluate_population(population, lecture_population)
                hard_dict[key] += sum(pen) if not c.is_soft else 0

            elif c.name in {"Course same semester constraint", "Course same semester lab constraint"}:
                is_lab = c.name == "Course same semester lab constraint"
                if is_lab:
                    viol = c.evaluate_population(population, lecture_population)
                else:
                    viol = c.evaluate_population(population)
                hard_dict[key] += viol   

        # ---------- individual‑level constraints ----------------------
        for idx, c in enumerate(self.constraints):
            key = f"{c.name}_{idx}"

            # skip population‑only constraints
            if c.name in {'Course same semester constraint', 'Course same semester lab constraint'}:
                continue

            for ind in population:
                value = c.evaluate(ind)     

                if getattr(c, "is_soft", False):
                    if value is not None:
                        soft_dict[key] += float(value)
                else:
                    hard_dict[key] += int(value)

        if verbose:
            _pretty_table("HARD VIOLATIONS", hard_dict, is_soft=False)
            _pretty_table("SOFT SATISFACTION", soft_dict, is_soft=True)

        return hard_dict, soft_dict

