class Individual:
    def __init__(self, course_id, group_id, bitstring, individual_type='lecture', room=None):
        self.course_id = course_id             # e.g., "CO2003"
        self.group_id = group_id               # e.g., "L01" or "LAB1-L01"
        self.bitstring = bitstring             # core encoding
        self.individual_type = individual_type # 'lecture' or 'lab'
        self.room = room                       # only used for labs or when needed
        self.violations = {}                   # constraint_name -> list of violations
        self.score = None                      # fitness or penalty score

    @property
    def name(self):
        return f"{self.course_id}-{self.group_id}"

    def add_violation(self, constraint_name, violating_courses):
        if constraint_name not in self.violations:
            self.violations[constraint_name] = {
                "violating_courses": set(violating_courses),
                "count": len(violating_courses)
            }
        else:
            current_courses = self.violations[constraint_name]["violating_courses"]
            current_courses.update(violating_courses)
            self.violations[constraint_name]["count"] = len(current_courses)

    def clear_violations(self):
        self.violations = {}

    def __str__(self):
        parts = [self.name]
        if self.room:
            parts.append(self.room)
        parts.append(self.bitstring)
        return "-".join(parts)
