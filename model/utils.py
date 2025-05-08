# Ex: lst = [1, 2, 3, 2, 4, 5, 1, 6, 3]
# Output:
# {
#     2: [1, 3],
#     1: [0, 6],
#     3: [2, 8]
# }
def find_duplicates(lst):
    duplicates = {}
    seen = {}
    
    for i, value in enumerate(lst):
        if value in seen:
            if value in duplicates:
                duplicates[value].append(i)
            else:
                duplicates[value] = [seen[value], i]
        else:
            seen[value] = i
    
    return duplicates

# Ex:     list = [1, 3, 5, 7]
#         num_sessions = [2, 2, 2, 2]
# Output: 0
# def count_overlap(list, num_sessions):
#     overlap_cases = 0
#     for i in range(len(list) - 1):
#         if num_sessions[i] == 0:
#             continue
#         for j in range(i + 1, len(list)):
#             if num_sessions[j] == 0:
#                 continue
#             start1, end1 = list[i], list[i] + num_sessions[i]
#             start2, end2 = list[j], list[j] + num_sessions[j]

#             if start1 < end2 and start2 < end1:
#                 overlap_cases += 1
#     return overlap_cases

def count_overlap(session_starts, num_sessions):
    session_ends = [start + num for start, num in zip(session_starts, num_sessions)]
    sorted_intervals = sorted(zip(session_starts, session_ends))
    overlaps = 0
    for i in range(len(sorted_intervals) - 1):
        if sorted_intervals[i][1] > sorted_intervals[i + 1][0]:
            overlaps += 1
    return overlaps