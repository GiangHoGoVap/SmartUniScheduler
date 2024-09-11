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

def count_overlap(list, num_sessions):
    overlap_cases = 0
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            start1, end1 = list[i], list[i] + num_sessions[i]
            start2, end2 = list[j], list[j] + num_sessions[j]

            if start1 < end2 and start2 < end1:
                overlap_cases += 1
    return overlap_cases