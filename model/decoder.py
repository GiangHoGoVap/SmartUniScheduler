from model.individual import Individual
from config import CLASS_BLOCK_SIZE, REVERSE_PROGRAM_ID
import pandas as pd
import re
import config

def decode_solution(solution, class_ids, is_lab=False, room_list=None):
    individuals = []

    for i, class_id in enumerate(class_ids):
        start = i * CLASS_BLOCK_SIZE
        chunk = solution[start:start + CLASS_BLOCK_SIZE]
        day, session, weeks = chunk[0], chunk[1], chunk[2:]

        day_bits = format(day, '04b')
        session_bits = format(session, '04b')
        week_bits = ''.join(str(w) for w in weeks)
        bitstring = day_bits + session_bits + week_bits

        if is_lab:
            course_id, lab_id, group_id = class_id.split('-')
            room = room_list[i] if room_list else None
            full_group_id = f"{lab_id}-{group_id}"
            individuals.append(Individual(course_id, full_group_id, bitstring, "lab", room))
        else:
            course_id, group_id = class_id.split('-')
            individuals.append(Individual(course_id, group_id, bitstring))

    return individuals

### --- MARK: Output Utilities ---
def decode_solution_to_dataframe(solution, class_ids, is_lab=False, room_list=None):
    decoded_rows = []

    for i, class_id in enumerate(class_ids):
        start = i * CLASS_BLOCK_SIZE
        chunk = solution[start:start + CLASS_BLOCK_SIZE]

        day = chunk[0]
        session = chunk[1]
        weeks = chunk[2:]

        if is_lab:
            course_id, lab_id, group_id = class_id.split('-')
        else:
            course_id, group_id = class_id.split('-')

        # Extract program info
        text, number = re.match(r'([a-zA-Z]+)(\d+)', group_id).groups()
        group_code = group_id.replace('CQ', 'L') if 'CQ' in group_id else group_id

        decoded = {
            'Mã môn học': course_id,
            'Tên môn học': config.COURSE_ID_TO_NAME.get(course_id, 'Unknown'),
            'Loại hình lớp': REVERSE_PROGRAM_ID.get(text, 'Unknown'),
            'Mã nhóm': group_code,
            'Thứ': day,
            'Tiết BD': session
        }

        if is_lab:
            decoded['Loại LAB'] = lab_id
            decoded['Phòng học'] = room_list[i] if room_list else 'Unknown'
        else:
            decoded['Loại phòng'] = config.ROOM_TYPE_ID.get(course_id, 'Unknown')

        # Add weeks
        for w_idx, week in enumerate(weeks):
            decoded[f'Tuần {w_idx + 1}'] = 'x' if week else None

        decoded_rows.append(decoded)

    return pd.DataFrame(decoded_rows)

def save_decoded_schedule(df_lecture=None, df_lab=None, output_path='output/result.xlsx'):
    with pd.ExcelWriter(output_path) as writer:
        if df_lecture is not None:
            df_lecture.to_excel(writer, sheet_name='LT', index=False)
        if df_lab is not None:
            df_lab.to_excel(writer, sheet_name='TH', index=False)
    print(f"Schedule saved to {output_path}")
