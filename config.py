import numpy as np

SHEET_FILE_NAME = 'data.xlsx'
SHEET_NAMES = ['Thống kê', 'KHMT', 'KTMT', 'Đề xuất']
CLASS_BLOCK_SIZE = 18
VALID_DAYS = range(2, 7)
VALID_DAYS_LAB = range(2, 7)
VALID_SESSIONS = range(2, 12)

ROOMS = {
    'CO3093': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-603', 'H6-604', 'H6-702', 'H6-703', 'H6-708']
    },
    'CO3097': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-603', 'H6-604', 'H6-702', 'H6-703', 'H6-708']
    },
    'CO2013': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    },
    'CO2007': {
        'CS1': ['C5-202'],
        'CS2': ['H6-601', 'H6-605']
    },
    'CO2003': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    },
    'CO1023': {
        'CS1': ['C5-202', 'C6-105'],
        'CS2': ['H6-601', 'H6-605']
    },
    'CO3053': {
        'CS1': ['C6-105'],
        'CS2': ['H6-601']
    },
    'CO2017': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    },
    'CO3009': {
        'CS1': ['C5-202', 'C6-105'],
        'CS2': ['H6-601']
    },
    'CO2037': {
        'CS1': ['C5-202', 'C6-105'],
        'CS2': ['H6-601', 'H6-605']
    },
    'CO1005': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    },
    'CO1025': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    },
    'CO1027': {
        'CS1': ['C6-102', 'C6-103', 'C6-104', 'C6-509'],
        'CS2': ['H6-701', 'H6-702', 'H6-703', 'H6-707', 'H6-603', 'H6-604']
    }
}

PROGRAM_ID = {
    "Chương trình giảng dạy bằng tiếng Anh": "CC",
    "Chương trình tiêu chuẩn": "CQ",
    "Chương trình định hướng Nhật Bản": "CN"
}
REVERSE_PROGRAM_ID = { v: k for k, v in PROGRAM_ID.items() }

COLUMN_MAP = {
    "Mã môn học": "course_id",
    "Tên môn học": "course_name",
    "Loại hình lớp": "program_id",
    "Số SV dự kiến": "num_students",
    "Số lượng nhóm": "num_groups",
    "Sỉ số SV max": "max_students",
    "Tổng LT": "num_lectures",
    "Tổng TH": "num_lab_lectures",
    "Số tiết LT": "num_sessions",
    "Số tiết TH": "num_lab_sessions",
    "Học kỳ": "semester",
    "Thực hành": "is_lab",
    "MSCB": "instructor_id",
    "Mã nhóm": "group_id",
    "Mô tả": "description"
}

COURSE_ID_TO_NAME = {}
ROOM_TYPE_ID = {}
PREFS = {}                # nested dict teacher → course-key → spec
COURSE_TEACHER = {}       # course-key → teacher  (needed by constraint)
PREFS_slot = {}

lab_rooms = []
lecture_population = []
df_course_lookup = None 
duration_lookup = {}

SLOTS = [(day, sess) for day in range(2, 8) for sess in range(2, 13)]       
UNIVERSE = np.arange(len(SLOTS)) 

slot_of = { ds: idx for idx, ds in enumerate(SLOTS) }  
day_sess_of_slot = dict(enumerate(SLOTS))   