import numpy as np
from tqdm import tqdm
import ray
from simulate_2048_v2 import Batch2048EnvSimulator
from ray.experimental import tqdm_ray

_LUT_LEFT_NEW: np.ndarray | None = None     # uint16[65536]
_LUT_RIGHT_NEW: np.ndarray | None = None    # uint16[65536]
_LUT_LEFT_REWARD: np.ndarray | None = None     # int32[65536]
_LUT_RIGHT_REWARD: np.ndarray | None = None    # int32[65536]

# 스폰 최적화용 클래스 정적 LUT들
_PC4: np.ndarray | None = None              # uint8[16], popcount
_PC16: np.ndarray | None = None             # uint8[65536], popcount
_LUT_EMPTY4_ROW: np.ndarray | None = None   # uint8[65536], 4비트 빈칸 마스크
_LUT_MASK_SELECT: np.ndarray | None = None  # uint8[16,4], (mask, nth)->col(0..3) or 255
_LUT_SELECT16: np.ndarray | None = None     # uint8[65536,16,2], (mask16,nth)->(row,col) or (255,255)

_LUT_TRANSPOSE: np.ndarray | None = None  # uint64[65536]

_LUT_CHOICE_INDEX: np.ndarray | None = None  # int8[16,4], (mask,nth)->index(0..3) or 0

_VALUE_TABLE: np.ndarray | None = None # float32[65536,4]

def able_move(boards):
    lut_left = ray.get(_LUT_LEFT_NEW)
    lut_right = ray.get(_LUT_RIGHT_NEW)
    num_envs = boards.shape[0]
    able_move = np.zeros((num_envs, 4), dtype=bool)
    
    boards_left = lut_left[boards]
    boards_right = lut_right[boards]
    able_move[:, 0] = (boards != boards_left).any(axis=1)
    able_move[:, 1] = (boards != boards_right).any(axis=1)

    boards_copy = boards.copy()
    _transpose_inplace(boards_copy)
    boards_up = lut_left[boards_copy]
    boards_down = lut_right[boards_copy]
    able_move[:, 2] = (boards_copy != boards_up).any(axis=1)
    able_move[:, 3] = (boards_copy != boards_down).any(axis=1)

    return able_move

def all_move_boards(boards):
    lut_left = ray.get(_LUT_LEFT_NEW)
    lut_right = ray.get(_LUT_RIGHT_NEW)
    num_envs = boards.shape[0]
    if num_envs == 0:
        return np.array([], dtype=np.uint16), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    moved_boards = np.zeros((num_envs, 4, 4), dtype=np.uint16)
    
    moved_boards[:, 0] = lut_left[boards]
    moved_boards[:, 1] = lut_right[boards]

    boards_copy = boards.copy()
    _transpose_inplace(boards_copy)
    up_boards = lut_left[boards_copy]
    down_boards = lut_right[boards_copy]
    _transpose_inplace(up_boards)
    _transpose_inplace(down_boards)
    moved_boards[:, 2] = up_boards
    moved_boards[:, 3] = down_boards

    mask = (moved_boards != np.expand_dims(boards, axis=1)).any(axis=2)

    return moved_boards[mask], np.nonzero(mask)[0], np.nonzero(mask)[1]

def empty_space_nums(boards):
    empty4 = Batch2048EnvSimulator._LUT_EMPTY4_ROW  # uint8[65536]
    pc16 = Batch2048EnvSimulator._PC16  # uint8[65536]

    # 1) 행 마스크 → 보드 마스크 16비트 (벡터)
    row_masks = empty4[boards]  # (M,4) uint8 (각 행 4bit)
    board_mask16 = (
        (row_masks[:, 0].astype(np.uint16) << 12) |
        (row_masks[:, 1].astype(np.uint16) << 8) |
        (row_masks[:, 2].astype(np.uint16) << 4) |
        (row_masks[:, 3].astype(np.uint16) << 0)
    )  # (M,) uint16

    # 2) 총 빈칸 수 & 유효 보드
    total_empty = pc16[board_mask16.astype(np.int64)].astype(np.int32)  # (M,)
    return total_empty

def all_next_boards(boards, only_2=False):
    if boards.shape[0] == 0:
        return np.array([], dtype=np.uint16), np.array([], dtype=np.int64), np.array([], dtype=np.uint8)
    
    empty4 = ray.get(_LUT_EMPTY4_ROW)  # uint8[65536]
    pc16 = ray.get(_PC16)  # uint8[65536]
    sel16 = ray.get(_LUT_SELECT16)  # uint8[65536,16,2]

    # 1) 행 마스크 → 보드 마스크 16비트 (벡터)
    row_masks = empty4[boards]  # (M,4) uint8 (각 행 4bit)
    board_mask16 = (
        (row_masks[:, 0].astype(np.uint16) << 12) |
        (row_masks[:, 1].astype(np.uint16) << 8) |
        (row_masks[:, 2].astype(np.uint16) << 4) |
        (row_masks[:, 3].astype(np.uint16) << 0)
    )  # (M,) uint16

    # 2) 총 빈칸 수 & 유효 보드
    total_empty = pc16[board_mask16.astype(np.int32)].astype(np.int32)  # (M,)
    valid = total_empty > 0
    if not np.any(valid):
        return np.array([], dtype=np.uint16), np.array([], dtype=np.int32), np.array([], dtype=np.uint8)

    v_rows = boards[valid]  # (Mv,4) uint16
    v_mask16 = board_mask16[valid].astype(np.int32)  # (Mv,) int64 index
    v_tot = total_empty[valid]  # (Mv,)

    mask = np.arange(16) < v_tot[:, None]
    rc = sel16[v_mask16][mask]  # (Mu,2) uint8
    rows = rc[:, 0].astype(np.int32)  # (Mu,)
    cols = rc[:, 1].astype(np.uint8)  # (Mu,)

    idx = np.nonzero(mask)[0]  # (Mu,)
    if not only_2:
        idx = idx.astype(np.int32).repeat(2)
    out = v_rows[idx].copy() 

    # 5) 니블 세팅 (scatter)
    shift = ((3 - cols).astype(np.uint16) << 2)
    clear_mask = (~(np.uint16(0xF) << shift) & np.uint16(0xFFFF)).astype(np.uint16)
    write_val_2 = (1 << shift).astype(np.uint16)
    if not only_2:
        write_val_4 = (2 << shift).astype(np.uint16)
    
    if only_2:
        old_rows = out[np.arange(out.shape[0], dtype=np.int32), rows]
        new_rows = (old_rows & clear_mask) | write_val_2
        out[np.arange(out.shape[0]), rows] = new_rows
        return out, idx, v_tot
    
    else:
        row_idx = np.arange(0, idx.shape[0], 2, dtype=np.int32)
        old_rows = out[row_idx, rows]
        new_rows = (old_rows & clear_mask) | write_val_2
        out[row_idx, rows] = new_rows

        row_idx += 1
        old_rows = out[row_idx, rows]
        new_rows = (old_rows & clear_mask) | write_val_4
        out[row_idx, rows] = new_rows

        return out, idx, v_tot * 2

def _transpose_inplace(boards, idx: np.ndarray | None = None):
    """
    비트연산 전치 (4x4 니블).
    boards[idx]: (M,4) uint16 의 각 보드를 전치하여 다시 (M,4)에 저장.
    """
    if idx is None:
        sub = boards
    else:
        sub = boards[idx]  # (M,4), 각 행은 0xABCD 니블들

    lut_trans = ray.get(_LUT_TRANSPOSE)
    t = lut_trans[sub[:, 0]] | (lut_trans[sub[:, 1]] >> 4) | (lut_trans[sub[:, 2]] >> 8) | (lut_trans[sub[:, 3]] >> 12)

    if idx is None:
        boards[:, 0] = t
        boards[:, 1] = (t >> 16)
        boards[:, 2] = (t >> 32)
        boards[:, 3] = (t >> 48)
    else:
        # use implicit type casting
        boards[idx, 0] = t
        boards[idx, 1] = (t >> 16)
        boards[idx, 2] = (t >> 32)
        boards[idx, 3] = (t >> 48)

def _pack_row(vals: np.ndarray) -> int:
    # vals: (4,) uint8  [a b c d] (a가 상위니블)
    return (int(vals[0]) << 12) | (int(vals[1]) << 8) | (int(vals[2]) << 4) | int(vals[3])

def _unpack_row(r: int) -> np.ndarray:
    return np.array([(r >> 12) & 0xF, (r >> 8) & 0xF, (r >> 4) & 0xF, r & 0xF], dtype=np.uint8)

def _slide_merge_left_row(vals: np.ndarray) -> np.uint16:
    # 보상, 결과 행만 반환 (LUT 용)
    comp = [int(v) for v in vals if v != 0]
    out = []
    i = 0
    reward = 0
    while i < len(comp):
        if i + 1 < len(comp) and comp[i] == comp[i + 1]:
            out.append(comp[i] + 1)
            reward += (1 << (comp[i] + 1))
            i += 2
        else:
            out.append(comp[i]); i += 1
    while len(out) < 4:
        out.append(0)
    return np.uint16(_pack_row(np.minimum(np.array(out[:4], dtype=np.uint8), 15))), reward

def _build_row_luts():
    """
    좌/우 결과, 보상 행 LUT 생성.
    - LEFT:  r -> left(r)
    - RIGHT: r -> reverse(left(reverse(r)))  (빌드 타임에 계산해 런타임 reverse 제거)
    """
    lut_left = np.zeros(65536, dtype=np.uint16)
    lut_right = np.zeros(65536, dtype=np.uint16)
    lut_left_reward = np.zeros(65536, dtype=np.int32)
    lut_right_reward = np.zeros(65536, dtype=np.int32)

    def reverse_row16(r: int) -> int:
        # abcd -> dcba
        return ((r & 0x000F) << 12) | ((r & 0x00F0) << 4) | ((r & 0x0F00) >> 4) | ((r & 0xF000) >> 12)

    for r in range(65536):
        orig = _unpack_row(r)
        # LEFT
        left_r, reward = _slide_merge_left_row(orig)
        lut_left[r] = left_r
        lut_left_reward[r] = reward

        # RIGHT (빌드 타임에 역방향 LUT 고정)
        rev = reverse_row16(r)
        rev_orig = _unpack_row(rev)
        rev_left, reward = _slide_merge_left_row(rev_orig)
        right_r = reverse_row16(int(rev_left))
        lut_right[r] = right_r
        lut_right_reward[r] = reward

    return lut_left, lut_right, lut_left_reward, lut_right_reward

def _build_transpose_luts():
    lut_trans = np.zeros(65536, dtype=np.uint64)
    for r in range(65536):
        lut_trans[r] = (r & 0xF000) | (r & 0x0F00) << 20 | (r & 0x00F0) << 40 | (r & 0x000F) << 60

    return lut_trans

def _init_spawn_luts():
    """
    스폰 최적화용 LUT를 한 번만 생성.
    - _LUT_EMPTY4_ROW[row16]: 4비트 마스크 (bit3=col0, ..., bit0=col3), 해당 니블==0이면 1
    - _PC4[v]: v(0..15)에서 1의 개수
    - _PC16[v]: v(0..65535)에서 1의 개수 (보드 마스크용)
    - _LUT_MASK_SELECT[mask4, n]: mask4에서 n번째(0-index) 1비트의 열 인덱스(0..3), 없으면 255
    - _LUT_SELECT16[mask16, nth] -> (row, col) (없으면 255,255)
    """
    # 1) 4비트 popcount
    pc4 = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)

    # 2) mask4 + nth -> col (O(1) 선택)
    lut_sel4 = np.full((16, 4), 255, dtype=np.uint8)
    for mask in range(16):
        cols = []
        for col in range(4):  # col=0..3 (왼→오)
            bit = 3 - col  # bit3↔col0, bit0↔col3
            if (mask >> bit) & 1:
                cols.append(col)
        for n, col in enumerate(cols):
            lut_sel4[mask, n] = col

    # 3) row16 -> empty 4bit mask
    empty4 = np.zeros(65536, dtype=np.uint8)
    for r in range(65536):
        m3 = 1 if ((r & 0xF000) == 0) else 0
        m2 = 1 if ((r & 0x0F00) == 0) else 0
        m1 = 1 if ((r & 0x00F0) == 0) else 0
        m0 = 1 if ((r & 0x000F) == 0) else 0
        empty4[r] = (m3 << 3) | (m2 << 2) | (m1 << 1) | m0

    # 4) 16비트 popcount (보드 마스크용)
    pc16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

    # 5) mask16 + nth -> (row, col)
    # mask16은 [row0(상위 4비트), row1, row2, row3(하위 4비트)]를 이어붙인 16비트
    lut_sel16 = np.full((1 << 16, 16, 2), 255, dtype=np.uint8)  # (row,col) = (255,255) = invalid
    for m in range(1 << 16):
        # 각 행의 4bit 추출
        m0 = (m >> 12) & 0xF  # row0
        m1 = (m >> 8) & 0xF  # row1
        m2 = (m >> 4) & 0xF  # row2
        m3 = (m >> 0) & 0xF  # row3
        c0 = int(pc4[m0])
        c1 = int(pc4[m1])
        c2 = int(pc4[m2])
        c3 = int(pc4[m3])

        # row0에서 가능한 n
        for n in range(c0):
            col = lut_sel4[m0, n]
            lut_sel16[m, n, 0] = 0
            lut_sel16[m, n, 1] = col
        # row1
        base = c0
        for n in range(c1):
            col = lut_sel4[m1, n]
            lut_sel16[m, base + n, 0] = 1
            lut_sel16[m, base + n, 1] = col
        # row2
        base += c1
        for n in range(c2):
            col = lut_sel4[m2, n]
            lut_sel16[m, base + n, 0] = 2
            lut_sel16[m, base + n, 1] = col
        # row3
        base += c2
        for n in range(c3):
            col = lut_sel4[m3, n]
            lut_sel16[m, base + n, 0] = 3
            lut_sel16[m, base + n, 1] = col
    # base+c3 == pc16[m]개의 유효 엔트리만 채워짐 (나머지는 255)

    return pc4, pc16, empty4, lut_sel4, lut_sel16



def init(params=None):
    global _LUT_LEFT_NEW, _LUT_RIGHT_NEW, _LUT_TRANSPOSE, _PC16, _LUT_EMPTY4_ROW, _LUT_SELECT16, _VALUE_TABLE
    if params is None:
        params = {
            "sum_weight" : 660,
            "sum_power" : 3.5,
            "locate_power" : 0,
            "locate_weight" : np.array([[0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [2.0, 3.0, 4.0, 5.0],
                    [8.0, 6.0, 5.0, 4.0]], dtype=np.float32),
            "monotonic_power" : 3.7,
            "monotonic_weight" : 28,
            "merge_weight" : 750,
            "empty_weight" : 300
        }

    _LUT_LEFT_NEW, _LUT_RIGHT_NEW, _LUT_LEFT_REWARD, _LUT_RIGHT_REWARD = _build_row_luts()
    _LUT_TRANSPOSE = _build_transpose_luts()
    _PC4, _PC16, _LUT_EMPTY4_ROW, _LUT_MASK_SELECT, _LUT_SELECT16 = _init_spawn_luts()
    _VALUE_TABLE = pre_evaluate(params)

    _LUT_LEFT_NEW = ray.put(_LUT_LEFT_NEW)
    _LUT_RIGHT_NEW = ray.put(_LUT_RIGHT_NEW)
    _LUT_TRANSPOSE = ray.put(_LUT_TRANSPOSE)
    _PC16 = ray.put(_PC16)
    _LUT_EMPTY4_ROW = ray.put(_LUT_EMPTY4_ROW)
    _LUT_SELECT16 = ray.put(_LUT_SELECT16)
    _VALUE_TABLE = ray.put(_VALUE_TABLE)


def find_best(board, depth=None, use_ray=False, split=32):
    if len(board.shape) == 1:
        board = np.expand_dims(board, axis=0)

    if depth == None:
        depth = max(2, (12 - empty_space_nums(board)[0]) // 3)

    if able_move(board).sum() == 0:
        return 255

    moved_boards, index, move = all_move_boards(board)
    next_boards, index2, num_cases = all_next_boards(moved_boards)
    
    if depth == 0:
        value = evaluate(next_boards)

    else:
        if use_ray and depth > 1:
            moved_boards_2, index_2, _ = all_move_boards(next_boards)
            next_boards_2, index2_2, num_cases_2 = all_next_boards(moved_boards_2)
            split_boards = np.array_split(next_boards_2, split)
            obj = []
            for i in range(split):
                obj.append(expectimax_ray.remote(split_boards[i], depth-2))

            value = np.concatenate(ray.get(obj)).flatten()
            value[0::2] *= 0.9
            value[1::2] *= 0.1
            value = np.bincount(index2_2, value) / num_cases_2
            n = next_boards.shape[0]
            out = np.full(n, -1e6, dtype=np.float32)
            np.maximum.at(out, index_2, value)
            value = out
            
        else:
            value = expectimax(next_boards, depth-1)

    value[0::2] *= 0.9
    value[1::2] *= 0.1
    value = np.bincount(index2, value) / num_cases
    return move[np.argmax(value)]

def expectimax(boards, depth):
    moved_boards, index, move = all_move_boards(boards)
    if moved_boards.shape[0] == 0:
        return np.full((boards.shape[0],), -1e6, dtype=np.float32)
    
    if depth == 0:
        next_boards, index2, num_cases = all_next_boards(moved_boards, only_2=True)
        value = evaluate(next_boards)

    else:
        next_boards, index2, num_cases = all_next_boards(moved_boards)
        value = expectimax(next_boards, depth-1)
        value[0::2] *= 0.9
        value[1::2] *= 0.1
    
    value = np.bincount(index2, value) / num_cases
    n = boards.shape[0]
    out = np.full(n, -1e6, dtype=np.float32)
    np.maximum.at(out, index, value)
    return out

@ray.remote(num_cpus=1)
def expectimax_ray(boards, depth):
    return expectimax(boards, depth)

def pre_evaluate(params):
    value_table = np.zeros((4, 2**16), dtype=np.float32)
    locate_weight = params["locate_weight"]
    for i in range(2 ** 16):
        line = [
            (i >> 0) % 16,
            (i >> 4) % 16,
            (i >> 8) % 16,
            (i >> 12) % 16
        ]
        value_table[0, i] = evaluate_line(line, locate_weight[0], params)
        value_table[1, i] = evaluate_line(line, locate_weight[1], params)
        value_table[2, i] = evaluate_line(line, locate_weight[2], params)
        value_table[3, i] = evaluate_line(line, locate_weight[3], params)
    return value_table

def evaluate_line(line, locate_weight, params):
    # sum_power = 3.5
    sum_weight = params["sum_weight"]
    locate_power = params["locate_power"]
    monotonic_power = params["monotonic_power"]
    monotonic_weight = params["monotonic_weight"]
    merge_weight = params["merge_weight"]
    empty_weight = params["empty_weight"]

    sum_value = 0
    locate_value = 0
    empty = 0
    merges = 0

    monotonic_left = 0
    monotonic_right = 0
    prev = 0

    for i in range(4):
        rank = line[i]
        locate_value += 2 ** rank * locate_weight[i]
        sum_value += 2 ** rank
        if rank == 0:
            empty += 1
        else:
            if prev == rank:
                merges += rank
            prev = rank

        if i > 0:
            prev_rank = line[i-1]
            if rank > prev_rank:
                monotonic_left += prev_rank ** monotonic_power - rank ** monotonic_power
            else:
                monotonic_right += rank ** monotonic_power - prev_rank ** monotonic_power

    monotonic = max(monotonic_left, monotonic_right)

    value = empty * empty_weight + merges * merge_weight \
        + monotonic * monotonic_weight + sum_value * sum_weight + locate_value

    return value

def evaluate(board):
    trans_board = board.copy()
    _transpose_inplace(trans_board)  
    return _evaluate(board) + _evaluate(trans_board, locate=False)

def _evaluate(rank, locate=True):
    value_table = ray.get(_VALUE_TABLE)
    if locate:
        return (value_table[0][rank[:, 0]] +
        value_table[1][rank[:, 1]] +
        value_table[2][rank[:, 2]] +
        value_table[3][rank[:, 3]])
    
    v_table = value_table[0]
    return v_table[rank].sum(axis=1)

init()
if __name__ == "__main__":
    num_env = 1
    # ray.init(num_cpus=32)
    # init()
    boards = Batch2048EnvSimulator.init_board(num_env)

    for i in tqdm(range(1000)):
        move = find_best(boards, depth=3, use_ray=True) # inside multi processing

    # move = find_best(boards, depth=3)
    # print(boards, move)
