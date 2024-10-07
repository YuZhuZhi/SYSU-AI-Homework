import copy
from ChessBoard import *


class Evaluate(object):
    # 棋子棋力得分
    single_chess_point = {
        'c': 989,   # 车
        'm': 439,   # 马
        'p': 442,   # 炮
        's': 226,   # 士
        'x': 210,   # 象
        'z': 55,    # 卒
        'j': 65536  # 将
    }
    # 红兵（卒）位置得分
    red_bin_pos_point = [
        [1, 3, 9, 10, 12, 10, 9, 3, 1],
        [18, 36, 56, 95, 118, 95, 56, 36, 18],
        [15, 28, 42, 73, 80, 73, 42, 28, 15],
        [13, 22, 30, 42, 52, 42, 30, 22, 13],
        [8, 17, 18, 21, 26, 21, 18, 17, 8],
        [3, 0, 7, 0, 8, 0, 7, 0, 3],
        [-1, 0, -3, 0, 3, 0, -3, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    # 红车位置得分
    red_che_pos_point = [
        [185, 195, 190, 210, 220, 210, 190, 195, 185],
        [185, 203, 198, 230, 245, 230, 198, 203, 185],
        [180, 198, 190, 215, 225, 215, 190, 198, 180],
        [180, 200, 195, 220, 230, 220, 195, 200, 180],
        [180, 190, 180, 205, 225, 205, 180, 190, 180],
        [155, 185, 172, 215, 215, 215, 172, 185, 155],
        [110, 148, 135, 185, 190, 185, 135, 148, 110],
        [100, 115, 105, 140, 135, 140, 105, 115, 110],
        [115, 95, 100, 155, 115, 155, 100, 95, 115],
        [20, 120, 105, 140, 115, 150, 105, 120, 20]
    ]
    # 红马位置得分
    red_ma_pos_point = [
        [80, 105, 135, 120, 80, 120, 135, 105, 80],
        [80, 115, 200, 135, 105, 135, 200, 115, 80],
        [120, 125, 135, 150, 145, 150, 135, 125, 120],
        [105, 175, 145, 175, 150, 175, 145, 175, 105],
        [90, 135, 125, 145, 135, 145, 125, 135, 90],
        [80, 120, 135, 125, 120, 125, 135, 120, 80],
        [45, 90, 105, 190, 110, 90, 105, 90, 45],
        [80, 45, 105, 105, 80, 105, 105, 45, 80],
        [20, 45, 80, 80, -10, 80, 80, 45, 20],
        [20, -20, 20, 20, 20, 20, 20, -20, 20]
    ]
    # 红炮位置得分
    red_pao_pos_point = [
        [190, 180, 190, 70, 10, 70, 190, 180, 190],
        [70, 120, 100, 90, 150, 90, 100, 120, 70],
        [70, 90, 80, 90, 200, 90, 80, 90, 70],
        [60, 80, 60, 50, 210, 50, 60, 80, 60],
        [90, 50, 90, 70, 220, 70, 90, 50, 90],
        [120, 70, 100, 60, 230, 60, 100, 70, 120],
        [10, 30, 10, 30, 120, 30, 10, 30, 10],
        [30, -20, 30, 20, 200, 20, 30, -20, 30],
        [30, 10, 30, 30, -10, 30, 30, 10, 30],
        [20, 20, 20, 20, -10, 20, 20, 20, 20]
    ]
    # 红将位置得分
    red_jiang_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 9750, 9800, 9750, 0, 0, 0],
        [0, 0, 0, 9900, 9900, 9900, 0, 0, 0],
        [0, 0, 0, 10000, 10000, 10000, 0, 0, 0],
    ]
    # 红相或士位置得分
    red_xiang_shi_pos_point = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 60, 0, 0, 0, 60, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [80, 0, 0, 80, 90, 80, 0, 0, 80],
        [0, 0, 0, 0, 0, 120, 0, 0, 0],
        [0, 0, 70, 100, 0, 100, 70, 0, 0],
    ]

    red_pos_point = {
        'z': red_bin_pos_point,
        'm': red_ma_pos_point,
        'c': red_che_pos_point,
        'j': red_jiang_pos_point,
        'p': red_pao_pos_point,
        'x': red_xiang_shi_pos_point,
        's': red_xiang_shi_pos_point
    }

    def __init__(self, team):
        self.team = team

    def get_single_chess_point(self, chess: Chess):
        if chess.team == self.team:
            return self.single_chess_point[chess.name]
        else:
            return -1 * self.single_chess_point[chess.name]

    def get_chess_pos_point(self, chess: Chess):
        red_pos_point_table = self.red_pos_point[chess.name]
        if chess.team == 'r':
            pos_point = red_pos_point_table[chess.row][chess.col]
        else:
            pos_point = red_pos_point_table[9 - chess.row][chess.col]
        if chess.team != self.team:
            pos_point *= -1
        return pos_point

    def evaluate(self, chessboard: ChessBoard):
        point = 0
        for chess in chessboard.get_chess():
            point += self.get_single_chess_point(chess)
            point += self.get_chess_pos_point(chess)
        return point


class ChessMap(object):
    def __init__(self, chessboard: ChessBoard):
        self.chess_map = copy.deepcopy(chessboard.chessboard_map)

class MyAI(object):
    def __init__(self, computer_team):
        self.team = computer_team
        self.evaluate_class = Evaluate(self.team)

    def get_next_step(self, chessboard: ChessBoard) -> list:
        origin_chessmap = chessboard.get_chessboard_str_map()
        if (True): result = self.DFSforRed(chessboard, -2e30, 2e30, True, 1)
        else: result = self.DFSforBlack(chessboard, -2e30, 2e30, False, 1)
        chessboard.set_chessboard_str_map(origin_chessmap)
        return [result[0].row, result[0].col, result[1][0], result[1][1]]
        
    def DFSforRed(self, chessboard: ChessBoard, alpha: int, beta: int, minmax: bool, depth: int) -> list:
        returner = [None, (), alpha, beta]
        thisalpha, thisbeta = alpha, beta
        origin_chessmap = chessboard.get_chessboard_str_map()
        if (depth == 3):
            if (thisalpha >= thisbeta): return returner
            for movablechess in chessboard.get_chess():
                if (movablechess.team == 'r'): # 只有红方允许走棋
                    for movablepos in chessboard.get_put_down_position(movablechess):
                        chessboard.set_chessboard_str_map(origin_chessmap)
                        self.VirtualMoveChess(chessboard, movablechess, movablepos)
                        cur_point = self.evaluate_class.evaluate(chessboard)
                        if (cur_point > thisalpha):
                            thisalpha = cur_point
                            returner[0], returner[1], returner[2] = movablechess, movablepos, cur_point
            chessboard.set_chessboard_str_map(origin_chessmap)
            return returner

        if (minmax): # minmax==true时是max结点
            if (thisalpha >= thisbeta): return returner # alpha剪枝
            for movablechess in chessboard.get_chess():
                if (movablechess.team == 'r'): # 只有红方允许走棋
                    for movablepos in chessboard.get_put_down_position(movablechess):
                        chessboard.set_chessboard_str_map(origin_chessmap)
                        self.VirtualMoveChess(chessboard, movablechess, movablepos)
                        temp = self.DFSforRed(chessboard, thisalpha, thisbeta, not minmax, depth + 1)
                        if (temp[3] > thisalpha or returner[0] is None):
                            thisalpha = temp[3]
                            returner[0], returner[1], returner[2] = movablechess, movablepos, thisalpha
            chessboard.set_chessboard_str_map(origin_chessmap)
            return returner

        else:
            if (thisbeta <= thisalpha): return returner # beta剪枝
            for movablechess in chessboard.get_chess():
                if (movablechess.team == 'b'): # 只有黑方允许走棋
                    for movablepos in chessboard.get_put_down_position(movablechess):
                        chessboard.set_chessboard_str_map(origin_chessmap)
                        self.VirtualMoveChess(chessboard, movablechess, movablepos)
                        temp = self.DFSforRed(chessboard, alpha, thisbeta, not minmax, depth + 1)
                        if (temp[2] < thisbeta):
                            thisbeta = temp[2]
                            returner[3] = thisbeta
            chessboard.set_chessboard_str_map(origin_chessmap)
            return returner
        
    def DFSforBlack(self, chessboard: ChessBoard, alpha: int, beta: int, minmax: bool, depth: int) -> list:
        returner = [None, (), alpha, beta]
        thisalpha, thisbeta = alpha, beta
        origin_chessmap = chessboard.get_chessboard_str_map()
        if (depth == 1):
            if (thisbeta <= thisalpha): return returner
            for movablechess in chessboard.get_chess():
                if (movablechess.team == 'b'): # 只有黑方允许走棋
                    for movablepos in chessboard.get_put_down_position(movablechess):
                        chessboard.set_chessboard_str_map(origin_chessmap)
                        self.VirtualMoveChess(chessboard, movablechess, movablepos)
                        cur_point = self.evaluate_class.evaluate(chessboard)
                        if (cur_point < thisbeta):
                            thisbeta = cur_point
                            returner[0], returner[1], returner[3] = movablechess, movablepos, thisbeta
            chessboard.set_chessboard_str_map(origin_chessmap)
            return returner

        if (minmax): # minmax==true时是max结点
            if (thisalpha >= thisbeta): return returner # alpha剪枝
            for movablechess in chessboard.get_chess():
                if (movablechess.team == 'r'): # 只有红方允许走棋
                    for movablepos in chessboard.get_put_down_position(movablechess):
                        chessboard.set_chessboard_str_map(origin_chessmap)
                        self.VirtualMoveChess(chessboard, movablechess, movablepos)
                        temp = self.DFSforBlack(chessboard, thisalpha, thisbeta, not minmax, depth + 1)
                        if (temp[3] > thisalpha or returner[0] is None):
                            thisalpha = temp[3]
                            returner[2] = thisalpha
            chessboard.set_chessboard_str_map(origin_chessmap)
            return returner

        else:
            if (thisbeta <= thisalpha): return returner # beta剪枝
            for movablechess in chessboard.get_chess():
                if (movablechess.team == 'b'): # 只有黑方允许走棋
                    for movablepos in chessboard.get_put_down_position(movablechess):
                        chessboard.set_chessboard_str_map(origin_chessmap)
                        self.VirtualMoveChess(chessboard, movablechess, movablepos)
                        temp = self.DFSforBlack(chessboard, alpha, thisbeta, not minmax, depth + 1)
                        if (temp[2] < thisbeta):
                            thisbeta = temp[2]
                            returner[0], returner[1], returner[3] = movablechess, movablepos, thisbeta
            chessboard.set_chessboard_str_map(origin_chessmap)
            return returner
    
    def VirtualMoveChess(self, chessboard: ChessBoard, move_chess: Chess, tar_coor: tuple) -> None:
        old_row, old_col = move_chess.row, move_chess.col
        new_row, new_col = tar_coor[0], tar_coor[1]
        chessboard.chessboard_map[new_row][new_col] = chessboard.chessboard_map[old_row][old_col]
        chessboard.chessboard_map[new_row][new_col].update_position(new_row, new_col)
        chessboard.chessboard_map[old_row][old_col] = None