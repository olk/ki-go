from kigo.types import Point


def is_point_an_eye(board, point, color):
    # an eye is an empty point
    if board.get(point) is not None:
        return False
    for neighbor in point.neighbors():
        if board.is_on_grid(neighbor):
            neighbor_color = board.get(neighbor)
            if neighbor_color != color:
                return False
    friendly_corners = 0
    off_board_corners = 0
    corners = [
            Point(point.row - 1, point.col - 1),
            Point(point.row - 1, point.col + 1),
            Point(point.row + 1, point.col - 1),
            Point(point.row + 1, point.col + 1)
    ]
    for corner in corners:
        if board.is_on_grid(corner):
            corner_color = board.get(corner)
            if corner_color == color:
                friendly_corners += 1
        else:
            off_board_corners += 1
    if 0 < off_board_corners:
        # point is on the edge or corner
        return 4 == off_board_corners + friendly_corners
    # point is in the middle
    return 3 <= friendly_corners
