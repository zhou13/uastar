#!/bin/env python3

import os
import sys
import curses

HELP = "Press [ARROW KEY] to move the tile, [s] to save current state, and [q] to quit"

def main(stdscr):
    screen_height, screen_width = stdscr.getmaxyx();
    if len(sys.argv) == 1:
        width = height = 4
    elif len(sys.argv) == 2:
        width = height = int(sys.argv[1])
    else:
        height = int(sys.argv[1]);
        width = int(sys.argv[2]);
    startx = (screen_width - width*3 + 1) // 2

    def draw_box(box):
        y = 2
        for row in box:
            x = startx
            for v in row:
                if v == 0:
                    out = '   '
                else:
                    out = "{:3d}".format(v)
                stdscr.addstr(y, x, out)
                x += 3
            y += 1
        stdscr.refresh()

    def find_zero(box):
        for i, row in enumerate(box):
            for j, v in enumerate(row):
                if v == 0:
                    return i, j
        return None

    curses.curs_set(0);

    stdscr.clear()
    stdscr.addstr(0, (screen_width - len(HELP))//2, HELP);

    box = [[x*width + y  + 1 for y in range(width)] for x in range(height)]
    box[height-1][height-1] = 0

    seq = []
    draw_box(box)
    while True:
        c = stdscr.getch()

        x, y = find_zero(box)
        if c == curses.KEY_LEFT:
            nx, ny = x, y+1
            seq.append('l')
        elif c == curses.KEY_RIGHT:
            nx, ny = x, y-1
            seq.append('r')
        elif c == curses.KEY_UP:
            nx, ny = x+1, y
            seq.append('u')
        elif c == curses.KEY_DOWN:
            nx, ny = x-1, y
            seq.append('d')
        elif c == ord('q'):
            return
        elif c == ord('s'):
            filename = "puzzle" + str(width) + '-'
            count = 1
            suffix = '.txt'
            while os.path.exists(filename + str(count) + suffix):
                count += 1
            filename = filename + str(count) + suffix
            with open(filename, "w") as f:
                for row in box:
                    for v in row:
                        f.write(str(v) + ' ')
                f.write('\n')
                for c in seq:
                    f.write(c)
                f.write('\n')
        else:
            pass

        if 0 <= nx and nx < height and 0 <= ny and ny < width:
            box[x][y], box[nx][ny] = box[nx][ny], box[x][y]
        draw_box(box)


if __name__ == '__main__':
    curses.wrapper(main)
