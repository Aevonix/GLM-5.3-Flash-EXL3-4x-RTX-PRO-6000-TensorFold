#!/usr/bin/env python3
"""Procedural Aevonix opener and CRT deck, using only the standard library.

animation_frames() is also the source for the README GIF. Each Cell carries one
terminal glyph plus foreground/background RGB; half blocks are exact pixels.
The shell dispatcher owns capability detection. No input or network is used.
"""
import argparse
import json
from pathlib import Path
from functools import lru_cache
from math import sqrt
import os
import random
import shutil
import signal
import sys
import time
import textwrap
from typing import NamedTuple


class Cell(NamedTuple):
    char: str
    fg: tuple
    bg: tuple


BG = (0, 0, 0)
GRAPHITE = (21, 22, 21)
CREAM = (243, 229, 205)
QUIET = (192, 180, 160)
AMBER = (238, 176, 126)
CORAL = (217, 146, 136)
TEAL = (169, 213, 206)

@lru_cache(maxsize=2048)
def blend(a, b, t):
    return tuple(round(x * (1 - t) + y * t) for x, y in zip(a, b))

# Twelve-pixel glyphs; two-pixel stems, crossbars and diagonals.
FONT = {
    'A': ['0011111100']*2 + ['0110000110'] + ['1100000011']*2 + ['1111111111']*2 + ['1100000011']*5,
    'E': ['1111111111']*2 + ['1100000000']*3 + ['1111111100']*2 + ['1100000000']*3 + ['1111111111']*2,
    'O': ['0011111100']*2 + ['0110000110'] + ['1100000011']*6 + ['0110000110'] + ['0011111100']*2,
    'I': ['111111']*2 + ['001100']*8 + ['111111']*2,
}
for letter in 'VNX':
    glyph = []
    for y in range(12):
        if letter == 'V':
            d = min(4, y*5//12)
            ink = {d,d+1,8-d,9-d}
        elif letter == 'X':
            d = round(min(y,11-y)*4/5)
            ink = {d,d+1,8-d,9-d}
        else:
            d = round(y*8/11)
            ink = {0,1,8,9,d,d+1}
        glyph.append(''.join('1' if x in ink else '0' for x in range(10)))
    FONT[letter] = glyph


def glyph_sheet(sheet):
    """Parse blocks of glyph names over bitmap rows; '#' is ink, '.' is empty."""
    glyphs = {}
    for block in sheet.strip('\n').split('\n\n'):
        names, *rows = block.split('\n')
        for i, name in enumerate(names.split()):
            glyphs[name] = [row.split()[i].replace('#', '1').replace('.', '0') for row in rows]
    return glyphs


# The rest of the title alphabet on the same grid: beveled corners, mid-height
# crossbars and a dotted zero, so model numbers never read as letters.
FONT.update(glyph_sheet('''
B          C          D          F          G          H          J          K
########.. ..######## ########.. ########## ..######## ##......## ........## ##......##
########.. ..######## ########.. ########## ..######## ##......## ........## ##.....##.
##.....##. .##....... ##.....##. ##........ .##....... ##......## ........## ##....##..
##......## ##........ ##......## ##........ ##........ ##......## ........## ##...##...
##.....##. ##........ ##......## ##........ ##........ ##......## ........## ##..##....
########.. ##........ ##......## ########.. ##...##### ########## ........## #####.....
########.. ##........ ##......## ########.. ##...##### ########## ........## #####.....
##.....##. ##........ ##......## ##........ ##......## ##......## ##......## ##..##....
##......## ##........ ##......## ##........ ##......## ##......## ##......## ##...##...
##.....##. .##....... ##.....##. ##........ .##....##. ##......## .##....##. ##....##..
########.. ..######## ########.. ##........ ..######.. ##......## ..######.. ##.....##.
########.. ..######## ########.. ##........ ..######.. ##......## ..######.. ##......##

L          M          P          Q          R          S          T          U
##........ ##......## ########.. ..######.. ########.. ..######## ########## ##......##
##........ ###....### ########.. ..######.. ########.. ..######## ########## ##......##
##........ ####..#### ##.....##. .##....##. ##.....##. .##....... ....##.... ##......##
##........ ##.####.## ##......## ##......## ##......## ##........ ....##.... ##......##
##........ ##..##..## ##.....##. ##......## ##.....##. .##....... ....##.... ##......##
##........ ##......## ########.. ##......## ########.. ..######.. ....##.... ##......##
##........ ##......## ########.. ##......## ########.. ..######.. ....##.... ##......##
##........ ##......## ##........ ##..##..## ##...##... .......##. ....##.... ##......##
##........ ##......## ##........ ##...####. ##....##.. ........## ....##.... ##......##
##........ ##......## ##........ .##...##.. ##.....##. .......##. ....##.... .##....##.
########## ##......## ##........ ..######## ##......## ########.. ....##.... ..######..
########## ##......## ##........ ..####..## ##......## ########.. ....##.... ..######..

W          Y          Z          0          1      2          3          4
##......## ##......## ########## ..######.. ..##.. ..######.. ..######.. ##....##..
##......## .##....##. ########## ..######.. .###.. ..######.. ..######.. ##....##..
##......## ..##..##.. .......##. .##....##. ####.. .##....##. .##....##. ##....##..
##......## ..##..##.. ......##.. ##......## ..##.. ........## ........## ##....##..
##......## ...####... .....##... ##......## ..##.. .......##. .......##. ##....##..
##......## ....##.... ....##.... ##..##..## ..##.. .....###.. ...#####.. ##....##..
##......## ....##.... ...##..... ##..##..## ..##.. ...###.... ...#####.. ##########
##..##..## ....##.... ..##...... ##......## ..##.. .###...... .......##. ##########
##.####.## ....##.... .##....... ##......## ..##.. ##........ ........## ......##..
####..#### ....##.... ##........ .##....##. ..##.. ##........ .##....##. ......##..
###....### ....##.... ########## ..######.. ###### ########## ..######.. ......##..
##......## ....##.... ########## ..######.. ###### ########## ..######.. ......##..

5          6          7          8          9          -      .
########## ..######.. ########## ..######.. ..######.. ...... ..
########## ..######.. ########## ..######.. ..######.. ...... ..
##........ .##....... ........## .##....##. .##....##. ...... ..
##........ ##........ .......##. ##......## ##......## ...... ..
##........ ##........ ......##.. .##....##. .##.....## ...... ..
########.. ########.. .....##... ..######.. ..######## ###### ..
########.. ########.. ....##.... ..######.. ..######## ###### ..
.......##. ##.....##. ....##.... .##....##. ........## ...... ..
........## ##......## ....##.... ##......## ........## ...... ..
.......##. .##....##. ....##.... .##....##. .......##. ...... ..
########.. ..######.. ....##.... ..######.. ..######.. ...... ##
########.. ..######.. ....##.... ..######.. ..######.. ...... ##
'''))
FONT[' '] = ['0000']*12

# Five-pixel capitals for the deck's title row: three terminal rows tall.
MINI = glyph_sheet('''
A     B     C     D     E     F     G     H     I   J     K     L     M
.###. ####. .#### ####. ##### ##### .#### #...# ### ....# #...# #.... #...#
#...# #...# #.... #...# #.... #.... #.... #...# .#. ....# #..#. #.... ##.##
##### ####. #.... #...# ####. ####. #..## ##### .#. ....# ###.. #.... #.#.#
#...# #...# #.... #...# #.... #.... #...# #...# .#. #...# #..#. #.... #...#
#...# ####. .#### ####. ##### #.... .###. #...# ### .###. #...# ##### #...#

N     O     P     Q     R     S     T     U     V     W     X     Y     Z
#...# .###. ####. .###. ####. .#### ##### #...# #...# #...# #...# #...# #####
##..# #...# #...# #...# #...# #.... ..#.. #...# #...# #...# .#.#. .#.#. ...#.
#.#.# #...# ####. #.#.# ####. .###. ..#.. #...# #...# #.#.# ..#.. ..#.. ..#..
#..## #...# #.... #..#. #..#. ....# ..#.. #...# .#.#. ##.## .#.#. ..#.. .#...
#...# .###. #.... .##.# #...# ####. ..#.. .###. ..#.. #...# #...# ..#.. #####

0     1   2     3     4     5     6     7     8     9     -   .
.###. .#. .###. ####. #..#. ##### .###. ##### .###. .###. ... .
#..## ##. #...# ....# #..#. #.... #.... ...#. #...# #...# ... .
#.#.# .#. ..##. .###. ##### ####. ####. ..#.. .###. .#### ### .
##..# .#. .#... ....# ...#. ....# #...# .#... #...# ....# ... .
.###. ### ##### ####. ...#. ####. .###. .#... .###. .###. ... #
''')
MINI[' '] = ['00']*5

class Frame:
    def __init__(self, w=108, h=None, duration_ms=0, phase="final"):
        h = h or DECK_HEIGHT
        self.duration_ms, self.phase = duration_ms, phase
        self.w, self.h = w, h
        self.rows = [[Cell(' ', BG, BG) for _ in range(w)] for _ in range(h)]

    def text(self, x, y, s, color=QUIET):
        assert x >= 0 and x + len(s) <= self.w, (x, s, self.w)
        for i, c in enumerate(s):
            self.rows[y][x+i] = Cell(c, color, BG)

    def center(self, y, s, color=QUIET):
        self.text((self.w-len(s))//2, y, s, color)

    def rule(self, y, left, right, middle='═', color=None):
        self.text(1, y, left + middle*(self.w-4) + right, color or blend(BG, AMBER, .48))

    def pixels(self, x, y, pixels):
        for j in range(0, len(pixels), 2):
            for i, top in enumerate(pixels[j]):
                bottom = pixels[j+1][i]
                if top == bottom:
                    cell = Cell(' ', top, top)
                elif top == BG:
                    cell = Cell('▄', bottom, top)
                else:
                    cell = Cell('▀', top, bottom)
                self.rows[y+j//2][x+i] = cell

def wordmask(gap, word='AEVONIX', font=FONT):
    mask = {}
    x = 0
    for index, letter in enumerate(word):
        glyph = font[letter]
        for y, line in enumerate(glyph):
            for i, c in enumerate(line):
                if c == '1':
                    mask[x+i, y] = index
        x += len(glyph[0])+gap
    return mask, x-gap


def title_lines(text, font=FONT, gaps=(3, 2), width=100, most=2):
    """(mask, width) per line: one line if it fits, else the most balanced two
    lines split at a space, trying wider letter gaps first. None if neither fits."""
    words = str(text or '').upper().split()
    if not words or not set(''.join(words)) <= set(font):
        return None
    layouts = [[' '.join(words)]], [[' '.join(words[:i]), ' '.join(words[i:])] for i in range(1, len(words))]
    widest = lambda masks: max(w for _, w in masks)
    for options in layouts[:most]:
        for gap in gaps:
            fits = [masks for masks in ([wordmask(gap, line, font) for line in lines] for lines in options)
                    if widest(masks) <= width]
            if fits:
                return min(fits, key=widest)
    return None

def logo(pixels, x0, y0, gap, crt=False):
    mask, width = wordmask(gap)
    def put(x, y, color):
        if 0 <= y < len(pixels) and 0 <= x < len(pixels[0]):
            pixels[y][x] = color
    if crt:
        # Two-pixel strokes surrounded by dim amber phosphor; no pseudo-readouts.
        for radius, intensity in [(2, .055), (1, .15)]:
            for x, y in mask:
                for dx in range(-radius, radius+1):
                    for dy in range(-radius, radius+1):
                        put(x0+x+dx, y0+y+dy, blend(BG, AMBER, intensity))
    else:
        # One dark outline and a single one-pixel extrusion, with upright faces.
        for x, y in mask:
            for dx in range(-1, 3):
                for dy in range(-1, 3):
                    put(x0+x+dx, y0+y+dy, GRAPHITE)
        for (x, y), index in mask.items():
            color = [AMBER, AMBER, CORAL, CORAL, TEAL, TEAL, TEAL][index]
            put(x0+x+1, y0+y+1, blend(BG, color, .40))
    for (x, y), index in mask.items():
        if crt:
            color = blend(AMBER, CREAM, .12) if y % 2 == 0 else AMBER
        else:
            base = [AMBER, AMBER, CORAL, CORAL, TEAL, TEAL, TEAL][index]
            color = blend(base, BG, max(0, y-3)*.013)
            if y < 4 and (x, y-1) not in mask:
                color = CREAM
        put(x0+x, y0+y, color)

def facts():
    return json.loads((Path(__file__).resolve().parents[1] / '.github/banner-facts.json').read_text())


FACTS = facts()
CHART = FACTS['chart']
def sequence_lines():
    lines, current = [], ''
    for step in FACTS['start_sequence']:
        candidate = current + (' → ' if current else '') + step
        if len(candidate) <= 76:
            current = candidate
        else:
            if current:
                lines.append(current)
            parts = textwrap.wrap(step, width=76) or ['']
            lines.extend(parts[:-1])
            current = parts[-1]
    if current:
        lines.append(current)
    return lines


START_LINES = sequence_lines()
START_ROWS = max(3, len(START_LINES))
# Optional model title: a block-letter title card in the opener and a title row
# in the deck. Without title_model every frame renders exactly as before.
TITLE = title_lines(FACTS.get('title_model'))
SUBTITLE = str(FACTS.get('title_subtitle') or '') if TITLE else ''
TITLE_GAP = 3
TITLE_ROWS = (3 + bool(SUBTITLE)) if TITLE else 0
TITLE_FIRST = FACTS.get('title_placement') == 'before-logo'
# The title moment: a four-step beam (60 ms each), an 80 ms flash, three 60 ms
# typing steps and the hold, in 100 ms frames so the floor keeps moving.
BEAM_ROWS = (3, 7, 11, 15)
TITLE_HOLD_MS = 1200
TITLE_MS = 60*len(BEAM_ROWS) + 80 + 60*3 + TITLE_HOLD_MS
DECK_HEIGHT = max(30, 14 + START_ROWS + max(len(FACTS['system']), len(CHART) + 2)) + TITLE_ROWS


def fitted(value, width):
    value = str(value)
    return value if len(value) <= width else value[:max(0, width-1)] + '…'


def credit(f, y):
    f.center(y, fitted(FACTS['credit'], f.w-4), CORAL)


def floor(f, first, last, progress=0):
    # Braille gives the perspective lines four vertical samples per text row.
    w, h = (f.w-4)*2, (last-first+1)*4
    points = set()
    cx = (w-1)/2
    for end in [-w*1.8, -w*.8, -w*.3, 0, w*.3, w*.8, w*1.8]:
        previous = round(cx+end*(1/(h+1)))
        for y in range(h):
            x = round(cx+end*((y+2)/(h+1)))
            for px in range(min(previous,x),max(previous,x)+1):
                if 0 <= px < w:
                    points.add((px,y))
            previous = x
    # Increasing spacing and a moving phase make the grid approach the viewer.
    for y in {round(((i / 4 + progress * .28) % 1) ** 2 * (h-1)) for i in range(4)}:
        points.update((x,y) for x in range(w))
    dots = {(0,0):1, (0,1):2, (0,2):4, (1,0):8, (1,1):16, (1,2):32, (0,3):64, (1,3):128}
    for row in range(last-first+1):
        color = blend(BG, TEAL, .43 + row*.065)
        for col in range(w//2):
            bits = sum(bit for (dx,dy), bit in dots.items() if (2*col+dx,4*row+dy) in points)
            if bits:
                f.text(2+col, first+row, chr(0x2800+bits), color)


def blockbar(value, width):
    maximum = max((row['value'] for row in CHART), default=1) or 1
    eighths = round(value/maximum*width*8)
    full, remainder = divmod(eighths, 8)
    bar = '█'*full + (' ▏▎▍▌▋▊▉'[remainder] if remainder else '')
    return bar.ljust(width)


def deck(w):
    f = Frame(w)
    border = blend(BG, AMBER, .58)
    divider = blend(BG, AMBER, .42)
    bottom = f.h - 3
    for y in range(1,bottom):
        f.text(1,y,'║',border)
        f.text(w-2,y,'║',border)
    f.rule(0, '╔', '╗', color=border)
    f.text(4, 0, ' AEVONIX RESEARCH ', AMBER)
    label = ' ' + fitted(FACTS['model'], w-27) + ' '
    f.text(w-3-len(label), 0, label, QUIET)
    p = [[BG for _ in range(w-4)] for _ in range(14)]
    gap = 3 if w == 108 else 2
    _, logo_width = wordmask(gap)
    logo(p, (w-4-logo_width)//2, 1, gap, crt=True)
    f.pixels(2, 1, p)
    top = 8 + title_band(f, w)
    split = 51 if w == 108 else 42
    f.rule(top, '╠', '╣', color=border)
    f.text(split, top, '╦', border)
    f.text(4, top, ' SYSTEM ', AMBER)
    f.text(split+3, top, ' MEASURED ', TEAL)
    section_end = bottom-START_ROWS-2
    for y in range(top+1,section_end):
        f.text(split,y,'│',divider)
    for y, row in enumerate(FACTS['system'], top+1):
        f.text(3,y,fitted(row['label'],7),QUIET)
        f.text(11,y,fitted(row['value'],split-12),CREAM)
    right = split+3
    f.text(right,top+1,fitted(FACTS['chart_title'],w-right-3),CREAM)
    f.text(right,top+2,'Aggregate decode · tok/s',QUIET)
    barwidth = w-right-24
    for i, row in enumerate(CHART):
        y = top+3+i
        color = (CORAL, TEAL, AMBER)[row['tone'] % 3]
        f.text(right,y,fitted(row['label'],10).ljust(10),color)
        f.text(right+11,y,blockbar(row['value'],barwidth),color)
        f.text(right+12+barwidth,y,f"{row['value']:>8,.1f}",CREAM)
    f.rule(section_end,'╠','╣',color=border)
    f.text(split,section_end,'╩',border)
    f.text(4,section_end,' START SEQUENCE ',AMBER)
    for i, line in enumerate(START_LINES, section_end+1):
        f.text(4,i,fitted(line,w-8),CREAM)
    f.center(bottom-1,fitted(FACTS['endpoint'],w-8),TEAL)
    f.rule(bottom,'╚','╝',color=border)
    credit(f,f.h-1)
    return f


def title_band(f, w, top=8):
    """The deck's title row: five-pixel capitals over the subtitle. Returns its rows."""
    if not TITLE:
        return 0
    lines = title_lines(FACTS['title_model'], MINI, (2, 1), w-8, 1)
    if lines:
        (mask, width), = lines
        p = [[BG for _ in range(w-4)] for _ in range(6)]
        x0 = (w-4-width)//2
        for x, y in mask:
            # Phosphor scanlines, as in the deck logo, in the title's cream ink.
            p[y+1][x0+x] = CREAM if y % 2 == 0 else blend(CREAM, AMBER, .35)
        f.pixels(2, top, p)
    else:
        f.center(top+1, fitted(str(FACTS['title_model']).upper(), w-8), CREAM)
    if SUBTITLE:
        f.center(top+3, fitted(SUBTITLE, w-8), TEAL)
    return TITLE_ROWS


def title_geometry():
    """Top pixel row of the title in the 34-row sky, and the subtitle's text row."""
    height = 12*len(TITLE) + TITLE_GAP*(len(TITLE)-1)
    top = max(1, (34 - height - (3 if SUBTITLE else 0))//2)
    return top, 3 + (top + height + 3)//2


def title_card(progress, flash=False, typed=None, cursor=False, phase='title', duration_ms=60, art=True):
    """The model title over the opener's stage, in the logo's extruded type."""
    f = Frame(phase=phase, duration_ms=duration_ms)
    f.text(2, 2, 'AEVONIX RESEARCH')
    corner = fitted(FACTS['corner'], f.w-24)
    f.text(f.w-2-len(corner), 2, corner)
    p = [[BG for _ in range(f.w)] for _ in range(34)]
    top, subtitle_row = title_geometry()
    placed = [(mask, (f.w-width)//2, top + i*(12+TITLE_GAP), (AMBER, TEAL)[min(i, 1)])
              for i, (mask, width) in enumerate(TITLE)] if art else []
    def put(x, y, color):
        if 0 <= y < len(p) and 0 <= x < f.w:
            p[y][x] = color
    for mask, x0, y0, _ in placed:
        for x, y in mask:
            for dx in range(-1, 3):
                for dy in range(-1, 3):
                    put(x0+x+dx, y0+y+dy, GRAPHITE)
    for mask, x0, y0, base in placed:
        for x, y in mask:
            put(x0+x+1, y0+y+1, blend(BG, base, .40))
    for mask, x0, y0, base in placed:
        for x, y in mask:
            color = blend(base, BG, max(0, y-3)*.013)
            if y < 4 and (x, y-1) not in mask:
                color = CREAM
            put(x0+x, y0+y, blend(color, CREAM, .7) if flash else color)
    f.pixels(0, 3, p)
    left = min((x0 for _, x0, _, _ in placed), default=f.w) - 2
    right = max((x0 + max(x for x, _ in mask) for mask, x0, _, _ in placed), default=0) + 3
    for x, y, glyph in [(8,5,'·'), (97,4,'+'), (20,8,'+'), (87,9,'·'), (30,4,'·')]:
        if not (left <= x <= right and 3 + (top-1)//2 <= y <= subtitle_row):
            f.text(x, y, glyph, blend(BG, CREAM, .8))
    if SUBTITLE and art:
        line = fitted(SUBTITLE, f.w-4)
        x = (f.w-len(line))//2
        shown = line[:len(line) if typed is None else typed]
        f.text(x, subtitle_row, shown, TEAL)
        if cursor and len(shown) < len(line):
            f.text(x+len(shown), subtitle_row, '▌', AMBER)
    f.text(2, 20, '─'*(f.w-4), CORAL)
    floor(f, 21, 24, progress)
    credit(f, 27)
    return f


def beam(old, new, row, phase='beam', duration_ms=60):
    """A CRT beam redraws the sky (rows 3-19) from the top: the new picture above
    the beam row, the old one decaying below it, the stage around it already new."""
    f = dimmed(new, 1, phase, duration_ms)
    f.rows[row] = [Cell('▀', blend(TEAL, CREAM, .65), c.bg) for c in new.rows[row]]
    f.rows[row+1:20] = dimmed(old, .45, phase, 0).rows[row+1:20]
    return f


def title_frames(previous, start):
    """Beam the title in over the previous picture, flash it, type the subtitle
    and hold: TITLE_MS in all. The stage floor runs from progress start at the
    opener's speed (1/880 progress per ms).
    """
    clock = start
    def card(duration_ms, **options):
        nonlocal clock
        frame = title_card(clock, duration_ms=duration_ms, **options)
        clock += duration_ms/880
        return frame
    for row in BEAM_ROWS:
        yield beam(previous, card(60, typed=0), row)
    yield card(80, flash=True, typed=0, phase='title-flash')
    for part in (1, 2, 3):
        yield card(60, typed=len(SUBTITLE)*part//3, cursor=part < 3, phase='title-type')
    for _ in range(TITLE_HOLD_MS//100):
        yield card(100, phase='title-hold')


def synth(progress):
    """The approved outrun composition, centered in the deck's reserved rows."""
    f = Frame(phase='opener', duration_ms=80)
    f.text(2, 2, 'AEVONIX RESEARCH')
    corner = fitted(FACTS['corner'], f.w-24)
    f.text(f.w-2-len(corner), 2, corner)
    p = [[BG for _ in range(f.w)] for _ in range(34)]
    rise = round(26 * (1 - min(1, progress / .8)) ** 2)
    for y in range(17):
        if y in (7, 11, 14, 16) or y+rise >= 34:
            continue
        half = 17 * sqrt(max(0, 1-((y-13)/13)**2))
        color = blend(AMBER, CORAL, y/12) if y <= 6 else CORAL if y <= 10 else TEAL
        for x in range(f.w):
            if abs(x-(f.w-1)/2) <= half:
                p[y+rise][x] = color
    if progress >= 4/11:
        letters = [[BG for _ in range(f.w)] for _ in range(34)]
        _, width = wordmask(3)
        logo(letters, (f.w-width)//2, 18, 3)
        reveal = 23 if progress < 5/11 else 34
        for y in range(reveal):
            for x, color in enumerate(letters[y]):
                if color != BG:
                    # The first complete face flashes, then settles to the palette.
                    if 5/11 <= progress < 6/11 and max(color) > 150:
                        color = blend(color, CREAM, .7)
                    p[y][x] = color
    f.pixels(0, 3, p)
    for x, y, glyph in [(8,5,'·'), (97,4,'+'), (20,8,'+'), (87,9,'·'), (30,4,'·')]:
        f.text(x, y, glyph, blend(BG, CREAM, min(.8, progress*2)))
    f.text(2, 20, '─'*(f.w-4), CORAL)
    floor(f, 21, 24, progress)
    f.center(25, fitted('[ ' + FACTS['opener'] + ' ]', f.w-4), CREAM)
    credit(f, 27)
    return f


def dimmed(source, brightness, phase, duration_ms):
    f = Frame(source.w, source.h, duration_ms, phase)
    def tint(color):
        if brightness <= 1:
            return blend(BG, color, brightness)
        return blend(color, CREAM, brightness-1) if color != BG else BG
    f.rows = [[Cell(c.char, tint(c.fg), tint(c.bg)) for c in row] for row in source.rows]
    return f


def glitch(source, index):
    f = dimmed(source, 1, 'glitch', 90)
    for y in ((6, 7, 12, 13, 18, 22) if index == 0 else (4, 10, 11, 16, 17, 25)):
        shift = (7 if y % 2 else -5) * (1 if index == 0 else -1)
        row = [Cell(' ', BG, BG) for _ in range(f.w)]
        # A one-cell chroma fringe on the leading edge of each torn shape.
        for x, cell in enumerate(source.rows[y]):
            dest = x + shift + (1 if shift > 0 else -1)
            if 0 <= dest < f.w and (cell.char != ' ' or cell.bg != BG):
                row[dest] = Cell(cell.char, TEAL, CORAL if cell.bg != BG else BG)
        for x, cell in enumerate(source.rows[y]):
            dest = x + shift
            if 0 <= dest < f.w and (cell.char != ' ' or cell.bg != BG):
                row[dest] = cell
        f.rows[y] = row
    return f


def collapse(source):
    f = Frame(phase='collapse', duration_ms=100)
    p = [[BG for _ in range(f.w)] for _ in range(8)]
    for y, row in enumerate(source.rows):
        dest = min(7, y*8//source.h)
        for x, cell in enumerate(row):
            color = cell.bg if cell.char == ' ' else cell.fg
            if sum(color) > sum(p[dest][x]):
                p[dest][x] = color
    f.pixels(0, 13, p)
    return f


def power_line(dot=False):
    f = Frame(phase='dot' if dot else 'line', duration_ms=70 if dot else 80)
    center = f.w//2
    for x in range(center-1, center+1) if dot else range(8, f.w-8):
        glow = max(.15, 1-abs(x-center)/(f.w/2))
        # Quantized shades keep both the wire format and GIF palette small.
        f.rows[14][x] = Cell('▄', blend(BG, CREAM, round(glow*5)/5), BG)
        if not dot:
            f.rows[15][x] = Cell('▀', blend(BG, TEAL, round(glow*3)/15), BG)
    return f


def static_noise():
    f = Frame(phase='static', duration_ms=80)
    rng = random.Random(53)
    glyphs = '     ░░▒▓.:/|+*'
    for y in range(f.h):
        for start in range(0, f.w, 6):
            color = blend(BG, rng.choice((QUIET, TEAL)), rng.choice((.12, .2, .28)))
            for x in range(start, min(start+6, f.w)):
                f.rows[y][x] = Cell(rng.choice(glyphs), color, BG)
    return f


def animation_frames():
    """Yield deterministic synthwave, glitch, power-off, static and CRT frames.

    A model title, when present, takes over after the logo, or opens the
    sequence and hands over to the sunrise when title_placement is before-logo.
    """
    if TITLE and TITLE_FIRST:
        # The floor ends where the sunrise starts, so the hand-over is seamless.
        title = list(title_frames(title_card(-TITLE_MS/880, art=False), -TITLE_MS/880))
        yield from title
        # The beam hands over to the sunrise's first frames.
        for i, row in enumerate(BEAM_ROWS):
            opening = beam(title[-1], synth(i/11), row)
            yield opening
        start = len(BEAM_ROWS)
    else:
        start = 0
    for i in range(start, 12):
        opening = synth(i/11)
        yield opening
    if TITLE and not TITLE_FIRST:
        for opening in title_frames(opening, 1 + 80/880):
            yield opening
    yield glitch(opening, 0)
    yield glitch(opening, 1)
    yield collapse(opening)
    yield power_line()
    yield power_line(dot=True)
    yield Frame(phase='black', duration_ms=70)
    final = deck(108)
    yield static_noise()
    yield dimmed(final, .22, 'flicker-dim', 60)
    yield Frame(phase='flicker-black', duration_ms=40)
    yield dimmed(final, 1.2, 'flicker-bright', 50)
    for scan in (round(final.h * part) for part in (.13, .4, .67, .93)):
        f = dimmed(final, .3, 'sweep', 70)
        f.rows[:scan] = [row[:] for row in final.rows[:scan]]
        f.rows[scan] = [Cell('▀', blend(TEAL, CREAM, .65), c.bg) for c in final.rows[scan]]
        yield f
    final.duration_ms = 200
    yield final


def plain_text():
    """Full facts without terminal clipping, color or animation."""
    lines = ['AEVONIX RESEARCH | ' + FACTS['model']]
    if FACTS.get('title_model'):
        lines.append(' | '.join(str(value) for value in (FACTS['title_model'], FACTS.get('title_subtitle')) if value))
    lines.extend(row['label'] + '  ' + row['value'] for row in FACTS['system'])
    lines.append(FACTS['chart_title'] + ' | aggregate decode, tok/s')
    lines.extend(row['label'] + '  ' + format(row['value'], ',.1f') for row in CHART)
    lines.extend(FACTS['start_sequence'])
    lines.extend([FACTS['endpoint'], FACTS['credit']])
    return '\n'.join(lines) + '\n'


def encode_cells(cells):
    """Emit RGB state changes only, coalescing adjacent cells of one color."""
    chunks = []
    foreground = background = None
    for cell in cells:
        codes = []
        if cell.fg != foreground and cell.char != ' ':
            codes.append('38;2;' + ';'.join(map(str, cell.fg)))
            foreground = cell.fg
        if cell.bg != background:
            codes.append('48;2;' + ';'.join(map(str, cell.bg)))
            background = cell.bg
        if codes:
            chunks.append('\x1b[' + ';'.join(codes) + 'm')
        chunks.append(cell.char)
    return ''.join(chunks)


def encode_update(frame, previous=None):
    # The saved origin is in the normal screen, after a single line reservation.
    # No newlines, clear-screen or alternate-screen operations during playback.
    chunks = ['\x1b[u']
    last_row = 0
    for y, row in enumerate(frame.rows):
        before = previous.rows[y] if previous is not None else None
        changed = [x for x, cell in enumerate(row) if before is None or cell != before[x]]
        if not changed:
            continue
        first, end = changed[0], changed[-1]+1
        if y != last_row:
            chunks.append(f'\x1b[{y-last_row}B')
        chunks.append('\r')
        if first:
            chunks.append(f'\x1b[{first}C')
        chunks.append(encode_cells(row[first:end]))
        last_row = y
    return ''.join(chunks)


def run_animation():
    reserved = False
    previous_handlers = {}

    def stop(signum, _frame):
        raise SystemExit(128+signum)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, stop)
        frames = list(animation_frames())
        updates = [encode_update(f, frames[i-1] if i else None) for i, f in enumerate(frames)]
        height = frames[0].h
        sys.stdout.write('\x1b[0m' + '\n'*height + f'\x1b[{height}A\r\x1b[s')
        reserved = True
        sys.stdout.write('\x1b[?25l')
        sys.stdout.flush()
        deadline = time.monotonic()
        skipped = False
        for frame, update in zip(frames, updates):
            deadline += frame.duration_ms/1000
            # Missed intermediate frames are disposable on slow terminals.
            if time.monotonic() < deadline or frame is frames[-1]:
                # Deltas require the immediately preceding frame. If output was
                # skipped, repaint this frame in full on the next opportunity.
                sys.stdout.write(update if not skipped else encode_update(frame))
                sys.stdout.flush()
                skipped = False
            else:
                skipped = True
            delay = deadline-time.monotonic()
            if delay > 0:
                time.sleep(delay)
    finally:
        # Ignore repeated signals during the tiny cleanup write, then restore the
        # caller's handlers. Saved origin also handles interruption during output.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        try:
            sys.stdout.write((f'\x1b[u\x1b[{DECK_HEIGHT}B\r' if reserved else '') + '\x1b[0m\x1b[?25h')
            sys.stdout.flush()
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static', action='store_true', help='print the final deck immediately')
    parser.add_argument('--plain', action='store_true', help='print just the deck facts, without ANSI')
    parser.add_argument('--columns', type=int, default=shutil.get_terminal_size().columns)
    args = parser.parse_args()
    if args.plain:
        sys.stdout.write(plain_text())
        return
    if not sys.stdout.isatty():
        return
    if 'NO_COLOR' in os.environ or args.columns < 86 or os.environ.get('TERM') == 'dumb':
        sys.stdout.write(plain_text())
    elif args.static or 'NO_ANIM' in os.environ or args.columns < 112:
        final = deck(108 if args.columns >= 112 else 84)
        for row in final.rows:
            sys.stdout.write(encode_cells(row) + '\x1b[0m\n')
    else:
        run_animation()


if __name__ == '__main__':
    main()
