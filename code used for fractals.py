#recipes 



import numpy as np
from matplotlib.patches import Circle as MplCircle

def transform_pairing_circles(C1, C2, u=complex(1, 0), v=complex(0, 0)):
    """Return mobius transformation that pairs the circles C1 and C2"""
    if abs(u)**2 - abs(v)**2 != 1:
        raise ValueError('|u|^2 - |v|^2 is not 1')

    P, r = C1.center, C1.radius
    Q, s = C2.center, C2.radius
    ubar = u.conjugate()
    vbar = v.conjugate()

    M = np.array([
        [ubar, -ubar * P + r * vbar],
        [u, -u * P + r * v]
    ])
    M = s * M + Q

    return MobiusTransformation(M)


def kissing_schottky(y, v):
    """Return generators and circles for symmetric kissing Schottky group"""
    assert np.isreal(y) and np.isreal(v)
    x = np.sqrt(1 + y ** 2)
    u = np.sqrt(1 + v ** 2)
    yv = y * v
    k = 1 / yv - np.sqrt(1 / yv ** 2 - 1)
    assert abs(k) < 1 or np.isclose(abs(k), 1)

    a = MobiusTransformation(u, 1j * k * v, -1j * v / k, u)
    b = MobiusTransformation(x, y, y, x)
    A = a.inv()
    B = b.inv()

    assert np.isclose(b(a(B(A))).M.trace(), -2)

    C_a = Circle(complex(0, k * u / v), k / v)
    C_A = Circle(complex(0, -k * u / v), k / v)
    C_b = Circle(complex(-x / y, 0), 1 / y)
    C_B = Circle(complex(x / y, 0), 1 / y)

    return [a, b, A, B], [C_a, C_b, C_A, C_B]


def solve_markov(t_a, t_b, use_negative=True):
    tatb = t_a * t_b
    root = np.sqrt(tatb ** 2 - 4 * (t_a ** 2 + t_b ** 2))
    if use_negative:
        t_ab = (tatb - root) / 2
    else:
        t_ab = (tatb + root) / 2
    return t_ab


def parabolic_commutator(t_a, t_b, use_negative=True):
    """Grandma's special parabolic commutator groups"""
    t_a = complex(t_a)
    t_b = complex(t_b)
    t_ab = solve_markov(t_a, t_b, use_negative)

    # fixed point of commutator abAB
    z0 = ((t_ab - 2) * t_b) / (t_b * t_ab - 2 * t_a + 2j * t_ab)

    # generator matrices with the correct traces
    a = MobiusTransformation(
        t_a / 2,
        (t_a * t_ab - 2 * t_b + 4j) / ((2 * t_ab + 4) * z0),
        ((t_a * t_ab - 2 * t_b - 4j) * z0) / (2 * t_ab - 4),
        t_a / 2
    )
    b = MobiusTransformation(
        (t_b - 2j) / 2,
        t_b / 2,
        t_b / 2,
        (t_b + 2j) / 2
    )

    A = a.inv()
    B = b.inv()
    
    assert np.isclose(a.M.trace(), t_a)
    assert np.isclose(b.M.trace(), t_b)
    assert np.isclose(a(b).M.trace(), t_ab)

    return [a, b, A, B]


def jorgensen(t_a, t_b, use_negative=True):
    """Jorgensen's recipe – fixed point of abAB at infinity"""
    t_a = complex(t_a)
    t_b = complex(t_b)
    t_ab = solve_markov(t_a, t_b, use_negative)

    a = MobiusTransformation(
        t_a - t_b / t_ab,
        t_a / t_ab**2,
        t_a,
        t_b / t_ab
    )
    b = MobiusTransformation(
        t_b - t_a / t_ab,
        -t_b / t_ab**2,
        -t_b, t_a / t_ab
    )

    A = a.inv()
    B = b.inv()

    assert np.isclose(np.linalg.det(a.M), 1)
    assert np.isclose(np.linalg.det(b.M), 1)
    assert np.all(np.isclose(a(b(A(B))).M, np.array([[-1, -2], [0, -1]])))

    return [a, b, A, B]


def riley(c):
    """Riley's recipe – two parabolic matrices"""
    c = complex(c)
    a = MobiusTransformation(1, 0, c, 1)
    b = MobiusTransformation(1, 2, 0, 1)
    return [a, b, a.inv(), b.inv()]


#common

from functools import reduce
from numbers import Number

import matplotlib.pyplot as plt

# constants
NUMERIC_EPS = 1e-12
VISUAL_EPS = 1e-4
MAX_LEVEL = 1e5

# helpers for going between words, tags, and functions
char_to_tag = {'a': 0, 'b': 1, 'A': 2, 'B': 3}
tag_to_char = ['a', 'b', 'A', 'B']


def tags_to_word(tags):
    return ''.join(tag_to_char[t] for t in tags)


def word_to_fct(s, gens):
    return reduce(lambda S, T: S(T), (gens[char_to_tag[c]] for c in s))


def tags_to_fct(tags, gens):
    return reduce(lambda S, T: S(T), (gens[t] for t in tags))


def word_to_tags(s):
    return [char_to_tag[c] for c in s]


class Circle:

    def __init__(self, center, radius, inside_pt=None):
        self.center = complex(center)
        if not np.isreal(radius) or radius <= 0:
            raise ValueError('Radius must be positive real')
        self.radius = radius

        # point that is "inside" the circle – defaults to center
        self.inside_pt = inside_pt
        if inside_pt is None:
            self.inside_pt = self.center

    def plot(self, ax, color='k'):
        circ = plt.Circle((self.center.real, self.center.imag), self.radius, color=color)
        ax.add_artist(circ)
        # TODO: how to fill the outside

    def __eq__(self, other):
        return np.isclose(self.center, other.center) and np.isclose(self.radius, other.radius)

    def __repr__(self):
        return f'Circle(center={self.center}, radius={self.radius})'


class Line:

    # lines as cos(a) * x + sin(a) * y = b,
    # for direction a in [0, 2pi) and offset b >= 0
    def __init__(self, direction, offset, inside_pt=None):
        
        if not np.isreal(direction) or direction < 0 or direction >= 2*np.pi:
            if not np.isreal(direction):
                raise ValueError('Direction must be in [0, 2pi)')
            direction = direction%(2*np.pi)
            
        self.direction = direction
        if not np.isreal(offset) or offset < 0:
            if np.isclose(offset,0):
                offset=0
            else:
                raise ValueError('Offset must be >= 0')
        self.offset = offset

        self.x_coef = np.cos(self.direction)
        self.y_coef = np.sin(self.direction)

        # point that is "inside" the line
        # defaults to (0, y+c) or (x+c, 0) depending on orientation
        self.inside_pt = inside_pt
        if inside_pt is None:
            if not self.is_vertical():
                self.inside_pt = self(0) + 1j
            else:
                self.inside_pt = complex(self.offset + 1, 0)

        # for generality this is useful to have
        self.radius = np.inf

    def is_horizontal(self):
        return np.isclose(self.x_coef, 0)

    def is_vertical(self):
        return np.isclose(self.y_coef, 0)

    def plot(self, ax, color='k'):
        if not self.is_vertical():
            xs = ax.get_xlim()
            ys = [self(x) for x in xs]
            ax.plot(xs, ys, color=color, lw=0.01)
        else:
            ys = ax.get_ylim()
            xs = [self(y) for y in ys]
            
            ax.fill_betweenx(ys, xs, color=color)
        # TODO: color the appropriate side

    def __call__(self, other):
        if isinstance(other, Number):
            other = complex(other)
            if not self.is_vertical():
                # we assume this is x and must compute y
                return (self.offset - self.x_coef * other) / self.y_coef
            # otherwise vertical, so assume this is y and return x
            return self.offset
        raise NotImplementedError('Transformation not supported')

    def __eq__(self, other):
        return np.isclose(self.direction, other.direction) and np.isclose(self.offset, other.offset)

    def __repr__(self):
        return f'Line(direction={self.direction}, offset={self.offset})'




#mobius


class MobiusTransformation:

    def __init__(self, a, b=None, c=None, d=None):
        # either initialize with matrix or its entries
        if b is None:
            self.M = a
        else:
            det = a * d - b * c
            if np.isclose(det, 0):
                raise ValueError('Determinant must be non-zero')
            self.M = 1 / det * np.array([[a, b], [c, d]])

    @property
    def a(self): return self.M[0, 0]

    @property
    def b(self): return self.M[0, 1]

    @property
    def c(self): return self.M[1, 0]

    @property
    def d(self): return self.M[1, 1]

    def inv(self):
        # Return inverse transformation
        return MobiusTransformation(self.d, -self.b, -self.c, self.a)

    def fps(self):
        # Return positive and negative fixed points (may be the same)
        denom = 2 * self.c
        if denom == 0:
            return np.inf
        diff = self.a - self.d
        root = np.sqrt(self.M.trace()**2 - 4)
        return (diff + root) / denom, (diff - root) / denom

    def multiplier(self):
        tr = self.M.trace()
        return ((tr + np.sqrt(tr**2 - 4)) / 2)**2

    def sink(self):
        # Return attracting fixed point (or the only one)
        pos_fp, neg_fp = self.fps()
        if pos_fp == neg_fp:
            return pos_fp
        k = self.multiplier()
        return pos_fp if abs(k) > 1 else neg_fp

    def source(self):
        # Return repelling fixed point (or the only one)
        pos_fp, neg_fp = self.fps()
        if pos_fp == neg_fp:
            return pos_fp
        k = self.multiplier()
        return neg_fp if abs(k) > 1 else pos_fp

    def conjugate(self, S):
        """Return conjugation by S, i.e. STS^-1"""
        return S(self(S.inv()))

    def __call__(self, other):
        # composition via matrix multiplication
        if isinstance(other, MobiusTransformation):
            return MobiusTransformation(self.M.dot(other.M))

        # application to points and circles
        if isinstance(other, Number):
            return self._apply_to_point(other)
        if isinstance(other, Circle):
            return self._apply_to_circle(other)
        if isinstance(other, Line):  # technically a special case of circles
            return self._apply_to_line(other)

        raise NotImplementedError(f'Transformation not supported: {other}')

    def _apply_to_point(self, z):
        """Apply Mobius transformation to complex number z"""
        if z == np.inf:
            return self.a / self.c if self.c != 0 else np.inf
        num = self.a * z + self.b
        den = self.c * z + self.d
        return num / den if den != 0 else np.inf

    def _apply_to_circle(self, C):
        """Apply Mobius transformation to circle C"""
        discrim = abs(self.d / self.c + C.center)

        if np.isclose(discrim, C.radius):  # image is a line
            return self._circle_to_line(C)

        if np.isclose(discrim, 0):  # image is a concentric circle
            new_cen = C.center
        else:  # general case
            z = C.center
            if self.c != 0:
                z -= C.radius**2 / (self.d / self.c + C.center).conjugate()
            new_cen = self._apply_to_point(z)

        new_rad = abs(new_cen - self._apply_to_point(C.center + C.radius))
        D = Circle(center=new_cen, radius=new_rad)
        return D

    def _apply_to_line(self, L):
        """Apply Mobius transformation to line L"""
        # 3 points on L
        denom = np.sin(L.direction) if not L.is_horizontal() else 1
        cosine = np.cos(L.direction)
        x1, y1 = 0, L.offset / denom
        x2, y2 = -1, (L.offset + cosine) / denom
        x3, y3 = 1, (L.offset - cosine) / denom

        # 3 points on T(L)
        z1 = self(complex(x1, y1))
        z2 = self(complex(x2, y2))
        z3 = self(complex(x3, y3))

        # solve for circle
        # TODO: need to tell if this also gives a line...
        w = z3 - z1
        w /= z2 - z1
        c = (z1 - z2) * (w - abs(w) ** 2) / 2j / w.imag - z1
        rad = abs(c + z1)
        return Circle(-c, rad)

    def _circle_to_line(self, C):
        """Get application to circle C, given that we know the result is a line"""
        z1 = self(C.center + C.radius)
        z2 = self(C.center - C.radius)
        x1, y1 = z1.real, z1.imag
        x2, y2 = z2.real, z2.imag

        direction = np.arctan((x2 - x1) / (y1 - y2))
        offset = np.cos(direction) * x1 + np.sin(direction) * y1
        return Line(direction, offset)

    def __eq__(self, other):
        return self.M == other.M

    def __repr__(self):
        return f'Mobius transformation:\n{str(self.M)}'


#plot limit set

from collections import deque
import itertools


def plot_limit_set(gens, beg_prefix='a', end_prefix='b', as_curve=True, ax=None, max_level=MAX_LEVEL, eps=VISUAL_EPS, debug=False, z = None, **kwargs):
    """
    Plot limit set of a generating set of Mobius transformations.
    The kwargs are passed on to matplotlib.

    :param gens: list of generating Mobius transformations
    :param beg_prefix: prefix to start at, as a string (default a)
    :param end_prefix: prefix to end at, as a string (default b)
    :param as_curve: whether to plot as a continuous curve (default) or individual points
    :param ax: optional axis for plotting
    :param max_level: max level to plot
    :param eps: tolerance for termination
    :param debug: debug prints
    :return: the axis used
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

    pts = list(dfs(gens, beg_prefix=beg_prefix, end_prefix=end_prefix, max_level=max_level, eps=eps, debug=debug))
    if not z == None:
        x = gens[0](z)
        #x2 = gens[1](z)
        #x3 = gens[2](z)
        #x4 = gens[3](z)
        pts.append(x)
        #pts.append(x2)
        #pts.append(x3)
        #pts.append(x4)
        print(x)
    xs = [x.real for x in pts]
    ys = [x.imag for x in pts]
    if as_curve:
        # connect last to first points, if we're plotting the whole curve
        if beg_prefix == 'a' and end_prefix == 'b':
            xs.append(xs[0])
            ys.append(ys[0])
        ax.plot(xs, ys, **kwargs) 
    else:
        ax.scatter(xs, ys, marker='.', s=0.01, **kwargs)

    return ax

def MTS(M):
    centre = -M.d/M.c
    radius = 1/np.abs(M.c)
    return Circle(complex(centre.real,centre.imag), radius)












def PLSWT(gens, beg_prefix = 'a', end_prefix = 'b', as_curve = True, ax=None, max_level = MAX_LEVEL, eps = VISUAL_EPS, debug=False, z=None, shift = False, **kwargs):
    #start with finding the isometric circles
#plot limit set with tiles
    
    #start with checking for perspective shift and finding the conjugating matrix

    
    base_circles=[]
    Circles = []
    
    if shift == False:
        
        for M in gens:
            Circles.append(MTS(M))
            base_circles.append(MTS(M))
    else:
        C_a = Line(np.pi/2, 0)
        C_A = gens[2](C_a)
        C_b = MTS(gens[1])
        C_B = MTS(gens[3])
        
        Circles = [C_a, C_b, C_A, C_B]
        base_circles = [C_a, C_b, C_A, C_B]
    
        
        
        
        
    for i in range(4):
        
        for j in range(4):
            if j == (i+2)%4:
                continue
            if j == i:
                circ = gens[i](base_circles[(j+2)%4])
                Circles.append(circ)
            else:
                
                circ = gens[i](base_circles[j])
                Circles.append(circ)
        
        
    if ax is None:
        fig=plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        
        ax.set_xlim((
            #min(C.center.real - C.radius for C in Circles if np.isfinite(C.radius)),
            #max(C.center.real + C.radius for C in Circles if np.isfinite(C.radius))
            -2,2
        ))
        ax.set_ylim((
            #min(C.center.imag - C.radius for C in Circles if np.isfinite(C.radius)),
            #max(C.center.imag + C.radius for C in Circles if np.isfinite(C.radius))
            -2,2
        ))

    

    for c in Circles:
        if isinstance(c, Line):
            c.plot(ax)
            
        else:
            
            x = c.center.real
            y = c.center.imag
        
            circle_patch = MplCircle(
                (x,y),
                c.radius,
                fill=False,
                edgecolor = 'black',
                linewidth = 1
                )
            ax.add_patch(circle_patch)
        
    if shift == True:
        #patching up the image circles by assuring C_aa is present
        x = C_A.center.real
        y = -1 * C_A.center.imag
        C_aa = Circle(complex(x,y), C_A.radius)
        circle_patch = MplCircle(
            (x,y),
            C_A.radius,
            fill=False,
            edgecolor = 'black',
            linewidth = 1
            )
        ax.add_patch(circle_patch)
        
        C_AA = gens[0](C_aa)
        x = C_AA.center.real
        y = -1 * C_AA.center.imag
        circle_patch = MplCircle(
            (x,y),
            C_AA.radius,
            fill=False,
            edgecolor = 'black',
            linewidth = 1
            )
        ax.add_patch(circle_patch)
        
        
    pts = list(dfs(gens, beg_prefix=beg_prefix, end_prefix=end_prefix, max_level=max_level, eps=eps, debug=debug))
    if not z == None:
        x = gens[0](z)
            #x2 = gens[1](z)
            #x3 = gens[2](z)
            #x4 = gens[3](z)
        pts.append(x)
            #pts.append(x2)
            #pts.append(x3)
            #pts.append(x4)
        print(x)
    xs = [x.real for x in pts]
    ys = [x.imag for x in pts]
    if as_curve:
        # connect last to first points, if we're plotting the whole curve
        if beg_prefix == 'a' and end_prefix == 'b':
            xs.append(xs[0])
            ys.append(ys[0])
        ax.plot(xs, ys, **kwargs)
    else:
        ax.scatter(xs, ys, marker='.', s=2, **kwargs)
    plt.grid()
        
    return ax
     

    














def get_commutator_fps(gens):
    n = len(gens)
    # will be useful later, for now just used for starting point
    beg_pts = [
        reduce(lambda S, T: S(T), (gens[(i + j) % n] for j in range(1, n + 1)))
        for i in range(n)
    ]
    beg_pts = [T.sink() for T in beg_pts]

    end_pts = [
        reduce(lambda S, T: S(T), (gens[(i - j) % n] for j in range(1, n + 1)))
        for i in range(n)
    ]
    end_pts = [T.sink() for T in end_pts]
    return beg_pts[-1], end_pts


def dfs(gens, beg_prefix='a', end_prefix='b', max_level=MAX_LEVEL, eps=VISUAL_EPS, debug=False):
    """
    Non-recursive DFS for plotting limit set (only for 4 generators).

    :param gens: list of generating Mobius transformations
    :param beg_prefix: prefix to start at, as a string (default a)
    :param end_prefix: prefix to end at, as a string (default b)
    :param max_level: max level to plot
    :param eps: tolerance for termination
    :param debug: debug prints
    :return: complex points to plot
    """
    beg_tags = word_to_tags(beg_prefix)
    end_tags = word_to_tags(end_prefix)
    if not precedes_or_equal(beg_tags, end_tags):
        raise ValueError("beginning prefix must precede end prefix in tree ordering")

    # start with the first word that starts with beg_prefix
    tags = deque([beg_tags[0]])
    words = deque([gens[beg_tags[0]]])
    if len(beg_tags) > 1:
        for t in beg_tags[1:]:
            tags.append(t)
            words.append(words[-1](gens[t]))

    if debug:
        print(tags_to_word(tags))
    level = len(tags)
    old_pt, fps = get_commutator_fps(gens)

    while True:
        # go forwards till the end of the branch
        while True:
            old_pt, branch_term = branch_termination(words[-1], fps[tags[-1]], old_pt, eps, level, max_level)
            if branch_term:
                break
            next_tag = right_of(tags[-1])
            next_word = words[-1](gens[next_tag])
            tags.append(next_tag)
            words.append(next_word)
            level += 1

        # we have a result!
        yield old_pt
        if debug:
            print(level)
            print(tags_to_word(tags))

        # stop if we're at the last word starting with end_prefix
        if starts_with(tags, end_tags) and all_lefts_from(tags, len(end_tags) - 1):
            break

        # go backwards till we have another turn or reach the root
        while True:
            last_tag = tags.pop()
            _ = words.pop()
            level -= 1
            if level == 0 or available_turn(last_tag, tags[-1]):
                break

        # turn and go forwards
        next_tag = left_of(last_tag)
        if level == 0:
            # if we're back to the first generator at the root, we're done!
            if next_tag == 0:
                break
            next_word = gens[next_tag]
        else:
            next_word = words[-1](gens[next_tag])
        tags.append(next_tag)
        words.append(next_word)
        level += 1


def available_turn(last_tag, curr_tag):
    """Return true if there's another turn to take from curr_tag"""
    return left_of(last_tag) != inverse_of(curr_tag)


def branch_termination(T, fp, old_pt, eps, level, max_level):
    """Return true if we should terminate branch"""
    new_pt = T(fp)
    if level > max_level or abs(new_pt - old_pt) < eps:
        return new_pt, True
    return old_pt, False


def right_of(tag):
    return (tag + 1) % 4


def left_of(tag):
    return (tag - 1) % 4


def inverse_of(tag):
    return (tag + 2) % 4


def starts_with(tags, prefix_tags):
    """Check whether tags starts with a given prefix"""
    return list(itertools.islice(tags, 0, len(prefix_tags))) == prefix_tags


def all_lefts_from(tags, idx):
    """Check whether tags is all left turns starting from idx"""
    for i in range(idx, len(tags)-1):
        if left_of(tags[i]) != tags[i+1]:
            return False
    return True


def precedes_or_equal(tags_1, tags_2):
    """Check whether tags_1 precedes tags_2 in the tree ordering (or is equal)"""
    # ordering is: a, B, A, b <=> 0, 3, 2, 1
    # to make things easier, we replace 0 with 4
    first = [x if x != 0 else 4 for x in tags_1]
    second = [x if x != 0 else 4 for x in tags_2]
    return recursive_precedes_or_equal(first, second)


def recursive_precedes_or_equal(first, second):
    if first == second or len(first) == 0:
        return True
    if len(second) == 0:
        return False
    return (
        first[0] > second[0]
        or (
            first[0] == second[0]
            and recursive_precedes_or_equal(first[1:], second[1:])
        )
    )

#plot tiles

def plot_tiles(gens, circs, ax=None, plot_level=None, eps=VISUAL_EPS):
    """
    Plot tiles generated by set of Mobius transformations.

    :param gens: list of generating Mobius transformations
    :param circs: seed circles to start with
    :param ax: optional axis for plotting
    :param plot_level: plot only circles of this level, or None to plot all
    :param eps: minimum radius size to return
    :return: the axis used
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        ax.set_xlim((
            min(C.center.real - C.radius for C in circs if np.isfinite(C.radius)),
            max(C.center.real + C.radius for C in circs if np.isfinite(C.radius))
            #-1,1
        ))
        ax.set_ylim((
            #min(C.center.imag - C.radius for C in circs if np.isfinite(C.radius)),
            #max(C.center.imag + C.radius for C in circs if np.isfinite(C.radius))
            -1.5,1.5
        ))
        
    colors = plt.cm.get_cmap('viridis', 20)  # TODO how to set this appropriately?

    # sort by level so that they plot in the correct order
    tiles = sorted(dfs_tiles(gens, circs, max_level=plot_level, eps=eps), key=lambda x: x[1])
    # if plot_level is not None:
    #     tiles = [x for x in tiles if x[1] == plot_level]
    for C, level in tiles:
        C.plot(ax, color=colors(level))
    
    plt.grid()
    return ax


def dfs_tiles(gens, circs, max_level, eps):
    """
    Iterate through tiles with depth-first search.

    :param gens: list of generating Mobius transformations
    :param circs: seed circles to start with
    :param eps: minimum radius size to return
    :return: circle and corresponding level
    """
    for k in range(len(gens)):
        yield circs[k], 0
        yield from explore_tree_tiles(gens[k], k, circs[k], 1, gens, max_level, eps)


def explore_tree_tiles(X, l, C, level, gens, max_level, eps):
    if max_level is not None and level > max_level:
        return
    n = len(gens)
    for k in range(l-1, l + 2):
        Y = X(gens[k % n])
        new_circ = Y(C)
        yield new_circ, level
        if new_circ.radius > eps:
            yield from explore_tree_tiles(Y, k, C, level + 1, gens, max_level, eps)



def bfs(gens, tags, max_level = 5):
    current = list(zip(gens, tags))
    invs = [2,3,0,1]
    for l in range(max_level + 1):
        next_level = []
        for seed, tag in current:
            yield seed, tag
            for i in range(len(gens)):
                if i == invs[tags.index(tag)]:
                    continue
                next_level.append((seed(gens[i]), tags[i]))
            current = next_level
            

def plot_circle(C, ax, color = 'k', fill = False):
    circ = plt.Circle((C.center.real, C.center.imag), C.radius, color = color, fill = fill)
    ax.add_artist(circ)

        
def exhaustive_plot_tiles(gens, invs, tags, circs, max_level = 5):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect = 'equal')
    ax.set_xlim((
        #max(C.center.real + C.radius for C in circs),
       # min(C.center.real + C.radius for C in circs)
       -1.5,1.5
        ))
    ax.set_ylim((
       # max(C.center.imag + C.radius for C in circs),
       # min(C.center.imag + C.radius for C in circs)
       -1.5,1.5
        ))
    
    for C, color in zip(circs, tags):
        plot_circle(C, ax, color = color)
    layer2 = []
    tag2 = []
    for G in gens:
        for C, color in zip(circs, tags):
            if not C == MTS(G):
                newcirc = G(C)
                layer2.append(newcirc)
                tag2.append(color)
                plot_circle(newcirc, ax, color=color)
        for C, color in zip(layer2, tag2):
            newcirc2 = G(C)
            plot_circle(newcirc2, ax, color=color)
    
    plt.grid()
    return ax
        
#def plot_limit_set_with_tiles(gens, circs, )    
        

tags = ['r', 'b', 'black', 'g']



import matplotlib.animation as animation

def animated(gens):
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.set_xlim((-1.3, 1.3))
    ax.set_ylim((-1.3, 1.3))
    
    pts = list(dfs(gens, max_level=100, eps=1e-2))
    
    xs = [x.real for x in pts] + [pts[0].real]
    ys = [x.imag for x in pts] + [pts[0].imag]
    
    n_frames = len(xs)
    
    line, = ax.plot([], [], linewidth=0.5)
    
    
    def init():  # only required for blitting to give a clean slate.
        line.set_data([], [])
        return line,
    
    
    def animate(i):
        line.set_data(xs[:i], ys[:i])
        return line,


    ani = animation.FuncAnimation(
        fig, animate,
        init_func=init,
        frames=n_frames,
        interval=2,
        blit=True,
    )
    
    # requires ffmpeg
    # ani.save("movie.mp4")
    
    plt.show()
    
    


def classic_schottky(y, v, epsilon):
    """Return generators and circles for disjoint schottky group based on separating the kissing schottky generators"""
    assert np.isreal(y) and np.isreal(v)
    x = np.sqrt(1 + y ** 2)
    u = np.sqrt(1 + v ** 2)
    yv = y * v
    k = 1 / yv - np.sqrt(1 / yv ** 2 - 1)
    assert abs(k) < 1 or np.isclose(abs(k), 1)
    assert x - y*epsilon > 1 and u + 1j*v/k * epsilon > 1

    a = MobiusTransformation(u + 1j*v*epsilon/k,  1j*k*((u + 1j*v*epsilon/k)**2 - 1) / v, -1j * v / k, u + 1j * v*epsilon/k)
    b = MobiusTransformation(x - y * epsilon, ((x-y*epsilon)**2 - 1)/y, y, x - y*epsilon)
    A = a.inv()
    B = b.inv()


    C_a = Circle(complex(0, k * u / v + epsilon), k / v)
    print(np.linalg.det(a.M), C_a)
    C_A = Circle(complex(0, -k * u / v - epsilon), k / v)
    print(np.linalg.det(A.M), C_A)
    C_b = Circle(complex(-x / y - epsilon, 0), 1 / y)
    print(np.linalg.det(b.M), C_b)
    C_B = Circle(complex(x / y + epsilon, 0), 1 / y)
    print(np.linalg.det(B.M), C_B)
    print(b(a(B(A))).M.trace())
    return [a, b, A, B], [C_a, C_b, C_A, C_B]


def classic_schottky2(theta):
    
    a = MobiusTransformation(1/np.sin(theta), 1j*np.cos(theta)/np.sin(theta), -1j*np.cos(theta)/np.sin(theta), 1/np.sin(theta))
    b = MobiusTransformation(1/np.sin(theta), 1/np.tan(theta), 1/np.tan(theta), 1/np.sin(theta))
    A = a.inv()
    B = b.inv()
    
    C_a = Circle(complex(0, 1/np.cos(theta)), np.tan(theta))
    C_A = Circle(complex(0, -1/np.cos(theta)), np.tan(theta))
    C_b = Circle(complex(1/np.cos(theta),0), np.tan(theta))
    C_B = Circle(complex(-1/np.cos(theta),0), np.tan(theta))
    
    print(np.linalg.det(a.M), np.linalg.det(b.M), b(a(B(A))).M.trace())
    return [a, b, A, B], [C_a, C_A, C_b, C_B]


gens, circs = kissing_schottky(1,1)
gens2 = parabolic_commutator(2+1j, 2+1j)


circs2= [MTS(gens2[0]), MTS(gens2[1]), MTS(gens2[2]), MTS(gens2[3])]
    
fp1, fp2 = gens[0].fps() 

#construct map which maps 'a' to a line
CM = MobiusTransformation(1, 0, 1, -fp1) #conjugating matrix
CM2 = CM.inv() #inv
newgens = []
for G in gens:
    G = CM(G(CM2)) #conjugate
    newgens.append(G)
    
newcircs= [MTS(newgens[0]), MTS(newgens[1]), MTS(newgens[2]), MTS(newgens[3])]

invs = [2,3,0,1]

def Apollonian():
    a = MobiusTransformation(1,0,-2j,1)
    b = MobiusTransformation(1-1j,1,1,1+1j)
    A = a.inv()
    B = b.inv()
    
    C_a = Line(np.pi/2, 0)
    C_b = Circle(complex(1,-1),1)
    C_A = A(C_a)
    C_B = Circle(complex(-1,-1),1)
    
    circs = [C_a, C_b, C_A, C_B]
    gens = [a,b,A,B]
    return gens, circs

def show_orthogonality(gens, circs, ax = None):
    base_circles=[]
    Circles = []
    
        

        
    for M in gens:
        Circles.append(MTS(M))
        base_circles.append(MTS(M))
    
        
        
        
        
    for i in range(4):
        
        for j in range(4):
            if j == (i+2)%4:
                continue
            if j == i:
                circ = gens[i](base_circles[(j+2)%4])
                Circles.append(circ)
            else:
                
                circ = gens[i](base_circles[j])
                Circles.append(circ)
        
        
    if ax is None:
        fig=plt.figure()
        ax = fig.add_subplot(111,aspect='equal')
        
        ax.set_xlim((
            #min(C.center.real - C.radius for C in Circles if np.isfinite(C.radius)),
            #max(C.center.real + C.radius for C in Circles if np.isfinite(C.radius))
            -2,2
        ))
        ax.set_ylim((
            #min(C.center.imag - C.radius for C in Circles if np.isfinite(C.radius)),
            #max(C.center.imag + C.radius for C in Circles if np.isfinite(C.radius))
            -2,2
        ))

    

    for c in Circles:
        if isinstance(c, Line):
            c.plot(ax)
            
        else:
            
            x = c.center.real
            y = c.center.imag
        
            circle_patch = MplCircle(
                (x,y),
                c.radius,
                fill=False,
                edgecolor = 'black',
                linewidth = 1
                )
            ax.add_patch(circle_patch)
            
    circle_patch_2 = MplCircle(
        (0,0),
        1,
        fill=False,
        edgecolor = 'blue',
        linewidth  = 2
        )
    ax.add_patch(circle_patch_2)
