
from DAMS import ANM_DAMS, DAM, measure

import what3words
import matplotlib.pyplot as plt

#### GET THE MASK OF BRAZIL #########

import regionmask
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely as sh
from shapely.geometry import Polygon

import pathlib
from numpy import random

def coord_lister(geom):
    coords = list(geom.exterior.coords)
    return (coords)


class Brazil(ANM_DAMS):
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    brazil = world.loc[world['name'] == 'Brazil']  # get Brazil row
    poly_border = brazil['geometry']
    coord_border = np.array(brazil.geometry.apply(coord_lister).values[0])
    bound_box = poly_border.bounds

    load_dams = True

    def __init__(self):
        x0, y0 = self.bound_box.values[0][0], self.bound_box.values[0][1]
        x1, y1 = self.bound_box.values[0][2], self.bound_box.values[0][3]

        self.bounds = [(x0,y0),(x1,y1)]
        self.cart_bound_box = np.array([measure(x0, y0, x1, y0), measure(x0, y0, x0, y1)])

        if self.load_dams:
            self._load_dam_data()

    def division(self,divisor):
        self._N_x, self._N_y = np.ceil(self.cart_bound_box/divisor)
        return self._N_x, self._N_y

    def get_centroid(self):
        ####   Centroid computed from analysis. 
        #tmp = self.brazil.centroid
        #return [tmp.x.values[0], tmp.y.values[0]]
        cx = self.bounds[0][0] + (self.bounds[0][1]-self.bounds[0][0])/2
        cy = self.bounds[1][0] + (self.bounds[1][1]-self.bounds[1][0])/2
        return cx,cy
    
    def get_boundary_corners(self):
        return np.array([(self.bounds[1]), (self.bounds[1][0], self.bounds[0][1]), (self.bounds[0]), (self.bounds[0][0], self.bounds[1][1])])

    def get_width(self):
        W = self.get_boundary_corners()
        return max(W[:,0]) - min(W[:,0])
    
    def get_height(self):
        H = self.get_boundary_corners()
        return max(H[:, 1]) - min(H[:, 1])

    def make_domain_square(self):
        if self.get_width() < self.get_height():
            print('width')
            d = 0 
            self.coord_border = self.coord_border[np.logical_and(self.coord_border[:, d] >= self.get_centroid()[0]-(self.get_width()/2),
                                                                 self.coord_border[:, d] <= self.get_centroid()[0]+(self.get_width()/2)), :]
                                                                 
        elif self.get_width() > self.get_height():
            print('height')
            d = 1 
            self.coord_border = self.coord_border[np.logical_and(self.coord_border[:, d] >= self.get_centroid()[1]-(self.get_height()/2),
                                                                 self.coord_border[:, d] <= self.get_centroid()[1]+(self.get_height()/2)), :]
        else:
            print('Domain is square')
            return
        x0, y0 = min(self.coord_border[:, 0]), min(self.coord_border[:, 1])
        x1, y1 = max(self.coord_border[:, 0]), max(self.coord_border[:, 1])

        self.bounds = [(x0, y0), (x1, y1)]
        self.cart_bound_box = np.array([measure(x0, y0, x1, y0), measure(x0, y0, x0, y1)])

    def _load_dam_data(self):
        super(ANM_DAMS).__init__()
        self._read_data()

    def filter_dams(self):
        dams = self._full_dams_df[['Long', 'Lat']].values
        cond_lon = np.logical_and(dams[:, 0]>self.bounds[0][0], dams[:, 0]<self.bounds[1][0])
        cond_lat = np.logical_and(dams[:, 1]>self.bounds[0][1], dams[:, 1]<self.bounds[1][1])
        return cond_lon*cond_lat

    def plot(self,ret=False):
        fig, ax = plt.subplots(figsize=[20,20])
        plt.plot(self.coord_border[:, 0], self.coord_border[:, 1], c='k')
        plt.plot(self.get_boundary_corners()[:, 0],self.get_boundary_corners()[:, 1], c='red')
        plt.plot([self.get_boundary_corners()[0][0],
                  self.get_boundary_corners()[-1][0]], 
                [self.get_boundary_corners()[0][1],
                self.get_boundary_corners()[-1][1]], c='red')

        if self.load_dams:
            ind = self.filter_dams()
            dams = self._full_dams_df[['Long', 'Lat']].values[ind,:]
            #self.dams_df[['Lat', 'Long']].values
            plt.scatter(dams[:,0], dams[:,1],marker='v', c='C0',s=100)

        plt.scatter(self.get_centroid()[0],self.get_centroid()[1],marker='+',c='red',s=10000)
        #self.boundaries.plot(ax=ax)
        #self.bound_box.plot(ax=ax)
        if ret:
            return ax
        plt.tight_layout()
        plt.show()
        

class Point():
    def __init__(self, *coords):
        if len(coords) == 2:
            self.x = coords[0]
            self.y = coords[1]
        elif len(coords) == 1:
            self.x = coords[0][0]
            self.y = coords[0][1]
        else:
            print('input must be a tupple of len 2 or 2 values')

    def add_payload(self, payload):
        self.payload = payload

    def __repr__(self):
        return '{}: {}'.format(str((self.x, self.y)), repr(self.payload))

    def distance_to(self, other):
        try:
            other_x, other_y = other.x, other.y
        except AttributeError:
            other_x, other_y = other
        return np.hypot(self.x - other_x, self.y - other_y)


class Rect:
    """A rectangle centred at (cx, cy) with width w and height h."""

    def __init__(self, cx, cy, w, h):
        self.cx, self.cy = cx, cy
        self.w, self.h = w, h
        self.west_edge, self.east_edge = cx - w/2, cx + w/2
        self.north_edge, self.south_edge = cy - h/2, cy + h/2

    def __repr__(self):
        return str((self.west_edge, self.east_edge, self.north_edge,
                    self.south_edge))

    def __str__(self):
        return '({:.2f}, {:.2f}, {:.2f}, {:.2f})'.format(self.west_edge,
                                                         self.north_edge, self.east_edge, self.south_edge)

    def contains(self, point):
        """Is point (a Point object or (x,y) tuple) inside this Rect?"""

        try:
            point_x, point_y = point.x, point.y
        except AttributeError:
            point_x, point_y = point

        return (point_x >= self.west_edge and
                point_x < self.east_edge and
                point_y >= self.north_edge and
                point_y < self.south_edge)

    def intersects(self, other):
        """Does Rect object other interesect this Rect?"""
        return not (other.west_edge > self.east_edge or
                    other.east_edge < self.west_edge or
                    other.north_edge > self.south_edge or
                    other.south_edge < self.north_edge)

    def draw(self, ax, c='k', lw=1, **kwargs):
        x1, y1 = self.west_edge, self.north_edge
        x2, y2 = self.east_edge, self.south_edge
        ax.plot([x1, x2, x2, x1, x1], [
                y1, y1, y2, y2, y1], c=c, lw=lw, **kwargs)


class QuadTree:
    """A class implementing a quadtree."""

    def __init__(self, boundary, max_points=4, depth=0):
        """Initialize this node of the quadtree.

        boundary is a Rect object defining the region from which points are
        placed into this node; max_points is the maximum number of points the
        node can hold before it must divide (branch into four more nodes);
        depth keeps track of how deep into the quadtree this node lies.

        """

        self.boundary = boundary
        self.max_points = max_points
        self.points = []
        self.depth = depth
        # A flag to indicate whether this node has divided (branched) or not.
        self.divided = False

    def __str__(self):
        """Return a string representation of this node, suitably formatted."""
        sp = ' ' * self.depth * 2
        s = str(self.boundary) + '\n'
        s += sp + ', '.join(str(point) for point in self.points)
        if not self.divided:
            return s
        return s + '\n' + '\n'.join([
            sp + 'nw: ' + str(self.nw), sp + 'ne: ' + str(self.ne),
            sp + 'se: ' + str(self.se), sp + 'sw: ' + str(self.sw)])

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""

        cx, cy = self.boundary.cx, self.boundary.cy
        w, h = self.boundary.w / 2, self.boundary.h / 2
        # The boundaries of the four children nodes are "northwest",
        # "northeast", "southeast" and "southwest" quadrants within the
        # boundary of the current node.
        self.nw = QuadTree(Rect(cx - w/2, cy - h/2, w, h),
                           self.max_points, self.depth + 1)
        self.ne = QuadTree(Rect(cx + w/2, cy - h/2, w, h),
                           self.max_points, self.depth + 1)
        self.se = QuadTree(Rect(cx + w/2, cy + h/2, w, h),
                           self.max_points, self.depth + 1)
        self.sw = QuadTree(Rect(cx - w/2, cy + h/2, w, h),
                           self.max_points, self.depth + 1)
        self.divided = True

    def insert(self, point):
        """Try to insert Point point into this QuadTree."""

        if not self.boundary.contains(point):
            # The point does not lie inside boundary: bail.
            return False
        if len(self.points) < self.max_points:
            # There's room for our point without dividing the QuadTree.
            self.points.append(point)
            return True

        # No room: divide if necessary, then try the sub-quads.
        if not self.divided:
            self.divide()

        return (self.ne.insert(point) or
                self.nw.insert(point) or
                self.se.insert(point) or
                self.sw.insert(point))

    def query(self, boundary, found_points):
        """Find the points in the quadtree that lie within boundary."""

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary ...
        for point in self.points:
            if boundary.contains(point):
                found_points.append(point)
        # ... and if this node has children, search them too.
        if self.divided:
            self.nw.query(boundary, found_points)
            self.ne.query(boundary, found_points)
            self.se.query(boundary, found_points)
            self.sw.query(boundary, found_points)
        return found_points

    def query_circle(self, boundary, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre.

        boundary is a Rect object (a square) that bounds the search circle.
        There is no need to call this method directly: use query_radius.

        """

        if not self.boundary.intersects(boundary):
            # If the domain of this node does not intersect the search
            # region, we don't need to look in it for points.
            return False

        # Search this node's points to see if they lie within boundary
        # and also lie within a circle of given radius around the centre point.
        for point in self.points:
            if (boundary.contains(point) and
                    point.distance_to(centre) <= radius):
                found_points.append(point)

        # Recurse the search into this node's children.
        if self.divided:
            self.nw.query_circle(boundary, centre, radius, found_points)
            self.ne.query_circle(boundary, centre, radius, found_points)
            self.se.query_circle(boundary, centre, radius, found_points)
            self.sw.query_circle(boundary, centre, radius, found_points)
        return found_points

    def query_radius(self, centre, radius, found_points):
        """Find the points in the quadtree that lie within radius of centre."""

        # First find the square that bounds the search circle as a Rect object.
        boundary = Rect(*centre, 2*radius, 2*radius)
        return self.query_circle(boundary, centre, radius, found_points)

    def __len__(self):
        """Return the number of points in the quadtree."""

        npoints = len(self.points)
        if self.divided:
            npoints += len(self.nw)+len(self.ne)+len(self.se)+len(self.sw)
        return npoints

    def draw(self, ax):
        """Draw a representation of the quadtree on Matplotlib Axes ax."""

        self.boundary.draw(ax)
        if self.divided:
            self.nw.draw(ax)
            self.ne.draw(ax)
            self.se.draw(ax)
            self.sw.draw(ax)


domain = Brazil()
domain.get_width()
domain.get_height()
domain.make_domain_square()
domain.plot()

Dom = Rect(domain.get_centroid()[0], domain.get_centroid()[1],domain.get_width(),domain.get_height())


qtree = QuadTree(Dom,1,0)
dams_ind = domain.filter_dams()
dams_df = domain._full_dams_df[dams_ind]

dams_keep = []
for i in dams_df.index:
    P = Point(dams_df['Long'].loc[i],dams_df['Lat'].loc[i])
    P.add_payload(dams_df.loc[i])
    dams_keep.append(P)

for point in dams_keep:
    qtree.insert(point)
ax = domain.plot(ret=True)
qtree.draw(ax)

QT = qtree.ne.sw.se
ax = domain.plot(ret=True)
QT.draw(ax)


QT1 = QT.ne
ax = domain.plot(ret=True)
QT1.draw(ax)






qtree.sw.ne.sw.depth

plt.show()


























filt = np.logical_and(domain.coord_border[:, d] >= domain.get_centroid()[0]-(domain.get_width()/2),
               domain.coord_border[:, d] <= domain.get_centroid()[0]+(domain.get_width()/2))
x = domain.coord_border[filt,0]
y = domain.coord_border[filt,1]
plt.figure(figsize=[20,20])
plt.plot(domain.coord_border[:, 0], domain.coord_border[:,1],color='red')
plt.plot(x,y)




class Node():
    def __init__(self,x0,y0,w,h,points):
        self.x0 = x0
        self.y0 = y0
        self.width = w
        self.height = h
        self.points = points
        self.children = []

    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def get_points(self):
        return self.points


def recursive_subdivide(node, k):
   if len(node.points) <= k:
       return

   w_ = float(node.width/2)
   h_ = float(node.height/2)

   p = contains(node.x0, node.y0, w_, h_, node.points)
   x1 = Node(node.x0, node.y0, w_, h_, p)
   recursive_subdivide(x1, k)

   p = contains(node.x0, node.y0+h_, w_, h_, node.points)
   x2 = Node(node.x0, node.y0+h_, w_, h_, p)
   recursive_subdivide(x2, k)

   p = contains(node.x0+w_, node.y0, w_, h_, node.points)
   x3 = Node(node.x0 + w_, node.y0, w_, h_, p)
   recursive_subdivide(x3, k)

   p = contains(node.x0+w_, node.y0+h_, w_, h_, node.points)
   x4 = Node(node.x0+w_, node.y0+h_, w_, h_, p)
   recursive_subdivide(x4, k)

   node.children = [x1, x2, x3, x4]


def contains(x, y, w, h, points):
   pts = []
   for point in points:
       if point.x >= x and point.x <= x+w and point.y >= y and point.y <= y+h:
           pts.append(point)
   return pts


def find_children(node):
   if not node.children:
       return [node]
   else:
       children = []
       for child in node.children:
           children += (find_children(child))
   return children


class QTree():
    def __init__(self, k, n):
        self.threshold = k
        self.points = [Point(random.uniform(
            0, 10), random.uniform(0, 10)) for x in range(n)]
        self.root = Node(0, 0, 10, 10, self.points)

    def add_point(self, x, y):
        self.points.append(Point(x, y))

    def get_points(self):
        return self.points

    def subdivide(self):
        recursive_subdivide(self.root, self.threshold)

    def graph(self):
        fig = plt.figure(figsize=(12, 8))
        plt.title("Quadtree")
        c = find_children(self.root)
        print("Number of segments: %d" % len(c))
        areas = set()
        for el in c:
            areas.add(el.width*el.height)
        print("Minimum segment area: %.3f units" % min(areas))
        for n in c:
            plt.gcf().gca().add_patch(patches.Rectangle(
                (n.x0, n.y0), n.width, n.height, fill=False))
        x = [point.x for point in self.points]
        y = [point.y for point in self.points]
        plt.plot(x, y, 'ro')  # plots the points as red dots
        plt.show()
        return


Q = QTree(1,1000)
Q.subdivide()
Q.graph()


domain.get_width() >domain.get_height()
d = 0 
domain.coord_border[np.logical_and(domain.coord_border[:, d] >= domain.get_centroid()[0]-(domain.get_width()/2),
                                   domain.coord_border[:, d] <= domain.get_centroid()[0]+(domain.get_width()/2)), :]


domain.get_width() > domain.get_height()

seed = Point(domain.get_centroid())
root = Node(seed.x, seed.y, domain.get_width(), domain.get_height())


domain.get_width() < domain.get_height()


domain.get_width() > domain.get_height()



















B = Brazil()
B._division(2)
B.plot()
B.get_centroid()


for i in range(len(B.coordinates)-1):
    p0 = B.coordinates[i]
    p1 = B.coordinates[i+1]


type(boundaries)


plt.plot(B.coordinates)


coordinates_array = np.asarray(Brazil['geometry'].exterior.coords)


A = np.asarray(B.boundaries)


B.boundaries._values

A = gpd.GeoDataFrame(B.boundaries)
coordinate_list = [(x, y) for x, y in zip(A['geometry'].x, A['geometry'].y)]


fig, ax = plt.subplots(figsize=[20, 20])
B.boundaries.plot(ax=ax)
#Polygon(B.get_boundary_corners(), ax=ax)
plt.show()
