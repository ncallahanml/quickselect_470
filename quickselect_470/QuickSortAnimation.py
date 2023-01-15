import os

os.system('pip install opencv-python')
os.system('pip install shapely')
os.system('pip install pygifsicle')
os.system('pip install imageio')
os.system('pip install matplotlib')
os.system('pip install numpy')

# import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio

import cv2
import time
import itertools

from dataclasses import dataclass
from pygifsicle import optimize
from IPython.display import clear_output
from copy import deepcopy
from math import floor


img_height = 400
img_width = 600

box_color = (255,255,255)
box_width = 80

gap = 10
n_boxes = 6
left_over = img_width - (n_boxes * box_width + (n_boxes - 1) * gap)
assert left_over >= 0, 'Layout exceeds background width'

purple = (floor(.698*255), floor(.345*255), floor(.584*255))
red = (floor(.2*255), floor(.2*255), floor(.804*255))

@dataclass
class Box():
    _width : int
    _center : tuple
    _number : int 
    _color : tuple
    thickness = 2
    font_scale = 1
    temp_color = None
#     color_gen = deepcopy(itertools.cycle([(0,0,255),(0,0,155),(0,0,55),None]))
    
    @property
    def points(self):
        xc, yc = self._center
        width = self._width
        start_point = ((xc - width // 2), (yc - width // 2))
        end_point = ((xc + width // 2), (yc + width // 2))
        return start_point, end_point
    
    @property
    def center(self):
        return self._center
    
    @property
    def number(self):
        return self._number
    
    @property
    def color(self):
        self.original_color = self.color
        return self._color
    
#     def revert_color(self):
#         self.color = self.original_color
#         return
    
    @center.setter
    def center(self, center : tuple):
        if not isinstance(center, tuple): raise TypeError(f'Center must be a tuple not {type(center)}')
        if not len(center) == 2: raise ValueError('Center must length 2')
        self._center = center
        return 
    
    @number.setter
    def number(self, number : int):
        if not isinstance(number, (int, np.int32, np.int64)): raise TypeError(f'Number must be an integer, not {type(number)}')
        self._number = number
        self.temp_color = (0,0,251)
        return
    
    @color.setter
    def color(self, color : tuple):
        if not isinstance(color, tuple): raise TypeError(f'Color must be a tuple, not {type(number)}')
        if not len(color) == 3: raise ValueError(f'Color must length 3, not {len(color)}')
        for item in color: assert isinstance(item, (int, np.int32, np.int64, np.uint8))
        self._color = color
        return
            
    def move_down(self, gap=10):
#         curr_center = self.center
#         self.center = (curr_center[0], curr_center[1] + gap + self._width)
        self.offset(y = gap + self._width)
        return
    
    def move_up(self, gap=10):
#         curr_center = self.center
#         self.center = (curr_center[0], curr_center[1] + gap + self._width)
        self.offset(y = -(gap + self._width))
        return
    
    def merge(self, box, xcmin=0, gap=10):
        self.move_up(gap=gap)
        box.move_up(gap=gap)
        
    def offset(self, x=0, y=0):
        self.center[0] += x
        self.center[1] += y
        return
    
    def align_horizontal(self, boxes, separation=None, xcmin=0, yc=0, gap=10):
        separation = boxes[0].width // 2 if separation is None else separation
        xc = -gap - separation // 2 + xcmin
        for box in boxes:
            xc += gap + separation
            box.center = (xc, yc)
        return
            
    def align_left(self, boxes, separation=None, xcmin=0, yc=0, gap=10):
        self.align_horizontal(boxes, separation=separation, xcmin=xcmin, yc=yc, gap=gap)
        return

    def align_right(self, boxes, separation=None, xcmax=0, yc=0, gap=10):
        separation = -boxes[0].width // 2 if separation is None else -separation
        #### this is probably wrong
        self.align_horizontal(boxes[::-1], separation=-separation, xcmin=xcmax + 2*gap + 2*separation, yc=yc, gap=gap)
        return
    
    def swap(self, box):
        self.center, box.center = box.center, self.center 
        return
        
    def graph(self, img):
        start_point, end_point = self.points
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.temp_color:
            cv2.rectangle(img, start_point, end_point, self.temp_color, -1)
#             self.temp_color = next(self.color_gen)
            if self.temp_color == (0,0,251): self.temp_color = (0,0,250)        
            elif self.temp_color == (0,0,250): self.temp_color = (0,0,175)
            elif self.temp_color == (0,0,175): self.temp_color = (0,0,100)
            elif self.temp_color == (0,0,100): self.temp_color = None
        else:
            cv2.rectangle(img, start_point, end_point, self._color, self.thickness)
        if isinstance(self._number, (str, int, np.int32, np.int64)):
            text = str(self._number)
            text_size = cv2.getTextSize(text, font, self.font_scale, self.thickness)[0]
            text_x = self.center[0] - text_size[0] // 2
            text_y = self.center[1] + text_size[1] // 2
            cv2.putText(img, text, (text_x, text_y), font, self.font_scale, self._color, self.thickness, cv2.LINE_AA)
        return img

def partition_gen(arr, left, right):
    x = arr[right]
    i = left
    indices = list()
    for j in range(right, left, -1):
        if arr[j] <= x:
            indices.append((i, j))
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
#             yield i - 1, j
    return indices

k = 3
show = True
animate = True
gif_path = 'quick_select.gif'
if not show:
    imgs = list()

img = np.zeros((img_height, img_width, 3), dtype='uint8')
# img = cv2.rectangle(img, start_point, end_point, color, thickness)
# nums = np.random.randint(100, size=n_boxes)
nums = [10, 13, 4, 3 ,1, 23]
boxes = list()
xc = -gap - box_width // 2 + left_over // 2
for num in nums:
    yc = img_height // 2
    xc += gap + box_width
    box = Box(box_width, (xc, yc), num, box_color)
    boxes.append(box)
    img = box.graph(img)

if show:
    plt.imshow(img[:,:,::-1])    
    plt.show()
else:
    imgs.append(img)
    
def graph():
    img = np.zeros((img_height, img_width, 3), dtype='uint8')
    for box in boxes:
        img = box.graph(img)
    print()
    if show:
        # if animate:clear_output(wait=True) # not available w/o notebook
        plt.imshow(img[:,:,::-1])    
        plt.show()
    else:
        imgs.append(img)
        
# for i, j in [(1,2),(0,3),(0,4)]:
for i, j in partition_gen(nums, 0, k):
    print(i, j)
#     color1, color2 = boxes[i].color, boxes[j].color
#     boxes[i].color, boxes[j].color = red, red
#     graph()
#     boxes[i].color, boxes[j].color = color1, color2
    boxes[i].number, boxes[j].number = boxes[i].number, boxes[j].number #this doesn't change the number, but does call the coloring
    graph()
#     boxes[i].number, boxes[j].number = boxes[j].number, boxes[i].number
    boxes[i].swap(boxes[j]) #swap is symmetric
    for _ in range(5): # these cycle colors, no number change
        if animate and show: time.sleep(1)
        graph()

if not show:
    frames = np.stack(imgs, axis=0)
    iio.imwrite(gif_path, frames)
    optimize(gif_path) # For overwriting the original one