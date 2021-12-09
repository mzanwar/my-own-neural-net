import pickle

import imageio
import numpy
import pygame.transform
import scipy.misc
import pygame as pg
from matplotlib import pyplot as plt

from main import NeuralNetwork

# pip install pygame==2.0.0.dev10

"""
With this program you can draw on the 
screen with pygame


pythonprogramming.altervista.org
"""

def init():
    global screen

    pg.init()
    screen = pg.display.set_mode((28, 28))
    mainloop()


drawing = False
last_pos = None
w = 1
color = (255, 255, 255)


def draw(event):
    global drawing, last_pos, w

    if event.type == pg.MOUSEMOTION:
        if (drawing):
            mouse_position = pg.mouse.get_pos()
            if last_pos is not None:
                pg.draw.line(screen, color, last_pos, mouse_position, w)
            last_pos = mouse_position
    elif event.type == pg.MOUSEBUTTONUP:
        mouse_position = (0, 0)
        drawing = False
        last_pos = None
    elif event.type == pg.MOUSEBUTTONDOWN:
        drawing = True


def mainloop():
    global screen

    with open("./trained_anns/ann-i784-h140-lr0.11-epochs5_0.97", 'rb') as pickle_file:
        ann = pickle.load(file=pickle_file)

    loop = 1
    while loop:
        # checks every user interaction in this list
        for event in pg.event.get():
            if event.type == pg.QUIT:
                loop = 0
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_s:
                    pg.image.save(screen, "image.png")
                if event.key == pg.K_c:
                    screen.fill((0,0,0))
                    pygame.display.update()
                if event.key == pg.K_n:
                    pg.image.save(screen, "image.png")
                    img_array = imageio.imread("image.png", as_gray=True)
                    img_array = numpy.resize(img_array, (28, 28, 1))
                    img_data = img_array.reshape(784)
                    img_data = (img_data / 255.0 * 0.99) + 0.01
                    outputs = ann.query(img_data)
                    label = numpy.argmax(outputs)
                    print(label)
            draw(event)
        pg.display.flip()
    pg.quit()

init()