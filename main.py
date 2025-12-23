import pygame
import torch
from consts import WIDTH, HEIGHT, BOARD_SIZE
from screens import Screen

def main():
    game = Screen()
    game.main_menu()


if __name__ == "__main__":
    main()