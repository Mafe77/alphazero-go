import sys
import pygame
from consts import WIDTH, HEIGHT
from game import Game
from button import Button

class Screen:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("AlphaGo PY")
        self.clock = pygame.time.Clock()  
        self.game = Game()  
    
    def get_font(self, size): # Returns Press-Start-2P in the desired size
        return pygame.font.Font("assets/font.ttf", size)

    def main_menu(self):
        bg = pygame.image.load("assets/goBG.png")
        playImage = pygame.image.load("assets/Start_Hovered.png")
        playImage = pygame.transform.scale(playImage, (330,40))
        playImage2 = pygame.image.load("assets/Start_Unhovered.png")
        playImage2 = pygame.transform.scale(playImage2, (330,40))

        settingImage = pygame.image.load("assets/Setting_Hovered.png")
        settingImage = pygame.transform.scale(settingImage, (330,40))
        settingImage2 = pygame.image.load("assets/Setting_Unhovered.png")
        settingImage2 = pygame.transform.scale(settingImage2, (330,40))
        
        while True:
            self.screen.blit(bg, (0, 0))
            
            MENU_MOUSE_POS = pygame.mouse.get_pos()
            PLAY_BUTTON = Button(hovered=playImage, unhovered=playImage2 , pos=(470, 700))
            SETTINGS_BUTTON = Button(hovered=settingImage, unhovered=settingImage2, pos=(470, 745))
            for button in [PLAY_BUTTON, SETTINGS_BUTTON]:
                button.changeHover(MENU_MOUSE_POS)
                button.update(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                        self.game.run()               


            pygame.display.update()
