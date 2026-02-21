import pygame
import sys
from game import Game
from button import Button

class Settings():
    def __init__(self):
        self.display_surface = pygame.display.get_surface()
        # self.color = color
        # self.komi = komi
        # self.board_size = board_size

        model_path = "model/best_model.pth"
        self.game = Game(model_path)

    def run(self):
        bg = pygame.image.load("assets/settingsBG.png").convert()
        playImage = pygame.image.load("assets/Start_Hovered.png")
        playImage = pygame.transform.scale(playImage, (330,40))
        playImage2 = pygame.image.load("assets/Start_Unhovered.png")
        playImage2 = pygame.transform.scale(playImage2, (330,40))

        unhovered = pygame.image.load("assets/SettingsButton.png")
        unhovered = pygame.transform.scale(unhovered, (100,100))
        hovered = pygame.image.load("assets/SettingsButtonPressed.png")
        hovered = pygame.transform.scale(hovered, (100,100))


        komi1 = Button(hovered=hovered, unhovered=unhovered, pos=(220,450))
        komi2 = Button(hovered=hovered, unhovered=unhovered, pos=(420,450))
        komi3 = Button(hovered=hovered, unhovered=unhovered, pos=(620,450))

        board1 = Button(hovered=hovered, unhovered=unhovered, pos=(220,650))
        board2 = Button(hovered=hovered, unhovered=unhovered, pos=(420,650))
        board3 = Button(hovered=hovered, unhovered=unhovered, pos=(620,650))

        

        while True:
            self.display_surface.blit(bg, (0, 0))
            MENU_MOUSE_POS = pygame.mouse.get_pos()
            PLAY_BUTTON = Button(hovered=playImage, unhovered=playImage2 , pos=(470, 800))
            
            

            # SETTINGS_BUTTON = Button(hovered=settingImage, unhovered=settingImage2, pos=(470, 745))
            for button in [PLAY_BUTTON, komi1, komi2, komi3, board1, board2, board3]:
                button.changeHover(MENU_MOUSE_POS)
                button.update(self.display_surface)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if PLAY_BUTTON.checkForInput(MENU_MOUSE_POS):
                        self.game.run()
                    if komi1.checkForInput(MENU_MOUSE_POS):
                        komi2.unPress()
                        komi3.unPress()
                        komi1.setPressed()
                        
                    if komi2.checkForInput(MENU_MOUSE_POS):
                       komi1.unPress()
                       komi3.unPress()
                       komi2.setPressed()
                        
                    if komi3.checkForInput(MENU_MOUSE_POS):
                        komi1.unPress()
                        komi2.unPress()
                        komi3.setPressed()
                    
                    if board1.checkForInput(MENU_MOUSE_POS):
                        board2.unPress()
                        board3.unPress()
                        board1.setPressed()
                        
                    if board2.checkForInput(MENU_MOUSE_POS):
                       board1.unPress()
                       board3.unPress()
                       board2.setPressed()
                        
                    if board3.checkForInput(MENU_MOUSE_POS):
                        board1.unPress()
                        board2.unPress()
                        board3.setPressed()
                    
                    
            

            pygame.display.update()