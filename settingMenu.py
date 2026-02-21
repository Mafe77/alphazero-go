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

    def select_button(self, buttons, selected):
        for button in buttons:
            button.unPress()
        
        selected.setPressed()

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


        komi_buttons = [
            Button(hovered=hovered, unhovered=unhovered, pos=(220, 450)),
            Button(hovered=hovered, unhovered=unhovered, pos=(420, 450)),
            Button(hovered=hovered, unhovered=unhovered, pos=(620, 450)),
        ]

        board_buttons = [
            Button(hovered=hovered, unhovered=unhovered, pos=(220, 650)),
            Button(hovered=hovered, unhovered=unhovered, pos=(420, 650)),
            Button(hovered=hovered, unhovered=unhovered, pos=(620, 650)),
        ]

        other = [
            Button(hovered=playImage, unhovered=playImage2 , pos=(470, 800)),
        ]
        

        while True:
            self.display_surface.blit(bg, (0, 0))
            MENU_MOUSE_POS = pygame.mouse.get_pos()

            for group in [komi_buttons, board_buttons, other]:
                for button in group:
                    button.changeHover(MENU_MOUSE_POS)
                    button.update(self.display_surface)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if other[0].checkForInput(MENU_MOUSE_POS):
                        self.game.run()
                    for group in [komi_buttons, board_buttons]:
                        for button in group:
                            if button.checkForInput(MENU_MOUSE_POS):
                                self.select_button(group, button)                  
                    
            
            pygame.display.update()