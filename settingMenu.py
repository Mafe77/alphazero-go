import pygame
import sys
from game import Game
from button import Button

class Settings():
    def __init__(self):
        self.display_surface = pygame.display.get_surface()
        # self.color = color
        self.komi = 6.5
        # self.board_size = 19
        self.handicap = 0

        self.model_path = "model/best_model.pth"
        # self.game = Game(model_path, self.board_size)

    def select_button(self, buttons, selected):
        for button in buttons:
            button.unPress()        
        selected.setPressed()
    
    def get_font(self, size):
        return pygame.font.Font("assets/font.ttf", size)

    def run(self):
        bg = pygame.image.load("assets/settings.png").convert()
        playImage = pygame.image.load("assets/Start_Hovered.png")
        playImage = pygame.transform.scale(playImage, (330,40))
        playImage2 = pygame.image.load("assets/Start_Unhovered.png")
        playImage2 = pygame.transform.scale(playImage2, (330,40))

        unhovered = pygame.image.load("assets/SettingsButton.png")
        unhovered = pygame.transform.scale(unhovered, (100,100))
        hovered = pygame.image.load("assets/SettingsButtonPressed.png")
        hovered = pygame.transform.scale(hovered, (100,100))


        komi_buttons = [
            Button(hovered=hovered, unhovered=unhovered, pos=(220, 450), 
            text_input="0", font=self.get_font(45)),
            Button(hovered=hovered, unhovered=unhovered, pos=(420, 450),
            text_input="6.5", font=self.get_font(20)),
            Button(hovered=hovered, unhovered=unhovered, pos=(620, 450),
            text_input="7.5", font=self.get_font(20)),
        ]
        komi_buttons[2].setPressed()
        komi_buttons[2].update(self.display_surface)

        handicap_buttons = [
            Button(hovered=hovered, unhovered=unhovered, pos=(220, 650),
            text_input="0", font=self.get_font(40)),
            Button(hovered=hovered, unhovered=unhovered, pos=(420, 650),
            text_input="2", font=self.get_font(40)),
            Button(hovered=hovered, unhovered=unhovered, pos=(620, 650),
            text_input="6", font=self.get_font(40)),
        ]
        handicap_buttons[0].setPressed()
        handicap_buttons[0].update(self.display_surface)

        other = [
            Button(hovered=playImage, unhovered=playImage2 , pos=(420, 800)),
        ]
        

        while True:
            self.display_surface.blit(bg, (0, 0))
            MENU_MOUSE_POS = pygame.mouse.get_pos()

            for group in [komi_buttons, handicap_buttons, other]:
                for button in group:
                    button.changeHover(MENU_MOUSE_POS)
                    button.update(self.display_surface)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if other[0].checkForInput(MENU_MOUSE_POS):
                        game = Game(self.model_path, self.handicap, self.komi)
                        game.run()
                   
                    for button in handicap_buttons:
                        if button.checkForInput(MENU_MOUSE_POS):
                            self.select_button(handicap_buttons, button)
                            # print(button.getValue())
                            self.handicap = int(button.getValue())

                    for button in komi_buttons:
                        if button.checkForInput(MENU_MOUSE_POS):
                            self.select_button(komi_buttons, button)
                            # print(button.getValue())
                            self.komi = float(button.getValue())                  
                    
            
            pygame.display.update()