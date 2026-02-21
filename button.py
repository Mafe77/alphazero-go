import pygame

class Button():
	def __init__(self, hovered, unhovered, pos, text_input=None, font=None):
		self.hovered, self.unhovered = hovered, unhovered
		self.font = font
		self.text_input = text_input
		self.x_pos = pos[0]
		self.y_pos = pos[1]
		if font is not None:
			self.text = self.font.render(self.text_input, True, "Black")
			self.text_rect = self.text.get_rect(center=(self.x_pos, self.y_pos-10))
		self.image = self.unhovered
		self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
		self.pressed = False
		

	def update(self, screen):
		screen.blit(self.image, self.rect)

		if self.text_input is not None:
			screen.blit(self.text, self.text_rect)

	def setPressed(self):
		self.pressed = True
		self.image = self.hovered
	
	def unPress(self):
		self.pressed = False
		self.image = self.unhovered

	def getPressed(self):
		return self.pressed		

	def checkForInput(self, position):
		if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
			return True
		return False

	def changeHover(self, position):
		if not self.pressed:
			if position[0] in range(self.rect.left, self.rect.right) and position[1] in range(self.rect.top, self.rect.bottom):
				self.image = self.hovered
				if self.text_input is not None:
					self.text_rect = self.text.get_rect(center=(self.x_pos, self.y_pos+3))
			else:
				self.image = self.unhovered
				if self.text_input is not None:
					self.text_rect = self.text.get_rect(center=(self.x_pos, self.y_pos-10))
		else:
			self.image = self.hovered

