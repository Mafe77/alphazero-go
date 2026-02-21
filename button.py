import pygame

class Button():
	def __init__(self, hovered, unhovered, pos):
		self.hovered, self.unhovered = hovered, unhovered
		self.x_pos = pos[0]
		self.y_pos = pos[1]
		self.image = self.unhovered
		self.rect = self.image.get_rect(center=(self.x_pos, self.y_pos))
		self.pressed = False
		

	def update(self, screen):
		screen.blit(self.image, self.rect)

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
			else:
				self.image = self.unhovered
		else:
			self.image = self.hovered

