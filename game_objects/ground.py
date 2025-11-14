import pygame
import os

floor_sprite = pygame.transform.scale2x(pygame.image.load(os.path.join("game_objects/assets","floor.png")).convert_alpha())

class ScrollingFloor:
    SCROLL_SPEED = 5
    TEXTURE_WIDTH = floor_sprite.get_width()
    TEXTURE = floor_sprite  
    
    def __init__(self, ground_level_y):
        self.y_position = ground_level_y    
        self.segment_1_x = 0
        self.segment_2_x = self.TEXTURE_WIDTH
        
    def update_position(self):
        self.segment_1_x -= self.SCROLL_SPEED
        self.segment_2_x -= self.SCROLL_SPEED
        
        if self.segment_1_x + self.TEXTURE_WIDTH < 0:
            self.segment_1_x = self.segment_2_x + self.TEXTURE_WIDTH
        if self.segment_2_x + self.TEXTURE_WIDTH < 0:
            self.segment_2_x = self.segment_1_x + self.TEXTURE_WIDTH
    
    def render_to_surface(self, surface):
        surface.blit(self.TEXTURE, (self.segment_1_x, self.y_position))
        surface.blit(self.TEXTURE, (self.segment_2_x, self.y_position))