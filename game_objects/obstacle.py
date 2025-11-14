import pygame
import os
import random

tube_sprite = pygame.transform.scale2x(pygame.image.load(os.path.join("game_objects/assets","tube.png")).convert_alpha())

class Pipe:
    SCROLL_VELOCITY = 5
    
    def __init__(self, start_x):
        self.pos_x = start_x   
        self.opening_height = 0
        self.gap_size = random.randint(135, 200)
        self.top_y = 0
        self.bottom_y = 0
        self.TUBE_INVERTED = pygame.transform.flip(tube_sprite, False, True)
        self.TUBE_UPRIGHT = tube_sprite
        self.is_passed = False
        self.randomize_height()
    
    def randomize_height(self):
        self.opening_height = random.randrange(50, 450)
        self.top_y = self.opening_height - self.TUBE_INVERTED.get_height()
        self.bottom_y = self.opening_height + self.gap_size
    
    def update_position(self):
        self.pos_x -= self.SCROLL_VELOCITY
        
    def render_tubes(self, surface):
        surface.blit(self.TUBE_INVERTED, (self.pos_x, self.top_y))
        surface.blit(self.TUBE_UPRIGHT, (self.pos_x, self.bottom_y))
        
    def check_crash(self, avatar):
        avatar_mask = avatar.get_pixel_mask()
        top_tube_mask = pygame.mask.from_surface(self.TUBE_INVERTED)
        bottom_tube_mask = pygame.mask.from_surface(self.TUBE_UPRIGHT)
        
        offset_top = (self.pos_x - avatar.pos_x, self.top_y - round(avatar.pos_y))
        offset_bottom = (self.pos_x - avatar.pos_x, self.bottom_y - round(avatar.pos_y))
        
        collision_bottom = avatar_mask.overlap(bottom_tube_mask, offset_bottom)
        collision_top = avatar_mask.overlap(top_tube_mask, offset_top)
        
        if collision_bottom or collision_top:
            return True

        return False