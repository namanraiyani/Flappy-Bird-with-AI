import pygame
import os

animation_frames = [pygame.transform.scale(pygame.image.load(os.path.join("game_objects/assets", "flapper" + str(i) + ".png")), (40,30)) for i in range(1,4)]

class FlapBird:
    MAX_UPWARD_TILT = 25
    FRAMES = animation_frames
    ROTATION_SPEED = 20
    FRAME_DURATION = 5
    
    def __init__(self, start_x, start_y):
        self.pos_x = start_x        
        self.pos_y = start_y        
        self.rotation_angle = 0
        self.air_time = 0
        self.vertical_velocity = 0
        self.launch_height = self.pos_y
        self.frame_tracker = 0
        self.current_sprite = self.FRAMES[0]
    
    def perform_flap(self):
        self.vertical_velocity = -10.5
        self.air_time = 0
        self.launch_height = self.pos_y
        
    def update_physics(self):
        self.air_time += 1
        
        vertical_delta = (
            self.vertical_velocity * self.air_time +
            0.5 * 3 * (self.air_time**2)
        )
        
        if vertical_delta >= 16:
            vertical_delta = 16
        
        self.pos_y = self.pos_y + vertical_delta
        
        if vertical_delta < 0 or self.pos_y < self.launch_height + 50:
            if self.rotation_angle < self.MAX_UPWARD_TILT:
                self.rotation_angle = self.MAX_UPWARD_TILT   
        elif self.rotation_angle > -90:
            self.rotation_angle -= self.ROTATION_SPEED
            
    def render_avatar(self, surface):
        self.frame_tracker += 1
        
        if self.frame_tracker <= self.FRAME_DURATION:
            self.current_sprite = self.FRAMES[0]
        elif self.frame_tracker <= self.FRAME_DURATION*2:
            self.current_sprite = self.FRAMES[1]
        elif self.frame_tracker <= self.FRAME_DURATION*3:
            self.current_sprite = self.FRAMES[2]
        elif self.frame_tracker <= self.FRAME_DURATION*4:
            self.current_sprite = self.FRAMES[1]
        elif self.frame_tracker == self.FRAME_DURATION*4 + 1:
            self.current_sprite = self.FRAMES[0]
            self.frame_tracker = 0
        
        if self.rotation_angle <= -80:
            self.current_sprite = self.FRAMES[1]
            self.frame_tracker = self.FRAME_DURATION*2
        
        rotate_and_blit(surface, self.current_sprite, (self.pos_x, self.pos_y), self.rotation_angle)
        
    def get_pixel_mask(self):
        return pygame.mask.from_surface(self.current_sprite)

def rotate_and_blit(target_surface, image, top_left_coords, angle_degrees):
    rotated_img = pygame.transform.rotate(image, angle_degrees)
    bounding_rect = rotated_img.get_rect(center = image.get_rect(topleft = top_left_coords).center)
    target_surface.blit(rotated_img, bounding_rect.topleft)