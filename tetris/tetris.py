#!/usr/bin/env python3

# File: tetris.py 
# Description: Main file with tetris game.
# Author: Pavel Benáček <pavel.benacek@gmail.com>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import time
from collections import defaultdict
import pygame
import random
import math
import block
import constants
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from constants import BlockType

class Tetris(object):
    """
    The class with implementation of tetris game logic.
    """

    def __init__(self, bx, by):
        """
        Initialize the tetris object.

        Parameters:
            - bx - number of blocks in x
            - by - number of blocks in y
        """
        self.bx = bx
        self.by = by
        # Compute the resolution of the play board based on the required number of blocks.
        self.resx = bx*constants.BWIDTH+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
        self.resy = by*constants.BHEIGHT+2*constants.BOARD_HEIGHT+constants.BOARD_MARGIN
        # Prepare the pygame board objects (white lines)
        self.board_up    = pygame.Rect(0,constants.BOARD_UP_MARGIN,self.resx,constants.BOARD_HEIGHT)
        self.board_down  = pygame.Rect(0,self.resy-constants.BOARD_HEIGHT,self.resx,constants.BOARD_HEIGHT)
        self.board_left  = pygame.Rect(0,constants.BOARD_UP_MARGIN,constants.BOARD_HEIGHT,self.resy)
        self.board_right = pygame.Rect(self.resx-constants.BOARD_HEIGHT,constants.BOARD_UP_MARGIN,constants.BOARD_HEIGHT,self.resy)
        # List of used blocks
        self.blk_list    = []
        # Compute start indexes for tetris blocks
        self.start_x = math.ceil(self.resx/2.0)
        self.start_y = constants.BOARD_UP_MARGIN + constants.BOARD_HEIGHT + constants.BOARD_MARGIN
        # Block data (shapes and colors). The shape is encoded in the list of [X,Y] points. Each point
        # represents the relative position. The true/false value is used for the configuration of rotation where
        # False means no rotate and True allows the rotation.
        self.block_data = (
            ([[0,0],[1,0],[2,0],[3,0]],constants.RED,True, BlockType.I_BLOCK),     # I block
            ([[0,0],[1,0],[0,1],[-1,1]],constants.GREEN,True, BlockType.S_BLOCK),  # S block
            ([[0,0],[1,0],[2,0],[2,1]],constants.BLUE,True, BlockType.L_BLOCK),    # L block
            ([[0,0],[0,1],[1,0],[1,1]],constants.ORANGE,False, BlockType.O_BLOCK), # O block
            ([[-1,0],[0,0],[0,1],[1,1]],constants.GOLD,True, BlockType.Z_BLOCK),   # Z block
            ([[0,0],[1,0],[2,0],[1,1]],constants.PURPLE,True, BlockType.T_BLOCK),  # T block
            ([[0,0],[1,0],[2,0],[0,1]],constants.CYAN,True, BlockType.J_BLOCK),    # J block
        )
        # Compute the number of blocks. When the number of blocks is even, we can use it directly but 
        # we have to decrese the number of blocks in line by one when the number is odd (because of the used margin).
        self.blocks_in_line = bx if bx%2 == 0 else bx-1
        self.blocks_in_pile = by
        # Score settings
        self.score = 0
        # Remember the current speed 
        self.speed = 1
        # The score level threshold
        self.score_level = constants.SCORE_LEVEL
        self.lines_cleared = 0
        self.x_positions = list(range(self.start_x % constants.BWIDTH, self.resx - self.start_x % constants.BWIDTH, constants.BWIDTH))
        self.y_positions = list(range(self.start_y % constants.BHEIGHT, self.resy - self.start_y % constants.BHEIGHT, constants.BHEIGHT))
        self.possible_block_states = defaultdict(list)


    def apply_action(self):
        """
        Get the event from the event queue and run the appropriate 
        action.
        """
        # Take the event from the event queue.
        for ev in pygame.event.get():
            # Check if the close button was fired.
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.unicode == 'q'):
                self.done = True
            # Detect the key evevents for game control.
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_DOWN:
                    self.active_block.move(0,constants.BHEIGHT)
                if ev.key == pygame.K_LEFT:
                    self.active_block.move(-constants.BWIDTH,0)
                if ev.key == pygame.K_RIGHT:
                    self.active_block.move(constants.BWIDTH,0)
                if ev.key == pygame.K_SPACE:
                    self.active_block.rotate()
                if ev.key == pygame.K_p:
                    self.pause()
                if ev.key == pygame.K_d:
                    self.drop_active_block()

            # Detect if the movement event was fired by the timer.
            if ev.type == constants.TIMER_MOVE_EVENT:
                self.active_block.move(0,constants.BHEIGHT)
       
    def pause(self):
        """
        Pause the game and draw the string. This function
        also calls the flip function which draws the string on the screen.
        """
        # Draw the string to the center of the screen.
        self.print_center(["PAUSE","Press \"p\" to continue"])
        pygame.display.flip()
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_p:
                    return
       
    def set_move_timer(self):
        """
        Setup the move timer to the 
        """
        # Setup the time to fire the move event. Minimal allowed value is 1
        speed = math.floor(constants.MOVE_TICK / self.speed)
        speed = max(1,speed)
        pygame.time.set_timer(constants.TIMER_MOVE_EVENT,speed)
 
    def run(self):
        # Initialize the game (pygame, fonts)
        pygame.init()
        pygame.font.init()
        self.myfont = pygame.font.SysFont(pygame.font.get_default_font(),constants.FONT_SIZE)
        self.screen = pygame.display.set_mode((self.resx,self.resy))
        pygame.display.set_caption("Tetris")
        # Setup the time to fire the move event every given time
        self.set_move_timer()
        # Control variables for the game. The done signal is used 
        # to control the main loop (it is set by the quit action), the game_over signal
        # is set by the game logic and it is also used for the detection of "game over" drawing.
        # Finally the new_block variable is used for the requesting of new tetris block. 
        self.done = False
        self.game_over = False
        self.new_block = True
        # Print the initial score
        self.print_status_line()
        while not(self.done) and not(self.game_over):
            # Get the block and run the game logic
            self.get_block()
            self.game_logic()
            self.draw_game()
        # Display the game_over and wait for a keypress
        if self.game_over:
            self.print_game_over()
        # Disable the pygame stuff
        pygame.font.quit()
        pygame.display.quit()

   
    def print_status_line(self):
        """
        Print the current state line
        """
        string = ["SCORE: {0}   SPEED: {1}x".format(self.score,self.speed)]
        self.print_text(string,constants.POINT_MARGIN,constants.POINT_MARGIN)        

    def print_game_over(self):
        """
        Print the game over string.
        """
        # Print the game over text
        self.print_center(["Game Over","Press \"q\" to exit"])
        # Draw the string
        pygame.display.flip()
        # Wait untill the space is pressed
        while True: 
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.unicode == 'q'):
                    return

    def print_text(self,str_lst,x,y):
        """
        Print the text on the X,Y coordinates. 

        Parameters:
            - str_lst - list of strings to print. Each string is printed on new line.
            - x - X coordinate of the first string
            - y - Y coordinate of the first string
        """
        prev_y = 0
        for string in str_lst:
            size_x,size_y = self.myfont.size(string)
            txt_surf = self.myfont.render(string,False,(255,255,255))
            self.screen.blit(txt_surf,(x,y+prev_y))
            prev_y += size_y 

    def print_center(self,str_list):
        """
        Print the string in the center of the screen.
        
        Parameters:
            - str_lst - list of strings to print. Each string is printed on new line.
        """
        max_xsize = max([tmp[0] for tmp in map(self.myfont.size,str_list)])
        self.print_text(str_list,self.resx/2-max_xsize/2,self.resy/2)

    def block_colides(self):
        """
        Check if the block colides with any other block.

        The function returns True if the collision is detected.
        """
        for blk in self.blk_list:
            # Check if the block is not the same
            if blk == self.active_block:
                continue 
            # Detect situations
            if(blk.check_collision(self.active_block.shape)):
                return True
        return False

    def game_logic(self):
        """
        Implementation of the main game logic. This function detects colisions
        and insertion of new tetris blocks.
        """
        # Remember the current configuration and try to 
        # apply the action
        self.active_block.backup()
        self.apply_action()
        # Border logic, check if we colide with down border or any
        # other border. This check also includes the detection with other tetris blocks. 
        down_board  = self.active_block.check_collision([self.board_down])
        any_border  = self.active_block.check_collision([self.board_left,self.board_up,self.board_right])
        block_any   = self.block_colides()
        # Restore the configuration if any collision was detected
        if down_board or any_border or block_any:
            self.active_block.restore()
        # So far so good, sample the previous state and try to move down (to detect the colision with other block). 
        # After that, detect the the insertion of new block. The block new block is inserted if we reached the boarder
        # or we cannot move down.
        self.active_block.backup()
        self.active_block.move(0,constants.BHEIGHT)
        can_move_down = not self.block_colides()  
        self.active_block.restore()
        # We end the game if we are on the respawn and we cannot move --> bang!
        if not can_move_down and (self.start_x == self.active_block.x and self.start_y == self.active_block.y):
            self.game_over = True
        # The new block is inserted if we reached down board or we cannot move down.
        if down_board or not can_move_down:     
            # Request new block
            self.new_block = True
            # Detect the filled line and possibly remove the line from the 
            # screen.
            self.detect_line()   
 
    def detect_line(self):
        """
        Detect if the line is filled. If yes, remove the line and
        move with remaining bulding blocks to new positions.
        """
        # Get each shape block of the non-moving tetris block and try
        # to detect the filled line. The number of bulding blocks is passed to the class
        # in the init function.
        for shape_block in self.active_block.shape:
            tmp_y = shape_block.y
            tmp_cnt = self.get_blocks_in_line(tmp_y)
            # Detect if the line contains the given number of blocks
            if tmp_cnt != self.blocks_in_line:
                continue 
            # Ok, the full line is detected!     
            self.remove_line(tmp_y)
            # Update the score.
            self.score += self.blocks_in_line * constants.POINT_VALUE 
            # Check if we need to speed up the game. If yes, change control variables
            if self.score > self.score_level:
                self.score_level *= constants.SCORE_LEVEL_RATIO
                self.speed       *= constants.GAME_SPEEDUP_RATIO
                # Change the game speed
                self.set_move_timer()

    def remove_line(self,y):
        """
        Remove the line with given Y coordinates. Blocks below the filled
        line are untouched. The rest of blocks (yi > y) are moved one level done.

        Parameters:
            - y - Y coordinate to remove.
        """ 
        # Iterate over all blocks in the list and remove blocks with the Y coordinate.
        for block in self.blk_list:
            block.remove_blocks(y)
        # Setup new block list (not needed blocks are removed)
        self.blk_list = [blk for blk in self.blk_list if blk.has_blocks()]

    def get_blocks_in_line(self,y):
        """
        Get the number of shape blocks on the Y coordinate.

        Parameters:
            - y - Y coordinate to scan.
        """
        # Iteraveovel all block's shape list and increment the counter
        # if the shape block equals to the Y coordinate.
        tmp_cnt = 0
        for block in self.blk_list:
            for shape_block in block.shape:
                tmp_cnt += (1 if y == shape_block.y else 0)            
        return tmp_cnt

    def draw_board(self, draw_status=True):
        """
        Draw the white board.
        """
        pygame.draw.rect(self.screen,constants.WHITE,self.board_up)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_down)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_left)
        pygame.draw.rect(self.screen,constants.WHITE,self.board_right)
        # Update the score
        if draw_status:
            self.print_status_line()

    def get_block(self):
        """
        Generate new block into the game if is required.
        """
        if self.new_block:
            # Get the block and add it into the block list(static for now)
            tmp = random.randint(0,len(self.block_data)-1)
            data = self.block_data[tmp]
            self.active_block = block.Block(data[0],self.start_x,self.start_y,self.screen, data[1], data[2], data[3])
            self.blk_list.append(self.active_block)
            self.new_block = False

    def draw_game(self, draw_status=True):
        """
        Draw the game screen.
        """
        # Clean the screen, draw the board and draw
        # all tetris blocks
        self.screen.fill(constants.BLACK)
        self.draw_board(draw_status=draw_status)
        for blk in self.blk_list:
            blk.draw()
        # Draw the screen buffer
        pygame.display.flip()

    # ======== !!! Below starts logic added for emulating the game, used for RL !!! ========

    def draw_game_rl(self, episode, loss, episode_reward):
        """
        Draw the game screen.
        """
        # Clean the screen, draw the board and draw
        # all tetris blocks
        self.screen.fill(constants.BLACK)
        self.draw_board(draw_status=False)
        self.print_learning_status(episode=episode, loss=loss, episode_reward=episode_reward)
        for blk in self.blk_list:
            blk.draw()
        # Draw the screen buffer
        pygame.display.flip()

    def remove_lines_emulator(self):
        """
        Detect if the line is filled. If yes, remove the line and
        move with remaining bulding blocks to new positions.
        """
        # Get each shape block of the non-moving tetris block and try
        # to detect the filled line. The number of bulding blocks is passed to the class
        # in the init function.
        curr_lines_cleared = 0
        for shape_block in self.active_block.shape:
            tmp_y = shape_block.y
            tmp_cnt = self.get_blocks_in_line(tmp_y)
            # Detect if the line contains the given number of blocks
            if tmp_cnt != self.blocks_in_line:
                continue
            # Ok, the full line is detected!
            self.remove_line(tmp_y)
            # Update the score.
            self.score += self.blocks_in_line * constants.POINT_VALUE
            curr_lines_cleared += 1

        self.lines_cleared += curr_lines_cleared
        return curr_lines_cleared

    def get_potential_lines_cleared(self):
        lines_cleared = 0
        for shape_block in self.active_block.shape:
            tmp_y = shape_block.y
            tmp_cnt = self.get_blocks_in_line(tmp_y)
            # Detect if the line contains the given number of blocks
            if tmp_cnt != self.blocks_in_line:
                continue
            lines_cleared += 1

        return lines_cleared

    def create_possible_block_states(self):
        x_positions_set = set(self.x_positions)

        for blk in self.block_data:
            self.active_block = block.Block(blk[0],self.start_x,self.start_y,self.screen, blk[1], blk[2], blk[3])
            self.blk_list.append(self.active_block)

            x_offsets = (np.array(self.x_positions) - self.active_block.x).tolist()

            if self.active_block.type in [BlockType.O_BLOCK]:
                rotations = [0]
            elif self.active_block.type in [BlockType.I_BLOCK, BlockType.S_BLOCK, BlockType.Z_BLOCK]:
                rotations = [0, 90]
            else:
                rotations = [0, 90, 180, 270]


            for rotation in rotations:
                for x_idx, x in enumerate(self.x_positions):
                    backup_cfg = self.active_block.backup_config()

                    self.active_block.rotate_by(rotation)
                    self.active_block.move(x_offsets[x_idx], 0)
                    active_block_rects = set([el.x for el in self.active_block.shape])

                    # If the block collided after movement move on
                    if self.active_block.check_collision([self.board_left, self.board_right, self.board_down]) \
                            or len(active_block_rects - x_positions_set) > 0:
                        self.active_block.restore_config(*backup_cfg)
                        continue

                    self.drop_active_block()

                    # tore
                    self.possible_block_states[blk[3]].append((x, x_idx, rotation))

                    # Restore the original config
                    self.active_block.restore_config(*backup_cfg)

            self.blk_list.clear()

    def get_next_states(self):
        x_offsets = (np.array(self.x_positions) - self.active_block.x).tolist()
        state_action_pairs = {}

        for x, x_idx, rotation in self.possible_block_states[self.active_block.type]:
            # Backup the current store
            backup_cfg = self.active_block.backup_config()
            self.active_block.rotate_by(rotation)
            self.active_block.move(x_offsets[x_idx], 0)
            self.drop_active_block()

            # Get the state and store
            lines_cleared = self.get_potential_lines_cleared()
            state = self.get_game_state(lines_cleared, skip_active_block=False)
            state_action_pairs[(x, rotation)] = state
            # Restore the original config
            self.active_block.restore_config(*backup_cfg)

        return state_action_pairs

    def get_next_display_states(self, display_state, draw_states=False):
        x_offsets = (np.array(self.x_positions) - self.active_block.x).tolist()
        state_action_pairs = {}

        for x, x_idx, rotation in self.possible_block_states[self.active_block.type]:
            # Backup the current store
            backup_cfg = self.active_block.backup_config()
            self.active_block.rotate_by(rotation)
            self.active_block.move(x_offsets[x_idx], 0)
            self.drop_active_block()
            self.draw_game()

            # Get the state and store   
            state = self.get_display_state() - display_state
            state_action_pairs[(x, rotation)] = state

            if draw_states:
                plt.figure()
                plt.imshow(state.cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none', cmap='gray')
                plt.show()

            # Restore the original config
            self.active_block.restore_config(*backup_cfg)
            self.draw_game()

        return state_action_pairs

    def get_next_grid_states(self, grid_state):
        x_offsets = (np.array(self.x_positions) - self.active_block.x).tolist()
        state_action_pairs = {}

        for x, x_idx, rotation in self.possible_block_states[self.active_block.type]:
            # Backup the current store
            backup_cfg = self.active_block.backup_config()
            self.active_block.rotate_by(rotation)
            self.active_block.move(x_offsets[x_idx], 0)
            self.drop_active_block()

            # Get the state and store 
            new_state = self.get_game_grid_state(skip_active_block=False)
            state = new_state - grid_state
            state_action_pairs[(x, rotation)] = state

            # Restore the original config
            self.active_block.restore_config(*backup_cfg)

        return state_action_pairs

    def perform_action(self, x, rotation):
        x_offsets = (np.array(self.x_positions) - self.active_block.x).tolist()
        offset_index = self.x_positions.index(x)
        offset = x_offsets[offset_index]

        self.active_block.rotate_by(rotation)
        self.active_block.move(offset, 0)

    def drop_active_block(self, episode=None, loss=None, episode_reward=None, draw=False):
        while True:
            self.active_block.backup()
            self.active_block.move(0, constants.BHEIGHT)
            down_board = self.active_block.check_collision([self.board_down])
            block_any = self.block_colides()

            if down_board or block_any:
                self.active_block.restore()
                break
            if draw:
                time.sleep(0.000001)
                self.draw_game_rl(episode, loss, episode_reward)


    def check_collisions(self):
        down_board = self.active_block.check_collision([self.board_down])
        any_border = self.active_block.check_collision([self.board_left, self.board_up, self.board_right])
        block_any = self.block_colides()
        return down_board, any_border, block_any

    def get_display_state(self):
        resize = T.Compose([T.ToPILImage(),
                 T.Resize(40, interpolation=Image.CUBIC),
                 T.ToTensor()])

        display = pygame.surfarray.array3d(pygame.display.get_surface())
        display = display.transpose([1, 0, 2])

        # Convert to grayscale.
        display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        display[display > 0] = 255
        
        # Remove score board and edges.
        img_h, img_w = display.shape
        display = display[
            constants.BOARD_UP_MARGIN + constants.BOARD_HEIGHT:img_h - constants.BOARD_HEIGHT,
            constants.BOARD_HEIGHT:img_w - constants.BOARD_HEIGHT
        ]

        display = np.ascontiguousarray(display, dtype=np.float32) / 255
        display = torch.from_numpy(display)
        display = resize(display).unsqueeze(0)
        display = display.permute(1, 0, 2, 3)

        return display

    def step(self, x, rotation, episode, loss, episode_reward, draw_game=True):

        self.reward = 0
        # Generate a new block into the game if required
        self.get_block()

        # Handle the events to allow pygame to handle internal actions
        pygame.event.pump()



        # Remember the current configuration and try to
        # Apply the action supplied by the agent
        self.active_block.backup()
        self.perform_action(x, rotation)

        can_move_down = not self.block_colides()
        if not can_move_down:
            self.game_over = True
            state = self.get_game_state(0)
            return state, -10, self.game_over

        self.drop_active_block(episode, loss, episode_reward, draw=True)
        _, any_border, block_any = self.check_collisions()
        if any_border or block_any:
            self.active_block.restore()

        # Move down by one each step - no matter what
        # self.active_block.backup()
        # self.active_block.move(0, constants.BHEIGHT)
        # down_board, any_border, block_any = self.check_collisions()
        # if down_board or any_border or block_any:
        #     self.active_block.restore()

        self.active_block.backup()
        self.active_block.move(0, constants.BHEIGHT)
        can_move_down = not self.block_colides()
        down_board, any_border, block_any = self.check_collisions()
        self.active_block.restore()
        # down_board, any_border, block_any = self.check_collisions()
        # We end the game if we are on the respawn and we cannot move --> bang!
        if not can_move_down and (self.start_x == self.active_block.x and self.start_y == self.active_block.y):
            self.game_over = True

        current_lines_cleared = 0
        # The new block is inserted if we reached down board or we cannot move down.
        if down_board or not can_move_down:
            # Request new block
            self.new_block = True

            # A block was placed --> add 1 reward point
            self.reward += 1
            # Detect the filled line and possibly remove the line from the
            # screen.
            current_lines_cleared = self.remove_lines_emulator()

            # Add the reward for lines cleared
            self.reward += (2*current_lines_cleared)**2 * self.bx

        if draw_game: self.draw_game_rl(episode, loss, episode_reward)

        done = self.done or self.game_over
        if done:
            self.reward -= 1
        state = self.get_game_state(current_lines_cleared)
        self.get_block()
        return state, self.reward, done

    def get_game_state(self, lines_cleared, skip_active_block=True):
        grid = self.get_game_grid(skip_active_block=skip_active_block)
        agg_height = self.aggregate_height(grid)
        n_holes = self.number_of_holes(grid)
        bumpiness, _, _ = self.bumpiness(grid)
        block_type = [idx for idx, el in enumerate(self.block_data) if el[3] == self.active_block.type][0]
        return np.array([agg_height, n_holes, bumpiness, lines_cleared, block_type])

    def get_game_grid(self, skip_active_block=True):
        grid = np.zeros((len(self.y_positions), len(self.x_positions)), dtype=np.int)
        try:
            for block in self.blk_list:
                # Skip the active block when building the grid
                if skip_active_block and block.x == self.active_block.x and block.y == self.active_block.y:
                    continue

                for block_shape in block.shape:
                    x_grid_idx = self.x_positions.index(block_shape.x)
                    y_grid_idx = self.y_positions.index(block_shape.y)
                    grid[y_grid_idx, x_grid_idx] = 1

        except Exception as e:
            print(e)
            print(self.x_positions)
            print(self.y_positions)

        return grid

    def get_game_grid_state(self, skip_active_block=True):
        game_grid_state = self.get_game_grid(skip_active_block)
        game_grid_state = np.ascontiguousarray(game_grid_state, dtype=np.float32) / 255
        game_grid_state = torch.from_numpy(game_grid_state).unsqueeze(0).unsqueeze(0)

        return game_grid_state

    def get_initial_grid_state(self):
        initial_grid = np.zeros((len(self.y_positions), len(self.x_positions)), dtype=np.int)
        initial_grid = np.ascontiguousarray(initial_grid, dtype=np.float32) / 255
        initial_grid = torch.from_numpy(initial_grid).unsqueeze(0).unsqueeze(0)

        return initial_grid

    def aggregate_height(self, grid):
        agg_height = 0
        for x in range(grid.shape[1]):
            top_y = np.argwhere(grid[:, x] == 1)
            if len(top_y) != 0:
                y_coord = top_y[0][0]
                y_height = grid.shape[0] - y_coord
                agg_height += y_height

        return agg_height

    def number_of_holes(self, grid):
        n_holes = 0
        for x in range(grid.shape[1]):
            top_y = np.argwhere(grid[:, x] == 1)
            if len(top_y) != 0:
                tile_coords = top_y.flatten()
                tile_coords = grid.shape[0] - tile_coords
                possible_tiles = set(range(1, max(tile_coords) + 1))
                hole_tiles = possible_tiles - set(tile_coords.tolist())
                n_holes += len(hole_tiles)

        return n_holes

    def bumpiness(self, grid):
        agg_height_arr = np.zeros(grid.shape[1])
        for x in range(grid.shape[1]):
            top_y = np.argwhere(grid[:, x] == 1)
            if len(top_y) != 0:
                y_coord = top_y[0][0]
                y_height = grid.shape[0] - y_coord
                agg_height_arr[x] = y_height

        max_height = np.max(agg_height_arr)
        min_height = np.min(agg_height_arr)
        bumpiness = np.diff(agg_height_arr)**2

        return np.sum(bumpiness), max_height, min_height

    def init_env(self):
        # Initialize the game (pygame, fonts)
        pygame.init()
        pygame.font.init()
        self.myfont = pygame.font.SysFont(pygame.font.get_default_font(), constants.FONT_SIZE)
        self.screen = pygame.display.set_mode((self.resx, self.resy))
        pygame.display.set_caption("Tetris")
        self.create_possible_block_states()

        # Setup the time to fire the move event every given time
        # self.set_move_timer()
        # Control variables for the game. The done signal is used
        # to control the main loop (it is set by the quit action), the game_over signal
        # is set by the game logic and it is also used for the detection of "game over" drawing.
        # Finally the new_block variable is used for the requesting of new tetris block.
        self.done = False
        self.game_over = False
        self.new_block = True
        # Print the initial score
        # self.print_status_line()
        self.lines_cleared = 0
        self.reward = 0
        self.get_block()

    def print_learning_status(self, episode=None, loss=None, episode_reward=None):
        """
        Print the current state line
        """
        string = ["Episode: {0} Reward: {1}".format(episode, episode_reward)]
        self.print_text(string,constants.POINT_MARGIN,constants.POINT_MARGIN)

    def reset_env(self):
        self.lines_cleared = 0
        self.reward = 0
        self.blk_list.clear()
        self.score = 0
        self.done = False
        self.game_over = False
        self.new_block = True
        self.get_block()

if __name__ == "__main__":
    Tetris(16,30).run()
