from pynput.keyboard import Key, Listener
import numpy as np
import math


class manual_control:
    def __init__(self, n_agents):
        self.controlled_agent = 0
        self.n_agents = n_agents
        self.cmd_vel = [0, 0]
        self.join = [0]
        self.angle = 0.4
        self.speed = 1.0

    def on_press(self, key):

        try:
            match (key.char):
                case "w":
                    self.cmd_vel = [self.speed, 0.0]
                case "q":
                    self.cmd_vel = [self.speed, self.angle]
                case "e":
                    self.cmd_vel = [self.speed, -self.angle]
                case "s":
                    self.cmd_vel = [-self.speed, 0.0]
                case "z":
                    self.cmd_vel = [-self.speed, -self.angle]
                case "c":
                    self.cmd_vel = [-self.speed, self.angle]
                case "d":
                    self.cmd_vel = [self.speed, -0.5]
                case "a":
                    self.cmd_vel = [self.speed, 0.5]
                case "8":
                    self.cmd_vel = [self.speed, 0.0]
                case "2":
                    self.cmd_vel = [-self.speed, 0.0]
                case "4":
                    self.cmd_vel = [self.speed, self.angle]
                case "6":
                    self.cmd_vel = [self.speed, -self.angle]
                case "j":
                    self.join = [0] if self.join[0] else [1]

        except AttributeError:
            if key == Key.up:
                self.cmd_vel = [self.speed, 0.0]
            elif key == Key.down:
                self.cmd_vel = [-self.speed, 0.0]
            elif key == Key.left:
                self.cmd_vel = [self.speed, self.angle]
            elif key == Key.right:
                self.cmd_vel = [self.speed, -self.angle]

    def on_release(self, key):

        self.cmd_vel = [0.0, 0.0]

        if key == Key.space:
            self.controlled_agent += 1
            if self.controlled_agent == self.n_agents:
                self.controlled_agent = 0


if __name__ == "__main__":
    mc = manual_control(n_agents=1)
    # Collect events until released
    with Listener(on_press=mc.on_press, on_release=mc.on_release) as listener:
        listener.join()
