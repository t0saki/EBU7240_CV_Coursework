import os
import cv2


class ImageWriter:
    def __init__(self, output_dir: str, write_frames: list):
        self.output_dir = output_dir
        self.write_frames = write_frames
        self.frame_count = 0

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def write(self, frame):
        self.frame_count += 1
        if self.frame_count in self.write_frames:
            cv2.imwrite(os.path.join(self.output_dir,
                        f'{self.frame_count}.png'), frame)
