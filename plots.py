from coordinate_conversion import get_time_and_coordinates
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ffmpeg

TIMES_PER_PLOT = 100

def make_vid(
    plot_frame: callable,
    vid_name: str,
    times: list[int],
    framerate: int = 30,
    ):
    
    this_folder = Path().resolve()
    frames_folder = this_folder / f'frames_{vid_name}'
    
    if not frames_folder.exists():
        frames_folder.mkdir()

    
    for time in tqdm(times):
        plot_frame(time)

        plt.title(f't={time}')

        plt.savefig(frames_folder / f'{time:04}.png')
        plt.close()
    
    (
        ffmpeg
        .input(str(frames_folder/'*.png'), pattern_type='glob', framerate=framerate)
        .output(str(this_folder / f'{vid_name}.mp4'))
        .run()
    )
    
    for img in frames_folder.iterdir():
        img.unlink()
    frames_folder.rmdir()

def plot_frame(time, r_1, r_2, times_per_plot=TIMES_PER_PLOT):
    
    plt.scatter(
        r_1[0, time:time+TIMES_PER_PLOT], 
        r_1[1, time:time+TIMES_PER_PLOT], 
        alpha=np.arange(TIMES_PER_PLOT) / TIMES_PER_PLOT
    )
    plt.scatter(
        r_2[0, time:time+TIMES_PER_PLOT], 
        r_2[1, time:time+TIMES_PER_PLOT], 
        alpha=np.arange(TIMES_PER_PLOT) / TIMES_PER_PLOT
    )

if __name__ == '__main__':
    t, r_1, r_2 = get_time_and_coordinates()
    times = np.arange(len(t) - TIMES_PER_PLOT)
    
    make_vid(partial(plot_frame, r_1=r_1, r_2=r_2), 'capture', times)