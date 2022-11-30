from coordinate_conversion import get_time_and_coordinates
from maximum_likelihood_scenario import M1, M2, get_dynamics
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ffmpeg
from scipy.interpolate import interp1d

VID_FRAMES = 1000
TIMES_PER_PLOT = 100

def make_vid(
    plot_frame: callable,
    vid_name: str,
    times: list[int],
    info: dict,
    framerate: int = 60,
    ):
    
    this_folder = Path().resolve()
    frames_folder = this_folder / f'frames_{vid_name}'
    
    if not frames_folder.exists():
        frames_folder.mkdir()

    
    for time in tqdm(times):
        plot_frame(time)

        plt.title(f"time=-{info['time'][time+TIMES_PER_PLOT]:.2f}s, energy={info['energy'][time+TIMES_PER_PLOT]:.4f}M")

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
        alpha=np.arange(TIMES_PER_PLOT) / TIMES_PER_PLOT,
        s=1.
    )
    plt.scatter(
        r_2[0, time:time+TIMES_PER_PLOT], 
        r_2[1, time:time+TIMES_PER_PLOT], 
        alpha=np.arange(TIMES_PER_PLOT) / TIMES_PER_PLOT,
        s=1.
    )
    plt.scatter(
        r_1[0, time+TIMES_PER_PLOT], 
        r_1[1, time+TIMES_PER_PLOT], 
        s=M1,
        c='black'
    )
    plt.scatter(
        r_2[0, time+TIMES_PER_PLOT], 
        r_2[1, time+TIMES_PER_PLOT], 
        s=M2,
        c='black'
    )
    plt.scatter(
        r_2[0, time:time+TIMES_PER_PLOT], 
        r_2[1, time:time+TIMES_PER_PLOT], 
        alpha=np.arange(TIMES_PER_PLOT) / TIMES_PER_PLOT,
        s=1.
    )
    plt.gca().set_aspect('equal')
    lims = 200 if time < .5*VID_FRAMES else 30
    plt.xlim(-lims, lims)
    plt.ylim(-lims, lims)
    plt.xlabel('x [M]')
    plt.ylabel('y [M]')

if __name__ == '__main__':
    t, r_1, r_2 = get_time_and_coordinates()
    dyn = get_dynamics()
    
    times = np.linspace(t[60], t[-35], num=VID_FRAMES+TIMES_PER_PLOT)
    times_index = np.arange(VID_FRAMES)
    times_to_merger = t[-1] - times
    
    r_1_interp = interp1d(t, r_1)(times)
    r_2_interp = interp1d(t, r_2)(times)
    E_interp = interp1d(t, dyn['E'])(times)
    
    M_in_seconds = 4.92549095e-06 * (M1+M2)
    
    make_vid(
        partial(plot_frame, r_1=r_1_interp, r_2=r_2_interp), 
        'capture', 
        times_index,
        info={
            'energy': E_interp,
            'time': times_to_merger * M_in_seconds
        },
    )
