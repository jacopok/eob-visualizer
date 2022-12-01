from coordinate_conversion import get_time_and_coordinates
from maximum_likelihood_scenario import M1, M2, get_dynamics, get_t_hp
from pathlib import Path
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import ffmpeg
from scipy.interpolate import interp1d

VID_FRAMES = 1000
TIMES_PER_PLOT = 100
TIME_AFTER_MERGER = 500

def get_limits(time: int):
    times = [0, .40*VID_FRAMES, .45*VID_FRAMES, VID_FRAMES]
    lims = [200, 200, 30, 30]
    return interp1d(times, lims, kind='linear')(time)


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

        plt.gcf().suptitle(
            (
                f"time=-{info['time'][time+TIMES_PER_PLOT]:.2f}s, "
                f"energy={info['energy'][time+TIMES_PER_PLOT]:.4f}M"
            ),
            y=.9
        )

        plt.savefig(
            frames_folder / f'{time:04}.png',
            # bbox_inches='tight', 
            # pad_inches = 0.5
        )
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

def plot_frame(time, r_1, r_2, times, hp, times_per_plot=TIMES_PER_PLOT):
    
    figsize_multip = 1.5
    fix, axs = plt.subplots(nrows=1, ncols=2, figsize=(16/figsize_multip, 9/figsize_multip), dpi=120*figsize_multip)
    
    axs[0].scatter(
        r_1[0, time:time+TIMES_PER_PLOT], 
        r_1[1, time:time+TIMES_PER_PLOT], 
        alpha=np.arange(TIMES_PER_PLOT) / TIMES_PER_PLOT,
        s=1.
    )
    axs[0].scatter(
        r_2[0, time:time+TIMES_PER_PLOT], 
        r_2[1, time:time+TIMES_PER_PLOT], 
        alpha=np.arange(TIMES_PER_PLOT) / TIMES_PER_PLOT,
        s=1.
    )
    axs[0].scatter(
        r_1[0, time+TIMES_PER_PLOT], 
        r_1[1, time+TIMES_PER_PLOT], 
        s=M1,
        c='black'
    )
    axs[0].scatter(
        r_2[0, time+TIMES_PER_PLOT], 
        r_2[1, time+TIMES_PER_PLOT], 
        s=M2,
        c='black'
    )
    axs[0].scatter(
        r_2[0, time:time+TIMES_PER_PLOT], 
        r_2[1, time:time+TIMES_PER_PLOT], 
        alpha=np.arange(TIMES_PER_PLOT) / TIMES_PER_PLOT,
        s=1.
    )
    axs[0].set_aspect('equal')
    lims = get_limits(time)
    axs[0].set_xlim(-lims, lims)
    axs[0].set_ylim(-lims, lims)
    axs[0].set_xlabel('x [M]')
    axs[0].set_ylabel('y [M]')
    axs[0].set_title('Physical space orbits')
    
    axs[1].plot(times[:time+TIMES_PER_PLOT], hp[:time+TIMES_PER_PLOT])
    axs[1].set_ylim(-1.5, 1.5)
    axs[1].set_xlabel('Time to merger [s]')
    axs[1].set_title('Plus polarized waveform')
    axs[1].set_aspect('equal')
    x_left, x_right = axs[1].get_xlim()
    y_low, y_high = axs[1].get_ylim()
    axs[1].set_aspect(abs((x_right-x_left)/(y_low-y_high)))
    # plt.tight_layout()
    # axs[1].set_ylabel('h_{+} [natural units]')

if __name__ == '__main__':
    t, r_1, r_2 = get_time_and_coordinates()
    dyn = get_dynamics()
    t_full, hp = get_t_hp()
    
    times = np.linspace(t[60], t[-1]+TIME_AFTER_MERGER, num=VID_FRAMES+TIMES_PER_PLOT)
    times_index = np.arange(VID_FRAMES)
    times_to_merger = t[-1] - times + TIME_AFTER_MERGER
    
    r_1_interp = interp1d(t[:-35], r_1[:,:-35], fill_value=0, bounds_error=False)(times)
    r_2_interp = interp1d(t[:-35], r_2[:,:-35], fill_value=0, bounds_error=False)(times)
    E_interp = interp1d(t, dyn['E'], fill_value=dyn['E'][-1], bounds_error=False)(times)
    hp_interp = interp1d(t_full, hp, kind='cubic', fill_value='extrapolate')(times)
    
    M_in_seconds = 4.92549095e-06 * (M1+M2)
    
    
    make_vid(
        partial(plot_frame, r_1=r_1_interp, r_2=r_2_interp, times=-times_to_merger * M_in_seconds, hp=hp_interp), 
        'capture', 
        times_index,
        info={
            'energy': E_interp,
            'time': times_to_merger * M_in_seconds
        },
    )
