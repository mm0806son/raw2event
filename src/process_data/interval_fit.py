import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import invgauss, levy


DEFAULT_SMOKE_SAMPLE = (
    "/tmp/raw2event_npz_smoke_one/"
    "42868_horse_1_8698_20251228_234242_filtered_raw.npz"
)


def load_event_array(events_or_path, event_key="events"):
    """Load an event array from memory or from a supported file on disk.

    Supported on-disk formats:
    - `.npz` with an `events` key (current train_class output)
    - `.npy` containing an `(N, 4)` event array
    - `.txt` / `.csv` containing numeric rows
    """
    if hasattr(events_or_path, "cpu"):
        events = events_or_path.cpu().numpy()
    elif isinstance(events_or_path, (str, Path)):
        path = Path(events_or_path)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if event_key in data.files:
                    events = data[event_key]
                elif len(data.files) == 1:
                    events = data[data.files[0]]
                else:
                    raise ValueError(
                        f"NPZ file '{path}' does not contain key '{event_key}'. "
                        f"Available keys: {list(data.files)}"
                    )
        elif suffix == ".npy":
            events = np.load(path, allow_pickle=False)
        elif suffix in {".txt", ".csv"}:
            events = np.loadtxt(path, delimiter=",")
        else:
            raise ValueError(
                f"Unsupported event file format '{suffix}' for '{path}'. "
                "Use .npz, .npy, .txt, or .csv."
            )
    else:
        events = np.asarray(events_or_path)

    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] < 4:
        raise ValueError(
            f"Event array must have shape (N, >=4), got {events.shape}."
        )
    return events


def _collect_eligible_pixel_timestamps(events, min_events_per_pixel=5, time_index=0):
    """Collect per-pixel timestamp lists for pixels with enough events."""
    x_coords = events[:, 1].astype(int)
    y_coords = events[:, 2].astype(int)
    pixel_timestamps = defaultdict(list)

    for idx in range(events.shape[0]):
        pixel_timestamps[(x_coords[idx], y_coords[idx])].append(events[idx, time_index])

    return [
        np.sort(np.asarray(ts_list))
        for ts_list in pixel_timestamps.values()
        if len(ts_list) >= min_events_per_pixel
    ]


def compute_event_intervals(events, mode="per-pixel", min_events_per_pixel=5, time_index=0):
    """Compute event time intervals in microseconds.

    Args:
        events: Event array or path to an event file.
        mode: `per-pixel` for per-pixel internal intervals, `global` for
            globally sorted consecutive events.
        min_events_per_pixel: Minimum per-pixel event count to be included
            in `per-pixel` mode.
        time_index: Timestamp column index.
    """
    events = load_event_array(events)

    if events.shape[0] < 2:
        return np.asarray([], dtype=np.float64)

    if mode == "global":
        timestamps = np.sort(events[:, time_index].astype(np.float64))
        intervals = np.diff(timestamps)
    elif mode == "per-pixel":
        eligible_pixels_timestamps = _collect_eligible_pixel_timestamps(
            events,
            min_events_per_pixel=min_events_per_pixel,
            time_index=time_index,
        )
        diffs = [np.diff(ts_list) for ts_list in eligible_pixels_timestamps if len(ts_list) > 1]
        if not diffs:
            return np.asarray([], dtype=np.float64)
        intervals = np.concatenate(diffs)
    else:
        raise ValueError(f"Unsupported interval mode '{mode}'.")

    return intervals[intervals > 0]


def plot_event_interval_histogram(
    events,
    output_path,
    mode="per-pixel",
    min_events_per_pixel=5,
    max_dt_us=100000,
    bins=100,
    title=None,
    event_key="events",
):
    """Plot and save an interval histogram for an event sample."""
    loaded_events = load_event_array(events, event_key=event_key)
    intervals = compute_event_intervals(
        loaded_events,
        mode=mode,
        min_events_per_pixel=min_events_per_pixel,
    )
    if intervals.size == 0:
        raise ValueError("No positive event intervals were found for plotting.")

    if max_dt_us is not None:
        intervals_for_plot = intervals[intervals <= max_dt_us]
        if intervals_for_plot.size == 0:
            raise ValueError(
                f"No intervals remain after applying max_dt_us={max_dt_us}."
            )
    else:
        intervals_for_plot = intervals

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.hist(intervals_for_plot, bins=bins, density=True, alpha=0.75, color="#1f77b4")
    plt.xlabel("Time Interval Δt (μs)")
    plt.ylabel("Probability Density")
    plt.title(title or f"Event Interval Distribution ({mode})")
    plt.grid(True, alpha=0.3)

    stats_text = (
        f"count={intervals_for_plot.size}\n"
        f"mean={intervals_for_plot.mean():.2f} μs\n"
        f"median={np.median(intervals_for_plot):.2f} μs\n"
        f"max={intervals_for_plot.max():.2f} μs"
    )
    plt.text(
        0.98,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

    return output_path


def analyze_per_pixel_event_intervals_combined(
    events,                      # Input event data
    min_events_per_pixel=5,      # Minimum events per pixel
    max_dt_us_for_plot=100000,   # Maximum time interval for plotting
    plot_bins=100,               # Number of histogram bins
    mean_threshold=1e-6,         # Drift rate threshold
    type='RGB'                   # Data type
):
    """Analyze per-pixel event time intervals and fit distributions"""
    events = load_event_array(events)
    eligible_pixels_timestamps = _collect_eligible_pixel_timestamps(
        events,
        min_events_per_pixel=min_events_per_pixel,
    )
    all_dt = compute_event_intervals(
        events,
        mode="per-pixel",
        min_events_per_pixel=min_events_per_pixel,
    )

    # Prepare fitting data
    tau_for_fit = all_dt.astype(np.float64)
    num_pixels = len(eligible_pixels_timestamps)
    if tau_for_fit.size == 0:
        return num_pixels, all_dt, np.nan, np.nan

    # Try fitting inverse Gaussian distribution
    try:
        ig_fit_shape, ig_fit_loc, ig_fit_scale = invgauss.fit(tau_for_fit, floc=0)
        ig_mu = ig_fit_shape * ig_fit_scale
        ig_lambda = ig_fit_scale
        mu = 1.0 / ig_mu if ig_mu > 0 else 0
        sigma = 1.0 / math.sqrt(ig_lambda) if ig_lambda > 0 else 0
        dist_name = "Inverse Gaussian"
    except Exception:
        # If inverse Gaussian fitting fails, try Lévy distribution
        try:
            levy_fit_loc, levy_fit_scale = levy.fit(tau_for_fit, floc=0)
            mu = 0.0
            sigma = 1.0 / math.sqrt(levy_fit_scale) if levy_fit_scale > 0 else 0
            final_scale_param = levy_fit_scale
            dist_name = "Lévy"
        except Exception:
            return num_pixels, all_dt, np.nan, np.nan
    
    # Plot distribution
    dt_for_plot = all_dt[all_dt <= max_dt_us_for_plot]
    if len(dt_for_plot) > 0:
        plt.figure(figsize=(12, 6))
        plt.hist(dt_for_plot, bins=plot_bins, density=True, alpha=0.6, label="Histogram")
        
        # Generate fitting curve
        x_plot = np.linspace(dt_for_plot.min(), dt_for_plot.max(), 500)
        if dist_name == "Inverse Gaussian":
            y_plot = invgauss.pdf(x_plot, ig_fit_shape, loc=0, scale=ig_fit_scale)
            plt.plot(x_plot, y_plot, 'r-', linewidth=2, 
                    label=f'IG Fit (ig_u={ig_mu:.5g}, ig_lambda={ig_lambda:.5g}, μ={mu:.5g}, σ={sigma:.5g})')
        else:
            y_plot = levy.pdf(x_plot, loc=0, scale=final_scale_param)
            plt.plot(x_plot, y_plot, 'r-', linewidth=2, 
                    label=f'Lévy Fit (scale={levy_fit_scale:.5g}, μ={mu:.5g}, σ={sigma:.5g})')
        
        # Set chart properties
        plt.xlabel("Time Interval Δt (μs)")
        plt.ylabel("Probability Density")
        plt.title(f"{type} Interval Distribution")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    return num_pixels, all_dt, mu, sigma


def _build_default_output_path(events_path, mode):
    """Build a default PNG path next to the input event file."""
    path = Path(events_path)
    return path.with_name(f"{path.stem}_interval_hist_{mode}.png")


def build_argparser():
    """Build CLI parser for minimal interval histogram generation."""
    parser = argparse.ArgumentParser(
        description="Plot a histogram of event time intervals from a current Raw2Event sample."
    )
    parser.add_argument(
        "--events",
        default=DEFAULT_SMOKE_SAMPLE,
        help=(
            "Path to the event file (.npz/.npy/.txt/.csv). "
            f"Defaults to the local smoke sample: {DEFAULT_SMOKE_SAMPLE}"
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path. Defaults to '<input>_interval_hist_<mode>.png'.",
    )
    parser.add_argument(
        "--mode",
        choices=("per-pixel", "global"),
        default="per-pixel",
        help="Interval definition to plot.",
    )
    parser.add_argument(
        "--min-events-per-pixel",
        type=int,
        default=5,
        help="Only used in per-pixel mode.",
    )
    parser.add_argument(
        "--max-dt-us",
        type=float,
        default=100000,
        help="Clip plotted intervals above this threshold. Use a negative value to disable clipping.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=100,
        help="Histogram bin count.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom plot title.",
    )
    parser.add_argument(
        "--event-key",
        default="events",
        help="NPZ key to load when reading .npz files.",
    )
    return parser


def main():
    """CLI entrypoint for interval histogram plotting."""
    parser = build_argparser()
    args = parser.parse_args()

    max_dt_us = None if args.max_dt_us is not None and args.max_dt_us < 0 else args.max_dt_us
    output_path = (
        Path(args.output)
        if args.output is not None
        else _build_default_output_path(args.events, args.mode)
    )

    loaded_events = load_event_array(args.events, event_key=args.event_key)
    intervals = compute_event_intervals(
        loaded_events,
        mode=args.mode,
        min_events_per_pixel=args.min_events_per_pixel,
    )
    saved_path = plot_event_interval_histogram(
        loaded_events,
        output_path=output_path,
        mode=args.mode,
        min_events_per_pixel=args.min_events_per_pixel,
        max_dt_us=max_dt_us,
        bins=args.bins,
        title=args.title,
        event_key=args.event_key,
    )

    print(f"Loaded events: {args.events}")
    print(f"Interval mode: {args.mode}")
    print(f"Positive intervals: {intervals.size}")
    if intervals.size > 0:
        print(
            "Interval stats (μs): "
            f"mean={intervals.mean():.2f}, median={np.median(intervals):.2f}, "
            f"min={intervals.min():.2f}, max={intervals.max():.2f}"
        )
    print(f"Saved histogram: {saved_path}")


if __name__ == "__main__":
    main()




def analyze_event_frequency_spectrum(events, max_freq_hz=100, bins=50, time_index=0, min_events_per_pixel=2):
    """
    Analyze frequency spectrum of event data
    
    Args:
        events (ndarray): Event array, format [t, x, y, ...]
        output_path (str): Output image path, None for display only
        max_freq_hz (float): Maximum analysis frequency (Hz)
        bins (int): Number of histogram bins
        time_index (int): Timestamp index position in event array
        min_events_per_pixel (int): Minimum events per pixel
    
    Returns:
        dict: Dictionary containing spectrum analysis results
    """
    if hasattr(events, 'cpu'):  # Convert PyTorch tensor to numpy array
        events = events.cpu().numpy()
    
    # Extract coordinate information
    x_coords = events[:, 1].astype(int)  # Extract x coordinates
    y_coords = events[:, 2].astype(int)  # Extract y coordinates
    
    # Collect timestamps for each pixel
    pixel_timestamps = {}
    for i in range(events.shape[0]):
        key = (x_coords[i], y_coords[i])
        if key not in pixel_timestamps:
            pixel_timestamps[key] = []
        pixel_timestamps[key].append(events[i, time_index])
    
    # Calculate frequency for each pixel
    pixel_frequencies = {}
    for pixel, timestamps in pixel_timestamps.items():
        if len(timestamps) >= min_events_per_pixel:
            ts = np.sort(timestamps)  # Sort timestamps
            intervals = np.diff(ts)   # Calculate intervals
            # Calculate frequency: 1/interval (convert microseconds to seconds)
            freqs = 1.0 / (intervals * 1e-6)
            freqs = freqs[freqs <= max_freq_hz]  # Limit maximum frequency
            if len(freqs) > 0:
                pixel_frequencies[pixel] = freqs
    
    # Merge frequency data from all pixels
    if not pixel_frequencies:
        print("Warning: No qualified pixel frequency data found")
        return None
    
    all_frequencies = np.concatenate(list(pixel_frequencies.values()))
    
    # Calculate basic statistics
    freq_stats = {
        'mean': np.mean(all_frequencies),
        'median': np.median(all_frequencies),
        'std': np.std(all_frequencies),
        'min': np.min(all_frequencies),
        'max': np.max(all_frequencies),
        'count': len(all_frequencies),
        'pixels': len(pixel_frequencies)
    }
    
    # Calculate frequency distribution histogram
    hist, bin_edges = np.histogram(all_frequencies, bins=bins, range=(0, max_freq_hz))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Plot spectrum
    plt.figure(figsize=(12, 6))
    plt.bar(bin_centers, hist, width=(max_freq_hz/bins)*0.8, alpha=0.7)
    plt.axvline(30, color='r', linestyle='--', label=f"Mean: {freq_stats['mean']:.2f} Hz")
    plt.axvline(freq_stats['median'], color='g', linestyle='--', label=f"Median: {freq_stats['median']:.2f} Hz")
    
    # Add statistics text
    info_text = (f"Pixel Count: {freq_stats['pixels']}\n"
                f"Event Frequency: {freq_stats['count']}\n"
                f"Mean: {freq_stats['mean']:.2f} Hz\n"
                f"Standard Deviation: {freq_stats['std']:.2f} Hz\n"
                f"Range: [{freq_stats['min']:.2f}, {freq_stats['max']:.2f}] Hz")
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set chart properties
    plt.xlabel('Frequency (Hz)(log scale)')
    plt.ylabel('Count')
    plt.title('Event Frequency Distribution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    
    return {
        'frequencies': all_frequencies,
        'histogram': {'bins': bin_centers, 'counts': hist},
        'stats': freq_stats
    }

def analyze_event_fft_spectrum(events, sampling_rate=1000, time_index=0, max_freq_hz=100):
    """Analyze FFT spectrum of event data"""
    if hasattr(events, 'cpu'):  # Convert PyTorch tensor to numpy array
        events = events.cpu().numpy()
    
    # Extract timestamps
    timestamps = events[:, time_index]
    if len(timestamps) < 10:  # Too few events for analysis
        print("Warning: Insufficient events for FFT analysis")
        return None
    
    # Calculate event rate signal
    t_min, t_max = np.min(timestamps), np.max(timestamps)
    duration = (t_max - t_min) * 1e-6  # Convert to seconds
    
    # Create uniform time axis
    num_bins = int(sampling_rate * duration)  # Determine bin count based on sampling rate
    num_bins = max(1024, num_bins)  # Ensure at least 1024 points for sufficient resolution
    
    # Calculate event rate time series
    hist, bin_edges = np.histogram(timestamps, bins=num_bins, range=(t_min, t_max))
    bin_width = (t_max - t_min) / num_bins  # μs
    event_rate = hist / (bin_width * 1e-6)  # Convert to events/second
    
    # Calculate FFT
    fft_result = np.fft.rfft(event_rate)  # Real FFT
    fft_magnitude = np.abs(fft_result) * 2.0 / num_bins  # Normalize magnitude
    fft_magnitude[0] /= 2  # DC component doesn't need ×2
    
    # Calculate frequency axis
    sample_spacing = duration / num_bins  # seconds
    fft_freqs = np.fft.rfftfreq(num_bins, d=sample_spacing)  # Frequency axis (Hz)
    
    # Limit frequency range
    valid_idx = fft_freqs <= max_freq_hz
    fft_freqs = fft_freqs[valid_idx]
    fft_magnitude = fft_magnitude[valid_idx]
    
    # Calculate statistics
    dominant_freq = fft_freqs[np.argmax(fft_magnitude[1:])+1] if len(fft_magnitude) > 1 else 0
    fft_stats = {
        'dominant_freq': dominant_freq,
        'mean_magnitude': np.mean(fft_magnitude),
        'max_magnitude': np.max(fft_magnitude),
        'total_power': np.sum(fft_magnitude**2)
    }
    
    # Plot FFT
    plt.figure(figsize=(12, 6))
    plt.plot(fft_freqs, fft_magnitude, linewidth=1.5)
    plt.axvline(dominant_freq, color='r', linestyle='--', 
                label=f'Main Frequency: {dominant_freq:.2f} Hz')
    
    # Add information text
    info_text = (f"main frequency: {dominant_freq:.2f} Hz\n"
                 f"mean magnitude: {fft_stats['mean_magnitude']:.2f}\n"
                 f"max magnitude: {fft_stats['max_magnitude']:.2f}\n"
                 f"total power: {fft_stats['total_power']:.2f}")
    plt.text(0.95, 0.95, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Set chart properties
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Event FFT Spectrum Analysis')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return {
        'freqs': fft_freqs,
        'magnitude': fft_magnitude,
        'stats': fft_stats,
        'event_rate': event_rate,
        'time_bins': bin_edges
    }
