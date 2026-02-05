"""
Depth Analysis Script for UE EXR Files
======================================
This script analyzes EXR depth files from Unreal Engine to help verify:
1. What depth units UE is outputting (normalized, linear, UE units)
2. Whether GT_TO_CENTIMETERS = 10000 is correct
3. Differences between SceneDepth vs Z channels

Usage:
    python analyze_depth.py ./data/HighresScreenshot00001_SceneDepth.exr
    python analyze_depth.py ./data/*.exr
"""

import os
import sys
import glob
import numpy as np
import OpenEXR
import Imath
import argparse


def analyze_exr_depth(filepath):
    """Analyze a single EXR depth file and print comprehensive statistics."""

    print(f"\n{'='*60}")
    print(f"FILE: {os.path.basename(filepath)}")
    print(f"{'='*60}")

    try:
        exr_file = OpenEXR.InputFile(filepath)
    except Exception as e:
        print(f"  ERROR: Could not open file - {e}")
        return None

    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    print(f"\n[Image Info]")
    print(f"  Resolution: {width} x {height}")
    print(f"  Pixel count: {width * height:,}")

    # List all available channels
    channels = list(header['channels'].keys())
    print(f"\n[Available Channels]")
    print(f"  {channels}")

    # Analyze each potentially useful depth channel
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    depth_channels = ['Z', 'SceneDepth', 'R', 'G', 'B', 'A']

    results = {}

    for chan_name in depth_channels:
        if chan_name not in channels:
            continue

        try:
            channel_str = exr_file.channel(chan_name, FLOAT)
            depth = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width).copy()

            # Replace inf/nan for stats
            valid_mask = np.isfinite(depth) & (depth > 0)
            valid_depth = depth[valid_mask]

            if len(valid_depth) == 0:
                print(f"\n[Channel '{chan_name}'] - No valid data")
                continue

            print(f"\n[Channel '{chan_name}']")
            print(f"  Valid pixels: {len(valid_depth):,} ({100*len(valid_depth)/(width*height):.1f}%)")
            print(f"  Min:    {valid_depth.min():.6f}")
            print(f"  Max:    {valid_depth.max():.6f}")
            print(f"  Mean:   {valid_depth.mean():.6f}")
            print(f"  Median: {np.median(valid_depth):.6f}")
            print(f"  Std:    {valid_depth.std():.6f}")

            # Percentiles for outlier analysis
            p1, p5, p25, p50, p75, p95, p99 = np.percentile(
                valid_depth, [1, 5, 25, 50, 75, 95, 99]
            )
            print(f"\n  Percentiles:")
            print(f"    1%:   {p1:.6f}")
            print(f"    5%:   {p5:.6f}")
            print(f"    25%:  {p25:.6f}")
            print(f"    50%:  {p50:.6f}")
            print(f"    75%:  {p75:.6f}")
            print(f"    95%:  {p95:.6f}")
            print(f"    99%:  {p99:.6f}")

            # Depth unit inference
            print(f"\n  Unit Analysis:")

            # Check if normalized (0-1 range)
            if valid_depth.max() <= 1.0 and valid_depth.min() >= 0.0:
                print(f"    -> Appears NORMALIZED (0-1 range)")

            # Check if in typical room-scale UE units
            # UE uses centimeters by default, so a room might be 100-10000 units
            if 10 < p50 < 100000:
                print(f"    -> Likely UE CENTIMETERS (typical indoor range)")
                print(f"    -> If GT_TO_CENTIMETERS=10000, median would be {p50/10000:.2f}m")
                print(f"    -> If GT_TO_CENTIMETERS=100, median would be {p50/100:.2f}m")
                print(f"    -> If GT_TO_CENTIMETERS=1, median would be {p50:.2f}cm = {p50/100:.2f}m")

            # Check if in meters (typical room 1-10m)
            if 0.1 < p50 < 100:
                print(f"    -> Could be METERS (typical indoor range)")

            # Store for comparison
            results[chan_name] = {
                'depth': depth,
                'valid_mask': valid_mask,
                'median': np.median(valid_depth),
                'mean': valid_depth.mean()
            }

        except Exception as e:
            print(f"\n[Channel '{chan_name}'] - Error reading: {e}")

    # Compare channels if multiple exist
    if len(results) > 1:
        print(f"\n[Channel Comparison]")
        channel_names = list(results.keys())
        for i, name1 in enumerate(channel_names):
            for name2 in channel_names[i+1:]:
                d1 = results[name1]['depth']
                d2 = results[name2]['depth']
                mask = results[name1]['valid_mask'] & results[name2]['valid_mask']

                if mask.sum() > 0:
                    diff = np.abs(d1[mask] - d2[mask])
                    ratio = d1[mask] / (d2[mask] + 1e-10)

                    print(f"\n  '{name1}' vs '{name2}':")
                    print(f"    Ratio (median): {np.median(ratio):.6f}")
                    print(f"    Diff (mean):    {diff.mean():.6f}")
                    print(f"    Diff (max):     {diff.max():.6f}")

                    if np.allclose(d1[mask], d2[mask], rtol=1e-5):
                        print(f"    -> Channels are IDENTICAL")
                    elif np.isclose(np.median(ratio), 1.0, rtol=0.01):
                        print(f"    -> Channels are VERY SIMILAR")

    return results


def estimate_gt_to_centimeters(median_depth, expected_room_depth_m=3.0):
    """
    Estimate what GT_TO_CENTIMETERS should be given an expected room depth.

    Args:
        median_depth: Median raw depth value from EXR
        expected_room_depth_m: Expected median depth in meters (default 3m for indoor)

    Returns:
        Estimated GT_TO_CENTIMETERS value
    """
    # If median depth in file is X, and we expect Y meters:
    # X * GT_TO_CENTIMETERS / 100 = Y
    # GT_TO_CENTIMETERS = Y * 100 / X
    expected_cm = expected_room_depth_m * 100
    gt_to_cm = median_depth / expected_cm
    return gt_to_cm


def main():
    parser = argparse.ArgumentParser(
        description='Analyze EXR depth files from Unreal Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_depth.py ./data/depth.exr
  python analyze_depth.py ./data/*.exr
  python analyze_depth.py ./data/depth.exr --expected-depth 4.0

This will help you verify:
  1. What channels contain depth data (Z, SceneDepth, R, etc.)
  2. What units the depth is in (normalized, UE centimeters, meters)
  3. Whether GT_TO_CENTIMETERS = 10000 is correct for your setup
        """
    )
    parser.add_argument('files', nargs='+', help='EXR file(s) to analyze (supports glob patterns)')
    parser.add_argument('--expected-depth', type=float, default=3.0,
                       help='Expected median depth in meters (default: 3.0m for typical indoor)')

    args = parser.parse_args()

    # Expand glob patterns
    all_files = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        if matches:
            all_files.extend(matches)
        elif os.path.exists(pattern):
            all_files.append(pattern)
        else:
            print(f"Warning: No files matched pattern '{pattern}'")

    if not all_files:
        print("Error: No valid EXR files found")
        sys.exit(1)

    print(f"\nAnalyzing {len(all_files)} file(s)...")
    print(f"Expected median room depth: {args.expected_depth}m")

    all_results = {}
    for filepath in all_files:
        if filepath.lower().endswith('.exr'):
            results = analyze_exr_depth(filepath)
            if results:
                all_results[filepath] = results

    # Summary and GT_TO_CENTIMETERS estimation
    print(f"\n{'='*60}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*60}")

    for filepath, results in all_results.items():
        print(f"\n{os.path.basename(filepath)}:")

        # Use the best channel (prefer Z or SceneDepth)
        best_channel = None
        for pref in ['Z', 'SceneDepth', 'R']:
            if pref in results:
                best_channel = pref
                break

        if best_channel:
            median = results[best_channel]['median']
            estimated_gt = estimate_gt_to_centimeters(median, args.expected_depth)

            print(f"  Best channel: '{best_channel}'")
            print(f"  Median raw value: {median:.2f}")
            print(f"\n  If your expected median depth is {args.expected_depth}m:")
            print(f"    -> Estimated GT_TO_CENTIMETERS = {estimated_gt:.2f}")
            print(f"\n  Verification with common GT_TO_CENTIMETERS values:")
            for gt_val in [1, 100, 1000, 10000, 100000]:
                depth_m = (median * gt_val) / 100
                print(f"    GT_TO_CENTIMETERS={gt_val:>6}: median depth = {depth_m:.2f}m")

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("""
1. Look at the 'Verification' table above
2. Find which GT_TO_CENTIMETERS gives a realistic room depth
3. Update GT_TO_CENTIMETERS in img_to_pointcloud.ipynb
4. Re-run the notebook and check point cloud alignment in UE

Typical values:
  - GT_TO_CENTIMETERS = 1      : Depth already in centimeters
  - GT_TO_CENTIMETERS = 100    : Depth in meters
  - GT_TO_CENTIMETERS = 10000  : Depth in some scaled UE format
""")


if __name__ == "__main__":
    main()
