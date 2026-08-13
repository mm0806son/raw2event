# batch_compress_local.py

import os
import argparse
import concurrent.futures
import subprocess
import shutil
import sys
import glob

def compress_file(dat_file, output, width, height, fps, remove_original=False):
    """Compress single .dat file to local output directory, optionally delete original file"""
    basename = os.path.basename(dat_file)
    if basename.startswith("raw"):
        mode = "raw_10bit"
    elif basename.startswith("rgb"):
        mode = "rgb"
    else:
        print(f"Skipped: {basename} (unknown prefix)")
        return False

    print(f"Processing {basename}...")
    
    # Step 1: Compress the file
    stem = os.path.splitext(basename)[0]
    # Use basename to include extension in temp folder name to avoid collision between file.dat and file.DAT
    safe_name = basename.replace('.', '_')
    temp_dir_name = f"temp_{safe_name}"
    cmd = [
        "python3", "dat2mkv.py",
        "--input", dat_file,
        "--output", output,
        "--width", str(width),
        "--height", str(height),
        "--fps", str(fps),
        "--mode", mode,
        "--temp_dir", temp_dir_name
    ]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Failed to compress: {basename}")
        return False
    
    print(f"Compressed: {basename}")
    
    # Step 2: Confirm generated mkv file(s)
    stem = os.path.splitext(basename)[0]
    mkv_files = glob.glob(os.path.join(output, f"{stem}*.mkv"))
    if not mkv_files:
        print(f"No .mkv file found for {basename}")
        return False
    print(f"Output files: {', '.join(os.path.basename(f) for f in mkv_files)}")
    
    # Step 3: Optionally remove original dat file
    if remove_original:
        try:
            os.remove(dat_file)
            # print(f"Removed original file: {basename}")
        except Exception as e:
            print(f"Could not remove original file {basename}: {e}")
    
    # Step 4: Clean up temporary directories created by dat2mkv.py
    if os.path.exists(temp_dir_name):
        try:
            shutil.rmtree(temp_dir_name)
            # print(f"Removed temp directory: {temp_dir_name}")
        except Exception as e:
            print(f"Could not remove temp directory {temp_dir_name}: {e}")
    
    print(f"Completed processing: {basename}")
    return True

def cleanup_output(output):
    """Clean up the output directory if it's empty"""
    if os.path.exists(output):
        try:
            # Check if directory is empty
            if not os.listdir(output):
                os.rmdir(output)
                # print(f"Removed empty output directory: {output}")
        except Exception as e:
            print(f"Could not remove output directory: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch compress .dat files to .mkv locally.")
    parser.add_argument('--input', required=True, help='Directory containing .dat files')
    parser.add_argument('--width', type=int, default=692)
    parser.add_argument('--height', type=int, default=520)
    parser.add_argument('--fps', type=int, default=60)
    parser.add_argument('--workers', type=int, default=1, help='Number of workers (default: 1 for sequential processing)')
    parser.add_argument('--output', type=str, default='.', help='Output directory for .mkv files')
    parser.add_argument('--remove_original', action='store_true', help='Remove original .dat files (default: keep after compression)')

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input directory does not exist: {args.input}")
        sys.exit(1)

    # Step 1: Gather .dat files
    dat_files = [os.path.join(args.input, f) for f in os.listdir(args.input)
                 if f.endswith(".dat") and (f.startswith("raw") or f.startswith("rgb"))]

    print(f"Found {len(dat_files)} .dat files")

    if not dat_files:
        print("No .dat files found to compress")
        sys.exit(0)

    os.makedirs(args.output, exist_ok=True)

    # Step 2: Process files one by one or in parallel
    if args.workers == 1:
        print("Processing files sequentially (compress → clean)...")
        successful = 0
        try:
            for dat_file in dat_files:
                if compress_file(dat_file, args.output, args.width, args.height, args.fps, args.remove_original):
                    successful += 1
        except KeyboardInterrupt:
            print("\n Process interrupted by user. Exiting...")
            sys.exit(1)
            
        print(f"Processing complete: {successful}/{len(dat_files)} files processed successfully")
    else:
        print(f"Processing files with {args.workers} workers...")
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)
        try:
            futures = [
                executor.submit(compress_file, dat_file, args.output, args.width, args.height, args.fps, args.remove_original)
                for dat_file in dat_files
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except concurrent.futures.CancelledError:
                    pass
            
            successful = sum(1 for result in results if result)
            print(f"Processing complete: {successful}/{len(dat_files)} files processed successfully")
            
        except KeyboardInterrupt:
            print("\n Process interrupted by user. Stopping workers...")
            # Cancel all pending futures
            for f in futures:
                f.cancel()
            # Shutdown executor immediately without waiting for running tasks
            executor.shutdown(wait=False, cancel_futures=True)
            print("Exiting...")
            sys.exit(1)
        finally:
            # Ensure executor is properly shut down
            executor.shutdown(wait=True)

    # Step 3: Final cleanup
    cleanup_output(args.output)
    
    print("All operations completed!")
