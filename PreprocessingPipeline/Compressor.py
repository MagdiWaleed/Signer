import os
import tarfile
from tqdm import tqdm
import time  


def compressFolder(base_folder, gloss_names):
  for gloss_name in gloss_names:
    print(f"///////////////////// Processing {gloss_name} //////////////////// ")
    orig_dir = os.path.join(base_folder, "ORIGINAL_DATA", gloss_name)
    dest_dir = os.path.join(base_folder, "COMPRESSED_DATA", gloss_name)

    os.makedirs(dest_dir, exist_ok=True)
    os.chmod(dest_dir, 0o777)  

    all_files = [f for f in os.listdir(orig_dir)
                if os.path.isfile(os.path.join(orig_dir, f))]
    all_files.sort()

    print(f"Found {len(all_files)} files in {orig_dir}")
    print("First 5 files:", all_files[:5])  

    chunk_size = 50
    total_chunks = (len(all_files) // chunk_size) + 1

    for chunk_idx in range(total_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size
        batch = all_files[start:end]

        if not batch:
            continue

        tar_filename = f"{gloss_name}_part{chunk_idx+1}.tar.gz"
        tar_path = os.path.join(dest_dir, tar_filename)

        print(f"\nProcessing chunk {chunk_idx+1}/{total_chunks} -> {tar_path}")
        print(f"Files in batch: {len(batch)}")

        try:
            start_time = time.time()

            with tarfile.open(tar_path, mode='w:gz', compresslevel=6) as tar:
                for fname in tqdm(batch, desc="Adding files"):
                    src_path = os.path.join(orig_dir, fname)
                    if not os.path.exists(src_path):
                        print(f"! Missing: {src_path}")
                        continue

                    try:
                        with open(src_path, 'rb') as test_file:
                            test_file.read(1)  
                    except IOError as e:
                        print(f"! Unreadable: {src_path} - {str(e)}")
                        continue

                    tar.add(src_path, arcname=fname, recursive=False)

            if not os.path.exists(tar_path):
                raise RuntimeError("Archive file not created!")

            archive_size = os.path.getsize(tar_path) / (1024*1024)  # in MB
            elapsed = time.time() - start_time

            print(f"✓ Success: {tar_filename} ({archive_size:.2f} MB)")
            print(f"  Time: {elapsed:.2f}s | {len(batch)/elapsed:.2f} files/sec")

        except Exception as e:
            print(f"❌ Critical error in chunk {chunk_idx+1}: {str(e)}")
            if 'tar_path' in locals() and os.path.exists(tar_path):
                os.remove(tar_path)

    print("\n=== Compression Report ===")
    created_files = [f for f in os.listdir(dest_dir) if f.endswith('.tar.gz')]
    if created_files:
        print(f"Created {len(created_files)} archives:")
        for f in sorted(created_files):
            size = os.path.getsize(os.path.join(dest_dir, f)) / (1024*1024)
            print(f"• {f} ({size:.2f} MB)")
    else:
        print("⚠️ No archives were created!")

