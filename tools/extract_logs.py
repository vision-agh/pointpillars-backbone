import os
import os.path as osp
import glob
import shutil

OUT_DIR = 'pp_logs'
RUNS_DIR = '.'

def main():
    runs_dirs = [ x for x in glob.glob(osp.join(RUNS_DIR, '*')) if osp.isdir(x) ]
    for run_dir in runs_dirs:
        out_run_dir = osp.join(OUT_DIR, osp.basename(run_dir))
        os.makedirs(out_run_dir, exist_ok=True)
        log_files = []
        log_files.extend(glob.glob(osp.join(run_dir, '*.log')))
        log_files.extend(glob.glob(osp.join(run_dir, '*.log.json')))
        log_files.extend(glob.glob(osp.join(run_dir, 'tf_logs')))
        for in_path in log_files:
            out_path = osp.join(
                out_run_dir,
                osp.basename(in_path),
            )
            if osp.isfile(in_path):
                shutil.copy(in_path, out_path)
            else:
                shutil.copytree(in_path, out_path)


if __name__ == "__main__":
    main()
