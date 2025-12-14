import os
import shutil

def clean_recursive(root_dir):
    # 3 类需要删除的文件夹名
    # folders_to_delete = {"casia_v1", "columbia", "coverage", "nist2016"}

    # 递归遍历
    for current_root, dirs, files in os.walk(root_dir, topdown=False):

        # ----------------------------
        # 删除规则 1：events.out.tfevents.*
        # ----------------------------
        for f in files:
            if f.startswith("events.out.tfevents"):
                fpath = os.path.join(current_root, f)
                print(f"[DELETE FILE] {fpath}")
                os.remove(fpath)

        # ----------------------------
        # 删除规则 2 & 3：文件夹
        # topdown=False 使得可以安全删除
        # ----------------------------
        # for d in dirs:
        #     dpath = os.path.join(current_root, d)

        #     # 规则 2：pixel-level F1*
        #     if d.startswith("pixel-level F1"):
        #         print(f"[DELETE DIR] {dpath}")
        #         shutil.rmtree(dpath)
        #         continue

        #     # 规则 3：指定文件夹名称
        #     if d in folders_to_delete:
        #         print(f"[DELETE DIR] {dpath}")
        #         shutil.rmtree(dpath)
        #         continue


if __name__ == "__main__":
    root = "output_dir_bs16"
    clean_recursive(root)
