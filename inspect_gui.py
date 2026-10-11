#!/usr/bin/env python
# coding=utf-8
"""
数据集可视化筛选工具（组级别筛选）

每次显示连续 4 帧（一组），独立决定是否保留这组。
最终只把被标记的那些帧复制到输出数据集。

每帧三联图：[ VI 彩色 ] [ IR 灰度 ] [ 融合彩色 ]

快捷键：
  ← / →      上一组 / 下一组
  Ctrl+← / → 上一个文件夹 / 下一个文件夹
  Enter       标记/取消 当前组
  Space       跳过当前组（不保留），继续下一组
  S           保存所有已标记帧到 LLVIP331
"""

import os
import sys
import shutil
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from collections import defaultdict

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageTk

# ============================================================
VI_ROOT  = r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\visible2"
IR_ROOT  = r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\infrared"
GT_ROOT  = r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP\Fusion"
OUT_ROOT = r"E:\paper2\codebase\STAR\Fusion\dataset\LLVIP331"

VI_NAME  = "visible2"
IR_NAME  = "infrared"
GT_NAME  = "Fusion"

GROUP    = 4
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp')
THUMB_W  = 310
THUMB_H  = 210
# ============================================================


def get_sorted_images(folder: Path):
    return sorted([f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTS])


def make_blend_color(vi_bgr, ir_bgr):
    h, w = vi_bgr.shape[:2]
    ir_bgr = cv2.resize(ir_bgr, (w, h))
    ir_gray = cv2.cvtColor(ir_bgr, cv2.COLOR_BGR2GRAY)
    ycrcb = cv2.cvtColor(vi_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    ycrcb[:, :, 0] = (ycrcb[:, :, 0] + ir_gray.astype(np.float32)) / 2
    return cv2.cvtColor(np.clip(ycrcb, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def bgr_to_tk(bgr, w, h):
    rgb = cv2.cvtColor(cv2.resize(bgr, (w, h)), cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class App(tk.Tk):
    def __init__(self, folders):
        super().__init__()
        self.title("数据集筛选工具（组级别）")
        self.configure(bg="#11111b")
        self.resizable(True, True)

        self.folders  = folders
        self.fidx     = 0
        self.gidx     = 0
        self.vi_files = []
        self.ir_files = []
        self.gt_files = []
        self.n_groups = 0
        self._refs    = []

        # kept: { folder_idx: set of group_idx }
        self.kept  = defaultdict(set)
        # saved: 已经写入磁盘的文件夹索引（不会重复保存）
        self.saved = set()

        self._build_ui()
        self._load_folder()

    # ──────────────────────────────────────────────────────────
    def _build_ui(self):
        # ── 顶栏 ──────────────────────────────────────────────
        top = tk.Frame(self, bg="#1e1e2e", pady=6)
        top.pack(fill=tk.X)

        self.lbl_folder = tk.Label(
            top, text="", font=("Consolas", 12, "bold"),
            fg="#cdd6f4", bg="#1e1e2e", anchor="w")
        self.lbl_folder.pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        self.lbl_stat = tk.Label(
            top, text="", font=("Consolas", 11),
            fg="#a6e3a1", bg="#1e1e2e")
        self.lbl_stat.pack(side=tk.RIGHT, padx=12)

        # ── 当前组状态栏 ──────────────────────────────────────
        self.frm_group_status = tk.Frame(self, bg="#181825", pady=4)
        self.frm_group_status.pack(fill=tk.X)

        self.lbl_group_info = tk.Label(
            self.frm_group_status, text="", font=("微软雅黑", 11, "bold"),
            fg="#f9e2af", bg="#181825")
        self.lbl_group_info.pack(side=tk.LEFT, padx=14)

        self.lbl_group_kept = tk.Label(
            self.frm_group_status, text="", font=("微软雅黑", 11),
            fg="#a6e3a1", bg="#181825")
        self.lbl_group_kept.pack(side=tk.LEFT, padx=20)

        # ── 列标题 ────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#181825")
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="", width=6, bg="#181825").pack(side=tk.LEFT)
        for title in ["VI  彩色", "IR  灰度", "融合  彩色"]:
            tk.Label(hdr, text=title, font=("微软雅黑", 10, "bold"),
                     fg="#89b4fa", bg="#181825",
                     width=THUMB_W // 8).pack(side=tk.LEFT, expand=True)

        # ── 图像区 ────────────────────────────────────────────
        self.img_frame = tk.Frame(self, bg="#11111b")
        self.img_frame.pack(fill=tk.BOTH, expand=True)

        self.rows = []
        for _ in range(GROUP):
            row_f = tk.Frame(self.img_frame, bg="#11111b", pady=2)
            row_f.pack(fill=tk.X)
            lbl_idx = tk.Label(row_f, text="", font=("Consolas", 9),
                               fg="#6c7086", bg="#11111b", width=7)
            lbl_idx.pack(side=tk.LEFT, padx=4)
            lbls = []
            for _ in range(3):
                lbl = tk.Label(row_f, bg="#11111b", relief="flat")
                lbl.pack(side=tk.LEFT, padx=3, pady=2)
                lbls.append(lbl)
            self.rows.append((lbl_idx, *lbls))

        # ── 进度条 ────────────────────────────────────────────
        prog = tk.Frame(self, bg="#1e1e2e", pady=2)
        prog.pack(fill=tk.X)
        self.pbar = ttk.Progressbar(prog, mode="determinate", length=600)
        self.pbar.pack(side=tk.LEFT, padx=12)
        self.lbl_pbar = tk.Label(prog, text="", font=("Consolas", 10),
                                 fg="#bac2de", bg="#1e1e2e")
        self.lbl_pbar.pack(side=tk.LEFT, padx=8)

        # ── 底部按钮 ──────────────────────────────────────────
        bot = tk.Frame(self, bg="#1e1e2e", pady=8)
        bot.pack(fill=tk.X)

        def btn(parent, text, bg, fg, cmd, width=11):
            return tk.Button(parent, text=text, bg=bg, fg=fg, command=cmd,
                             font=("微软雅黑", 10, "bold"), width=width,
                             relief="flat", cursor="hand2", activebackground=bg)

        btn(bot, "◀ 上一组", "#313244", "#cdd6f4", self._prev_group).pack(side=tk.LEFT, padx=6)
        btn(bot, "下一组 ▶", "#313244", "#cdd6f4", self._next_group).pack(side=tk.LEFT, padx=2)

        tk.Label(bot, text="  |  ", fg="#45475a", bg="#1e1e2e",
                 font=("Consolas", 12)).pack(side=tk.LEFT)

        self.btn_keep = btn(bot, "✓ 保留本组", "#a6e3a1", "#1e1e2e", self._toggle_keep, width=12)
        self.btn_keep.pack(side=tk.LEFT, padx=6)

        btn(bot, "跳过本组 →", "#45475a", "#cdd6f4", self._skip_group, width=12
            ).pack(side=tk.LEFT, padx=2)

        tk.Label(bot, text="  |  ", fg="#45475a", bg="#1e1e2e",
                 font=("Consolas", 12)).pack(side=tk.LEFT)

        btn(bot, "◀◀ 上个夹", "#45475a", "#cdd6f4", self._prev_folder).pack(side=tk.LEFT, padx=6)
        btn(bot, "下个夹 ▶▶", "#45475a", "#cdd6f4", self._next_folder).pack(side=tk.LEFT, padx=2)

        self.lbl_total_kept = tk.Label(bot, text="已选: 0 组", font=("微软雅黑", 11),
                                       fg="#fab387", bg="#1e1e2e")
        self.lbl_total_kept.pack(side=tk.RIGHT, padx=8)

        # 存本文件夹（主要操作，快捷键 D）
        self.btn_save_cur = btn(
            bot, "💾 存本文件夹", "#f9e2af", "#1e1e2e",
            self._save_current_folder, width=14)
        self.btn_save_cur.pack(side=tk.RIGHT, padx=6)

        self.pbar_save = ttk.Progressbar(bot, length=100, mode="determinate")
        self.pbar_save.pack(side=tk.RIGHT, padx=4)

        # 快捷键
        self.bind("<Left>",          lambda e: self._prev_group())
        self.bind("<Right>",         lambda e: self._next_group())
        self.bind("<Control-Left>",  lambda e: self._prev_folder())
        self.bind("<Control-Right>", lambda e: self._next_folder())
        self.bind("<Return>",        lambda e: self._toggle_keep())
        self.bind("<space>",         lambda e: self._skip_group())
        self.bind("d",               lambda e: self._save_current_folder())

    # ──────────────────────────────────────────────────────────
    def _load_folder(self):
        vi_sub, ir_sub, gt_sub = self.folders[self.fidx]
        self.vi_files = get_sorted_images(vi_sub)
        self.ir_files = get_sorted_images(ir_sub)
        self.gt_files = get_sorted_images(gt_sub) if gt_sub and gt_sub.exists() else []
        self.n_groups = max(1, (len(self.vi_files) + GROUP - 1) // GROUP)
        self.gidx = 0
        self._render()

    def _render(self):
        self._refs.clear()
        vi_sub, _, _ = self.folders[self.fidx]
        n_vi = len(self.vi_files)

        # 当前组是否已保留
        is_kept = self.gidx in self.kept[self.fidx]

        # 顶栏
        folder_kept  = len(self.kept[self.fidx])
        is_saved     = self.fidx in self.saved
        saved_mark   = "  ✅ 已存盘" if is_saved else ""
        self.lbl_folder.config(
            text=f"[{self.fidx+1}/{len(self.folders)}]  {vi_sub.name}"
                 + (f"  （已选 {folder_kept} 组）" if folder_kept else "")
                 + saved_mark
        )
        self.lbl_stat.config(text=f"共 {n_vi} 帧 / {self.n_groups} 组")

        # 存本文件夹按钮状态
        if is_saved:
            self.btn_save_cur.config(bg="#45475a", text="✅ 本夹已存盘")
        elif folder_kept:
            self.btn_save_cur.config(bg="#f9e2af", text="💾 存本文件夹")
        else:
            self.btn_save_cur.config(bg="#313244", text="（本夹无选组）")

        # 组状态栏
        self.lbl_group_info.config(
            text=f"第 {self.gidx+1} / {self.n_groups} 组  "
                 f"（帧 {self.gidx*GROUP:04d} ~ {min((self.gidx+1)*GROUP-1, n_vi-1):04d}）"
        )
        if is_kept:
            self.lbl_group_kept.config(text="✓ 本组已保留", fg="#a6e3a1")
            self.btn_keep.config(bg="#313244", text="取消保留")
        else:
            self.lbl_group_kept.config(text="  本组未保留", fg="#6c7086")
            self.btn_keep.config(bg="#a6e3a1", text="✓ 保留本组")

        # 进度条
        self.pbar["maximum"] = self.n_groups
        self.pbar["value"]   = self.gidx + 1
        self.lbl_pbar.config(text=f"第 {self.gidx+1} / {self.n_groups} 组")

        # 全局已选计数
        total = sum(len(v) for v in self.kept.values())
        self.lbl_total_kept.config(text=f"已选: {total} 组")

        # 图像渲染
        base = self.gidx * GROUP
        for r in range(GROUP):
            fidx_frame = base + r
            lbl_idx, lbl_vi, lbl_ir, lbl_blend = self.rows[r]

            if fidx_frame >= n_vi:
                lbl_idx.config(text="  ---")
                for lbl in (lbl_vi, lbl_ir, lbl_blend):
                    lbl.config(image="", bg="#11111b",
                               width=THUMB_W, height=THUMB_H)
                continue

            ir_idx = min(fidx_frame, len(self.ir_files) - 1)
            vi_bgr = cv2.imread(str(self.vi_files[fidx_frame]))
            ir_bgr = cv2.imread(str(self.ir_files[ir_idx]))
            if vi_bgr is None or ir_bgr is None:
                lbl_idx.config(text=" ERR")
                continue

            h, w = vi_bgr.shape[:2]
            ir_resized = cv2.resize(ir_bgr, (w, h))
            ir_gray3   = cv2.cvtColor(
                cv2.cvtColor(ir_resized, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            blend = make_blend_color(vi_bgr, ir_bgr)

            # 已保留的组加绿色边框
            border = 4 if is_kept else 0
            color  = (100, 220, 100)

            lbl_idx.config(text=f" {fidx_frame:04d}")
            for lbl, img in zip((lbl_vi, lbl_ir, lbl_blend),
                                (vi_bgr, ir_gray3, blend)):
                display = img.copy()
                if border:
                    cv2.rectangle(display, (0, 0),
                                  (display.shape[1]-1, display.shape[0]-1),
                                  color, border)
                ph = bgr_to_tk(display, THUMB_W, THUMB_H)
                lbl.config(image=ph)
                lbl.image = ph
                self._refs.append(ph)

    # ──────────────────────────────────────────────────────────
    def _prev_group(self):
        if self.gidx > 0:
            self.gidx -= 1
            self._render()

    def _next_group(self):
        if self.gidx < self.n_groups - 1:
            self.gidx += 1
            self._render()

    def _toggle_keep(self):
        if self.gidx in self.kept[self.fidx]:
            self.kept[self.fidx].discard(self.gidx)
        else:
            self.kept[self.fidx].add(self.gidx)
        self._render()

    def _skip_group(self):
        """不保留当前组，直接去下一组。"""
        self.kept[self.fidx].discard(self.gidx)
        self._next_group()

    def _prev_folder(self):
        if self.fidx > 0:
            self.fidx -= 1
            self._load_folder()

    def _next_folder(self):
        if self.fidx < len(self.folders) - 1:
            self.fidx += 1
            self._load_folder()

    # ──────────────────────────────────────────────────────────
    def _save_current_folder(self):
        """保存当前文件夹已选的组，存完标记为已存，不会重复。"""
        if self.fidx in self.saved:
            messagebox.showinfo("提示", "本文件夹已经存过了，无需重复保存。")
            return
        group_set = self.kept[self.fidx]
        if not group_set:
            messagebox.showwarning("提示", "本文件夹还没有选任何组！")
            return

        vi_sub, ir_sub, gt_sub = self.folders[self.fidx]
        gt_root = Path(GT_ROOT) if GT_ROOT and Path(GT_ROOT).exists() else None
        out = Path(OUT_ROOT)
        name = vi_sub.name

        vi_files = self.vi_files
        ir_files = self.ir_files
        gt_files = get_sorted_images(gt_sub) \
            if gt_root and gt_sub and gt_sub.exists() else []

        out_vi = out / VI_NAME / name
        out_ir = out / IR_NAME / name
        out_gt = (out / GT_NAME / name) if gt_files else None
        out_vi.mkdir(parents=True, exist_ok=True)
        out_ir.mkdir(parents=True, exist_ok=True)
        if out_gt:
            out_gt.mkdir(parents=True, exist_ok=True)

        n = len(group_set)
        self.pbar_save["maximum"] = n
        self.pbar_save["value"]   = 0

        for i, gidx in enumerate(sorted(group_set)):
            base = gidx * GROUP
            for r in range(GROUP):
                fi    = base + r
                ir_fi = min(fi, len(ir_files) - 1)
                if fi >= len(vi_files):
                    break
                shutil.copy2(str(vi_files[fi]),    str(out_vi / vi_files[fi].name))
                shutil.copy2(str(ir_files[ir_fi]), str(out_ir / ir_files[ir_fi].name))
                if gt_files and fi < len(gt_files):
                    shutil.copy2(str(gt_files[fi]), str(out_gt / gt_files[fi].name))
            self.pbar_save["value"] = i + 1
            self.update_idletasks()

        self.saved.add(self.fidx)
        self.pbar_save["value"] = 0
        self._render()   # 刷新顶栏显示"已存盘"

    # ──────────────────────────────────────────────────────────
    def _save_dataset(self):
        """保存所有尚未存盘的已选文件夹（跳过已存过的）。"""
        pending = {k: v for k, v in self.kept.items()
                   if v and k not in self.saved}
        total_groups = sum(len(v) for v in pending.values())
        if total_groups == 0:
            messagebox.showinfo("提示", "所有已选文件夹均已存盘，无需重复保存。")
            return

        out = Path(OUT_ROOT)
        gt_root = Path(GT_ROOT) if GT_ROOT and Path(GT_ROOT).exists() else None

        def do_copy():
            self.pbar_save["maximum"] = total_groups
            self.pbar_save["value"]   = 0
            count = 0

            for folder_idx, group_set in sorted(pending.items()):
                if not group_set:
                    continue
                vi_sub, ir_sub, gt_sub = self.folders[folder_idx]
                name = vi_sub.name

                vi_files = get_sorted_images(vi_sub)
                ir_files = get_sorted_images(ir_sub)
                gt_files = get_sorted_images(gt_sub) \
                    if gt_root and gt_sub and gt_sub.exists() else []

                # 输出子目录
                out_vi = out / VI_NAME / name
                out_ir = out / IR_NAME / name
                out_gt = (out / GT_NAME / name) if gt_files else None
                out_vi.mkdir(parents=True, exist_ok=True)
                out_ir.mkdir(parents=True, exist_ok=True)
                if out_gt:
                    out_gt.mkdir(parents=True, exist_ok=True)

                for gidx in sorted(group_set):
                    base = gidx * GROUP
                    for r in range(GROUP):
                        fi = base + r
                        if fi >= len(vi_files):
                            break
                        ir_fi = min(fi, len(ir_files) - 1)

                        shutil.copy2(str(vi_files[fi]),
                                     str(out_vi / vi_files[fi].name))
                        shutil.copy2(str(ir_files[ir_fi]),
                                     str(out_ir / ir_files[ir_fi].name))
                        if gt_files and fi < len(gt_files):
                            shutil.copy2(str(gt_files[fi]),
                                         str(out_gt / gt_files[fi].name))

                    count += 1
                    self.pbar_save["value"] = count
                    self.update_idletasks()

                self.saved.add(folder_idx)

            messagebox.showinfo(
                "完成",
                f"已保存 {total_groups} 组（{total_groups * GROUP} 帧）到:\n{OUT_ROOT}"
            )
            self.pbar_save["value"] = 0
            self._render()

        threading.Thread(target=do_copy, daemon=True).start()


# ============================================================
def main():
    vi_root = Path(VI_ROOT)
    ir_root = Path(IR_ROOT)
    gt_root = Path(GT_ROOT) if GT_ROOT and Path(GT_ROOT).exists() else None

    vi_subdirs = sorted([d for d in vi_root.iterdir() if d.is_dir()])
    folders = []
    for vi_sub in vi_subdirs:
        ir_sub = ir_root / vi_sub.name
        if not (ir_sub.exists() and ir_sub.is_dir()):
            continue
        gt_sub = (gt_root / vi_sub.name) if gt_root else None
        folders.append((vi_sub, ir_sub, gt_sub))

    if not folders:
        print("未找到匹配子文件夹，请检查路径。")
        sys.exit(1)

    print(f"共 {len(folders)} 个子文件夹")
    print("← → 翻组  Ctrl+← → 换文件夹  Enter=保留本组  Space=跳过  S=保存")
    App(folders).mainloop()


if __name__ == "__main__":
    main()
