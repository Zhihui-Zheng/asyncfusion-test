"""检查数据集文件夹匹配性：子文件夹名称、图片数量、图片文件名"""
import os
from pathlib import Path
from collections import defaultdict


def check_dataset(root_dir, folder_names, dataset_name="Dataset"):
    """
    检查一个数据集下多个文件夹是否匹配
    
    Args:
        root_dir: 数据集根目录
        folder_names: 子文件夹名称列表 (如 ['visible1', 'infrared', 'Fusion_good', 'Fusion_bad'])
        dataset_name: 数据集名称 (用于打印)
    """
    print("=" * 70)
    print(f"  检查数据集: {dataset_name}")
    print(f"  路径: {root_dir}")
    print(f"  文件夹: {folder_names}")
    print("=" * 70)
    
    all_ok = True
    
    # 1. 收集每个文件夹的子文件夹和图片信息
    folder_data = {}  # folder_name -> {subfolder_name -> [image_names]}
    
    for fname in folder_names:
        fpath = Path(root_dir) / fname
        if not fpath.exists():
            print(f"  ❌ 文件夹不存在: {fpath}")
            all_ok = False
            continue
        
        subfolders = {}
        for sub in sorted(fpath.iterdir()):
            if sub.is_dir():
                images = sorted([
                    f.name for f in sub.iterdir()
                    if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                ])
                subfolders[sub.name] = images
        
        folder_data[fname] = subfolders
        print(f"\n  [{fname}] {len(subfolders)} 个子文件夹")
    
    if len(folder_data) < 2:
        print("\n  ⚠️ 有效文件夹不足2个，无法比较")
        return
    
    # 2. 检查子文件夹名称是否匹配
    print(f"\n{'─' * 70}")
    print("  [1] 子文件夹名称匹配检查")
    print(f"{'─' * 70}")
    
    all_subfolder_sets = {fname: set(data.keys()) for fname, data in folder_data.items()}
    reference_name = folder_names[0]
    reference_set = all_subfolder_sets.get(reference_name, set())
    
    for fname, subfolder_set in all_subfolder_sets.items():
        if fname == reference_name:
            continue
        
        only_in_ref = reference_set - subfolder_set
        only_in_other = subfolder_set - reference_set
        
        if only_in_ref:
            print(f"  ❌ 在 [{reference_name}] 有但 [{fname}] 没有: {sorted(only_in_ref)}")
            all_ok = False
        if only_in_other:
            print(f"  ❌ 在 [{fname}] 有但 [{reference_name}] 没有: {sorted(only_in_other)}")
            all_ok = False
    
    common_subfolders = set.intersection(*all_subfolder_sets.values()) if all_subfolder_sets else set()
    print(f"  共同子文件夹: {len(common_subfolders)} 个")
    
    if common_subfolders == reference_set and all(s == reference_set for s in all_subfolder_sets.values()):
        print(f"  ✅ 所有文件夹的子文件夹名称完全一致 ({len(common_subfolders)} 个)")
    
    # 3. 检查每个子文件夹内的图片数量
    print(f"\n{'─' * 70}")
    print("  [2] 图片数量匹配检查")
    print(f"{'─' * 70}")
    
    count_mismatches = []
    for sub in sorted(common_subfolders):
        counts = {fname: len(data[sub]) for fname, data in folder_data.items()}
        values = list(counts.values())
        if len(set(values)) > 1:
            count_mismatches.append((sub, counts))
    
    if count_mismatches:
        print(f"  ❌ {len(count_mismatches)} 个子文件夹图片数量不一致:")
        for sub, counts in count_mismatches:
            counts_str = ", ".join(f"{k}={v}" for k, v in counts.items())
            print(f"     {sub}: {counts_str}")
        all_ok = False
    else:
        # 统计总图片数
        total_per_folder = {fname: sum(len(imgs) for imgs in data.values()) for fname, data in folder_data.items()}
        totals_str = ", ".join(f"{k}={v}" for k, v in total_per_folder.items())
        print(f"  ✅ 所有子文件夹图片数量一致")
        print(f"     总图片数: {totals_str}")
    
    # 4. 检查图片文件名是否匹配
    print(f"\n{'─' * 70}")
    print("  [3] 图片文件名匹配检查")
    print(f"{'─' * 70}")
    
    name_mismatches = []
    for sub in sorted(common_subfolders):
        image_sets = {fname: set(data[sub]) for fname, data in folder_data.items()}
        all_same = len(set(frozenset(s) for s in image_sets.values())) == 1
        
        if not all_same:
            # 找出差异
            all_images = set.union(*image_sets.values())
            common_images = set.intersection(*image_sets.values())
            diff_count = len(all_images) - len(common_images)
            name_mismatches.append((sub, diff_count, image_sets))
    
    if name_mismatches:
        print(f"  ❌ {len(name_mismatches)} 个子文件夹图片文件名不一致:")
        for sub, diff_count, image_sets in name_mismatches[:10]:  # 最多显示10个
            print(f"     {sub} ({diff_count} 个差异):")
            ref_set = image_sets[reference_name]
            for fname, img_set in image_sets.items():
                if fname == reference_name:
                    continue
                only_ref = ref_set - img_set
                only_other = img_set - ref_set
                if only_ref:
                    print(f"       在[{reference_name}]有但[{fname}]没有: {sorted(only_ref)[:5]}{'...' if len(only_ref)>5 else ''}")
                if only_other:
                    print(f"       在[{fname}]有但[{reference_name}]没有: {sorted(only_other)[:5]}{'...' if len(only_other)>5 else ''}")
        if len(name_mismatches) > 10:
            print(f"     ... 还有 {len(name_mismatches) - 10} 个")
        all_ok = False
    else:
        print(f"  ✅ 所有子文件夹的图片文件名完全一致")
    
    # 5. 检查图片是否能正确分成4帧一组 (组内连续, 组间可跳)
    print(f"\n{'─' * 70}")
    print("  [4] 4帧分组检查 (组内必须连续, 组间允许跳号)")
    print(f"{'─' * 70}")
    
    import re
    group_issues = []
    
    for sub in sorted(common_subfolders):
        ref_fname = list(folder_data.keys())[0]
        images = folder_data[ref_fname][sub]
        
        if len(images) == 0:
            continue
        
        # 提取帧编号
        indices = []
        for img in images:
            nums = re.findall(r'(\d+)', os.path.splitext(img)[0])
            if nums:
                indices.append(int(nums[-1]))
        
        if not indices:
            continue
        
        indices.sort()
        
        # 检查总数是否为4的倍数
        not_div4 = len(indices) % 4 != 0
        
        # 每4帧分一组, 检查组内是否连续
        broken_groups = []
        num_groups = len(indices) // 4
        remainder = len(indices) % 4
        
        for g in range(num_groups):
            group = indices[g*4 : g*4+4]
            # 组内必须连续 (步长=1)
            is_consecutive = all(group[i+1] == group[i] + 1 for i in range(3))
            if not is_consecutive:
                broken_groups.append((g, group))
        
        # 如果有余数, 检查尾部
        if remainder > 0:
            tail = indices[num_groups*4:]
            is_consecutive = all(tail[i+1] == tail[i] + 1 for i in range(len(tail)-1))
            if not is_consecutive:
                broken_groups.append(('tail', tail))
        
        if broken_groups or not_div4:
            group_issues.append({
                'subfolder': sub,
                'total': len(indices),
                'num_groups': num_groups,
                'remainder': remainder,
                'broken_groups': broken_groups,
                'not_div4': not_div4,
            })
    
    if group_issues:
        has_broken = [g for g in group_issues if g['broken_groups']]
        has_div4 = [g for g in group_issues if g['not_div4']]
        
        if has_broken:
            print(f"  ❌ {len(has_broken)} 个子文件夹存在组内不连续:")
            for g in has_broken[:15]:
                for gid, frames in g['broken_groups'][:3]:
                    print(f"     {g['subfolder']} 第{gid}组: {frames} ← 不连续!")
            if len(has_broken) > 15:
                print(f"     ... 还有 {len(has_broken) - 15} 个")
            all_ok = False
        else:
            print(f"  ✅ 所有子文件夹的4帧分组内部均连续")
        
        if has_div4:
            print(f"  ⚠️  {len(has_div4)} 个子文件夹图片数不是4的倍数:")
            for g in has_div4[:10]:
                print(f"     {g['subfolder']}: {g['total']}张 ({g['total']}%4={g['total']%4})")
            if len(has_div4) > 10:
                print(f"     ... 还有 {len(has_div4) - 10} 个")
    else:
        print(f"  ✅ 所有子文件夹均可正确分成4帧一组, 组内连续")
    
    # 6. 检查跨文件夹的帧对齐 (同一子文件夹中, 各文件夹的第N张图是否对应)
    print(f"\n{'─' * 70}")
    print("  [5] 跨文件夹帧对齐检查 (第N张图是否跨文件夹对应)")
    print(f"{'─' * 70}")
    
    align_issues = []
    for sub in sorted(common_subfolders):
        # 每个文件夹中该子文件夹的图片列表 (已排序)
        all_lists = {fname: folder_data[fname][sub] for fname in folder_data}
        
        # 取所有列表长度
        lengths = {fname: len(lst) for fname, lst in all_lists.items()}
        if len(set(lengths.values())) > 1:
            continue  # 数量不一致的已经在前面检查过了
        
        n = list(lengths.values())[0]
        if n == 0:
            continue
        
        # 逐帧检查: 提取编号部分比较
        for i in range(n):
            frame_names = {fname: all_lists[fname][i] for fname in all_lists}
            # 提取编号
            frame_nums = {}
            for fname, img_name in frame_names.items():
                nums = re.findall(r'(\d+)', os.path.splitext(img_name)[0])
                frame_nums[fname] = nums[-1] if nums else img_name
            
            # 检查所有文件夹的编号是否一致
            unique_nums = set(frame_nums.values())
            if len(unique_nums) > 1:
                align_issues.append((sub, i, frame_names))
    
    if align_issues:
        print(f"  ❌ {len(align_issues)} 处帧对齐不一致:")
        for sub, idx, frame_names in align_issues[:10]:
            names_str = ", ".join(f"{k}={v}" for k, v in frame_names.items())
            print(f"     {sub}[{idx}]: {names_str}")
        if len(align_issues) > 10:
            print(f"     ... 还有 {len(align_issues) - 10} 处")
        all_ok = False
    else:
        print(f"  ✅ 所有帧跨文件夹完全对齐")
    
    # 总结
    print(f"\n{'=' * 70}")
    if all_ok:
        print(f"  ✅ {dataset_name} 检查全部通过!")
    else:
        print(f"  ❌ {dataset_name} 存在不匹配项，请检查上方详情")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    # 检查 M3SVD_cropped (4个文件夹)
    check_dataset(
        root_dir=r"E:\paper2\codebase\STAR\Fusion\dataset\train\M3SVD_cropped",
        folder_names=["visible1", "infrared", "Fusion_good", "Fusion_bad"],
        dataset_name="M3SVD_cropped"
    )
    
    # 检查 LLVIP2 (3个文件夹)
    check_dataset(
        root_dir=r"E:\paper2\codebase\STAR\Fusion\dataset\train\LLVIP2",
        folder_names=["visible", "infrared", "Fusion"],
        dataset_name="LLVIP2"
    )

