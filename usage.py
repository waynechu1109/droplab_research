import sys
import argparse
import open3d as o3d
sys.path.append('./dust3r')

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import numpy as np
import trimesh

VIS = False

# images_list = [
#     'dtu_scan24/images/000000.png', 
#     'dtu_scan24/images/000001.png'
#     'dtu_scan24/images/000002.png'
# ]

# images_list = ['dust3r/croco/assets/Chateau1.png', 'dust3r/croco/assets/Chateau2.png']

images_list = [
    # f'dust3r/data/co3d_subset/apple/110_13051_23361/images/frame{num:06d}.jpg'
    # for num in range(1,202,30)

    # f'dust3r/data/co3d_subset/car/621_101777_202473/images/frame{num:06d}.jpg'
    # for num in range(1,202,30)

    # 'data/church_1.jpg',
    # 'data/church_2.jpg'

    # 'data/arc_1.jpg',
    # 'data/arc_2.jpg'

    'data/shoes.jpg',
    'data/shoes.jpg'
]

def save_ply(filename, points):
    with open(filename, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def save_colored_ply(filename, points, colors):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dust3r inference and point cloud downsampling')
    parser.add_argument('--max_points', type=int, default=75000,
                        help='the max number of points after downsample')
    parser.add_argument('--voxel_size', type=float, default=0.00001,
                        help='Voxel Grid 大小，建議佔 bbox 對角線的 1% 左右')
    args = parser.parse_args()


    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    # load the model
    model_name = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory
    images = load_images(images_list, size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)

    # inference here
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    # next we'll use the global_aligner to align the predictions
    # depending on your task, you may be fine with the raw output and not need it
    # with only two input images, you could use GlobalAlignerMode.PairViewer: it would just convert the output
    # if using GlobalAlignerMode.PairViewer, no need to run compute_global_alignment
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # print(f'view1: {view1}')
    # print(f'view2: {view2}')
    # print(f'pred1: {pred1}')
    # print(f'pred2: {pred2}')

    # print(f'type(focals): {type(focals)}')
    # print(f'size(focals): {focals.size()}')
    # print(f'focals: {focals}')

    # print(f'type(poses): {type(poses)}')
    # print(f'size(poses): {poses.size()}')
    # print(f'poses: {poses}')

    # print(f'type(pts3d): {type(pts3d)}')
    # for i, p in enumerate(pts3d):
    #     print(f'shape of pts3d[{i}]: {p.shape}')
    # print(f'pts3d: {pts3d}')

    # print(f'type(confidence_masks): {type(confidence_masks)}')
    # for i, p in enumerate(confidence_masks):
    #     print(f'shape of confidence_masks[{i}]: {p.shape}')
    # print(f'confidence_masks: {confidence_masks}')



    # visualize reconstruction
    # scene.show()

    # save the point cloud
    all_pts3d = []
    all_colors = []
    for i in range(len(images_list)):
        pts3d_tensor = pts3d[i].detach().cpu()
        mask = confidence_masks[i].cpu().numpy()
        pts3d_np = pts3d_tensor.numpy()
        pts3d_filtered = pts3d_np[mask]
        colors = (imgs[i][mask] * 255).astype(np.uint8)

        all_pts3d.append(pts3d_filtered)
        all_colors.append(colors)

    # combined to a single array
    all_pts3d = np.concatenate(all_pts3d, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # # 歸一化到 [-1, 1]
    # # 計算每個維度的最小與最大值
    # mins = all_pts3d.min(axis=0)
    # maxs = all_pts3d.max(axis=0)
    # # 中心與半範圍
    # centres = (maxs + mins) / 2.0
    # scales = (maxs - mins) / 2.0
    # # 防止除以零
    # scales[scales == 0] = 1.0
    # # 執行歸一化
    # all_pts3d = (all_pts3d - centres) / scales

    # --- 下采樣開始 ---
    # 隨機抽樣
    N = all_pts3d.shape[0]
    if N > args.max_points:
        idx = np.random.choice(N, size=args.max_points, replace=False)
        all_pts3d = all_pts3d[idx]
        all_colors = all_colors[idx]

    # Voxel Grid Down Sampling
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts3d)
    pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(float) / 255.0)

    # 先下採樣
    pcd_down = pcd.voxel_down_sample(voxel_size=args.voxel_size)

    # 再估算法向量（重要！）
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    pcd_down.orient_normals_consistent_tangent_plane(k=30)

    # 儲存帶顏色與法向量的點雲
    o3d.io.write_point_cloud("data/output_pointcloud_.ply", pcd_down)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(all_pts3d)
    # pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(float) / 255.0)

    # # Calculate Normals
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    # pcd_down = pcd.voxel_down_sample(voxel_size=args.voxel_size)
    # all_pts3d = np.asarray(pcd_down.points)
    # all_colors = (np.asarray(pcd_down.colors) * 255).astype(np.uint8)
    # # --- 下采樣結束 ---

    # # save_colored_ply("data/output_pointcloud_.ply", all_pts3d, all_colors)
    # o3d.io.write_point_cloud("data/output_pointcloud_.ply", all_pts3d)
        


    if VIS:   
        # find 2D-2D matches between the two images
        from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
        pts2d_list, pts3d_list = [], []
        for i in range(len(images_list)):
            conf_i = confidence_masks[i].cpu().numpy()
            pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
            pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
        reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
        print(f'found {num_matches} matches')
        matches_im1 = pts2d_list[1][reciprocal_in_P2]
        matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]


        # visualize a few matches
        from matplotlib import pyplot as pl
        n_viz = 10
        match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
        viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

        H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
        img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = np.concatenate((img0, img1), axis=1)
        pl.figure()
        pl.imshow(img)
        cmap = pl.get_cmap('jet')
        for i in range(n_viz):
            (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
            pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
        pl.show(block=True)


