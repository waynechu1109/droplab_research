import open3d as o3d

filepath = "data/output_pointcloud_all.ply"
pcd = o3d.io.read_point_cloud(filepath)

app = o3d.visualization.gui.Application.instance
app.initialize()

win = o3d.visualization.O3DVisualizer(filepath, 1024, 768)
win.add_geometry("pointcloud", pcd)
app.add_window(win)
app.run()
