from simulator import *
import os
import numpy as np

res_x = 1024 # pixels
res_y = 1024
# res_y = 1920 # pixels

# normalized focal length
FOV = 10
f = 0.5 / np.tan(np.deg2rad(FOV) / 2)

# pixel aspect ratio
pixel_ar = 1

# normalized principal point
ppx = 0.5
ppy = 0.5

# gaussian_noise_sigma_stack = np.arange(0.000035, 0.00035, 0.000035) # rad
# gaussian_noise_sigma_stack = 0
gaussian_noise_sigma = 0

cam = 0

#%%

# magnitude parameters

A_pixel = 525 # photonelectrons/s mm
sigma_pixel = 525 # photonelectrons/s mm

sigma_psf = 0.5 # pixel
t_exp = 0.2 # s
aperture = 15 # mm

base_photons = 19100 # photoelectrons per mm² and second of a magnitude 0 G2 star

# magnitude_gaussian = 0.3 # mag
magnitude_gaussian = 0.00 # mag

#%%

# star count

min_true = 2
max_true = 100
min_false = 0
max_false = 0
min_stars = 2

catalog = StarCatalog()
hip = catalog.catalog.HIP.values.tolist()

cat_star_vec = catalog.star_vectors
cat_star_mag = catalog.magnitudes

cameras = [
    RectilinearCamera,
    EquidistantCamera,
    EquisolidAngleCamera,
    StereographicCamera,
    OrthographicCamera,
]

camera = cameras[cam](f, (res_x, res_y), pixel_ar, (ppx, ppy))

#%%

detector = StarDetector(A_pixel, sigma_pixel, sigma_psf, t_exp, aperture, base_photons)
num_scenes = 1

# save text file
file = open("HIP_catalog_vectors_with_mag_less_than_6","a")
min_mag = np.min(cat_star_mag)
#
output_dir = "./output"

for scene_id in range(num_scenes):
    scene = Scene.random(catalog, camera, detector, min_true, max_true, min_false, max_false, min_stars, gaussian_noise_sigma=gaussian_noise_sigma, magnitude_gaussian=magnitude_gaussian)

    # retrieve groundtruth star_vector
    vec_ids = []
    gt_mag = []
    gt_vec = np.zeros((len(scene.ids), 3))
    for i in range(len(scene.ids)):
        if (scene.ids[i] == -1):
            gt_mag.append(0)
            gt_vec[i, :] = (np.array([0, 0, 0]))
            vec_ids.append(-1)
        else:
            vec_ids.append(hip.index(scene.ids[i]))
            gt_mag.append((cat_star_mag[vec_ids[i]] - min_mag) + 0.1) # try to scale the magnitude to have a minimum of 0.1
            gt_vec[i, :] = cat_star_vec[vec_ids[i]]

    gt_rotation_matrix = scene.orientation
    gt_rotation_matrix = np.transpose(gt_rotation_matrix)

    # project xy to vec, then rotate it
    cam_coord_vec = []
    rotated_coord_vec = []
    dist = []
    for i in range(len(scene.pos)):
        curr_ang = camera.to_angles(scene.pos[i][np.newaxis, :])
        curr_vec = angles_to_vector(curr_ang[0], curr_ang[1])
        mag_curr_vec = (scene.magnitudes[i] - min_mag + 0.1) * curr_vec # try to scale the magnitude to have a minimum of 0.1
        cam_coord_vec.append(mag_curr_vec)
        # the groundtruth rotation will align the generated stars to the groundtruth position (in a fixed coordinate)

    # comparing eul distance
    # save generated scene and GT ID and GT orientation
    # create file directory

    os.makedirs(output_dir, exist_ok=True)
    # tmp = curr_path + "/" + str(scene_id)
    stars_file = open(output_dir + "/stars_" + str(scene_id),"w")
    GT_id_file = open(output_dir + "/id_" + str(scene_id),"w")
    #
    for i in range(len(cam_coord_vec)):
        curr_x = cam_coord_vec[i][0][0]
        curr_y = cam_coord_vec[i][0][1]
        curr_z = cam_coord_vec[i][0][2]
        tmp_str = str(curr_x) + " " + str(curr_y) + " " + str(curr_z) + "\n"
        stars_file.write(tmp_str)
        GT_id_file.write(str(vec_ids[i]) + "\n")

    stars_file.close()
    GT_id_file.close()

    GT_rot_file = open(output_dir + "/rot_" + str(scene_id),"w")
    GT_rot_file.write(str(gt_rotation_matrix[0, 0]) + " " + str(gt_rotation_matrix[0, 1]) + " " + str(gt_rotation_matrix[0, 2]) +"\n")
    GT_rot_file.write(str(gt_rotation_matrix[1, 0]) + " " + str(gt_rotation_matrix[1, 1]) + " " + str(gt_rotation_matrix[1, 2]) +"\n")
    GT_rot_file.write(str(gt_rotation_matrix[2, 0]) + " " + str(gt_rotation_matrix[2, 1]) + " " + str(gt_rotation_matrix[2, 2]) +"\n")
    GT_rot_file.close()
