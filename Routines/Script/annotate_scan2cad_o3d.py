import sys

assert sys.version_info >= (3, 5)

import numpy as np
from pathlib import Path
import os
import shutil
import glob
import JSONHelper
import CSVHelper
import csv
import quaternion
from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import pywavefront
import argparse

import open3d as o3d

# params
parser = argparse.ArgumentParser()
parser.add_argument('--out', default="./meshes/", help="outdir")
opt = parser.parse_args()


def get_catid2index(filename):
    catid2index = {}
    csvfile = open(filename)
    spamreader = csv.DictReader(csvfile, delimiter='\t')
    for row in spamreader:
        try:
            catid2index[row["wnsynsetid"][1:]] = int(row["nyu40id"])
        except:
            pass
    csvfile.close()

    return catid2index


def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M


def calc_Mbbox(model):
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M


def decompose_mat4(M):
    R = M[0:3, 0:3]
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:, 0] /= sx;
    R[:, 1] /= sy;
    R[:, 2] /= sz;
    q = quaternion.from_rotation_matrix(R[0:3, 0:3])

    t = M[0:3, 3]
    return t, q, s

def parse_scan2d_dataset(scan2d_folder, output_dir):
    params_json = Path(scan2d_folder) / "./Parameters.json"
    params = JSONHelper.read(params_json)  # <-- read parameter file (contains dataset paths)

    filename_json = Path(scan2d_folder) / "full_annotations.json"
    if not os.path.exists(filename_json):
        filename_json = Path(scan2d_folder) / "example_annotation.json"

    for r in JSONHelper.read(filename_json):
        id_scan = r["id_scan"]
        if id_scan != "scene0470_00":
            continue

        outdir = os.path.abspath(opt.out + "/" + id_scan)
        Path(outdir).mkdir(parents=True, exist_ok=True)

        ## read & transform scan
        scan_file = Path(scan2d_folder) / Path(params["scannet"]) / id_scan / (id_scan + "_vh_clean_2.ply")
        Mscan = make_M_from_tqs(r["trs"]["translation"], r["trs"]["rotation"], r["trs"]["scale"])
        scan = o3d.io.read_point_cloud(str(scan_file))
        scan.transform(Mscan)

        ray_casting_scene = o3d.t.geometry.RaycastingScene()

        to_draw = [scan]

        for model in r["aligned_models"]:
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = model["trs"]["scale"]

            id_cad = model["id_cad"]
            catid_cad = model["catid_cad"]

            ## read cad file
            cad_file = Path(scan2d_folder) / Path(params["shapenet"]) / catid_cad / (id_cad + "/models/model_normalized.obj")
            mesh = o3d.io.read_triangle_mesh(str(cad_file))
            Mcad = make_M_from_tqs(t, q, s)
            mesh.transform(Mcad)
            color = [50, 200, 50]
            mesh.paint_uniform_color(np.array(color) / 100)

            ## add cad to raycasting
            mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            ray_casting_scene.add_triangles(mesh_legacy)

            ## add bounding box
            meshbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(mesh.vertices))
            meshbox.color = [0, 0, 0]

            to_draw.extend([mesh, meshbox])

    o3d.visualization.draw_geometries(to_draw)

    distances_to_cad_model = ray_casting_scene.compute_signed_distance(np.asarray(scan.points, dtype=np.float32)).numpy()

if __name__ == '__main__':
    parse_scan2d_dataset(scan2d_folder="other_data/Scan2CAD/Routines/Script", output_dir=opt.out)




