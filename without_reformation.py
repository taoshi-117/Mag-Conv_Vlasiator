import sys
sys.path.append('/users/taoshi11/analysator/')
import pytools as pt
import numpy as np
# sys.path.append('/users/taoshi11/code/Fieldline_Trace')
# from field_trace_vg import field_line_trace_vg
sys.path.append('/users/taoshi11/analysator/pyCalculations/fieldtracer')
from fieldtracer import static_field_tracer_3d
import time
import cProfile
from pstats import SortKey

# shape of points(N,3); shape of mag_mon (1,3)
def get_b_dip(points, mag_mom=np.array([0., 0., -8e22])[np.newaxis, :]):
    mu_0 = 4e-7 * np.pi
    
    r_mag = np.linalg.norm(points, axis = 1, keepdims = True)
    r_hat = points / r_mag
    
    dot_product = np.sum(mag_mom * r_hat, axis=1, keepdims=True)
    
    B = mu_0 / (4*np.pi) * (3 * dot_product * r_hat - mag_mom) / (r_mag)**3
    # check from
    return B



# Find barycenter
def get_triangle_barycenter(triangles):
    bary_center = (triangles[:,0,:] + triangles[:,1,:] + triangles[:,2,:])/3
    return bary_center


# Area calculation for a triangle 
# points: (n,3,3) shape[0]:number of triangles shape[1]:triangle vertex, shape[2]:x,y,z
def get_triangle_area(triangles):
    vec_1 = triangles[:,1,:] - triangles[:,0,:]
    vec_2 = triangles[:,2,:] - triangles[:,0,:]
    tri_area = np.linalg.norm(np.cross(vec_1, vec_2), axis = 1) / 2.
    return tri_area

# Calculate open flux for completely open triangles
def Calculate_all_open_flux(vertex_coords):
    open_area = get_triangle_area(vertex_coords)
    bary_center = get_triangle_barycenter(vertex_coords)
    B_bary = get_b_dip(bary_center)

    r = np.linalg.norm(bary_center,axis = 1)
    theta = np.arccos(bary_center[:,2]/r)
    phi = np.arctan2(bary_center[:,1], bary_center[:,0])

    Br = B_bary[:,0] * np.sin(theta) * np.cos(phi) + B_bary[:,1] * np.sin(theta) * np.sin(phi) + B_bary[:,2] * np.cos(theta)
    print("shape of Br",Br.shape)
    print("shape of open area",open_area.shape)
    
    # flux = np.sum(np.linalg.norm(B_bary, axis = 1) * open_area)
    flux = np.sum(Br * open_area)
    return flux

# 1611
t = np.arange(800,1611,1)
# t = np.arange(1001,1056,1)

flux_total = np.full(t.shape,np.nan)

for i in range(len(t)):
    str_t = str(int(t[i]))


    filepath = '/scratch/project_2000203/3D/FHA/bulk1/'
    filename = 'bulk1.'+str_t.rjust(7,'0')+'.vlsv'
    name = filepath+filename
    f = pt.vlsvfile.VlsvReader(file_name = name)

    up_coords = f.read_variable("ig_upmappednodecoords")
    # Find the field lines that are closed within 5 Re, set their seed points as nan
    indices_not_needed = np.where(up_coords[:,2] == 0)[0]
    up_coords[indices_not_needed,:] = np.nan

    c = f.get_ionosphere_element_corners()
    p = up_coords[c,:]

    # io_open_closed = f.read_variable("ig_openclosed")
    if t[i]>= 1001 and t[i] <= 1055:
        io_open_closed = np.load(f"/scratch/project_2000203/taoshi/field_line_outputs/FHA_{str_t}/oc_values.npy")
    else:
        io_open_closed = f.read_variable("ig_openclosed")
    

    io_connect = io_open_closed[c]

    # 0th order refinement
    if t[i]<=1055:
        indices_all_open = np.where(np.all(io_connect == 1, axis = 1))[0]
    else:
        indices_all_open = np.where(np.all(io_connect == 2, axis = 1))[0]
    
    io_connect_all_open = io_connect[indices_all_open,:]
    io_coords_all_open = p[indices_all_open,:,:]


    io_coords_open_n = io_coords_all_open[np.all(io_coords_all_open[:,:,2] > 0,axis = 1),:,:]

    print(io_coords_open_n)


    total_io_open_flux_n = Calculate_all_open_flux(io_coords_open_n)
    flux_total[i] = total_io_open_flux_n

file_directory = '/scratch/project_2000203/taoshi/io_refine/'
file_name = 'flux_without_refine_800_1611.npy'
# file_name = 'flux_without_refine_1001_1055.npy'
np.save(file_directory+file_name, flux_total)