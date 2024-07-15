import numpy as np
import sys
sys.path.insert(0, '/home/taoshi/analysator')
import pytools as pt
sys.path.insert(0, '/home/taoshi/analysator/pyCalculations')
from fieldtracer import static_field_tracer_3d

def simple_condition(f, points):
    Re = 6371000
    r = np.linalg.norm(points, axis=1)
    [xmin, ymin, zmin, xmax, ymax, zmax] = f.get_spatial_mesh_extent()
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    return (r < (Re + 1e5)) | (x < xmin) | (x > xmax) | (y < ymin) | (y > ymax) | (z < zmin) | (z > zmax)

def Flux_tube_entropy(f):
    """
    Get the closed coordinates of the equatorial plane
    """

    # # Sort cellids and connection variable
    # connection = f.read_variable('vg_connection')
    # cellids = f.read_variable("CellID")

    # indexids = cellids.argsort()
    # cellids = cellids[indexids]
    # connection = connection[indexids]

    # # find the cells that vg_connection is 0
    # connection_zero_indices = np.where(connection == 0)[0]
    # cellids_closed = cellids[connection_zero_indices]
    # coords_closed = f.read_variable("vg_coordinates", cellids_closed)

    # cellsize_closed = f.read_variable("vg_dx", cellids_closed)
    # eq_indices = np.where((coords_closed[:, 2] < cellsize_closed[:, 2]) & (coords_closed[:, 2] > 0))[0]  # Determine the "close to equatorial" cells
    # coords_used = coords_closed[eq_indices][:]
    # cellids_used = cellids_closed[eq_indices]

    # Use a cut of equatorial plane from -20 to 20 Re, and find the corresponding values use read_interpolated_variable
    Re = 6371000
    xx = np.linspace(-20,20,200)
    yy = np.linspace(-20,20,200)

    XX, YY = np.meshgrid(xx, yy)

    # Flatten the meshgrid and scale by Earth's radius
    coords = np.vstack((XX.flatten(), YY.flatten(), np.zeros(200 * 200))).T * Re
    print(coords.shape)

    connection = f.read_interpolated_variable("vg_connection",coords)
    mask = connection == 0
    coords_used = coords[mask]


    """
    Use the vg field line trace for getting the flux tube
    Note that the shape of coords_fl is (coords_used.shape[0], max_iteration, 3)
    """

    max_iterations = 20000
    dx = 4e4
    chunk_size = 100  # Process 100 points at a time to reduce memory usage


    # Both the northern and southern hemisphere.
    # coords_fl = static_field_tracer_3d(f, coords_used, max_iterations, dx, direction='+-',
    #                                   grid_var='vg_b_vol', stop_condition=simple_condition)
    
    # Only northern hemisphere
    coords_fl = static_field_tracer_3d(f, coords_used, max_iterations, dx, direction='-',
                                        grid_var='vg_b_vol', stop_condition=simple_condition)
    
    """
    Entropy Parameter = (\integra P^(3/5) ds / B) ^(5/3)        (Be consistent with those from RCM)
    Unit: nPa (Re / nT) ^(5/3)
    """
    Re = 6371000
    epsilon = 1e-12  # Small constant to avoid division by zero
    # coordinates and entropy shape (coord_used.shape[0], 4)
    Entropy_total = np.full((coords_used.shape[0], 4), np.nan, dtype=np.float32)
    Entropy_total[:, 0:3] = coords_used

    num_chunks = (coords_used.shape[0] + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, coords_used.shape[0])

        # Compute ds and ds_abs
        ds = np.diff(coords_fl[start_idx:end_idx], axis=1) / Re  # in Re
        ds_abs = np.linalg.norm(ds, axis=2)

        # use mid point of the coordinates
        mid_coords_fl = (coords_fl[start_idx:end_idx, 1:, :] + coords_fl[start_idx:end_idx, :-1, :]) / 2
        mid_coords_fl_reshaped = mid_coords_fl.reshape(-1, mid_coords_fl.shape[-1])

        B = f.read_interpolated_variable("vg_b_vol", mid_coords_fl_reshaped).reshape(mid_coords_fl.shape) * 1e9  # in nT
        B_nT = np.linalg.norm(B, axis=2)
        P_diag = f.read_interpolated_variable('proton/vg_ptensor_diagonal', mid_coords_fl_reshaped).reshape(mid_coords_fl.shape)
        P_nPa = (P_diag[:, :, 0] + P_diag[:, :, 1] + P_diag[:, :, 2]) / 3 * 1e9  # in nPa

        # create mask for nan values
        mask_nan_B = np.isnan(B_nT)
        mask_nan_P = np.isnan(P_nPa)

        B_nT[mask_nan_B] = np.inf 
        P_nPa[mask_nan_P] = 0.0

        Entropy_coords = P_nPa ** (3/5) * ds_abs / B_nT
        Entropy_coords[mask_nan_B | mask_nan_P] = 0.0

        Entropy_total[start_idx:end_idx, 3] = np.sum(Entropy_coords, axis=1) ** (5/3)

    return Entropy_total

def main():
    filepath = '/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/'
    filename = 'bulk1.0001000.vlsv'
    name = filepath + filename
    f = pt.vlsvfile.VlsvReader(file_name=name)

    Entropy = Flux_tube_entropy(f)
    # np.save("Entropy_1000s.npy", Entropy)
    np.save("Entropy_1000s_new.npy", Entropy)
    # np.save("Entropy_1000s_n.npy",Entropy)

if __name__ == "__main__":
    main()
