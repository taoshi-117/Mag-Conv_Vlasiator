import sys
sys.path.append('/users/taoshi11/analysator/')
import pytools as pt
import numpy as np
sys.path.append('/users/taoshi11/analysator/pyCalculations')
# import cutthrough.py

def get_bulk(name):
    filepath = '/scratch/project_2000203/3D/FHA/bulk1/'
    filename = filepath+name
    
    f = pt.vlsvfile.VlsvReader(file_name = filename)
    # filename = 'bulk1.0000600.vlsv'

    return f

def get_filtered_indice(f,coords):
    """
    1. closed: vg_connection == 0
    2. inner magnetosphere: 0 < vg_beta < 1
    """
    cellids0 = f.get_cellid(coords)
    
    p_beta = np.full(cellids0.shape,np.nan)
    connection = np.full(cellids0.shape,np.nan)
    
    mask = (cellids0 != 0)

    p_beta[mask] = f.read_variable("vg_beta", cellids0[mask])
    connection[mask] = f.read_variable("vg_connection",cellids0[mask])

    # filtered_indices = np.where((connection == 0) & (p_beta < 1) & (p_beta > 0))[0]
    # filtered_indices = np.where((connection == 0))[0]
    mask2 = (connection == 0)
    
    return mask2


def flux_transport_rate(f, coords, distances):

    # Determine Er and Etheta
    # E = f.read_variable("vg_e_vol",cellids)
    E = f.read_interpolated_variable("vg_e_vol",coords)
    Ex, Ey = E[:,0], E[:,1]
    x,y = coords[:,0], coords[:,1]

    dr = distances
    theta = np.arctan2(y,x)

    Er = Ex * np.cos(theta)+ Ey * np.sin(theta)


    Erdr = Er * dr

    return Erdr

def generate_points(points_array1, points_array2,step = 1.0e6/2):
    direction = points_array2-points_array1
    distance = np.linalg.norm(points_array2-points_array1,axis = 1)
    unit_vector = direction/distance[0]

    num_steps = int(distance[0]/step) # This assumes the same distance!

    points = [points_array1 + i * step * unit_vector for i in range(num_steps + 1)]

    # if distance % step != 0:
    #     points.append(points_array2)

    return np.array(points) # a 3d array shape[0]: the num_steps, shape[1]: size_phi, shape[2]: 3 



def main():
    # create the origin and the boundary points of the circle
    size_phi = 100
    phi = np.linspace(0,2*np.pi,size_phi)
    Re = 6371000
    r = 20 * Re
    x, y = r * np.cos(phi), r * np.sin(phi)
    points_circle = np.zeros((size_phi,3))
    points_circle[:,0] = x
    points_circle[:,1] = y

    r0 = 5 * Re
    x0, y0 = r0 * np.cos(phi), r0 * np.sin(phi)
    points_origin = np.zeros((size_phi,3))
    points_origin[:,0] = x0
    points_origin[:,1] = y0


    # get the min cell length, set distance as celllength/2
    # that is dx = 1e6?

    distance = 1.0e6/2.
    total_points = generate_points(points_origin,points_circle,step = distance)

    # reshape the points to 2d
    points_reshaped = total_points.reshape(-1,3)

    # I = np.arange(800,1612,1)
    I = np.arange(1150,1151,1)

    azimuthal_all = np.full((I.shape[0],size_phi),np.nan)

    for i in range(len(I)):
        bulkname = 'bulk1.'+str(I[i]).rjust(7,'0')+'.vlsv'
        f = get_bulk(bulkname)

        mask0 = get_filtered_indice(f,points_reshaped)
        points_reshaped[~mask0] = np.nan

        az_conv_point_reshaped = flux_transport_rate(f,points_reshaped,distance)
        az_conv_point = az_conv_point_reshaped.reshape(-1, size_phi)

        az_conv_point[np.isnan(az_conv_point)] = 0

        az_conv_MLT = np.sum(az_conv_point,axis = 0) # Now it is a shape: size_phi array
        azimuthal_all[i,:] = az_conv_MLT
    
    np.savetxt('../../output/Bflux/Bconvection_rate/azimuthal_convection_rate_1150_new.csv',az_conv_MLT,delimiter=',')
    # np.save('../../output/Bflux/Bconvection_rate/azimuthal_convection_rate_all_new.npy',azimuthal_all)



if __name__ =="__main__":
    main()
