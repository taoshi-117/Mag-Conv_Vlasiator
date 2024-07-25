import sys
#sys.path.append('/users/taoshi11/analysator/')
sys.path.insert(0, '/home/taoshi/analysator')
import pytools as pt
import numpy as np

def get_bulk(name):
    #filepath = '/scratch/project_2000203/3D/FHA/bulk1/'
    filepath = '/wrk-vakka/group/spacephysics/vlasiator/3D/FHA/bulk1/'
    filename = filepath+name
    
    f = pt.vlsvfile.VlsvReader(file_name = filename)
    # filename = 'bulk1.0000600.vlsv'

    return f

def get_MLT_value(f,cellids):
    coords = f.read_variable("vg_coordinates",cellids)
    xx, yy = coords[:, 0], coords[:, 1]

    angles = np.arctan2(yy,xx)
    MLT_values = 12 + angles *(12/np.pi)

    MLT_values = np.where((xx == 0) & (yy > 0), 18, MLT_values)
    MLT_values = np.where((xx == 0) & (yy < 0), 6, MLT_values)


    return MLT_values


def transform_to_polar(coords):
    x, y = coords[:, 0], coords[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def get_cellids_coordinates_distances( vlsvReader, point1, point2 ):
    

    point1 = np.array(point1, copy=False)
    point2 = np.array(point2, copy=False)


    distances = [0]

    cellids = [vlsvReader.get_cellid(point1)]

    coordinates = [point1]

    epsilon = sys.float_info.epsilon*2

    iterator = point1

   # Special case: both points in the same cell. 
   #if(vlsvReader.get_cellid(point1) == vlsvReader.get_cellid(point2)):
   #   cellids.append(vlsvReader.get_cellid(point2))
      
    

    unit_vector = (point2 - point1) / np.linalg.norm(point2 - point1 + epsilon)

    while True:

      # Get the cell id
        cellid = vlsvReader.get_cellid(iterator)
        cell_lengths = vlsvReader.read_variable("vg_dx",cellid)

        if cellid == 0:
            print("ERROR, invalid cell id!")
            return
      # Get the max and min boundaries:
        min_bounds = vlsvReader.get_cell_coordinates(cellid) - 0.5 * cell_lengths
        max_bounds = np.array([min_bounds[i] + cell_lengths[i] for i in range(0,3)])
        
    
      # Check which boundary we hit first when we move in the unit_vector direction:
        coefficients_min = np.divide((min_bounds - iterator), unit_vector, out=np.full(min_bounds.shape, sys.float_info.max), where=np.logical_not(unit_vector==0.0))
        coefficients_max = np.divide((max_bounds - iterator), unit_vector, out=np.full(min_bounds.shape, sys.float_info.max), where=np.logical_not(unit_vector==0.0))


      # Condition that make sure the coefficints are valid
        condition_min = np.all((coefficients_min <= 0) | (coefficients_min == sys.float_info.max))
        condition_max = np.all((coefficients_max <= 0) | (coefficients_max == sys.float_info.max))

        if condition_min and condition_max:
            coefficients_min = np.abs(coefficients_min)
            coefficients_max = np.abs(coefficients_max)

            
    # Remove negative coefficients:
    
        
        for i in range(3):
            if coefficients_min[i] <= 0:
                coefficients_min[i] = sys.float_info.max
            if coefficients_max[i] <= 0:
                coefficients_max[i] = sys.float_info.max



    # Find the minimum coefficient for calculating the minimum distance from a boundary
        coefficient = min([min(coefficients_min), min(coefficients_max)]) * 1.000001


    # Get the cell id in the new coordinates
        newcoordinate = iterator + coefficient * unit_vector

          # Check to make sure the iterator is not moving past point2 - if it is, use point2 instead:
        if min((point2 - newcoordinate)* unit_vector) < 0:
            newcoordinate = point2

        newcellid = vlsvReader.get_cellid( newcoordinate )
          # Check if valid cell id:
        if newcellid == 0:
            break
          # Append the cell id:
        cellids.append( newcellid )


          # Append the coordinate:
        coordinates.append( newcoordinate )

          # Append the distance:


        distances.append( np.linalg.norm( newcoordinate - point1))
        point1 = newcoordinate


          # Move the iterator to the next cell. Note: Epsilon is here to ensure there are no bugs with float rounding
        iterator = iterator + coefficient * unit_vector
          # Check to make sure the iterator is not moving past point2:
        if min((point2 - iterator)* unit_vector) < 0:
            break


    return [np.array(cellids, copy=False), np.array(distances, copy=False), np.array(coordinates, copy=False)]


def find_boundary_points(f,coords, estimated_radius, tolerance,cellids):
    MLT_values = get_MLT_value(f,cellids)
    r, theta = transform_to_polar(coords)
    
    
    boundary_indices = np.where((r >= estimated_radius - tolerance) & (r <= estimated_radius + tolerance))[0]
    boundary_points = coords[boundary_indices]
    boundary_r = r[boundary_indices]
    boundary_theta = theta[boundary_indices]
    boundary_cellids = cellids[boundary_indices]
    boundary_MLTs = MLT_values[boundary_indices]

    return boundary_r, boundary_theta, boundary_cellids,boundary_indices,boundary_MLTs

def radial_convection_rate(f,coords,cellids,distances):
    
    E = f.read_variable("vg_e_vol", cellids)
    
    Ex, Ey = E[:,0], E[:,1]
    x, y = coords[:,0], coords[:,1]
    theta = np.arctan2(y,x)

    Etheta = -Ex * np.sin(theta) + Ey * np.cos(theta)
    Etheta_dtheta = Etheta * distances

    return Etheta,Etheta_dtheta

def find_closedandeq_coords(f):

    cellids = f.read_variable("CellID")
    connection = f.read_variable("vg_connection",cellids)

    indexids = cellids.argsort()
    cellids = cellids[indexids]
    connection = connection[indexids]

    connection_indices = np.where(connection == 0)[0]
    cellids_closed = cellids[connection_indices]

    coords_closed = f.read_variable("vg_coordinates",cellids_closed)
    cellsize_closed = f.read_variable("vg_dx",cellids_closed)

    # Find the coords that close to equatorial plane
    eq_indices = np.where((coords_closed[:,2] < cellsize_closed[:,2]) & (coords_closed[:,2] > 0))[0] 
    coords_used = coords_closed[eq_indices][:]
    cellids_used = cellids_closed[eq_indices]


    return cellids_used,coords_used




def main():
    
    # I = np.arange(1000,1001,1)
    # I = np.arange(800,1612,1)
    I = np.arange(501,1613,1)
    # I = np.array([800,900,1000])

    K = 24
    flux_change = np.zeros((I.shape[0],K))

    Re = 6371000
    r = 5 * Re
    dtheta = 0.1
    theta_min = 0
    theta_max = 360
    theta = np.arange(theta_min,theta_max,dtheta)*np.pi/180

    r_column = np.full_like(theta,r)
    coords_r_theta = np.column_stack((r_column,theta))
    coords_x,coords_y = coords_r_theta[:,0] * np.cos(coords_r_theta[:,1]), coords_r_theta[:,0]*np.sin(coords_r_theta[:,1])
    coords_z = np.zeros((coords_x.shape))
    points_new = np.column_stack((coords_x,coords_y,coords_z))

    for i in range(len(I)):
        bulkname = 'bulk1.'+str(I[i]).rjust(7,'0')+'.vlsv'

        f = get_bulk(bulkname)
        cellids_used,coords_used = find_closedandeq_coords(f)

        points1_new = points_new[0:-1]
        points2_new = points_new[1:]

        cellids_crossed_new = []
        distances_crossed_new = []
        Etheta_dtheta_crossed_new = []
        Etheta_crossed_new = []
        for j in range(points1_new.shape[0]):
            point1_tmp = points1_new[j,:]
            point2_tmp = points2_new[j,:]
            a = get_cellids_coordinates_distances(f,point1_tmp,point2_tmp)
            Etheta_tmp,Ethetadtheta_tmp = radial_convection_rate(f,a[2],a[0],a[1])
            cellids_crossed_new = np.hstack((cellids_crossed_new,a[0]))
            distances_crossed_new = np.hstack((distances_crossed_new,a[1]))
            Etheta_dtheta_crossed_new = np.hstack((Etheta_dtheta_crossed_new,Ethetadtheta_tmp))

        MLT_corresponded = get_MLT_value(f,cellids_crossed_new)


        
        # Find the boundary transported flux in each MLT sector
        for k in range(K):
            idx_tmp = np.where((MLT_corresponded >= k) & (MLT_corresponded < k+1))[0] # 1 hour MLT interval
            flux_change[i,k] = Etheta_dtheta_crossed_new[idx_tmp].sum()



        # idx_03 = np.where((MLT_corresponded >= 0) & (MLT_corresponded < 3))[0]
        # idx_36 = np.where((MLT_corresponded >= 3) & (MLT_corresponded < 6))[0]
        # idx_69 = np.where((MLT_corresponded >= 6) & (MLT_corresponded < 9))[0]
        # idx_912 = np.where((MLT_corresponded >= 9) & (MLT_corresponded < 12))[0]
        # idx_1215 = np.where((MLT_corresponded >= 12) & (MLT_corresponded < 15))[0]
        # idx_1518 = np.where((MLT_corresponded >= 15) & (MLT_corresponded < 18))[0]
        # idx_1821 = np.where((MLT_corresponded >=18) & (MLT_corresponded < 21))[0]
        # idx_2124 = np.where((MLT_corresponded >=21) & (MLT_corresponded < 24))[0]


        # Etheta_dtheta_crossed_03 = Etheta_dtheta_crossed_new[idx_03].sum()
        # Etheta_dtheta_crossed_36 = Etheta_dtheta_crossed_new[idx_36].sum()
        # Etheta_dtheta_crossed_69 = Etheta_dtheta_crossed_new[idx_69].sum()
        # Etheta_dtheta_crossed_912 = Etheta_dtheta_crossed_new[idx_912].sum()
        # Etheta_dtheta_crossed_1215 = Etheta_dtheta_crossed_new[idx_1215].sum()
        # Etheta_dtheta_crossed_1518 = Etheta_dtheta_crossed_new[idx_1518].sum()
        # Etheta_dtheta_crossed_1821 = Etheta_dtheta_crossed_new[idx_1821].sum()
        # Etheta_dtheta_crossed_2124 = Etheta_dtheta_crossed_new[idx_2124].sum()


        # flux_change[i,0] = I[i]
        # flux_change[i,1] = Etheta_dtheta_crossed_03
        # flux_change[i,2] = Etheta_dtheta_crossed_36
        # flux_change[i,3] = Etheta_dtheta_crossed_69
        # flux_change[i,4] = Etheta_dtheta_crossed_912
        # flux_change[i,5] = Etheta_dtheta_crossed_1215
        # flux_change[i,6] = Etheta_dtheta_crossed_1518
        # flux_change[i,7] = Etheta_dtheta_crossed_1821
        # flux_change[i,8] = Etheta_dtheta_crossed_2124
        

    # np.savetxt('../../output/Bflux/B_radial_conv_rate/radial_conv_rate.csv',flux_change,delimiter=',')
    np.save('/home/taoshi/output/radial_conv_rate_MLT1h.npy',flux_change)













if __name__ == "__main__":
    main()
