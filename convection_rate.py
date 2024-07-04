import sys
sys.path.append('/users/taoshi11/analysator/')
import pytools as pt
import numpy as np
# import matplotlib.pyplot as plt

# Flux change rate in each MLT sector
def flux_change_rate(Bflux_total):

    Bflux0 = Bflux_total[0:-1,1:]
    Bflux1 = Bflux_total[1:,1:]

    t0 = Bflux_total[0:-1,0]
    t1 = Bflux_total[1:,0]
    delta_t = t1-t0

    rate = (Bflux1-Bflux0)/delta_t[:,np.newaxis]
    return rate

def get_bulk(name):
    filepath = '/scratch/project_2000203/3D/FHA/bulk1/'
    filename = filepath+name
    
    f = pt.vlsvfile.VlsvReader(file_name = filename)
    # filename = 'bulk1.0000600.vlsv'

    return f

# Flux transport rate between MLT boundaries
# Functions in analysator:get cellids of the cells between two points(cut0through line)
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
        # print(cell_lengths).shape

        # if cellid == 0:
        #     print("ERROR, invalid cell id!")
        #     return

      # Get the max and min boundaries:
        min_bounds = vlsvReader.get_cell_coordinates(cellid) - 0.5 * cell_lengths
        max_bounds = np.array([min_bounds[i] + cell_lengths[i] for i in range(0,3)])
        # max_bounds = min_bounds + cell_lengths
        
        
    
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


def find_closed_indice(f,cellids0):

    connection = f.read_variable("vg_connection",cellids0)
    connection_indices = np.where(connection == 0)[0]
    return connection_indices

def MLT_line_conv_rate(f,point1,point2):
    MLT_par = get_cellids_coordinates_distances(f,point1,point2)
    cellids = MLT_par[0]
    distances = MLT_par[1]
 

    indice_closed = find_closed_indice(f,cellids)
    cellids_closed = cellids[indice_closed]
    distances_closed = distances[indice_closed]

    coords_closed = f.read_variable('vg_coordinates',cellids_closed)
    def flux_transport_rate(f, coords, cellids, distances):


        # Determine Er and Etheta
        # E = f.read_variable("vg_e_vol",cellids)
        E = f.read_interpolated_variable("vg_e_vol",coords)
        Ex, Ey = E[:,0], E[:,1]
        x,y = coords[:,0], coords[:,1]

        dr = distances
        theta = np.arctan2(y,x)

        Er = Ex * np.cos(theta)+ Ey * np.sin(theta)
        Etheta = -Ex * np.sin(theta) + Ey * np.cos(theta)

        Erdr = Er * dr
        

        return Erdr
    Erdr = flux_transport_rate(f,coords_closed,cellids_closed,distances_closed)
    return Erdr

def main():


    # # Plot the flux change rate in each MLT sector
    # Bflux_total = np.loadtxt('../../output/Bflux/Bflux_total_new.csv')
    # rate = flux_change_rate(Bflux_total)
    # legends = ['0-3','3-6','6-9','9-12','12-15','15-18','18-21','21-0']
    # plt.figure(figsize=(12, 8))

    # for i in range(rate.shape[1]):
    #     rate_tmp = rate[:,i]
        
    #     t0 = Bflux_total[:-1,0]
    #     plt.subplot(4,2,(i+1))
    #     plt.scatter(t0,rate_tmp,s=4)
    #     plt.xlabel('time(s)',fontsize = 8)
    #     plt.ylabel('rate Wb/s',fontsize = 8)
    #     plt.title(f'MLT:{legends[i]}',fontsize = 10)
    #     plt.ylim([-250000,250000])

    # plt.tight_layout()
    # plt.show()

    # size_phi = 100
    # phi = np.linspace(0,2*np.pi,size_phi)
    # Re = 6371000
    # r = 20 * Re
    # x, y = r * np.cos(phi), r * np.sin(phi)
    # points_circle = np.zeros((size_phi,3))
    # points_circle[:,0] = x
    # points_circle[:,1] = y

    # r0 = 5 * Re
    # x0, y0 = r0 * np.cos(phi), r0 * np.sin(phi)
    # points_origin = np.zeros((size_phi,3))
    # points_origin[:,0] = x0
    # points_origin[:,1] = y0


    # Calculate the flux transport rate between MLT sectors
    I = np.arange(800,1612,1)
    # I = np.arange(900,901,1)
    # I = np.arange(1150,1151,1)

    # azimuthal_convection_rate_all = np.full((size_phi,I.shape[0]),np.nan)
    flux_change = np.zeros((I.shape[0],25))

    points = []
    MLT_hours = 24
    Re = 6371000
    for hour in range(MLT_hours):
        angle = 2 * np.pi * hour / MLT_hours
        x = -20 * Re * np.cos(angle)
        y = -20 * Re * np.sin(angle)
        points.append([x, y, 0])
    
    for i in range(len(I)):

        bulkname = 'bulk1.'+str(I[i]).rjust(7,'0')+'.vlsv'
        f = get_bulk(bulkname)


        Re = 6371000
        point_origin = [0,0,0]
        Erdr_MLT = [MLT_line_conv_rate(f, point_origin, point) for point in points]
        flux_change[i, 0] = I[i]
        for hour in range(MLT_hours):
            next_hour = (hour + 1) % MLT_hours
            flux_change_tmp = -Erdr_MLT[hour].sum() + Erdr_MLT[next_hour].sum()
            flux_change[i, hour + 1] = flux_change_tmp

        # point_12 = [20*Re,0,0]
        # point_15 = [20*Re,20*Re,0]
        # point_18 = [0,20*Re,0]
        # point_21 = [-20*Re,20*Re,0]
        # point_0 = [-20*Re,0,0]
        # point_3 = [-20*Re,-20*Re,0]
        # point_6 = [0,-20*Re,0]
        # point_9 = [20*Re,-20*Re,0]

        # Erdr_MLT_0 = MLT_line_conv_rate(f,point_origin,point_0)
        # Erdr_MLT_3 = MLT_line_conv_rate(f,point_origin,point_3)
        # Erdr_MLT_6 = MLT_line_conv_rate(f,point_origin,point_6)
        # Erdr_MLT_9 = MLT_line_conv_rate(f,point_origin,point_9)
        # Erdr_MLT_12 = MLT_line_conv_rate(f,point_origin,point_12)

        # Erdr_MLT_15 = MLT_line_conv_rate(f,point_origin,point_15)
        # Erdr_MLT_18 = MLT_line_conv_rate(f,point_origin,point_18)
        # Erdr_MLT_21 = MLT_line_conv_rate(f,point_origin,point_21)

        # # Dawnside

        # flux_change_0_3_tmp = -Erdr_MLT_0.sum()+Erdr_MLT_3.sum()
        # flux_change_3_6_tmp = -Erdr_MLT_3.sum()+Erdr_MLT_6.sum()
        # flux_change_6_9_tmp = -Erdr_MLT_6.sum()+Erdr_MLT_9.sum()
        # flux_change_9_12_tmp = -Erdr_MLT_9.sum()+Erdr_MLT_12.sum()

        # # Duskside
        # flux_change_12_15_tmp = -Erdr_MLT_12.sum()+Erdr_MLT_15.sum()
        # flux_change_15_18_tmp = -Erdr_MLT_15.sum()+Erdr_MLT_18.sum()
        # flux_change_18_21_tmp = -Erdr_MLT_18.sum()+Erdr_MLT_21.sum()
        # flux_change_21_0_tmp = -Erdr_MLT_21.sum()+Erdr_MLT_0.sum()
        
        # flux_change[i,0] = I[i]
        # flux_change[i,1] = flux_change_0_3_tmp
        # flux_change[i,2] = flux_change_3_6_tmp
        # flux_change[i,3] = flux_change_6_9_tmp
        # flux_change[i,4] = flux_change_9_12_tmp
        # flux_change[i,5] = flux_change_12_15_tmp
        # flux_change[i,6] = flux_change_15_18_tmp
        # flux_change[i,7] = flux_change_18_21_tmp
        # flux_change[i,8] = flux_change_21_0_tmp

        # for j in range(size_phi):

        #     Erdr_MLT_temp = MLT_line_conv_rate(f,points_origin[j,:],points_circle[j,:])
        #     print(Erdr_MLT_temp.shape)

        #     azimuthal_convection_rate_all[j][i] = Erdr_MLT_temp.sum()



        # print(Erdr_MLT_15.sum()-Erdr_MLT_12.sum())
    # np.savetxt('../../output/Bflux/Bconvection_rate/convection_rate_total.csv',flux_change,delimiter=',')
    # np.savetxt('../../output/Bflux/Bconvection_rate/azimuthal_convection_rate.csv',azimuthal_convection_rate_all,delimiter=',')
    # np.savetxt('../../output/Bflux/Bconvection_rate/azimuthal_convection_rate_1150.csv',azimuthal_convection_rate_all,delimiter=',')
    np.save('../../output/Bflux/Bconvection_rate/az_conv_MLT.npy',flux_change)


if __name__ == "__main__":
    main()

