""" All the functions necessary for the clustering of the Poincare solutions.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from seasonality_odes import getr
from rk4_solver import rk4solver

r = getr()

def clustering(f, poinc_sol, aS, nu, plotit = True, saveplot=False, savepath=os.getcwd()):
    """ Groups the poincare grid solution into clusters of the same long-term 
        behaviour. 
        First, we cut the trajectories so that we only use the last 50 entries
        for each trajectory (as then, all transience should be vanished).
        Then, using the K-means clustering algorithm we group all the trajectories
        into clusters, which get classified as cycles (of a certain period) or
        chaotic/quasiperiodic, using just a single representative of each initial
        cluster (using advancedclustering).
        Finally, the trajectories belonging to the chaotic/quasiperiodic clusters
        get classified as chaotic if the median of the FTLE of a single randomly
        chosen point (for a few different FTLE calculation lengths) is above a
        certain threshold and otherwise as quasiperiodic.
        In the end we have classified all the long-term behaviour that's present 
        
    Args:
        f: our right hand side of the ODE - i.e. np_odes
        poinc_sol:  3D array consisting of our solution trajectories
                    where each trajectory i at time t has entries [n,p,t]
        aS: double the summer length
        nu: generalist predator density dependence parameter
        plotit: boolean for plotting the clustered graph
        saveplot: boolean for saving the plot
        savepath: path to save the plot

    Return: 8 outputs, namely
        aS: double the summer length
        nu: generalist predator density dependence parameter
        newclusters:    list with entries [i, x, per]
                        where i denotes the index of the new cluster that point
                        x (consisting of n and p component) belongs to.
                        This cluster i is a cycle of length per.
                        I.e. for a cycle of length per, we will have per entries
        chaosclusterlist:   List of points [n,p] that belong to chaos
        cyclelist:      List of points [n,p] that belong to a quasiperiodic orbit
        notsurelist:    List of points that might belong to a cycle of length 
                        > tmax or points that we're not sure about
        FTLE_average: the average of FTLE of a random point within our chaos/cycle cluster for the last 10 years, starting from 50 years
        FTLE_tolerance: tolerance used to determine whether we have chaos or cycle
    """
    
    n = poinc_sol.shape[0]
    m = poinc_sol.shape[1]
    
    Xcopy = np.transpose(poinc_sol, (0, 2, 1))
    
    # ensuring we only take max. 100 orbits
    if n>100:
        # we have too many orbits. Select 100 out of them by random
        selected_traj = np.random.choice(n, 100, replace=False)
        #X = poinc_sol[selected_traj, -50:, :2] # we also cut the time from our array
        X = Xcopy[selected_traj, -50:, :2] # we also cut the time from our array
    else:
        #X = poinc_sol[:, -50:, :2]
        X = Xcopy[:, -50:, :2]
    
    n = X.shape[0]
    m = X.shape[1]
    
    ###########################################
    # Setting up the matrices
    ###########################################
    
    # the similarity matrix
    bigK = 1e4
    W = np.zeros((n, n))
    
    for i in range(0, n):
        for j in range(i + 1, n):
            sum = 0
            for k in range(0, m-1):
                sum += np.sqrt((X[i][k+1][0] - X[j][k+1][0])**2 + (X[i][k+1][1] - X[j][k+1][1])**2) + np.sqrt((X[i][k][0] - X[j][k][0])**2 + (X[i][k][1] - X[j][k][1])**2)
            rval = sum/((m-1)*2)
            if rval < 1e-4:
                #print('basically no difference')
                W[i][j] = bigK
                W[j][i] = bigK
            else:
                W[i][j] = 1/rval
                W[j][i] = 1/rval
    
    # now the diagonal elements
    for i in range(n):
        W[i][i] = bigK
    
    # and the diagonal matrix
    
    D = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            D[i][i] += W[i][j]
    
    # Then, we're ready to solve the eigenvalue problem
    L = D - W
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(D), L))
    
    if plotit == True:
        # Plotting the eigenvalues
        fig = plt.figure(figsize=(6,4))
        plt.xlabel("i")
        plt.ylabel("abs(lambda_i)")
        plt.grid()
      
        if n > 25:
            ind = np.arange(1,26)
            plt.title("First 25 eigenvalues of the clustering problem")
            plt.plot(ind, eigenvalues[:25])
        else:
            ind = np.arange(1,n+1)
            plt.title(f"First {n} eigenvalues of the clustering problem")
            plt.plot(ind, eigenvalues)
    
    
    ###########################################
    # Getting the optimal cutoff point
    ###########################################
    
    indexlist = []
    
    for i in range(len(eigenvalues)):
        if eigenvalues[i] < 0.3:
            indexlist.extend([i])
    
    bestk = len(indexlist)
    
    # getting our matrix U
    U = eigenvectors[:,indexlist].real
    
    ###########################################
    # Performing the kmeans algorithm and getting the clusters
    ###########################################
    
    #print(U)
    #print(bestk)
    centroids, clusters = kmeans(U, bestk)
    #print(centroids, clusters)
    
    
    # we now have our list of clusters for each of our (max.) 100 (randomly chosen) trajectories
    # we want to check for cycles
    
    # create a new list of pairs [i,j]=[trajectory, cluster]
    clusterlist = []
    for i in range(X.shape[0]):
        clusterlist.extend([[i, clusters[i]]])
    
    # However, now I only want to keep one representative per cluster
    # Dictionary to store the first occurrence of each j
    unique_tuples = {}
    for i, j in clusterlist:
        if j not in unique_tuples:
            unique_tuples[j] = i
    
    # Convert the dictionary back to a list of tuples
    croppedlist = [[i, j] for j, i in unique_tuples.items()] #maybe need int(j) in the brackets
    
    #print(croppedlist)
    
    
    ###########################################
    # Clustering into periodic solutions
    ###########################################
    
    # with these new unique representatives, we can now check for periodicity.
    
    newclusters, chaosclusterlist, cyclelist, notsurelist, FTLE_average, FTLE_tolerance = advancedclustering(f, croppedlist, X, 50, 1e-5, aS, nu)
    # NOTE: due to the new ODE-solving function, I recognized that the solution is not too accurate, so I set the tolerance down a bit (5e-4 or 1e-5 instead of 1e-6)
    
    ###########################################
    # plotting
    ###########################################
    
    if plotit == True:
        nplt = []
        pplt = []
        labelplt = []
        yearplt = []
      
        fig = plt.figure(figsize=(12,8))
        plt.xlabel("Prey n")
        plt.ylabel("Predator p")
        plt.title(f"Poincare map clusters in the prey-predator space for T_S = {np.round(aS/2, 4)} and nu = {nu}")
        plt.vlines(0, ymin=0, ymax=0.2, color='lightgrey')
        plt.hlines(0, xmin=0, xmax=1.0, color='lightgrey')
      
        # first, the chaos:
        nchaos = []
        pchaos = []
      
        for it in range(0, len(chaosclusterlist)):
            nchaos.append(chaosclusterlist[it][0])
            pchaos.append(chaosclusterlist[it][1])
      
        chaoscmap = plt.get_cmap('gray', len(chaosclusterlist))
      
        scatterchaos = plt.scatter(nchaos, pchaos, c='black', cmap=chaoscmap, s=1)
      
      
        # then, the limit cycles:
        ncycles = []
        pcycles = []
      
        for it in range(0, len(cyclelist)):
            ncycles.append(cyclelist[it][0])
            pcycles.append(cyclelist[it][1])
      
        cyclecmap = plt.get_cmap('gray', len(cyclelist))
      
        scattercycles = plt.scatter(ncycles, pcycles, c='orange', cmap=cyclecmap, s=1)
      
      
        # now the not sure list:
        nnotsure = []
        pnotsure = []
      
        for it in range(0, len(notsurelist)):
            nnotsure.append(notsurelist[it][1][0])
            pnotsure.append(notsurelist[it][1][1])
      
        scatternotsure = plt.scatter(nnotsure, pnotsure, c='gray', s=1)
      
        # now the cycles (or non-chaos):
        if newclusters:
            differentclusters = [newclusters[0][2]]
            formercyclelength = newclusters[0][2]
        
            for it in range(0, len(newclusters)):
                # decoding the entries in the newclusters list
                labelplt.append(newclusters[it][0])
                nplt.append(newclusters[it][1][0])
                pplt.append(newclusters[it][1][1])
                currentcyclelength = newclusters[it][2]
                yearplt.append(currentcyclelength)
                  
                if currentcyclelength != formercyclelength:
                  differentclusters.append(currentcyclelength)
                  
                formercyclelength = currentcyclelength
        
            num_clusters = len(differentclusters)
        
            cmap = plt.get_cmap('rainbow', num_clusters)
            scatter = plt.scatter(nplt, pplt, c=labelplt, cmap=cmap, s=4)
        
            # Create colorbar with ticks corresponding to clusters
            cbar = plt.colorbar(scatter, ticks=np.arange(num_clusters))
            cbar.set_ticks(np.arange(num_clusters))
            cbar.set_ticklabels([f'Cycle of length {i}' for i in differentclusters])
        
        
        else:
            # we don't have any cyclic points, so just chaos or cycles without nice period
            # Create colorbar with black color
            cbar = plt.colorbar(scatterchaos)
            if cyclelist:
                if chaosclusterlist:
                    cbar.set_label('Chaos and limit cycle')
                else:
                    cbar.set_label('Limit cycle only')
            else:
                cbar.set_label('Chaos only')
            cbar.set_ticks([])  # Remove ticks
      
        # saving the plot?
      
        if saveplot == True:
            fig.savefig(f"{savepath}/Poincaregridclusters_TS_{np.round(aS/2, 4)}_nu_{nu}.png")
    
    return aS, nu, newclusters, chaosclusterlist, cyclelist, notsurelist, FTLE_average, FTLE_tolerance



def advancedclustering(f, croppedlist, X, tmax, tol, aS, nu):
    """ This function takes one representative for each cluster and then 
        decides whether this representative (and thus: all the trajectories
        belonging to this cluster) is a cyclic point - if so: of what length -
        or not - then it will be either chaos or a quasiperiodic point (or a 
        cycle of length > tmax). Furthermore, trajectories that belong to the 
        same cyclic point of length > 1, but are representatives of different 
        clusters initially, get grouped together into a new cluster.
        For example: representatives ABCD and BCDA obviously both belong to the
        same cycle of length 4, but start at different points, so are initially
        not in the same cluster.
        We collect all those k points of a cycle of length k in a new clusterlist
        called newclusters.

    Args:
        f: our right hand side of the ODE
        croppedlist:    list of 2-dimensional elements [i,j] where i represents
                        the index of a cluster and j the index of a trajectory
                        belonging to this cluster.
                        This means, that if we have k clusters, then each
                        index 0,...k-1 is mapped to a single representative 
                        trajectory (with index j) belonging to this cluster.
        X:  3D array of our trajectories of solutions. X has dimensions 
            n*50*2, where we have n trajectories, the last 50 entries for each
            and the n- and p-components.
        tmax: the maximum length of cycles we should search for. Should be <= 50
        tol: the threshold for determining whether two points are equal or not.
             Needed when checking for cyclic points.
        aS: double the summer length (one of the bifurcation parameters)
        nu: generalist predator density dependence parameter (second bif. param)

    Return: the four different lists as well as the FTLE values and the threshold
        newclusters:    list with entries [i, x, per]
                        where i denotes the index of the new cluster that point
                        x (consisting of n and p component) belongs to.
                        This cluster i is a cycle of length per.
                        I.e. for a cycle of length per, we will have per entries
        chaosclusterlist:   List of points [n,p] that belong to chaos
        cyclelist:      List of points [n,p] that belong to a quasiperiodic orbit
        notsurelist:    List of points that might belong to a cycle of length 
                        > tmax
        chaosornot[1]: the resulting median of the FTLE calculation
        chaosornot[2]: the tolerance in the FTLE calculation
    """
    
    stilltocheck = croppedlist
    newclusters = []
    newclustercount = 0
    chaosclusterlist = []
    cyclelist = []
    notsurelist = []
    notsurecount = 0
    
    while stilltocheck:
        [ti, ci] = stilltocheck[0]
        #print('new item to check')
        #print([ti, ci])
        currentcheck = [[ti, ci]]
        #print(X[ti])
        for t in range(1, tmax):
            matching_traj = [tup for tup in stilltocheck if np.linalg.norm(X[ti][t] - X[tup[0]][0])<tol ]
            #print(matching_traj)
            if matching_traj:
                # have found our next value in our cycle
                if len(matching_traj)>1:
                    print('found multiple indices matching -> chaos or cycle')
                    for it in range(len(matching_traj)):
                        currentcheck.append(matching_traj[it])
                        for tnew in range(0, t):
                            chaosclusterlist.append(X[ti][tnew])
            
                    stilltocheck = [tup for tup in stilltocheck if tup not in currentcheck]
                    currentcheck = []
                    break
                else:
                    # we only have found one item in matching_traj
                    if abs(ti-matching_traj[0][0]) < 0.1: #ti == matching_traj[0][0]:
                        # NOTE: here we assume that we always only find one matching index
                        # we have found our starting point again, i.e. fulfilled the cycle
                        # our cycle has length t
                        for tnew in range(0, t):
                            newclusters.append([newclustercount, X[ti][tnew], t])
                            # add all t points on our trajectory to the list with the same new
                            # cluster ID and the length t of the cycle
                        newclustercount += 1
                        stilltocheck = [tup for tup in stilltocheck if tup not in currentcheck]
                        # remove all of the found points on the current trajectory from
                        # the stilltocheck list
                        currentcheck = []
                        break
                    else:
                        # have found another point (of a different former cluster) which lies
                        # on our trajectory but are not finished with the cycle
                        currentcheck.append(matching_traj[0])
                        if t == tmax-1:
                            # have reached the end of our checking interval, but are not
                            # finished with the cycle
                            #print('not finished with the cycle but found a new point')
                            for tnew in range(0, t):
                                notsurelist.append([notsurecount, X[ti][tnew], tmax + 0.1])
                                # we add all t points on our traejctory to the list with the same
                                # new cluster ID and the length tmax + 0.1 (which represents that
                                # we're not finished with the cycle)
                            notsurecount += 1
                            stilltocheck = [tup for tup in stilltocheck if tup not in currentcheck]
                            # remove all of the found points on the current trajectory from
                            # the stilltocheck list
                            currentcheck = []
            else:
                # have found a point on the trajectory which was not yet in our
                # list of representatives.
                # Could be that this is indeed a point on a multi-year cycle that we
                # didn't find before, but could also just be a point on a trajectory
                # belonging to 'chaos'
                # so: keep on searching unless we are at our tmax time
                if t == tmax-1:
                    # have not found a closed/periodic cycle. I.e. either chaos or an
                    # aperiodic cycle like in T_S \approx 0.76. For now, treat it as chaos
                    # and check later what it is.
                    # Here we assume, that chaos and a cycle cannot coexist.
                    for tnew in range(0, t):
                        chaosclusterlist.append(X[ti][tnew])
            
                    stilltocheck = [tup for tup in stilltocheck if tup not in currentcheck]
                    currentcheck = []
    
    if chaosclusterlist:
        # we have either chaos or cycles. Test which we have
        Xtest = chaosclusterlist[0]
        chaosornot = chaosorcycle(f, Xtest, 50, 10, aS, nu)
        if chaosornot[0] == 'cycle':
            print('cycle')
            cyclelist = chaosclusterlist
            chaosclusterlist = []
        else:
            print('chaos')
            # do not need to change anything
      
        # we now return the four different lists as well as the FTLE values and the threshold
        return newclusters, chaosclusterlist, cyclelist, notsurelist, chaosornot[1], chaosornot[2]
    else:
        # we didn't have chaos nor limit cycles, so chaosornot is not defined. Return empty arrays instead
        return newclusters, chaosclusterlist, cyclelist, notsurelist, [], []



def chaosorcycle(f, X_0, startyear, years, aS, nu, tol=1e-2):
    """ This is the function to distinguish between chaos and quasiperiodic
        orbits. 
        We calculate the FTLE for {startyears, startyears + 1, ... startyears + years - 1}
        and then take the arithmetic mean and the median of those values
        to distinguish between chaos and quasiperiodicity.

    Args:
        f: the ODE's of our system
        X_0: an initial condition for which we should check the FTLE
        startyear: the year we start our simulations from
        years:  how many FTLE calculation we should run (i.e. the last one
                runs for (startyear + years - 1) time)
        aS: double the summer length (one of the bifurcation parameters)
        nu: generalist predator density dependence parameter (second bif. param)
        tol: the tolerance that is used to distinguish between chaos and quasiper.

    Return: 3 things, namely:
        'chaos' or 'cycle': a text showing directly which type of behaviour we
                            have
        FTLE_result: the median of our FTLE values
        tol:    the tolerance that was used for distinguishing between chaos and
                quasiperiodicity
    """
    # we want to compute the FTLE of the point X_0 for years years, starting at
    # startyear and averaging over all the years. Then, if the average is below
    # tol, we have a cycle and else we have chaos
    # we return 'chaos' resp. 'cycle' as text
      
    FTLE_values = []
    n0 = X_0[0]
    p0 = X_0[0]
    t0 = 0.25*r
      
    for i in range(startyear+1, startyear+1+years):
        res = FTLE_new(f, np.array([[n0, p0, t0]]), 1e-5, aS, nu, i)
        #print(res)
        FTLE_values.append(res[0])
    
    FTLE_result = np.median(FTLE_values)
    FTLE_mean = np.mean(FTLE_values)
      
    print(FTLE_result, FTLE_mean, tol)
    current_directory = os.getcwd()
    #print(current_directory)
    nu_folder = os.path.join(current_directory, f"nu_{nu:.2f}")
    aS_folder = os.path.join(nu_folder, f"aS_{aS:.2f}")
    simulation_file_path = os.path.join(aS_folder, "FTLE_values.txt")
    with open(simulation_file_path, 'w') as filepath:
        filepath.write(f'starting at the {startyear+1}th year and simulating for {years} years\n')
        filepath.write(f'median: {FTLE_result}\n')
        filepath.write(f'mean: {FTLE_mean}\n')
        filepath.write(f'tolerance: {tol}\n')
    
    
    if FTLE_result < tol:
        return 'cycle', FTLE_result, tol
    else:
        return 'chaos', FTLE_result, tol



def FTLE_new(f, initial_conditions, delta, aS, nu, years=1, dt=0.01):
    """ The FTLE calculation that is used to determine between chaos and
        quasiperiodic orbits

    Args:
        f: the ODE's of our system
        initial_conditions: vector of initial conditions, where the FTLE should
                            be calculated at
        delta: the distance which we use for the finite difference approximation
        aS: double the summer length (one of the bifurcation parameters)
        nu: generalist predator density dependence parameter (second bif. param)
        years: number of years that the FTLE simulation should run for
        dt: the step size used when calculating the solutions

    Return: 
        lam: the FTLE value that was calculated
    """

    N = len(initial_conditions)
    #print(N)
    
    initial_vector = np.zeros((4*N, 3))
    deltas = delta
    for i in range(N):
        n0 = initial_conditions[i][0]
        p0 = initial_conditions[i][1]
        if n0-delta < 0 or p0-delta < 0:
            deltas = n0/100
            # making sure that the delta is not too big
            print('had to make the deltas smaller, as we are too close to the border')
    
    for i in range(N):
        n0 = initial_conditions[i][0]
        p0 = initial_conditions[i][1]
        t0 = initial_conditions[i][2]
        # sol1pos
        initial_vector[4*i][0] = n0 + deltas
        initial_vector[4*i][1] = p0
        initial_vector[4*i][2] = t0
        # sol1neg
        initial_vector[4*i+1][0] = n0 - deltas
        initial_vector[4*i+1][1] = p0
        initial_vector[4*i+1][2] = t0
        # sol2pos
        initial_vector[4*i+2][0] = n0
        initial_vector[4*i+2][1] = p0 + deltas
        initial_vector[4*i+2][2] = t0
        # sol2neg
        initial_vector[4*i+3][0] = n0
        initial_vector[4*i+3][1] = p0-deltas
        initial_vector[4*i+3][2] = t0
    
    # In order to be able to use the new parallelized RK4 method, I need to cut the last column of the initial vector (time)
    
    # Remove the last column
    initial_vector_reduced = initial_vector[:, :-1]
    sol_y = rk4solver(aS, nu, r, int(1/dt), years, initial_vector_reduced, f)
    
    # Now, we need to shift the shape again in order to be able to use it.
    sol_y_copy = np.transpose(sol_y, (0, 2, 1))
    
    
    # now calculating the matrix entries
    lam = []
    for i in range(int(sol_y.shape[0]/4)):
        
        # NEW
        sol1pos = sol_y_copy[4*i][-1]
        sol1neg = sol_y_copy[4*i+1][-1]
        sol2pos = sol_y_copy[4*i+2][-1]
        sol2neg = sol_y_copy[4*i+3][-1]
      
        M11 = (sol1pos[0] - sol1neg[0])/(2*np.abs(deltas))
        M12 = (sol2pos[0] - sol2neg[0])/(2*np.abs(deltas))
        M21 = (sol1pos[1] - sol1neg[1])/(2*np.abs(deltas))
        M22 = (sol2pos[1] - sol2neg[1])/(2*np.abs(deltas))
      
        # creating the matrix
        C = np.array([[M11*M11 + M21*M21, M11*M12 + M21*M22], [M12*M11 + M22*M21, M12*M12 + M22*M22]])
        ev = np.linalg.eig(C)
      
        # now the calculation
        if abs(ev.eigenvalues[0]) > abs(ev.eigenvalues[1]):
            lam.append(1/(2*years*r)*np.log(abs(ev.eigenvalues[0])))
        else:
            lam.append(1/(2*years*r)*np.log(abs(ev.eigenvalues[1])))
    
    return lam



def kmeans(X, k, max_iters=100):
    """ The K-means algorithm sorts the rows of the matrix X into k clusters,
        where the rows (i.e. corresponding to the trajectories) are being 
        grouped in an optimal way - i.e. lowest within-group distance and 
        high distance to the other clusters.

    Args:
        X:  2D matrix, coming from the eigenvalue problem in the clustering
            algorithm (it's called U there). X has dimensions n*k, this
            means that we have n eigenvectors of length k
        k:  number of clusters that should be constructed
        max_iters: maximal number of iterations to find the optimal clustering

    Return: 
        centroids:  vectors which represent the 'centers' of the clusters. 
                    They are of the same dimension as the rows of X
        clusters:   vector of length n, where the entries correspond to the cluster
                    that the corresponding rows of X (indices of the rows of X = 
                    indices of the original trajectories) belong to.
    """
    
    # Randomly initialize centroids ensuring they are unique within a tolerance
    centroids = np.zeros((k, X.shape[1]))
    chosen_indices = set()
    chosen_values = []
    tol = 1e-4
    for i in range(k):
        while True:
            index = np.random.choice(X.shape[0])
            value_tuple = tuple(X[index])
            if all(np.linalg.norm(np.array(value_tuple) - np.array(chosen_value)) > tol for chosen_value in chosen_values):
                chosen_indices.add(index)
                chosen_values.append(value_tuple)
                centroids[i] = X[index]
                break
    
    for _ in range(max_iters):
        # Assign clusters based on closest centroid
        clusters = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            min_dist = float('inf')
            for j in range(k):
                dist = np.linalg.norm(X[i] - centroids[j])
                if dist < min_dist:
                    min_dist = dist
                    clusters[i] = j
    
        # Calculate new centroids
        new_centroids = np.zeros((k, X.shape[1]))
        for j in range(k):
            points = [X[i] for i in range(X.shape[0]) if clusters[i] == j]
            if len(points) > 0:  # Check if there are points in the cluster
                # Calculate the mean of the points in the cluster
                sum_points = np.zeros(X.shape[1])
                for point in points:
                    sum_points += point
                new_centroids[j] = sum_points / len(points)
            else:
                new_centroids[j] = centroids[j]  # Keep the old centroid if no points are assigned
    
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
    
        centroids = new_centroids
    
    return centroids, clusters