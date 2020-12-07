# Function used for simulating state prep B in CeNTREX

#Import necessary packages:
import sys
import pickle
import numpy as np
from tqdm.notebook import tqdm
from scipy.linalg import expm
from scipy.linalg.lapack import zheevd
from functools import partial
import timeit


#Custom classes for defining molecular states and some convenience functions for them
sys.path.append('./molecular-state-classes-and-functions/')
from classes import UncoupledBasisState, CoupledBasisState, State
from functions import (make_hamiltonian, make_QN, ni_range, find_closest_state,
                       reorder_evecs, find_state_idx_from_state)

#Define the time integrator function
def simulate_SPB(R_t = None, #Position of molecule as function of time in XYZ coordinate system
                 T = None, #Total time for which molecule is simulated
                 E_R = None, B_R = None, #Spatial profiles of the static electric and magnetic fields
                 microwave_fields = None, #List of microwave fields for the simulation
                 initial_states = None, #Initial states for the system (can simulate multiple initial states with little additional cost)
                 final_state = None, #Desired final state after the state preparation
                 N_steps = int(1e4), #Number of timesteps for time-evolution
                 H_path = None, #Path to Hamiltonian if not the default
                 verbose = False, #Whether or not need to print a bunch of output
                 plots = False #Whether or not to save plots
                 ): 

    """
    Function that simulates state preparation B in CeNTREX.

    The structure of the cod is as follows:
    0. Define the position of the molecule as a function of time. Position is defined in
       the capitalized coordinate system(XYZ where Z is along the beamline). Helps with
       defining time-dependence of e.g. microwave intensities and electric and magnetic fields.
    
    1. Define electric and magnetic fields as a function of time. This is done based on 
       spatial profiles of the EM-fields. The coordinate system for the direction of the
       the fields is different from the coordinate system used for defining the spatial
       profile:
        - For defining spatial profile use XYZ (Z is along beamline)
        - For defining direction of fields use xyz where z is along the electric field
          in the Main Interaction region
       For electric fields should use V/cm and for B-fields G.

    2. Define slowly time-varying Hamiltonian. This is the Hamiltonian due to the internal
       dynamics of the molecule and the external DC electric and magnetic fields
       (maybe split into internal Hamiltonian and EM-field Hamiltonian?)

    3. Define the microwave Hamiltonian. Simply the couplings due to the microwaves.
       Rotating wave approximation is applied and the rotating frame is used.

    4. Time-evolve the system by using matrix exponentiation for solving Schroedinger eqn.

    5. Make plots etc. if needed
    """
    ### 0. Position as function of time
    #Check if position as function of time was provided, otherwise use a default trajectory
    if R_t is None:
        def molecule_position(t, r0, v):
            """
            Functions that returns position of molecule at a given time for given initial position and velocity.
            inputs:
            t = time in seconds
            r0 = position of molecule at t = 0 in meters
            v = velocity of molecule in meters per second
            
            returns:
            r = position of molecule in metres
            """
            r =  r0 + v*t
            
            return r

        #Define the position of the molecule as a function of time
        r0 = np.array((0,0,-100e-3))
        v = np.array((0,0,200))
        R_t = lambda t: molecule_position(t, r0, v)

        #Define the total time for which the molecule is simulated
        z0 = r0[2]
        vz = v[2]
        T = np.abs(2*z0/vz)


    ### 1. Electric and magnetic fields as function of time
    #Here we determine the E- and B-fields as function of time based on provided spatial
    #profiles of the fields
    
    #Electric field:
    if E_R is not None:
        E_t = lambda t: E_R(R_t(t))

    #If spatial profile is not provided, assume a uniform 50 V/cm along z
    else:
        E_t = lambda t: np.array((0,0,50.))

    #Magnetic field:
    if B_R is not None:
        B_t = lambda t: B_R(R_t(t))

    #If spatial profile is not provided assume uniform 0.001 G along z
    else:
        B_t = lambda t: np.array((0,0,0.001))

    
    ### 2. Slowly time-varying Hamiltonian
    #Make list of quantum numbers that defines the basis for the matrices
    QN = make_QN(0,3,1/2,1/2)
    dim = len(QN)

    #Get H_0 from file (H_0 should be in rad/s)
    if H_path is None:
        H_0_EB = make_hamiltonian("./hamiltonians/TlF_X_state_hamiltonian0to3_2020_03_03.py")
    else:
        H_0_EB = make_hamiltonian(H_path)

    #Calculate the field free Hamiltonian
    H_0 = H_0_EB(np.array((0,0,0)), np.array((0,0,0)))

    #Calculate part of the Hamiltonian that is due to electromagnetic fields
    H_EB_t = lambda t: H_0_EB(E_t(t), B_t(t)) - H_0

    #Check that H_0 is hermitian
    H_test = H_0 + H_EB_t(T/2)
    is_herm = np.allclose(H_test,np.conj(H_test.T))
    if not is_herm:
        print("H_0 is non-hermitian!")


    ### 3. Microwave Hamiltonians
    #Generate the part of the Hamiltonian that describes the effect of the microwave
    #field(s)
    if microwave_fields is not None:
        counter = 0
        #Containers for microwave coupling Hamiltonians and the microwave frequencies
        microwave_couplings = []
        omegas = []
        D_mu = np.zeros(H_0.shape)
        for microwave_field in microwave_fields:
            if verbose:
                counter +=1
                print("\nMicrowave {:}:".format(counter))
            #Find out where the peak intensity of the field is located 
            Z_peak = microwave_field.z0
            R_peak = np.array((0,0,Z_peak))

            #Determine the Hamiltonian at z0
            H_peak = H_0_EB(E_R(R_peak), B_R(R_peak))

            #Find the exact ground and excited states for the field at z_peak
            microwave_field.find_closest_eigenstates(H_peak, QN)

            #Calculate angular part of matrix element for main transition
            ME_main = microwave_field.calculate_ME_main(pol_vec =microwave_field.p_t(0)) #Angular part of ME for main transition

            #Find the coupling matrices due to the microwave
            H_list = microwave_field.generate_couplings(QN)
            H_list = [H/ME_main for H in H_list]
            Hu_list = [np.triu(H) for H in H_list]
            Hl_list = [np.tril(H) for H in H_list]

            #Find some necessary parameters and then define the coupling matrix as a function of time
            Omega_t = microwave_field.find_Omega_t(R_t) #Time dependence of Rabi rate
            p_t = microwave_field.p_t #Time dependence of polarization of field
            omega = np.abs(microwave_field.calculate_frequency(H_peak, QN)) #Calculate frequency of transition
            omegas.append(omega)
            D_mu += microwave_field.generate_D(np.sum(omegas), H_peak, QN) #Matrix that shifts energies for rotating frame
    
            #Define the coupling matrix as function of time
            def H_mu_t_func(Hu_list, Hl_list, p_t, Omega_t, t):
                return (1/2*Omega_t(t)
                        *(Hu_list[0]*p_t(t)[0] + Hu_list[1]*p_t(t)[1] + Hu_list[2]*p_t(t)[2]
                          + Hl_list[0]*p_t(t)[0].conj() + Hl_list[1]*p_t(t)[1].conj() + Hl_list[2]*p_t(t)[2].conj()))

            H_mu_t = partial(H_mu_t_func, Hu_list, Hl_list, p_t, Omega_t)
            microwave_couplings.append(H_mu_t)

            #Print output for checks
            if verbose:
                print("ME_main = {:.3E}".format(ME_main))
                print("muW frequency: {:.9E} GHz".format(omega/(2*np.pi*1e9)))
                print("ground_main:")
                ground_main = microwave_field.ground_main
                ground_main.print_state()
                print("excited_main:")
                excited_main = microwave_field.excited_main
                excited_main.print_state()
                ME = excited_main.state_vector(QN).conj().T @ H_mu_t(T/2) @ ground_main.state_vector(QN)
                print("Omega_T/2 = {:.3E}".format(ME/(2*np.pi)))


        #Generate function that gives couplings due to all microwaves
        def H_mu_tot_t(t):
            H_mu_tot = microwave_couplings[0](t)
            if len(microwave_couplings) > 1:
                for H_mu_t in microwave_couplings[1:]:
                    H_mu_tot = H_mu_tot + H_mu_t(t)
            return H_mu_tot

    else:
        H_mu_tot_t = lambda t: np.zeros(H_0.shape)
    
    if verbose:
        time = timeit.timeit("H_mu_tot_t(T/2)", number = 10, globals = locals())/10
        print("Time to generate H_mu_tot_t: {:.3e} s".format(time))
        H_test = H_mu_tot_t(T/2)
        non_zero = H_test[np.abs(H_test) > 0].shape[0]
        print("Non-zero elements at T/2: {}".format(non_zero))
        print("Rotating frame energy shifts:")
        print(np.diag(D_mu)/(2*np.pi*1e9))
        print(np.max(np.abs(H_mu_tot_t(T/2)))/(2*np.pi))


    ### 4. Time-evolution
    #Now time-evolve the system by repeatedly applying the time-evolution operator with small 
    # timestep
    
    #Make array of times at which time-evolution is performed
    t_array = np.linspace(0,T,N_steps)

    #Calculate timestep
    dt = T/N_steps

    #Set the state vector to the initial state
    H_t0 = H_0+H_EB_t(0)
    psis = np.zeros((len(initial_states), H_t0.shape[0]), dtype = complex)
    ini_index = np.zeros(len(initial_states))
    for i, initial_state in enumerate(initial_states):
        initial_state = find_closest_state(H_t0, initial_state, QN)
        psis[i,:] = initial_state.state_vector(QN)
        #Also figure out the state indices that correspond to the initial states
        ini_index[i] = find_state_idx_from_state(H_t0, initial_state, QN)

    #Set up containers to store results
    state_probabilities = np.zeros((len(initial_states),len(t_array),H_t0.shape[0]))
    state_energies = np.zeros((len(t_array),H_t0.shape[0]))
    psi_t = np.zeros((len(initial_states), len(t_array),H_t0.shape[0]),dtype = complex)

    #Initialize the reference matrix of eigenvectors
    E_ref, V_ref = np.linalg.eigh(H_t0)
    E_ref_0 = E_ref
    V_ref_0 = V_ref

    #Loop over timesteps to evolve system in time
    for i, t in enumerate(tqdm(t_array)):
        #Calculate the necessary Hamiltonians at this time
        H_t = H_0+H_EB_t(t)
        H_mu = H_mu_tot_t(t)
        
        #Diagonalize H_t and transform to that basis
        D_t, V_t, info_t = zheevd(H_t)
        if info_t !=0:
            print("zheevd didn't work for H_0")
            D_t, V_t = np.linalg.eigh(H_t)

        #Make intermediate hamiltonian by transforming H to the basis where H_0 is diagonal
        H_I = V_t.conj().T @ H_t @ V_t    
        
        #Sort the eigenvalues so they are in ascending order
        index = np.argsort(D_t)
        D_t = D_t[index]
        V_t = V_t[:,index]
        
        #Find the microwave coupling matrix in the new basis:
        H_mu = V_t.conj().T @ H_mu @ V_t
        
        #Make total intermediate hamiltonian
        H_I = H_I + H_mu
                
        #Transform H_I to the rotating basis
        H_R = H_I + D_mu
        
        #Diagonalize H_R
        D_R, V_R, info_R = zheevd(H_R)
        if info_R !=0:
            print("zheevd didn't work for H_R")
            D_R, V_R = np.linalg.eigh(H_R)

        #Reorder the eigenstates so that a given index corresponds to the same "adiabatic" state
        energies, evecs = D_t, V_t
        energies, evecs = reorder_evecs(evecs,energies,V_ref)
        
        #Propagate state vector in time
        propagator = V_t @ V_R @ np.diag(np.exp(-1j*D_R*dt)) @ V_R.conj().T @ V_t.conj().T
        for j in range(0,len(initial_states)):
            psis[j,:] = propagator @ psis[j,:]
                    
            #Calculate the overlap of the state vector with all of the eigenvectors of the Hamiltonian
            overlaps = np.dot(np.conj(psis[j,:]).T,evecs)
            probabilities = overlaps*np.conj(overlaps)
            
            #Store the probabilities, energies and the state vector
            state_probabilities[j,i,:] = np.real(probabilities)
            psi_t[j,i,:] = psis[j,:].T

        state_energies[i,:] = energies
        V_ref = evecs

    #Find the exact form of the desired final state and calculate overlap with state vector
    final_state = find_closest_state(H_0+H_EB_t(T), final_state, QN)
    final_state_vec = final_state.state_vector(QN)
    
    if len(initial_states) == 1:
        overlap = final_state_vec.conj().T @ psis[0,:]
        probability = np.abs(overlap)**2

        if verbose:
            print("Probability to end up in desired final state: {:.3f}".format(probability))
            print("desired final state|fin> = ")
            final_state.print_state()

    return t_array, state_probabilities, V_ref, QN



